use super::data_source::DataSource;
use super::embedding::Embedding;
use super::tensor::{FromPiecesDirection, Tensor, TensorDType};
#[cfg(feature = "opencl")]
use super::tensor_opencl_support::OpenCL;
use super::tokenizer::TokenId;

use super::unpickler::UnpicklingError;
use indicatif::ProgressBar;
use num_complex::Complex;
use rayon::prelude::*;

use std::sync::{Arc, OnceLock, RwLock};

type FreqsCis = Vec<Vec<Complex<f64>>>;

#[allow(dead_code)]
pub struct Transformer {
    freqs_cis: FreqsCis,
    emb: Embedding,
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,

    norm: RMSNorm,
    output: Tensor,

    layers: Vec<TransformerBlock>,

    data_settings: DataSettings,
}

// Clone is cheap
#[derive(Clone)]
pub struct DataSettings {
    #[cfg(feature = "opencl")]
    percentage_to_gpu: f32,
    #[cfg(feature = "opencl")]
    use_opencl_for_feedforward: bool,
    #[cfg(feature = "opencl")]
    use_opencl_for_attention: bool,
    #[cfg(feature = "opencl")]
    cl: Option<OpenCL>,

    force_f16: bool,
}

// OpenCL is safe to send to threads but Rust doesn't know that
unsafe impl Send for DataSettings {}
unsafe impl Sync for DataSettings {}

impl DataSettings {
    #[cfg(feature = "opencl")]
    pub fn new(cl: Option<OpenCL>) -> Self {
        DataSettings {
            use_opencl_for_feedforward: false,
            use_opencl_for_attention: false,
            force_f16: false,
            percentage_to_gpu: 1.0,
            cl: cl.clone(),
        }
    }

    #[allow(clippy::new_without_default)]
    #[cfg(not(feature = "opencl"))]
    pub fn new() -> Self {
        DataSettings {
            force_f16: false,
        }
    }

    #[cfg(feature = "opencl")]
    pub fn use_opencl(mut self) -> DataSettings {
        if self.cl.is_none() {
            panic!("OpenCL is not available, cannot call use_opencl() on DataSettings.");
        }
        self.use_opencl_for_feedforward = true;
        self.use_opencl_for_attention = true;
        self
    }

    #[cfg(feature = "opencl")]
    pub fn dont_use_opencl(mut self) -> DataSettings {
        self.use_opencl_for_feedforward = false;
        self.use_opencl_for_attention = false;
        self
    }

    #[cfg(feature = "opencl")]
    pub fn percentage_to_gpu(mut self, percentage: f32) -> DataSettings {
        self.percentage_to_gpu = percentage;
        if self.percentage_to_gpu >= 1.0 {
            self.percentage_to_gpu = 1.0;
        }
        if self.percentage_to_gpu < 0.0 {
            self.percentage_to_gpu = 0.0;
        }
        if self.percentage_to_gpu.is_nan() {
            self.percentage_to_gpu = 0.0;
        }
        self
    }

    pub fn force_f16(mut self) -> DataSettings {
        self.force_f16 = true;
        self
    }

}

pub struct TransformerCaches {
    layer_caches: Vec<AttentionCache>,
}

pub struct TransformerBlock {
    feed_forward: FeedForward,
    attn: Attention,
    ffn_norm: RMSNorm,
    attention_norm: RMSNorm,
}

pub struct AttentionCache {
    cache_k: Vec<Arc<RwLock<Tensor>>>,
    cache_v: Vec<Arc<RwLock<Tensor>>>,
    data_settings: DataSettings,
}

impl AttentionCache {
    fn new(
        max_seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        data_settings: &DataSettings,
    ) -> Self {
        let mut cache_k = Vec::with_capacity(n_kv_heads);
        let mut cache_v = Vec::with_capacity(n_kv_heads);

        let dtype = if data_settings.force_f16 {
            TensorDType::Float16
        } else {
            TensorDType::Float32
        };
        for _ in 0..n_kv_heads {
            cache_k.push(Arc::new(RwLock::new(Tensor::zeros(
                head_dim as i64,
                max_seq_len as i64,
                dtype,
            ))));
            cache_v.push(Arc::new(RwLock::new(Tensor::zeros(
                head_dim as i64,
                max_seq_len as i64,
                dtype,
            ))));
        }
        AttentionCache {
            cache_k,
            cache_v,
            data_settings: data_settings.clone(),
        }
    }

    /// Cloning AttentionCache normally just makes new references to the same cache.
    /// This creates a true clone with copied tensors.
    fn true_clone(&self) -> AttentionCache {
        let mut cache_k = Vec::with_capacity(self.cache_k.len());
        let mut cache_v = Vec::with_capacity(self.cache_v.len());
        for idx in 0..self.cache_k.len() {
            let old_k = self.cache_k[idx].read().unwrap();
            cache_k.push(Arc::new(RwLock::new(old_k.clone())));
            let old_v = self.cache_v[idx].read().unwrap();
            cache_v.push(Arc::new(RwLock::new(old_v.clone())));
        }
        AttentionCache {
            cache_k,
            cache_v,
            data_settings: self.data_settings.clone(),
        }
    }

    fn shift_left(&mut self, shifts: usize) {
        for _ in 0..shifts {
            for idx in 0..self.cache_k.len() {
                let mut k = self.cache_k[idx].write().unwrap();
                let mut v = self.cache_v[idx].write().unwrap();
                let k_rows = k.rows();
                let k_cols = k.cols();
                for head_idx in 0..k_rows {
                    for seq_idx in 0..k_cols - 1 {
                        let kval = k.get_f32(head_idx, seq_idx + 1);
                        let vval = v.get_f32(head_idx, seq_idx + 1);
                        k.set_f32(head_idx, seq_idx, kval);
                        v.set_f32(head_idx, seq_idx, vval);
                    }
                }
            }
        }
    }
}

impl TransformerCaches {
    pub fn shift_left(&mut self, shifts: usize) {
        for layer in self.layer_caches.iter_mut() {
            layer.shift_left(shifts);
        }
    }

    pub fn true_clone(&self) -> TransformerCaches {
        let mut layer_caches = Vec::with_capacity(self.layer_caches.len());
        for layer in self.layer_caches.iter() {
            layer_caches.push(layer.true_clone());
        }
        TransformerCaches { layer_caches }
    }
}

pub struct RMSNorm {
    eps: f64,
    weight: Tensor,
}

#[allow(dead_code)]
pub struct Attention {
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    n_local_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_norm_weight: Option<Tensor>,
    k_norm_weight: Option<Tensor>,
    eps: f64,
    data_settings: DataSettings,
    #[cfg(feature = "opencl")]
    freqs_gpu: OnceLock<(Tensor, Tensor)>,
}

#[allow(dead_code)]
pub struct FeedForward {
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
    data_settings: DataSettings,
}

impl Transformer {
    #[allow(clippy::too_many_arguments)]
    pub fn from_unpickled(
        emb: Embedding,
        dim: usize,
        n_layers: usize,
        n_heads: usize,
        max_seq_len: usize,
        eps: f64,
        data_settings: DataSettings,
        data_source: DataSource,
        head_dim: Option<usize>,
        n_kv_heads: Option<usize>,
        rope_theta: Option<f64>,
    ) -> Result<Transformer, UnpicklingError> {
        let head_dim = head_dim.unwrap_or_else(|| dim / n_heads);
        let n_kv_heads = n_kv_heads.unwrap_or(n_heads);
        let n_local_heads = n_heads;
        let rope_theta = rope_theta.unwrap_or(10000.0);

        let progress_bar = ProgressBar::new(n_layers as u64);
        let layers: Vec<TransformerBlock> = (0..n_layers)
            .into_par_iter()
            .map(|layer_id| {
                let data_settings = {
                    #[cfg(feature = "opencl")]
                    {
                        let max_layers = n_layers;
                        let last_layer_on_gpu = (data_settings.percentage_to_gpu
                            * (max_layers - 1) as f32)
                            .round() as usize;
                        if layer_id > last_layer_on_gpu {
                            data_settings.clone().dont_use_opencl()
                        } else {
                            data_settings.clone()
                        }
                    }
                    #[cfg(not(feature = "opencl"))]
                    {
                        data_settings.clone()
                    }
                };

                let result = TransformerBlock::from_unpickled(
                    layer_id,
                    eps,
                    n_local_heads,
                    n_kv_heads,
                    head_dim,
                    dim,
                    data_settings,
                    data_source.clone(),
                );
                progress_bar.inc(1);
                result
            })
            .collect::<Result<Vec<TransformerBlock>, UnpicklingError>>()?;
        std::mem::drop(progress_bar);

        let norm = RMSNorm::from_unpickled(
            "norm.weight".to_string(),
            "model.norm.weight".to_string(),
            eps,
            data_source.clone(),
        )?;
        let output = Tensor::from_unpickled_pieces1(
            "output.weight",
            data_source.clone(),
            FromPiecesDirection::Rows,
        )
        .or_else(|_| {
            Tensor::from_unpickled_pieces1(
                "lm_head.weight",
                data_source.clone(),
                FromPiecesDirection::Rows,
            )
        })
        .or_else(|_| {
            Tensor::from_unpickled_pieces1(
                "model.embed_tokens.weight",
                data_source.clone(),
                FromPiecesDirection::Rows,
            )
        })?
        .to_f32();

        Ok(Transformer {
            freqs_cis: compute_freqs_cis(head_dim, max_seq_len, rope_theta),
            data_settings: data_settings.clone(),
            emb,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            max_seq_len,
            head_dim,

            norm,
            output,

            layers,
        })
    }

    pub fn make_caches(&self) -> TransformerCaches {
        let mut result = vec![];
        for _ in 0..self.n_layers {
            result.push(AttentionCache::new(
                self.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
                &self.data_settings,
            ));
        }
        TransformerCaches {
            layer_caches: result,
        }
    }

    pub fn forward(
        &self,
        tokens: &[TokenId],
        start_pos: usize,
        caches: &mut TransformerCaches,
    ) -> Tensor {
        assert!(caches.layer_caches.len() == self.n_layers);
        let mask: Option<Tensor> = if tokens.len() > 1 {
            Some(Tensor::full_triu(
                tokens.len() as i64,
                tokens.len() as i64,
                start_pos as i64 + 1,
                TensorDType::Float32,
                std::f32::NEG_INFINITY,
            ))
        } else {
            None
        };
        let mut embs: Vec<&Tensor> = Vec::with_capacity(tokens.len());
        for token in tokens.iter() {
            let emb = self.emb.get_embedding(*token as usize);
            embs.push(emb);
        }
        let mut emb_tensor: Tensor = Tensor::concat(&embs);
        std::mem::drop(embs);

        for (idx, layer) in self.layers.iter().enumerate() {
            emb_tensor = layer.forward(
                &emb_tensor,
                start_pos,
                &self.freqs_cis,
                &mask,
                &mut caches.layer_caches[idx],
            );
        }

        let out = self.norm.forward(&emb_tensor, "final");
        let out = out.row(out.rows() - 1);

        let logits = self.output.matrix_mul_transposed(&out);
        logits
    }
}

impl TransformerBlock {
    pub fn from_unpickled(
        layer_id: usize,
        eps: f64,
        n_local_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dim: usize,
        data_settings: DataSettings,
        data_source: DataSource,
    ) -> Result<Self, UnpicklingError> {
        let ff = FeedForward::from_unpickled(layer_id, data_source.clone(), data_settings.clone())?;
        let attn = Attention::from_unpickled(
            layer_id,
            n_local_heads,
            n_kv_heads,
            head_dim,
            dim,
            eps,
            data_settings,
            data_source.clone(),
        )?;
        let ffn_norm = RMSNorm::from_unpickled(
            format!("layers.{}.ffn_norm.weight", layer_id),
            format!("model.layers.{}.post_attention_layernorm.weight", layer_id),
            eps,
            data_source.clone(),
        )?;
        let attn_norm = RMSNorm::from_unpickled(
            format!("layers.{}.attention_norm.weight", layer_id),
            format!("model.layers.{}.input_layernorm.weight", layer_id),
            eps,
            data_source,
        )?;
        Ok(Self {
            feed_forward: ff,
            attn,
            ffn_norm,
            attention_norm: attn_norm,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        freqs_cis: &FreqsCis,
        mask: &Option<Tensor>,
        attention_cache: &mut AttentionCache,
    ) -> Tensor {
        let mut attnorm_out = self.attention_norm.forward(x, "attn");
        let att_out = self.attn.forward(
            &mut attnorm_out,
            start_pos,
            freqs_cis,
            mask,
            attention_cache,
        );
        std::mem::drop(attnorm_out);

        let h = x.add(&att_out);
        let mut att_out = self.ffn_norm.forward(&h, "ffn");
        let att_out = self.feed_forward.forward(&mut att_out).transpose();
        h.add(&att_out)
    }
}

impl RMSNorm {
    pub fn from_unpickled(
        name: String,
        name2: String,
        eps: f64,
        data_source: DataSource,
    ) -> Result<RMSNorm, UnpicklingError> {
        let weights = match Tensor::from_unpickled_pieces1(
            name,
            data_source.clone(),
            FromPiecesDirection::Rows,
        ) {
            Ok(w) => w,
            Err(_) => Tensor::from_unpickled_pieces1(
                name2,
                data_source.clone(),
                FromPiecesDirection::Rows,
            )?,
        };
        let weights = weights.to_f32();

        Ok(Self {
            eps,
            weight: weights,
        })
    }

    fn forward(&self, x: &Tensor, label: &str) -> Tensor {
        let inner = x.pow(2.0).mean_cols().add_scalar(self.eps as f32);
        let out1 = x.scalar_multiply_broadcast(&inner.rsqrt());
        let weight = &self.weight;
        if out1.cols() != weight.cols() {
            panic!(
                "RMSNorm[{}] shape mismatch: input {}x{} vs weight {}x{}",
                label,
                out1.rows(),
                out1.cols(),
                weight.rows(),
                weight.cols()
            );
        }
        out1.hadamard_product_broadcast(weight)
    }
}

impl FeedForward {
    pub fn from_unpickled(
        layer_id: usize,
        data_source: DataSource,
        data_settings: DataSettings,
    ) -> Result<FeedForward, UnpicklingError> {
        let mut w1 = Tensor::from_unpickled_pieces2(
            format!("layers.{}.feed_forward.w1.weight", layer_id),
            format!("model.layers.{}.mlp.gate_proj.weight", layer_id),
            data_source.clone(),
            FromPiecesDirection::Rows,
        )?;
        let mut w2 = Tensor::from_unpickled_pieces2(
            format!("layers.{}.feed_forward.w2.weight", layer_id),
            format!("model.layers.{}.mlp.down_proj.weight", layer_id),
            data_source.clone(),
            FromPiecesDirection::Cols,
        )?;
        let mut w3 = Tensor::from_unpickled_pieces2(
            format!("layers.{}.feed_forward.w3.weight", layer_id),
            format!("model.layers.{}.mlp.up_proj.weight", layer_id),
            data_source.clone(),
            FromPiecesDirection::Rows,
        )?;

        if data_settings.force_f16 {
            w1 = w1.to_f16();
            w2 = w2.to_f16();
            w3 = w3.to_f16();
        }

        #[cfg(feature = "opencl")]
        {
            if data_settings.use_opencl_for_feedforward {
                w1 = w1.to_f16();
                w2 = w2.to_f16();
                w3 = w3.to_f16();
                let ds = data_settings.clone();
                w1.to_gpu_inplace(&ds.cl.as_ref().unwrap().clone()).unwrap();
                w2.to_gpu_inplace(&ds.cl.as_ref().unwrap().clone()).unwrap();
                w3.to_gpu_inplace(&ds.cl.unwrap()).unwrap();
            }
        }
        // w1, w2, w3 maybe be f32 or f16 depending on source data.

        Ok(Self {
            w1,
            w2,
            w3,
            data_settings,
        })
    }

    pub fn forward(&self, x: &mut Tensor) -> Tensor {
        let _original_x_dtype = x.dtype();
        if x.dtype() != self.w1.dtype() {
            *x = x.to_same_type(&self.w1);
        }
        #[cfg(feature = "opencl")]
        let x_was_on_cpu: bool;
        #[cfg(feature = "opencl")]
        {
            x_was_on_cpu = x.is_on_cpu();
            if self.data_settings.use_opencl_for_feedforward {
                x.to_gpu_inplace(self.data_settings.cl.as_ref().unwrap())
                    .unwrap();
            }
        }
        let (mut w1_out, mut w3_out) = rayon::join(
            || self.w1.matrix_mul_transposed(x),
            || self.w3.matrix_mul_transposed(x),
        );

        // Float16 not supported for some of these ops on CPU.
        if w1_out.is_on_cpu() && w1_out.dtype() == TensorDType::Float16 {
            w1_out = w1_out.to_f32();
            w3_out = w3_out.to_f32();
        }
        let w1_out = w1_out.silu();
        let mut w1w3_out = w1_out.hadamard_product(&w3_out).transpose();
        if w1w3_out.dtype() != self.w2.dtype() {
            w1w3_out = w1w3_out.to_same_type(&self.w2);
        }
        #[cfg(not(feature = "opencl"))]
        {
            self.w2
                .matrix_mul_transposed(&w1w3_out)
                .into_dtype(_original_x_dtype)
        }
        #[cfg(feature = "opencl")]
        {
            let mut result = self.w2.matrix_mul_transposed(&w1w3_out);
            if x_was_on_cpu {
                result.to_cpu_inplace().unwrap();
                result
            } else {
                result
            }
        }
    }
}

impl Attention {
    pub fn from_unpickled(
        layer_id: usize,
        n_local_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dim: usize,
        eps: f64,
        data_settings: DataSettings,
        data_source: DataSource,
    ) -> Result<Attention, UnpicklingError> {
        let mut wq = Tensor::from_unpickled_pieces2(
            format!("layers.{}.attention.wq.weight", layer_id),
            format!("model.layers.{}.self_attn.q_proj.weight", layer_id),
            data_source.clone(),
            FromPiecesDirection::Rows,
        )?;
        let mut wk = Tensor::from_unpickled_pieces2(
            format!("layers.{}.attention.wk.weight", layer_id),
            format!("model.layers.{}.self_attn.k_proj.weight", layer_id),
            data_source.clone(),
            FromPiecesDirection::Rows,
        )?;
        let mut wv = Tensor::from_unpickled_pieces2(
            format!("layers.{}.attention.wv.weight", layer_id),
            format!("model.layers.{}.self_attn.v_proj.weight", layer_id),
            data_source.clone(),
            FromPiecesDirection::Rows,
        )?;
        let mut wo = Tensor::from_unpickled_pieces2(
            format!("layers.{}.attention.wo.weight", layer_id),
            format!("model.layers.{}.self_attn.o_proj.weight", layer_id),
            data_source.clone(),
            FromPiecesDirection::Cols,
        )?;

        if data_source.need_to_do_antitranspose() {
            wq = wq.huggingface_llama_model_antitranspose(n_local_heads, dim);
            wk = wk.huggingface_llama_model_antitranspose(n_local_heads, dim);
        }

        if data_settings.force_f16 {
            wq = wq.to_f16();
            wk = wk.to_f16();
            wv = wv.to_f16();
            wo = wo.to_f16();
        }

        #[cfg(feature = "opencl")]
        {
            if data_settings.use_opencl_for_attention {
                wq = wq.to_f16();
                wk = wk.to_f16();
                wv = wv.to_f16();
                wo = wo.to_f16();
                let ds = data_settings.clone();
                wq.to_gpu_inplace(&ds.cl.as_ref().unwrap().clone()).unwrap();
                wk.to_gpu_inplace(&ds.cl.as_ref().unwrap().clone()).unwrap();
                wv.to_gpu_inplace(&ds.cl.as_ref().unwrap().clone()).unwrap();
                wo.to_gpu_inplace(&ds.cl.unwrap()).unwrap();
            }
        }

        let q_norm_weight = Tensor::from_unpickled_pieces1(
            format!("model.layers.{}.self_attn.q_norm.weight", layer_id),
            data_source.clone(),
            FromPiecesDirection::Rows,
        ).ok().map(|t| t.to_f32());
        let k_norm_weight = Tensor::from_unpickled_pieces1(
            format!("model.layers.{}.self_attn.k_norm.weight", layer_id),
            data_source.clone(),
            FromPiecesDirection::Rows,
        ).ok().map(|t| t.to_f32());

        let q_bias = Tensor::from_unpickled_pieces1(
            format!("model.layers.{}.self_attn.q_proj.bias", layer_id),
            data_source.clone(),
            FromPiecesDirection::Rows,
        ).ok().map(|t| t.to_f32());
        let k_bias = Tensor::from_unpickled_pieces1(
            format!("model.layers.{}.self_attn.k_proj.bias", layer_id),
            data_source.clone(),
            FromPiecesDirection::Rows,
        ).ok().map(|t| t.to_f32());
        let v_bias = Tensor::from_unpickled_pieces1(
            format!("model.layers.{}.self_attn.v_proj.bias", layer_id),
            data_source.clone(),
            FromPiecesDirection::Rows,
        ).ok().map(|t| t.to_f32());

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_bias,
            k_bias,
            v_bias,
            n_local_heads,
            n_kv_heads,
            head_dim,
            q_norm_weight,
            k_norm_weight,
            eps,
            data_settings,
            #[cfg(feature = "opencl")]
            freqs_gpu: OnceLock::new(),
        })
    }

    fn forward(
        &self,
        x: &mut Tensor,
        start_pos: usize,
        freqs_cis: &FreqsCis,
        mask: &Option<Tensor>,
        attention_cache: &mut AttentionCache,
    ) -> Tensor {
        let original_x_dtype = x.dtype();
        if x.dtype() != self.wq.dtype() {
            *x = x.to_same_type(&self.wq);
        }

        #[cfg(feature = "opencl")]
        {
            if self.data_settings.use_opencl_for_attention {
                x.to_gpu_inplace(self.data_settings.cl.as_ref().unwrap())
                    .unwrap();
            }
        }

        let seq_len = x.rows();
        #[cfg(feature = "opencl")]
        let (mut xq_out, mut xk_out, mut xv_out) = {
            let mut xq_out = x.matrix_mul_transposed(&self.wq);
            let mut xk_out = x.matrix_mul_transposed(&self.wk);
            let mut xv_out = x.matrix_mul_transposed(&self.wv);

            if self.data_settings.use_opencl_for_attention {
                // Apply per-head RMSNorm on GPU (avoids GPU→CPU→GPU round-trips)
                let cl = self.data_settings.cl.as_ref().unwrap();
                if let Some(ref q_norm) = self.q_norm_weight {
                    let mut q_norm_f16 = q_norm.to_f16();
                    q_norm_f16.to_gpu_inplace(cl).unwrap();
                    xq_out.rms_norm_gpu(
                        &q_norm_f16,
                        self.n_local_heads as i32,
                        self.head_dim as i32,
                        self.eps as f32,
                    );
                }
                if let Some(ref k_norm) = self.k_norm_weight {
                    let mut k_norm_f16 = k_norm.to_f16();
                    k_norm_f16.to_gpu_inplace(cl).unwrap();
                    xk_out.rms_norm_gpu(
                        &k_norm_f16,
                        self.n_kv_heads as i32,
                        self.head_dim as i32,
                        self.eps as f32,
                    );
                }

                // Apply RoPE on GPU (frequency tensors cached in OnceLock)
                let half = self.head_dim / 2;
                let max_seq_len = freqs_cis.len();
                let group_size = (self.n_local_heads / self.n_kv_heads) as i32;
                let freqs = self.freqs_gpu.get_or_init(|| {
                    let cl_init = self.data_settings.cl.as_ref().unwrap().clone();
                    let mut cos_t = Tensor::zeros(max_seq_len as i64, half as i64, TensorDType::Float16);
                    let mut sin_t = Tensor::zeros(max_seq_len as i64, half as i64, TensorDType::Float16);
                    for pos in 0..max_seq_len {
                        for col in 0..half {
                            let c = freqs_cis[pos][col];
                            cos_t.set_f32(pos as i64, col as i64, c.re as f32);
                            sin_t.set_f32(pos as i64, col as i64, c.im as f32);
                        }
                    }
                    cos_t.to_gpu_inplace(&cl_init).unwrap();
                    sin_t.to_gpu_inplace(&cl_init).unwrap();
                    (cos_t, sin_t)
                });
                xq_out.rope_gpu(
                    &mut xk_out,
                    &freqs.0,
                    &freqs.1,
                    self.n_local_heads as i32,
                    self.head_dim as i32,
                    group_size,
                    start_pos as i32,
                );
            }

            xq_out.to_cpu_inplace().unwrap();
            xk_out.to_cpu_inplace().unwrap();
            xv_out.to_cpu_inplace().unwrap();
            (xq_out.to_f32(), xk_out.to_f32(), xv_out.to_f32())
        };

        #[cfg(not(feature = "opencl"))]
        let (mut xq_out, (mut xk_out, mut xv_out)) = rayon::join(
            || x.matrix_mul_transposed(&self.wq).to_f32(),
            || {
                rayon::join(
                    || x.matrix_mul_transposed(&self.wk).to_f32(),
                    || x.matrix_mul_transposed(&self.wv).to_f32(),
                )
            },
        );

        if let Some(ref q_bias) = self.q_bias {
            xq_out = xq_out.add_broadcast_row(q_bias);
        }
        if let Some(ref k_bias) = self.k_bias {
            xk_out = xk_out.add_broadcast_row(k_bias);
        }
        if let Some(ref v_bias) = self.v_bias {
            xv_out = xv_out.add_broadcast_row(v_bias);
        }

        let mut xq_views: Vec<Tensor> = Vec::with_capacity(seq_len as usize);
        let mut xk_views: Vec<Tensor> = Vec::with_capacity(seq_len as usize);
        let mut xv_views: Vec<Tensor> = Vec::with_capacity(seq_len as usize);

        #[cfg(feature = "opencl")]
        let use_gpu_norm_rope = self.data_settings.use_opencl_for_attention;

        for idx in 0..seq_len {
            #[cfg(feature = "opencl")]
            if use_gpu_norm_rope {
                // RMSNorm and RoPE already applied on GPU above
                let xq_row = xq_out
                    .row(idx)
                    .view(self.n_local_heads as i64, self.head_dim as i64);
                let xk_row = xk_out
                    .row(idx)
                    .view(self.n_kv_heads as i64, self.head_dim as i64);
                let xv_row = xv_out
                    .row(idx)
                    .view(self.n_kv_heads as i64, self.head_dim as i64);
                xq_views.push(xq_row);
                xk_views.push(xk_row);
                xv_views.push(xv_row);
            } else {
                let mut xq_row = xq_out
                    .row(idx)
                    .view(self.n_local_heads as i64, self.head_dim as i64);
                let mut xk_row = xk_out
                    .row(idx)
                    .view(self.n_kv_heads as i64, self.head_dim as i64);
                let xv_row = xv_out
                    .row(idx)
                    .view(self.n_kv_heads as i64, self.head_dim as i64);
                if let Some(ref q_norm) = self.q_norm_weight {
                    xq_row = per_head_rms_norm(&xq_row, q_norm, self.eps, "q");
                }
                if let Some(ref k_norm) = self.k_norm_weight {
                    xk_row = per_head_rms_norm(&xk_row, k_norm, self.eps, "k");
                }
                let (xq_row, xk_row) = apply_rotary_emb(
                    &xq_row, &xk_row, freqs_cis, idx as usize, start_pos, self.n_kv_heads,
                );
                xq_views.push(xq_row);
                xk_views.push(xk_row);
                xv_views.push(xv_row);
            }
            #[cfg(not(feature = "opencl"))]
            {
                let mut xq_row = xq_out
                    .row(idx)
                    .view(self.n_local_heads as i64, self.head_dim as i64);
                let mut xk_row = xk_out
                    .row(idx)
                    .view(self.n_kv_heads as i64, self.head_dim as i64);
                let xv_row = xv_out
                    .row(idx)
                    .view(self.n_kv_heads as i64, self.head_dim as i64);

                if let Some(ref q_norm) = self.q_norm_weight {
                    xq_row = per_head_rms_norm(&xq_row, q_norm, self.eps, "q");
                }
                if let Some(ref k_norm) = self.k_norm_weight {
                    xk_row = per_head_rms_norm(&xk_row, k_norm, self.eps, "k");
                }

                let (xq_row, xk_row) =
                    apply_rotary_emb(&xq_row, &xk_row, freqs_cis, idx as usize, start_pos, self.n_kv_heads);

                xq_views.push(xq_row);
                xk_views.push(xk_row);
                xv_views.push(xv_row);
            }
        }
        let group_size = self.n_local_heads / self.n_kv_heads;

        // Phase 1: Write KV cache using bulk operations (much faster than element-wise)
        for kv_idx in 0..self.n_kv_heads {
            // Concatenate all K vectors for this KV head: [seq_len, head_dim]
            let mut concat_vec: Vec<Tensor> = Vec::with_capacity(seq_len as usize);
            for idx2 in 0..seq_len {
                concat_vec.push(xk_views[idx2 as usize].row(kv_idx as i64));
            }
            let concat_vec2: Vec<&Tensor> = concat_vec.iter().collect();
            let xk_row = Tensor::concat(&concat_vec2).transpose(); // [head_dim, seq_len]

            // Concatenate all V vectors for this KV head: [seq_len, head_dim]
            concat_vec.clear();
            for idx2 in 0..seq_len {
                concat_vec.push(xv_views[idx2 as usize].row(kv_idx as i64));
            }
            let concat_vec2: Vec<&Tensor> = concat_vec.iter().collect();
            let xv_row = Tensor::concat(&concat_vec2); // [seq_len, head_dim]

            // Bulk update cache using copy_rows_from instead of element-wise loops
            let mut cache_k = attention_cache.cache_k[kv_idx].write().unwrap();
            let mut cache_v = attention_cache.cache_v[kv_idx].write().unwrap();
            
            // Transpose V first, then convert dtypes to match cache
            let xv_row_transposed = xv_row.transpose();
            let cache_dtype = cache_k.dtype();
            let xk_row = if xk_row.dtype() != cache_dtype {
                xk_row.into_dtype(cache_dtype)
            } else {
                xk_row
            };
            let xv_row_transposed = if xv_row_transposed.dtype() != cache_dtype {
                xv_row_transposed.into_dtype(cache_dtype)
            } else {
                xv_row_transposed
            };
            
            // Use bulk copy instead of element-wise get/set
            cache_k.copy_rows_from(start_pos as i64, &xk_row);
            cache_v.copy_rows_from(start_pos as i64, &xv_row_transposed);
            
            std::mem::drop(cache_k);
            std::mem::drop(cache_v);
        }

        // Phase 2: Parallel attention computation — read-only cache access, no lock contention
        let output: Vec<Tensor> = (0..self.n_local_heads)
                .into_par_iter()
                .map(|idx| {
                    let kv_idx = idx / group_size;
                    let mut concat_vec: Vec<Tensor> = vec![];
                    for idx2 in 0..seq_len {
                        concat_vec.push(xq_views[idx2 as usize].row(idx as i64));
                    }
                    let concat_vec2: Vec<&Tensor> = concat_vec.iter().collect();
                    let xq_row = Tensor::concat(&concat_vec2);

                    let cache_k = attention_cache.cache_k[kv_idx].read().unwrap();
                    let cache_v = attention_cache.cache_v[kv_idx].read().unwrap();
                    let keys = cache_k.clip_cols(start_pos + seq_len as usize);
                    let values = cache_v.clip_cols(start_pos + seq_len as usize);
                    std::mem::drop(cache_k);
                    std::mem::drop(cache_v);

                    let keys = keys.into_same_type(&xq_row);
                    let values = values.into_same_type(&xq_row);

                    let m = xq_row
                        .matrix_mul(&keys)
                        .scalar_multiply_f32(1.0 / (self.head_dim as f32).sqrt());

                    match mask {
                        Some(ref mask) => m
                            .add(mask)
                            .to_f32()
                            .softmax()
                            .matrix_mul_transposed(&values),
                        None => m.softmax().matrix_mul_transposed(&values),
                    }
                })
                .collect();

        let output2: Vec<Tensor> = (0..seq_len)
            .into_par_iter()
            .map(|idx| {
                let combined_dim = self.n_local_heads * self.head_dim;
                let mut concat_vec: Vec<Tensor> = vec![];
                for output in &output {
                    concat_vec.push(output.row(idx));
                }
                let concat_vec2: Vec<&Tensor> = concat_vec.iter().collect();
                #[cfg(not(feature = "opencl"))]
                {
                    let xq_row = Tensor::concat(&concat_vec2).view(1, combined_dim as i64);
                    xq_row
                        .into_same_type(&self.wo)
                        .matrix_mul_transposed(&self.wo)
                }
                #[cfg(feature = "opencl")]
                {
                    let mut xq_row = Tensor::concat(&concat_vec2)
                        .view(1, combined_dim as i64)
                        .into_same_type(&self.wo);
                    if self.wo.is_on_gpu() {
                        xq_row
                            .to_gpu_inplace(&self.data_settings.cl.as_ref().unwrap())
                            .unwrap();
                        let mut result = xq_row.matrix_mul_transposed(&self.wo);
                        result.to_cpu_inplace().unwrap();
                        result.to_f32()
                    } else {
                        xq_row.matrix_mul_transposed(&self.wo)
                    }
                }
            })
            .collect();

        let output3: Vec<&Tensor> = output2.iter().collect();
        let output2: Tensor = Tensor::concat(&output3);
        output2.into_dtype(original_x_dtype)
    }
}

fn apply_rotary_emb(
    xq: &Tensor,
    xk: &Tensor,
    freqs_cis: &FreqsCis,
    seq_idx: usize,
    start_pos: usize,
    n_kv_heads: usize,
) -> (Tensor, Tensor) {
    assert!(xq.cols() % 2 == 0);
    assert!(xk.cols() % 2 == 0);
    let mut xq_out: Tensor = xq.clone();
    let mut xk_out: Tensor = xk.clone();
    let half = xq.cols() / 2;
    let group_size = if n_kv_heads > 0 { (xq.rows() as usize) / n_kv_heads } else { 1 };
    for row in 0..xq.rows() as usize {
        let kv_row = (row / group_size) as i64;
        let row = row as i64;
        for col in 0..half {
            let f_real = freqs_cis[seq_idx + start_pos][col as usize].re as f32;
            let f_imag = freqs_cis[seq_idx + start_pos][col as usize].im as f32;
            // HuggingFace split layout: first half paired with second half
            let xq_real = xq.get_f32(row, col);
            let xq_imag = xq.get_f32(row, col + half);
            let xk_real = xk.get_f32(kv_row, col);
            let xk_imag = xk.get_f32(kv_row, col + half);

            let xq_realpart = xq_real * f_real - xq_imag * f_imag;
            let xq_imagpart = xq_real * f_imag + xq_imag * f_real;
            let xk_realpart = xk_real * f_real - xk_imag * f_imag;
            let xk_imagpart = xk_real * f_imag + xk_imag * f_real;

            xq_out.set_f32(row, col, xq_realpart);
            xq_out.set_f32(row, col + half, xq_imagpart);
            xk_out.set_f32(kv_row, col, xk_realpart);
            xk_out.set_f32(kv_row, col + half, xk_imagpart);
        }
    }
    (xq_out, xk_out)
}

fn per_head_rms_norm(x: &Tensor, weight: &Tensor, eps: f64, label: &str) -> Tensor {
    let inner = x.pow(2.0).mean_cols().add_scalar(eps as f32);
    let out1 = x.scalar_multiply_broadcast(&inner.rsqrt());
    if out1.cols() != weight.cols() {
        panic!(
            "per_head_rms_norm[{}] shape mismatch: input {}x{} vs weight {}x{}",
            label,
            out1.rows(),
            out1.cols(),
            weight.rows(),
            weight.cols()
        );
    }
    out1.hadamard_product_broadcast(weight)
}

fn compute_freqs_cis(dim: usize, end: usize, theta: f64) -> FreqsCis {
    let mut freqs = Vec::new();
    for idx in 0..(dim / 2) {
        let freq = 1.0 / (theta.powf(idx as f64 * 2.0 / dim as f64));
        freqs.push(freq);
    }

    let mut result: Vec<Vec<f64>> = Vec::new();
    for x in 0..end {
        let mut row = Vec::new();
        for freq in freqs.iter() {
            let freq = freq * (x as f64);
            row.push(freq);
        }
        result.push(row);
    }

    let mut resultc: Vec<Vec<Complex<f64>>> = Vec::new();
    for row in result.into_iter() {
        let mut rowc = Vec::new();
        for freq in row {
            let cis = Complex::from_polar(1.0, freq);
            rowc.push(cis);
        }
        resultc.push(rowc);
    }
    resultc
}


