// Canonical clean transformer module.
use crate::nn::Linear;
use crate::nn::Module;
use crate::ops::{ChunkedAttention, FlashAttentionRef};
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use std::sync::Arc;

/// Attention variants supported by MultiHeadAttention
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionVariant {
    Baseline,
    FlashRef,
    Chunked { chunk_size: usize },
}

/// Bias function used for NL-OOB (non-local out-of-bounds) distance biases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasFunction {
    Logarithmic,
    Gaussian,
}

/// Compute simple ALiBi slopes
pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
    let mut slopes = Vec::with_capacity(n_heads);
    for i in 0..n_heads {
        let x = (i as f32) / (n_heads as f32 + 0.0f32);
        slopes.push(2f32.powf(-x));
    }
    slopes
}

#[derive(Clone)]
pub struct MultiHeadAttention {
    pub linear_q: Linear,
    pub linear_k: Linear,
    pub linear_v: Linear,
    pub linear_o: Linear,
    pub num_heads: usize,
    pub d_model: usize,
    pub kv_heads: usize,
    pub use_rope: bool,
    pub use_alibi: bool,
    pub alibi_slopes: Option<Vec<f32>>,
    pub relative_bias: Option<Tensor>,
    pub attention_variant: AttentionVariant,
    // NL-OOB fields
    pub nl_oob_config: Option<BiasFunction>,
    pub nl_oob_max_scale: Option<f32>,
    pub slopes: Option<Tensor>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
    }
    pub fn new_with_kv_and_rope(
        d_model: usize,
        num_heads: usize,
        kv_heads: usize,
        use_rope: bool,
    ) -> Self {
        MultiHeadAttention {
            linear_q: Linear::new(d_model, d_model, true),
            linear_k: Linear::new(d_model, d_model, true),
            linear_v: Linear::new(d_model, d_model, true),
            linear_o: Linear::new(d_model, d_model, true),
            num_heads,
            d_model,
            kv_heads,
            use_rope,
            use_alibi: false,
            alibi_slopes: None,
            relative_bias: None,
            attention_variant: AttentionVariant::Baseline,
            nl_oob_config: None,
            nl_oob_max_scale: None,
            slopes: None,
        }
    }
    pub fn new_with_nl_oob(
        d_model: usize,
        num_heads: usize,
        config: BiasFunction,
        max_scale: f32,
    ) -> Self {
        let mut s = MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, num_heads, false);
        // create slopes as a per-head parameter shaped (1, num_heads, 1, 1)
        let arr = ndarray::Array::from_shape_vec((1, num_heads, 1, 1), vec![1.0f32; num_heads])
            .unwrap()
            .into_dyn();
        let slopes_t = Tensor::new(arr * max_scale, true);
        s.slopes = Some(slopes_t);
        s.nl_oob_config = Some(config);
        s.nl_oob_max_scale = Some(max_scale);
        s
    }
    pub fn with_alibi(mut self) -> Self {
        self.use_alibi = true;
        self.alibi_slopes = Some(compute_alibi_slopes(self.num_heads));
        self
    }
    pub fn with_relative_bias(mut self, bias: Tensor) -> Self {
        self.relative_bias = Some(bias);
        self
    }
    pub fn set_attention_variant(&mut self, var: AttentionVariant) {
        self.attention_variant = var;
    }

    pub fn forward_impl(&self, x: &Tensor) -> Tensor {
        self.forward_with_causal(x, false, None)
    }

    pub fn forward_with_causal(
        &self,
        x: &Tensor,
        causal: bool,
        causal_offset: Option<usize>,
    ) -> Tensor {
        let mut q = self.linear_q.forward(x);
        let mut k = self.linear_k.forward(x);
        let v = self.linear_v.forward(x);
        // Apply RoPE (rotary positional embeddings) to q/k if configured.
        if self.use_rope {
            q = q.rope(self.num_heads);
            k = k.rope(self.num_heads);
        }
        let shape = q.lock().storage.shape();
        if shape.len() != 3 {
            return x.clone();
        }
        let b = shape[0];
        let seq = shape[1];
        let head_dim = self.d_model / self.num_heads;
        let q = match q.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MultiHeadAttention forward: reshape q to (b, seq, num_heads, head_dim) failed: {}", e);
                return x.clone();
            }
        };
        let q = q.permute(vec![0, 2, 1, 3]);
        let q2 = match q.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!(
                    "MultiHeadAttention forward: reshape q after permute failed: {}",
                    e
                );
                return x.clone();
            }
        };
        let k = match k.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MultiHeadAttention forward: reshape k to (b, seq, num_heads, head_dim) failed: {}", e);
                return x.clone();
            }
        };
        let k = k.permute(vec![0, 2, 1, 3]);
        let k2 = match k.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!(
                    "MultiHeadAttention forward: reshape k after permute failed: {}",
                    e
                );
                return x.clone();
            }
        };
        let v = match v.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MultiHeadAttention forward: reshape v to (b, seq, num_heads, head_dim) failed: {}", e);
                return x.clone();
            }
        };
        let v = v.permute(vec![0, 2, 1, 3]);
        let v2 = match v.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!(
                    "MultiHeadAttention forward: reshape v after permute failed: {}",
                    e
                );
                return x.clone();
            }
        };

        let out = match self.attention_variant {
            AttentionVariant::Baseline => {
                let k2t = k2.permute(vec![0, 2, 1]);
                let qk = q2.batched_matmul(&k2t);
                let scale = 1.0f32 / (head_dim as f32).sqrt();
                let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
                let scaled = qk.mul(&scalar_tensor);
                let mut scaled_logits = scaled.clone();
                if self.use_alibi {
                    let slopes = if let Some(s) = &self.alibi_slopes {
                        s.clone()
                    } else {
                        compute_alibi_slopes(self.num_heads)
                    };
                    let mut bias_arr =
                        ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                    for batch in 0..b {
                        for h in 0..self.num_heads {
                            let slope = slopes[h];
                            for i in 0..seq {
                                for j in 0..seq {
                                    let dist = (j as isize - i as isize) as f32;
                                    bias_arr[[batch * self.num_heads + h, i, j]] = -slope * dist;
                                }
                            }
                        }
                    }
                    let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                    scaled_logits = scaled_logits.add(&bias_t);
                }
                if let Some(rb) = &self.relative_bias {
                    let shape = rb.lock().storage.shape();
                    if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] {
                        scaled_logits = scaled_logits.add(rb);
                    }
                }
                if causal {
                    let mut mask_arr =
                        ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                    for i in 0..(b * self.num_heads) {
                        for r in 0..seq {
                            for c2 in (r + 1)..seq {
                                if let Some(offset) = causal_offset {
                                    let r_is_text = r >= offset;
                                    let c2_is_text = c2 >= offset;
                                    if r_is_text && c2_is_text {
                                        mask_arr[[i, r, c2]] = -1e9_f32;
                                    }
                                } else {
                                    mask_arr[[i, r, c2]] = -1e9_f32;
                                }
                            }
                        }
                    }
                    let mask_t = crate::tensor::Tensor::new(mask_arr, false);
                    scaled_logits = scaled_logits.add(&mask_t);
                }
                let attn = scaled_logits.softmax(2);
                attn.batched_matmul(&v2)
            }
            AttentionVariant::FlashRef => {
                let flash = FlashAttentionRef::new(head_dim);
                Tensor::apply(Arc::new(flash), &[q2.clone(), k2.clone(), v2.clone()])
            }
            AttentionVariant::Chunked { chunk_size } => {
                let op = ChunkedAttention::new(head_dim, chunk_size);
                Tensor::apply(Arc::new(op), &[q2.clone(), k2.clone(), v2.clone()])
            }
        };
        let out2 = match out.reshape(vec![b, self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MultiHeadAttention forward: reshape out to (b, num_heads, seq, head_dim) failed: {}", e);
                return x.clone();
            }
        };
        let out3 = out2.permute(vec![0, 2, 1, 3]);
        let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MultiHeadAttention forward: reshape out after permute to (b, seq, d_model) failed: {}", e);
                return x.clone();
            }
        };
        self.linear_o.forward(&out4)
    }

    /// Forward with distance matrix integrating NL-OOB distances as additional attention bias.
    /// `dist` may be 2D (seq x seq) or 3D (batch x seq x seq).
    pub fn forward_with_distance(&self, x: &Tensor, dist: &Tensor) -> Tensor {
        let q = self.linear_q.forward(x);
        let k = self.linear_k.forward(x);

        let v = self.linear_v.forward(x);

        let shape = q.lock().storage.shape();
        if shape.len() != 3 {
            return x.clone();
        }
        let b = shape[0];
        let seq = shape[1];
        let head_dim = self.d_model / self.num_heads;
        // Prepare distance tensor by extracting ndarray copy first to avoid lock-ordering issues.
        let dist_arr = dist.to_f32_array();
        let dist_shape = dist_arr.shape().to_vec();
        if !(dist_shape == [seq, seq]
            || (dist_shape.len() == 3
                && dist_shape[0] == b
                && dist_shape[1] == seq
                && dist_shape[2] == seq))
        {
            // mismatched shapes -> return input unchanged
            return x.clone();
        }
        let q2 = match q.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };

        let k2 = match k.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };
        let v2 = match v.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };
        let k2t = k2.permute(vec![0, 2, 1]);
        let qk = q2.batched_matmul(&k2t);

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let scalar_tensor = Tensor::new(ndarray::Array::from_elem(IxDyn(&[1]), scale), false);
        let scaled = qk.mul(&scalar_tensor);
        let mut scaled_logits = scaled.clone();
        // We will compute bias via Tensor operations so gradients flow back to slopes
        // Shape the scaled logits into (b, num_heads, seq, seq) to add a (1, num_heads, seq, seq) bias via broadcasting
        let scaled_logits4 = match scaled_logits.reshape(vec![b, self.num_heads, seq, seq]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };

        // Reshape/expand distance into (b, 1, seq, seq) or (1, 1, seq, seq)
        // We operate on ndarray copies to avoid repeated Mutex locks on Tensor storage.
        if !(dist_shape == [seq, seq]
            || (dist_shape.len() == 3
                && dist_shape[0] == b
                && dist_shape[1] == seq
                && dist_shape[2] == seq))
        {
            return x.clone();
        }
        let slopes_t = self.slopes.as_ref().unwrap().clone();
        // slopes_t should be shaped (1, num_heads, 1, 1)
        // Compute NL-OOB bias efficiently using ndarray and a single Tensor multiply so gradient flows to slopes.
        let bias4 = if let Some(cfg) = self.nl_oob_config {
            // compute f(dist) as ndarray:
            let mut fdist = if dist_shape.len() == 2 {
                // shape (seq, seq) -> expand to (1, 1, seq, seq)
                let arr = ndarray::Array::from_shape_vec(
                    (1, 1, seq, seq),
                    dist_arr.iter().cloned().collect(),
                )
                .unwrap();
                arr
            } else {
                // shape (b, seq, seq) -> expand to (b, 1, seq, seq)
                let raw: Vec<f32> = dist_arr.iter().cloned().collect();
                let arr = ndarray::Array::from_shape_vec((b, 1, seq, seq), raw).unwrap();
                arr
            };
            // apply f depending on cfg
            if cfg == BiasFunction::Logarithmic {
                // fdist = ln(dist + 1)
                fdist = fdist.mapv(|v| (v + 1.0f32).ln());
            } else {
                // Gaussian: fdist = dist^2
                fdist = fdist.mapv(|v| v * v);
            }
            // create Tensor from fdist (non-diff) and multiply with slopes to get bias (diff wrt slopes)
            let fdist_t = Tensor::new(fdist.into_dyn(), false);
            let bias_t = slopes_t.mul(&fdist_t);

            bias_t
        } else {
            Tensor::new(
                ndarray::Array::zeros(IxDyn(&[1, self.num_heads, 1, 1])),
                false,
            )
        };

        // Add bias4 (1,num_heads,seq,seq) or (b,num_heads,seq,seq) with scaled_logits4 (b,num_heads,seq,seq)
        let scaled_with_bias4 = scaled_logits4.add(&bias4);

        // reshape back to (b * num_heads, seq, seq)
        let scaled_with_bias = match scaled_with_bias4.reshape(vec![b * self.num_heads, seq, seq]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };

        scaled_logits = scaled_with_bias;

        let attn = scaled_logits.softmax(2);

        let out = attn.batched_matmul(&v2);
        let out2 = match out.reshape(vec![b, self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };
        let out3 = out2.permute(vec![0, 2, 1, 3]);
        let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };
        self.linear_o.forward(&out4)
    }

    pub fn parameters_impl(&self) -> Vec<Tensor> {
        let mut p = self.linear_q.parameters();
        p.extend(self.linear_k.parameters());
        p.extend(self.linear_v.parameters());
        p.extend(self.linear_o.parameters());
        p
    }
    pub fn named_parameters_impl(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(
            self.linear_q
                .named_parameters(&format!("{}.linear_q", prefix)),
        );
        out.extend(
            self.linear_k
                .named_parameters(&format!("{}.linear_k", prefix)),
        );
        out.extend(
            self.linear_v
                .named_parameters(&format!("{}.linear_v", prefix)),
        );
        out.extend(
            self.linear_o
                .named_parameters(&format!("{}.linear_o", prefix)),
        );
        out
    }
    pub fn load_state_dict_impl(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.linear_q
            .load_state_dict(state, &format!("{}.linear_q", prefix))?;
        self.linear_k
            .load_state_dict(state, &format!("{}.linear_k", prefix))?;
        self.linear_v
            .load_state_dict(state, &format!("{}.linear_v", prefix))?;
        self.linear_o
            .load_state_dict(state, &format!("{}.linear_o", prefix))?;
        Ok(())
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.parameters_impl();
        if let Some(s) = &self.slopes {
            p.push(s.clone());
        }
        p
    }
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = self.named_parameters_impl(prefix);
        if let Some(s) = &self.slopes {
            out.push((format!("{}.nl_oob.slopes", prefix), s.clone()));
        }
        out
    }
    pub fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.load_state_dict_impl(state, prefix)?;
        // Load NL-OOB config and slopes if present
        let key_cfg = format!("{}.nl_oob.config", prefix);
        if let Some(cfg) = state.get(&key_cfg) {
            // cfg should be a scalar float 0/1 mapping to BiasFunction
            let arr = cfg.lock().storage.to_f32_array();
            if arr.ndim() == 1 && arr.len() > 0 {
                let v = arr.into_dimensionality::<ndarray::Ix1>().unwrap()[0];
                if v == 1.0 {
                    self.nl_oob_config = Some(BiasFunction::Gaussian);
                } else {
                    self.nl_oob_config = Some(BiasFunction::Logarithmic);
                }
            }
        }
        let key_slopes = format!("{}.nl_oob.slopes", prefix);
        if let Some(s) = state.get(&key_slopes) {
            // ensure requires_grad is true on loaded slopes
            let mut slock = s.lock();
            slock.requires_grad = true;
            self.slopes = Some(s.clone());
        }
        Ok(())
    }
}

impl crate::nn::Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_impl(input)
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.named_parameters(prefix)
    }
    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.load_state_dict(state, prefix)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    } fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

#[derive(Clone)]
pub struct TransformerBlock {
    pub mha: MultiHeadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
    pub causal: bool,
    // Llama-style pre-norm mode uses RMSNorm; store gamma parameters when enabled
    pub llama_style: bool,
    pub rms_attn_gamma: Option<Tensor>,
    pub rms_ffn_gamma: Option<Tensor>,
}
impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self {
        TransformerBlock {
            mha: MultiHeadAttention::new(d_model, num_heads),
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
            causal: false,
            llama_style: false,
            rms_attn_gamma: None,
            rms_ffn_gamma: None,
        }
    }
    pub fn new_with_kv_and_rope(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
        use_rope: bool,
    ) -> Self {
        TransformerBlock {
            mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope),
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
            causal: false,
            llama_style: false,
            rms_attn_gamma: None,
            rms_ffn_gamma: None,
        }
    }
    pub fn new_with_nl_oob(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        config: BiasFunction,
        max_scale: f32,
    ) -> Self {
        let mut t =
            TransformerBlock::new_with_kv_and_rope(d_model, d_ff, num_heads, num_heads, false);
        t.mha = MultiHeadAttention::new_with_nl_oob(d_model, num_heads, config, max_scale);
        t
    }

    /// Create a Llama-style TransformerBlock.
    /// Defaults:
    /// - `bias`: whether to include biases in linear layers. Set to `false` for Llama-style biasless dense layers.
    /// - `use_rope`: apply RoPE to q/k during attention.
    pub fn new_llama_style(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
        use_rope: bool,
        bias: bool,
    ) -> Self {
        // linear1 must output 2*d_ff for SwiGLU splitting
        let linear1 = Linear::new(d_model, d_ff * 2, bias);
        let linear2 = Linear::new(d_ff, d_model, bias);
        let gamma_attn = Tensor::new(ndarray::Array::from_elem(IxDyn(&[d_model]), 1.0f32), true);
        let gamma_ffn = Tensor::new(ndarray::Array::from_elem(IxDyn(&[d_model]), 1.0f32), true);
        TransformerBlock {
            mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope),
            linear1,
            linear2,
            causal: false,
            llama_style: true,
            rms_attn_gamma: Some(gamma_attn),
            rms_ffn_gamma: Some(gamma_ffn),
        }
    }
    pub fn new_decoder(d_model: usize, d_ff: usize, num_heads: usize) -> Self {
        let mut t = TransformerBlock::new(d_model, d_ff, num_heads);
        t.causal = true;
        t
    }
    pub fn forward_block_impl(&self, x: &Tensor) -> Tensor {
        if self.llama_style {
            // Pre-norm RMSNorm -> Attention -> Residual -> Pre-norm RMSNorm -> SwiGLU FFN
            let gamma_attn = self
                .rms_attn_gamma
                .as_ref()
                .expect("llama_style requires rms_attn_gamma to be set");
            // RMSNorm along the last axis
            let x_norm = x.rmsnorm(gamma_attn, 2, 1e-5);
            let attn_out = self.mha.forward_with_causal(&x_norm, self.causal, None);

            let x2 = x.add(&attn_out);
            let gamma_ffn = self
                .rms_ffn_gamma
                .as_ref()
                .expect("llama_style requires rms_ffn_gamma to be set");
            let x2_norm = x2.rmsnorm(gamma_ffn, 2, 1e-5);
            // linear1 outputs 2*d_ff, SwiGLU will split it to produce d_ff activation
            let ff = self.linear1.forward(&x2_norm).swiglu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        } else {
            let attn_out = self.mha.forward_with_causal(x, self.causal, None);
            let x2 = x.add(&attn_out);
            let dim = x.lock().storage.shape()[2];
            let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
            let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
            let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
            let ff = self.linear1.forward(&x2norm).relu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        }
    }
    /// Backwards-compatible wrapper for older tests expecting `forward_block` method name.
    pub fn forward_block(&self, x: &Tensor) -> Tensor {
        self.forward_block_impl(x)
    }
    /// Backwards-compatible wrapper for older API that accepted a causal offset.
    pub fn forward_block_with_causal_offset(
        &self,
        x: &Tensor,
        causal_offset: Option<usize>,
    ) -> Tensor {
        if self.llama_style {
            let gamma_attn = self
                .rms_attn_gamma
                .as_ref()
                .expect("llama_style requires rms_attn_gamma to be set");
            let x_norm = x.rmsnorm(gamma_attn, 2, 1e-5);
            let attn_out = self.mha.forward_with_causal(&x_norm, self.causal, causal_offset);
            let x2 = x.add(&attn_out);
            let gamma_ffn = self
                .rms_ffn_gamma
                .as_ref()
                .expect("llama_style requires rms_ffn_gamma to be set");
            let x2_norm = x2.rmsnorm(gamma_ffn, 2, 1e-5);
            let ff = self.linear1.forward(&x2_norm).swiglu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        } else {
            let attn_out = self.mha.forward_with_causal(x, self.causal, causal_offset);
            let x2 = x.add(&attn_out);
            let dim = x.lock().storage.shape()[2];
            let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
            let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
            let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
            let ff = self.linear1.forward(&x2norm).relu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        }
    }
    pub fn forward_block_with_distance(&self, x: &Tensor, dist: &Tensor) -> Tensor {
        if self.llama_style {
            let gamma_attn = self
                .rms_attn_gamma
                .as_ref()
                .expect("llama_style requires rms_attn_gamma to be set");
            let x_norm = x.rmsnorm(gamma_attn, 2, 1e-5);
            let attn_out = self.mha.forward_with_distance(&x_norm, dist);
            let x2 = x.add(&attn_out);
            let gamma_ffn = self
                .rms_ffn_gamma
                .as_ref()
                .expect("llama_style requires rms_ffn_gamma to be set");
            let x2_norm = x2.rmsnorm(gamma_ffn, 2, 1e-5);
            let ff = self.linear1.forward(&x2_norm).swiglu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        } else {
            let attn_out = self.mha.forward_with_distance(x, dist);
            let x2 = x.add(&attn_out);
            let dim = x.lock().storage.shape()[2];
            let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
            let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
            let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
            let ff = self.linear1.forward(&x2norm).relu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        }
    }
    pub fn parameters_impl(&self) -> Vec<Tensor> {
        let mut p = self.mha.parameters();
        p.extend(self.linear1.parameters());
        p.extend(self.linear2.parameters());
        if let Some(g) = &self.rms_attn_gamma { p.push(g.clone()); }
        if let Some(g) = &self.rms_ffn_gamma { p.push(g.clone()); }
        p
    }
    pub fn named_parameters_impl(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(self.mha.named_parameters(&format!("{}.mha", prefix)));
        out.extend(
            self.linear1
                .named_parameters(&format!("{}.linear1", prefix)),
        );
        out.extend(
            self.linear2
                .named_parameters(&format!("{}.linear2", prefix)),
        );
        if let Some(g) = &self.rms_attn_gamma {
            out.push((format!("{}.rms_attn_gamma", prefix), g.clone()));
        }
        if let Some(g) = &self.rms_ffn_gamma {
            out.push((format!("{}.rms_ffn_gamma", prefix), g.clone()));
        }
        out
    }
    pub fn load_state_dict_impl(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.mha
            .load_state_dict(state, &format!("{}.mha", prefix))?;
        self.linear1
            .load_state_dict(state, &format!("{}.linear1", prefix))?;
        self.linear2
            .load_state_dict(state, &format!("{}.linear2", prefix))?;
        // Load optional RMS gamma params
        let key_attn = format!("{}.rms_attn_gamma", prefix);
        if let Some(g) = state.get(&key_attn) {
            let mut glock = g.lock();
            glock.requires_grad = true;
            self.rms_attn_gamma = Some(g.clone());
        }
        let key_ffn = format!("{}.rms_ffn_gamma", prefix);
        if let Some(g) = state.get(&key_ffn) {
            let mut glock = g.lock();
            glock.requires_grad = true;
            self.rms_ffn_gamma = Some(g.clone());
        }
        Ok(())
    }
}
impl crate::nn::Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_block_impl(input)
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.parameters_impl()
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.named_parameters_impl(prefix)
    }
    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.load_state_dict_impl(state, prefix)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    } fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

// Simple Encoder-Decoder wrapper using encoder and decoder TransformerBlock sequences.
#[derive(Clone)]
pub struct EncoderDecoderTransformer {
    pub encoder_blocks: Vec<TransformerBlock>,
    pub decoder_blocks: Vec<TransformerBlock>,
}
impl EncoderDecoderTransformer {
    pub fn new(
        encoder_blocks: Vec<TransformerBlock>,
        decoder_blocks: Vec<TransformerBlock>,
    ) -> Self {
        EncoderDecoderTransformer {
            encoder_blocks,
            decoder_blocks,
        }
    }
}
impl crate::nn::Module for EncoderDecoderTransformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut enc = input.clone();
        for blk in &self.encoder_blocks {
            enc = blk.forward_block(&enc);
        }
        let mut dec = enc.clone();
        for blk in &self.decoder_blocks {
            dec = blk.forward_block(&dec);
        }
        dec
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for (i, b) in self.encoder_blocks.iter().enumerate() {
            p.extend(
                b.named_parameters(&format!("encoder.blocks.{}", i))
                    .into_iter()
                    .map(|(_, t)| t),
            );
        }
        for (i, b) in self.decoder_blocks.iter().enumerate() {
            p.extend(
                b.named_parameters(&format!("decoder.blocks.{}", i))
                    .into_iter()
                    .map(|(_, t)| t),
            );
        }
        p
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
