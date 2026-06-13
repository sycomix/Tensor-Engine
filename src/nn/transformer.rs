// Canonical clean transformer module.
#![allow(non_snake_case)]
use crate::nn::linear_dispatch::LinearLayer;
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
    SlidingWindow { window_size: usize },
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
    pub linear_q: LinearLayer,
    pub linear_k: LinearLayer,
    pub linear_v: LinearLayer,
    pub linear_o: LinearLayer,
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
    // RoPE base frequency (theta) for rotary embeddings
    pub rope_theta: f32,
    pub rope_scale: f32,
}

/// Dedicated Grouped Query Attention layer wrapper.
///
/// This wraps `MultiHeadAttention` with `kv_heads < num_heads` to expose
/// an explicit GQA module-level API similar to PyTorch ecosystem patterns.
#[derive(Clone)]
pub struct GroupedQueryAttention {
    pub mha: MultiHeadAttention,
}

/// Dedicated cross-attention wrapper built on top of `MultiHeadAttention`.
///
/// Query projections are computed from the query input while key/value
/// projections are computed from the context input.
#[derive(Clone)]
pub struct CrossAttention {
    pub mha: MultiHeadAttention,
}

/// Dedicated sliding-window attention wrapper (Mistral-style locality).
///
/// This wraps `MultiHeadAttention` and configures a bounded local context
/// window in the attention logits path.
#[derive(Clone)]
pub struct SlidingWindowAttention {
    pub mha: MultiHeadAttention,
}

impl GroupedQueryAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        kv_heads: usize,
        use_rope: bool,
        rope_theta: f32,
        rope_scale: f32,
        bias: bool,
    ) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "GroupedQueryAttention::new: d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }
        if kv_heads == 0 {
            return Err("GroupedQueryAttention::new: kv_heads must be > 0".to_string());
        }
        if !num_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "GroupedQueryAttention::new: num_heads ({}) must be divisible by kv_heads ({})",
                num_heads, kv_heads
            ));
        }

        Ok(Self {
            mha: MultiHeadAttention::new_with_kv_and_rope(
                d_model, num_heads, kv_heads, use_rope, rope_theta, rope_scale, bias,
            ),
        })
    }

    pub fn forward_with_causal(
        &self,
        x: &Tensor,
        causal: bool,
        causal_offset: Option<usize>,
    ) -> Tensor {
        self.mha.forward_with_causal(x, causal, causal_offset, None)
    }
}

impl CrossAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        kv_heads: usize,
        use_rope: bool,
        rope_theta: f32,
        rope_scale: f32,
        bias: bool,
    ) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "CrossAttention::new: d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }
        if kv_heads == 0 {
            return Err("CrossAttention::new: kv_heads must be > 0".to_string());
        }
        if !num_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "CrossAttention::new: num_heads ({}) must be divisible by kv_heads ({})",
                num_heads, kv_heads
            ));
        }

        Ok(Self {
            mha: MultiHeadAttention::new_with_kv_and_rope(
                d_model, num_heads, kv_heads, use_rope, rope_theta, rope_scale, bias,
            ),
        })
    }

    pub fn forward_cross(&self, query: &Tensor, context: &Tensor, mask: Option<&Tensor>) -> Tensor {
        self.mha.forward_cross(query, context, mask)
    }
}

impl SlidingWindowAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        kv_heads: usize,
        window_size: usize,
        use_rope: bool,
        rope_theta: f32,
        rope_scale: f32,
        bias: bool,
    ) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "SlidingWindowAttention::new: d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }
        if kv_heads == 0 {
            return Err("SlidingWindowAttention::new: kv_heads must be > 0".to_string());
        }
        if !num_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "SlidingWindowAttention::new: num_heads ({}) must be divisible by kv_heads ({})",
                num_heads, kv_heads
            ));
        }
        if window_size == 0 {
            return Err("SlidingWindowAttention::new: window_size must be > 0".to_string());
        }

        let mut mha = MultiHeadAttention::new_with_kv_and_rope(
            d_model, num_heads, kv_heads, use_rope, rope_theta, rope_scale, bias,
        );
        mha.set_attention_variant(AttentionVariant::SlidingWindow { window_size });
        Ok(Self { mha })
    }

    pub fn forward_with_causal(
        &self,
        x: &Tensor,
        causal: bool,
        causal_offset: Option<usize>,
    ) -> Tensor {
        self.mha.forward_with_causal(x, causal, causal_offset, None)
    }
}

impl Module for GroupedQueryAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.mha.forward_impl(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.mha.parameters()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.mha.named_parameters(prefix)
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.mha.load_state_dict(state, prefix)
    }

    fn set_training(&mut self, training: bool) {
        self.mha.set_training(training);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Module for CrossAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.mha.forward_impl(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.mha.parameters()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.mha.named_parameters(prefix)
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.mha.load_state_dict(state, prefix)
    }

    fn set_training(&mut self, training: bool) {
        self.mha.set_training(training);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Module for SlidingWindowAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.mha.forward_impl(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.mha.parameters()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.mha.named_parameters(prefix)
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.mha.load_state_dict(state, prefix)
    }

    fn set_training(&mut self, training: bool) {
        self.mha.set_training(training);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false, 10000.0, 1.0, true)
    }
    pub fn new_with_kv_and_rope(
        d_model: usize,
        num_heads: usize,
        kv_heads: usize,
        use_rope: bool,
        rope_theta: f32,
        rope_scale: f32,
        bias: bool,
    ) -> Self {
        let head_dim = d_model / num_heads;
        let kv_dim = kv_heads * head_dim;
        MultiHeadAttention {
            linear_q: LinearLayer::new_f32(d_model, d_model, bias),
            linear_k: LinearLayer::new_f32(d_model, kv_dim, bias),
            linear_v: LinearLayer::new_f32(d_model, kv_dim, bias),
            linear_o: LinearLayer::new_f32(d_model, d_model, bias),
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
            rope_theta,
            rope_scale,
        }
    }
    pub fn new_with_nl_oob(
        d_model: usize,
        num_heads: usize,
        config: BiasFunction,
        max_scale: f32,
    ) -> Self {
        println!(
            "[MHA] new_with_nl_oob start d_model={} heads={} max_scale={}",
            d_model, num_heads, max_scale
        );
        let mut s = MultiHeadAttention::new_with_kv_and_rope(
            d_model, num_heads, num_heads, false, 10000.0, 1.0, true,
        );
        // create slopes as a per-head parameter shaped (1, num_heads, 1, 1)
        let arr =
            match Array::from_shape_vec((1, num_heads, 1, 1), vec![1.0f32; num_heads]) {
                Ok(a) => a.into_dyn(),
                Err(e) => {
                    log::error!(
                        "MultiHeadAttention new_with_nl_oob: failed to construct slopes array: {}",
                        e
                    );
                    Array::from_elem(IxDyn(&[1, num_heads, 1, 1][..]), 1.0f32)
                }
            };
        let slopes_t = Tensor::new(arr * max_scale, true);
        s.slopes = Some(slopes_t);
        s.nl_oob_config = Some(config);
        s.nl_oob_max_scale = Some(max_scale);
        println!("[MHA] new_with_nl_oob done");
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
        self.forward_with_causal(x, false, None, None)
    }

    pub fn forward_with_causal(
        &self,
        x: &Tensor,
        causal: bool,
        causal_offset: Option<usize>,
        distance: Option<&Tensor>,
    ) -> Tensor {
        // Backward-compatible wrapper: no KV cache
        self.forward_with_caching(x, causal, causal_offset, None, None, distance)
    }

    pub fn forward_with_caching(
        &self,
        x: &Tensor,
        causal: bool,
        causal_offset: Option<usize>,
        kv_cache: Option<&mut crate::nn::KVCache>,
        mask: Option<&Tensor>,
        distance: Option<&Tensor>,
    ) -> Tensor {
        // Compute q and the new k/v chunk for the current input x, handling transposed weights as needed
        let mut q = self.linear_q.forward(x);
        // new k chunk
        // Check for transposed weights in k/v/o linear layers (common issue with some loaders)
        // Only applies if we are using standard F32 Linear layers.
        let (k_shape, v_shape) =
            if let (Some(lk), Some(lv)) = (self.linear_k.as_f32(), self.linear_v.as_f32()) {
                (
                    lk.weight.lock().storage.shape().to_vec(),
                    lv.weight.lock().storage.shape().to_vec(),
                )
            } else {
                // If quantized, we assume weights are already packed/correct shape.
                (vec![], vec![])
            };

        // Compute new_v chunk first so it's available for KV cache append in error paths.
        let new_v = if !v_shape.is_empty()
            && v_shape.len() == 2
            && v_shape[0] != self.d_model
            && v_shape[1] == self.d_model
        {
            log::debug!("MHA.forward_with_caching: detected transposed v_proj weight shape {:?}, fixing on-the-fly", v_shape);
            if let Some(lv) = self.linear_v.as_f32() {
                let arr = lv.weight.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                let w_fixed = Tensor::new(arr_t.into_dyn(), false);
                let shape_x = x.lock().storage.shape().to_vec();
                let b = shape_x[0];
                let seq = shape_x[1];
                let last = shape_x[2];
                let batch = b * seq;
                match x.reshape(vec![batch, last]).and_then(|t| t.matmul(&w_fixed).reshape(out_shape_check(&t, &w_fixed))) {
                    Ok(v) => v,
                    Err(e) => {
                        log::error!("MHA.forward_with_caching: failed to compute transposed v output: {}", e);
                        self.linear_v.forward(x)
                    }
                }
            } else {
                self.linear_v.forward(x)
            }
        } else {
            self.linear_v.forward(x)
        };

        let mut new_k = if !k_shape.is_empty()
            && k_shape.len() == 2
            && k_shape[0] != self.d_model
            && k_shape[1] == self.d_model
        {
            log::debug!("MHA.forward_with_caching: detected transposed k_proj weight shape {:?}, fixing on-the-fly", k_shape);
            if let Some(lk) = self.linear_k.as_f32() {
                let arr = lk.weight.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                let w_fixed = Tensor::new(arr_t.into_dyn(), false);
                let shape_x = x.lock().storage.shape().to_vec();
                let b = shape_x[0];
                let seq = shape_x[1];
                let last = shape_x[2];
                let batch = b * seq;
                match x.reshape(vec![batch, last]).and_then(|t| t.matmul(&w_fixed).reshape(out_shape_check(&t, &w_fixed))) {
                    Ok(k) => k,
                    Err(e) => {
                        log::error!("MHA.forward_with_caching: failed to compute transposed k output: {}", e);
                        // Append new_k/new_v to KV cache before returning.
                        if let Some(kvc) = kv_cache {
                            let _ = kvc.append_packed(&self.linear_k.forward(x), &new_v);
                        }
                        return x.clone();
                    }
                }
            } else {
                self.linear_k.forward(x)
            }
        } else {
            self.linear_k.forward(x)
        };

        // Helper closure for shape computation (avoids shadowing issues)
        fn out_shape_check(t: &Tensor, w: &Tensor) -> Vec<usize> {
            vec![t.lock().storage.shape()[0], t.lock().storage.shape()[1], w.lock().storage.shape()[1]]
        }

        // Apply RoPE to q and new_k if configured
        if self.use_rope {
            let cache_len = kv_cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);
            let offset = causal_offset.unwrap_or(cache_len);
            log::debug!(
                "MHA RoPE: cache_len={}, causal_offset={:?}, final_offset={}, q_shape={:?}, new_k_shape={:?}",
                cache_len, causal_offset, offset,
                q.lock().storage.shape(),
                new_k.lock().storage.shape()
            );
            q = q.rope(self.num_heads, self.rope_theta, self.rope_scale, offset);
            new_k = new_k.rope(self.kv_heads, self.rope_theta, self.rope_scale, offset);
        }

        // If a KV cache is provided, append new_k/new_v to packed storage and use the cached full keys/values
        let (k_total, v_total) = if let Some(ref kvc) = kv_cache {
            let cache_len_before = kvc.seq_len();
            log::debug!(
                "KV cache before append: seq_len={}, new_k_shape={:?}, new_v_shape={:?}",
                cache_len_before,
                new_k.lock().storage.shape(),
                new_v.lock().storage.shape()
            );

            // append packed; this will initialize packed storage if necessary
            if let Err(e) = kvc.append_packed(&new_k, &new_v) {
                log::error!("KV cache append failed: {}", e);
                // fallback to using the new_k/new_v only
                (new_k.clone(), new_v.clone())
            } else {
                let cache_len_after = kvc.seq_len();
                // read back the packed storage
                match (kvc.packed_keys(), kvc.packed_values()) {
                    (Some(pk), Some(pv)) => {
                        log::debug!(
                            "KV cache after append: seq_len={} (was {}), k_total_shape={:?}, v_total_shape={:?}",
                            cache_len_after, cache_len_before,
                            pk.lock().storage.shape(),
                            pv.lock().storage.shape()
                        );
                        (pk, pv)
                    }
                    _ => {
                        log::error!("KV cache append succeeded but packed storage is None");
                        (new_k.clone(), new_v.clone())
                    }
                }
            }
        } else {
            (new_k.clone(), new_v.clone())
        };

        // Debug shapes early
        log::debug!("MHA.forward_with_caching: pre-rope shapes q={:?} k={:?} v={:?} d_model={} num_heads={} kv_heads={}", q.lock().storage.shape(), k_total.lock().storage.shape(), v_total.lock().storage.shape(), self.d_model, self.num_heads, self.kv_heads);

        let shape_q = q.lock().storage.shape().to_vec();
        if shape_q.len() != 3 {
            log::debug!(
                "MHA.forward_with_caching: q expected 3D tensor, got {:?}",
                shape_q
            );
            // Append to KV cache before returning to maintain consistency.
            if let Some(kvc) = kv_cache {
                let _ = kvc.append_packed(&new_k, &new_v);
            }
            return x.clone();
        }
        let b = shape_q[0];
        let q_seq = shape_q[1];
        let head_dim = self.d_model / self.num_heads;
        let q = match q.reshape(vec![b, q_seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MultiHeadAttention forward: reshape q to (b, seq, num_heads, head_dim) failed: {}", e);
                return x.clone();
            }
        };
        let q = q.permute(vec![0, 2, 1, 3]);
        let q2 = match q.reshape(vec![b * self.num_heads, q_seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!(
                    "MultiHeadAttention forward: reshape q after permute failed: {}",
                    e
                );
                return x.clone();
            }
        };

        // reshape k_total/v_total into per-head tiled forms, expanding kv_heads if needed
        let k_total_shape = k_total.lock().storage.shape().to_vec();
        let kv_seq = k_total_shape[1];
        // Attempt reshape to (b, kv_seq, num_heads, head_dim)
        let k_try_num = k_total.reshape(vec![b, kv_seq, self.num_heads, head_dim]);
        let k = match k_try_num {
            Ok(t) => t.permute(vec![0, 2, 1, 3]),
            Err(_) => {
                // Try reshape with kv_heads and expand
                let k_try_kv = match k_total.reshape(vec![b, kv_seq, self.kv_heads, head_dim]) {
                    Ok(t) => t.permute(vec![0, 2, 1, 3]),
                    Err(e) => {
                        log::error!("MultiHeadAttention forward: reshape k_total to (b, kv_seq, num_heads or kv_heads, head_dim) failed: {}", e);
                        return x.clone();
                    }
                };
                // Expand k from (b, kv_heads, kv_seq, head_dim) to (b, num_heads, kv_seq, head_dim)
                let repeat = self.num_heads / self.kv_heads;
                let arr = k_try_kv.lock().storage.to_f32_array();
                let mut new = ndarray::ArrayD::<f32>::zeros(IxDyn(
                    &[b, self.num_heads, kv_seq, head_dim][..],
                ));
                for batch in 0..b {
                    let batch_view = arr.index_axis(ndarray::Axis(0), batch);
                    for i in 0..self.kv_heads {
                        let src = batch_view.index_axis(ndarray::Axis(0), i).to_owned(); // [kv_seq, head_dim]
                        for r in 0..repeat {
                            let dest_idx = i * repeat + r;
                            new.index_axis_mut(ndarray::Axis(0), batch)
                                .index_axis_mut(ndarray::Axis(0), dest_idx)
                                .assign(&src);
                        }
                    }
                }
                Tensor::new(new.into_dyn(), false)
            }
        };
        let k2 = match k.reshape(vec![b * self.num_heads, kv_seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!(
                    "MultiHeadAttention forward: reshape k after permute failed: {}",
                    e
                );
                return x.clone();
            }
        };
        // v
        let v_try_num = v_total.reshape(vec![b, kv_seq, self.num_heads, head_dim]);
        let v = match v_try_num {
            Ok(t) => t.permute(vec![0, 2, 1, 3]),
            Err(_) => {
                let v_try_kv = match v_total.reshape(vec![b, kv_seq, self.kv_heads, head_dim]) {
                    Ok(t) => t.permute(vec![0, 2, 1, 3]),
                    Err(e) => {
                        log::error!("MultiHeadAttention forward: reshape v_total to (b, kv_seq, num_heads or kv_heads, head_dim) failed: {}", e);
                        return x.clone();
                    }
                };
                let repeat = self.num_heads / self.kv_heads;
                let arr = v_try_kv.lock().storage.to_f32_array();
                let mut new = ndarray::ArrayD::<f32>::zeros(IxDyn(
                    &[b, self.num_heads, kv_seq, head_dim][..],
                ));
                for batch in 0..b {
                    let batch_view = arr.index_axis(ndarray::Axis(0), batch);
                    for i in 0..self.kv_heads {
                        let src = batch_view.index_axis(ndarray::Axis(0), i).to_owned();
                        for r in 0..repeat {
                            let dest_idx = i * repeat + r;
                            new.index_axis_mut(ndarray::Axis(0), batch)
                                .index_axis_mut(ndarray::Axis(0), dest_idx)
                                .assign(&src);
                        }
                    }
                }
                Tensor::new(new.into_dyn(), false)
            }
        };
        let v2 = match v.reshape(vec![b * self.num_heads, kv_seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!(
                    "MultiHeadAttention forward: reshape v after permute failed: {}",
                    e
                );
                // Append to KV cache before returning to avoid corrupting inference state.
                if let Some(kvc) = kv_cache {
                    let _ = kvc.append_packed(&new_k, &new_v);
                }
                return x.clone();
            }
        };

        let effective_variant = if mask.is_some() {
            AttentionVariant::Baseline
        } else {
            self.attention_variant
        };
        let sliding_window = match effective_variant {
            AttentionVariant::SlidingWindow { window_size } => Some(window_size),
            _ => None,
        };
        let out = match effective_variant {
            AttentionVariant::Baseline | AttentionVariant::SlidingWindow { .. } => {
                let k2t = k2.permute(vec![0, 2, 1]);
                let qk = q2.batched_matmul(&k2t);
                let scale = 1.0f32 / (head_dim as f32).sqrt();
                let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1][..]), scale), false);
                let scaled = qk.mul(&scalar_tensor);
                let mut scaled_logits = scaled.clone();
                if self.use_alibi {
                    let slopes = if let Some(s) = &self.alibi_slopes {
                        s.clone()
                    } else {
                        compute_alibi_slopes(self.num_heads)
                    };
                    // bias shape: (b*num_heads, q_seq, kv_seq)
                    let mut bias_arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(
                        &[b * self.num_heads, q_seq, kv_seq][..],
                    ));
                    // If kv_seq == q_seq and new_start == 0 this reduces to previous behavior
                    let new_start = kv_seq.saturating_sub(q_seq);
                    for batch in 0..b {
                        for h in 0..self.num_heads {
                            let slope = slopes[h];
                            for i in 0..q_seq {
                                for j in 0..kv_seq {
                                    let dist = (j as isize - (new_start + i) as isize) as f32;
                                    bias_arr[[batch * self.num_heads + h, i, j]] = -slope * dist;
                                }
                            }
                        }
                    }
                    let bias_t = Tensor::new(bias_arr.into_dyn(), false);
                    scaled_logits = scaled_logits.add(&bias_t);
                }
                if let Some(rb) = &self.relative_bias {
                    let shape = rb.lock().storage.shape().to_vec();
                    // accept shapes (1, q_seq, kv_seq) or (num_heads, q_seq, kv_seq)
                    if (shape.len() == 3 && shape[1] == q_seq && shape[2] == kv_seq)
                        && (shape[0] == 1 || shape[0] == self.num_heads)
                    {
                        scaled_logits = scaled_logits.add(rb);
                    }
                }
                if causal {
                    // mask shape: (b*num_heads, q_seq, kv_seq)
                    let mut mask_arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(
                        &[b * self.num_heads, q_seq, kv_seq][..],
                    ));
                    let new_start = kv_seq.saturating_sub(q_seq);
                    for i in 0..(b * self.num_heads) {
                        for r in 0..q_seq {
                            for c2 in 0..kv_seq {
                                let global_r = new_start + r;
                                if c2 > global_r {
                                    if let Some(offset) = causal_offset {
                                        let r_is_text = global_r >= offset;
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
                    }
                    let mask_t = Tensor::new(mask_arr.into_dyn(), false);
                    log::debug!(
                        "Causal mask applied: q_seq={}, kv_seq={}, new_start={}, causal_offset={:?}",
                        q_seq, kv_seq, new_start, causal_offset
                    );
                    scaled_logits = scaled_logits.add(&mask_t);
                }
                if let Some(window_size) = sliding_window {
                    let mut window_mask_arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(
                        &[b * self.num_heads, q_seq, kv_seq][..],
                    ));
                    let new_start = kv_seq.saturating_sub(q_seq);
                    for i in 0..(b * self.num_heads) {
                        for r in 0..q_seq {
                            let global_r = new_start + r;
                            for c2 in 0..kv_seq {
                                let should_mask = if causal {
                                    c2 > global_r || global_r.saturating_sub(c2) > window_size
                                } else {
                                    c2.abs_diff(global_r) > window_size
                                };
                                if should_mask {
                                    if causal {
                                        if let Some(offset) = causal_offset {
                                            let r_is_text = global_r >= offset;
                                            let c2_is_text = c2 >= offset;
                                            if r_is_text && c2_is_text {
                                                window_mask_arr[[i, r, c2]] = -1e9_f32;
                                            }
                                        } else {
                                            window_mask_arr[[i, r, c2]] = -1e9_f32;
                                        }
                                    } else {
                                        window_mask_arr[[i, r, c2]] = -1e9_f32;
                                    }
                                }
                            }
                        }
                    }
                    let window_mask_t =
                        Tensor::new(window_mask_arr.into_dyn(), false);
                    scaled_logits = scaled_logits.add(&window_mask_t);
                }
                if let Some(dist) = distance {
                    // Port NL-OOB logic here
                    let dist_arr = dist.to_f32_array();
                    let dist_shape = dist_arr.shape().to_vec();
                    if dist_shape == [q_seq, kv_seq]
                        || (dist_shape.len() == 3
                            && dist_shape[0] == b
                            && dist_shape[1] == q_seq
                            && dist_shape[2] == kv_seq)
                    {
                        if let (Some(slopes_t), Some(cfg)) = (&self.slopes, self.nl_oob_config) {
                            let mut fdist_arr = if dist_shape.len() == 2 {
                                let raw: Vec<f32> = dist_arr.iter().cloned().collect();
                                Array::from_shape_vec((1, 1, q_seq, kv_seq), raw)
                                    .unwrap_or_else(|_| {
                                        Array::zeros((1, 1, q_seq, kv_seq))
                                    })
                            } else {
                                let raw: Vec<f32> = dist_arr.iter().cloned().collect();
                                Array::from_shape_vec((b, 1, q_seq, kv_seq), raw)
                                    .unwrap_or_else(|_| {
                                        Array::zeros((b, 1, q_seq, kv_seq))
                                    })
                            };

                            if cfg == BiasFunction::Logarithmic {
                                fdist_arr = fdist_arr.mapv(|v| (v + 1.0f32).ln());
                            } else {
                                fdist_arr = fdist_arr.mapv(|v| v * v);
                            }
                            let fdist_t = Tensor::new(fdist_arr.into_dyn(), false);
                            let nl_bias = slopes_t.mul(&fdist_t);

                            // nl_bias is (1 or b, num_heads, q_seq, kv_seq)
                            // If it's (1, num_heads, q_seq, kv_seq) and b > 1, we need to broadcast it
                            // before flattening to (b * num_heads, q_seq, kv_seq)
                            let nl_bias_flat = if dist_shape.len() == 2 && b > 1 {
                                // Emulate broadcast by repeating or using broadcast_to if Tensor supported it better.
                                // For now, let's just reshape to (num_heads, q_seq, kv_seq) and let sub handle broadcasting
                                // If scaled_logits allowed it. But scaled_logits is (b*num_heads, q_seq, kv_seq).
                                // So we MUST expand to b first.
                                let mut expanded =
                                    Vec::with_capacity(b * self.num_heads * q_seq * kv_seq);
                                let single_batch_data = nl_bias.to_f32_array();
                                for _ in 0..b {
                                    expanded.extend(single_batch_data.iter().cloned());
                                }
                                Tensor::new(
                                    Array::from_shape_vec(
                                        (b * self.num_heads, q_seq, kv_seq),
                                        expanded,
                                    )
                                    .unwrap()
                                    .into_dyn(),
                                    false,
                                )
                            } else {
                                nl_bias
                                    .reshape(vec![b * self.num_heads, q_seq, kv_seq])
                                    .unwrap_or_else(|_| nl_bias.clone())
                            };
                            scaled_logits = scaled_logits.sub(&nl_bias_flat);
                        }
                    }
                }
                if let Some(m) = mask {
                    scaled_logits = scaled_logits.add(m);
                }
                let attn = scaled_logits.softmax(2);
                attn.batched_matmul(&v2)
            }
            AttentionVariant::FlashRef => {
                let flash = FlashAttentionRef::new(head_dim);
                Tensor::apply(Arc::new(flash), &[q2.clone(), k2.clone(), v2.clone()][..])
            }
            AttentionVariant::Chunked { chunk_size } => {
                let op = ChunkedAttention::new(head_dim, chunk_size);
                Tensor::apply(Arc::new(op), &[q2.clone(), k2.clone(), v2.clone()][..])
            }
        };
        let out2 = match out.reshape(vec![b, self.num_heads, q_seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MultiHeadAttention forward: reshape out to (b, num_heads, q_seq, head_dim) failed: {}", e);
                // Append to KV cache before returning to avoid corrupting inference state.
                if let Some(kvc) = kv_cache {
                    let _ = kvc.append_packed(&new_k, &new_v);
                }
                return x.clone();
            }
        };
        let out3 = out2.permute(vec![0, 2, 1, 3]);
        let out4 = match out3.reshape(vec![b, q_seq, self.d_model]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MultiHeadAttention forward: reshape out after permute to (b, q_seq, d_model) failed: {}", e);
                // Append to KV cache before returning.
                if let Some(kvc) = kv_cache {
                    let _ = kvc.append_packed(&new_k, &new_v);
                }
                return x.clone();
            }
        };
        self.linear_o.forward(&out4)
    }

    pub fn forward_cross(&self, query: &Tensor, context: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let mut q = self.linear_q.forward(query);
        let mut k_total = self.linear_k.forward(context);
        let v_total = self.linear_v.forward(context);

        if self.use_rope {
            q = q.rope(self.num_heads, self.rope_theta, self.rope_scale, 0);
            k_total = k_total.rope(self.kv_heads, self.rope_theta, self.rope_scale, 0);
        }

        let shape_q = q.lock().storage.shape().to_vec();
        let shape_k = k_total.lock().storage.shape().to_vec();
        if shape_q.len() != 3 || shape_k.len() != 3 {
            log::error!(
                "MHA.forward_cross expects 3D query/context tensors, got q={:?}, k={:?}",
                shape_q,
                shape_k
            );
            return query.clone();
        }

        let b = shape_q[0];
        let q_seq = shape_q[1];
        let b_ctx = shape_k[0];
        if b != b_ctx {
            log::error!(
                "MHA.forward_cross batch mismatch: query batch={}, context batch={}",
                b,
                b_ctx
            );
            return query.clone();
        }

        let head_dim = self.d_model / self.num_heads;
        let q = match q.reshape(vec![b, q_seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MHA.forward_cross: reshape q failed: {}", e);
                return query.clone();
            }
        };
        let q = q.permute(vec![0, 2, 1, 3]);
        let q2 = match q.reshape(vec![b * self.num_heads, q_seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MHA.forward_cross: reshape q2 failed: {}", e);
                return query.clone();
            }
        };

        let kv_seq = shape_k[1];

        let k_try_num = k_total.reshape(vec![b, kv_seq, self.num_heads, head_dim]);
        let k = match k_try_num {
            Ok(t) => t.permute(vec![0, 2, 1, 3]),
            Err(_) => {
                let k_try_kv = match k_total.reshape(vec![b, kv_seq, self.kv_heads, head_dim]) {
                    Ok(t) => t.permute(vec![0, 2, 1, 3]),
                    Err(e) => {
                        log::error!("MHA.forward_cross: reshape k failed: {}", e);
                        return query.clone();
                    }
                };
                let repeat = self.num_heads / self.kv_heads;
                let arr = k_try_kv.lock().storage.to_f32_array();
                let mut new = ndarray::ArrayD::<f32>::zeros(IxDyn(
                    &[b, self.num_heads, kv_seq, head_dim][..],
                ));
                for batch in 0..b {
                    let batch_view = arr.index_axis(ndarray::Axis(0), batch);
                    for i in 0..self.kv_heads {
                        let src = batch_view.index_axis(ndarray::Axis(0), i).to_owned();
                        for r in 0..repeat {
                            let dest_idx = i * repeat + r;
                            new.index_axis_mut(ndarray::Axis(0), batch)
                                .index_axis_mut(ndarray::Axis(0), dest_idx)
                                .assign(&src);
                        }
                    }
                }
                Tensor::new(new.into_dyn(), false)
            }
        };
        let k2 = match k.reshape(vec![b * self.num_heads, kv_seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MHA.forward_cross: reshape k2 failed: {}", e);
                return query.clone();
            }
        };

        let v_try_num = v_total.reshape(vec![b, kv_seq, self.num_heads, head_dim]);
        let v = match v_try_num {
            Ok(t) => t.permute(vec![0, 2, 1, 3]),
            Err(_) => {
                let v_try_kv = match v_total.reshape(vec![b, kv_seq, self.kv_heads, head_dim]) {
                    Ok(t) => t.permute(vec![0, 2, 1, 3]),
                    Err(e) => {
                        log::error!("MHA.forward_cross: reshape v failed: {}", e);
                        return query.clone();
                    }
                };
                let repeat = self.num_heads / self.kv_heads;
                let arr = v_try_kv.lock().storage.to_f32_array();
                let mut new = ndarray::ArrayD::<f32>::zeros(IxDyn(
                    &[b, self.num_heads, kv_seq, head_dim][..],
                ));
                for batch in 0..b {
                    let batch_view = arr.index_axis(ndarray::Axis(0), batch);
                    for i in 0..self.kv_heads {
                        let src = batch_view.index_axis(ndarray::Axis(0), i).to_owned();
                        for r in 0..repeat {
                            let dest_idx = i * repeat + r;
                            new.index_axis_mut(ndarray::Axis(0), batch)
                                .index_axis_mut(ndarray::Axis(0), dest_idx)
                                .assign(&src);
                        }
                    }
                }
                Tensor::new(new.into_dyn(), false)
            }
        };
        let v2 = match v.reshape(vec![b * self.num_heads, kv_seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MHA.forward_cross: reshape v2 failed: {}", e);
                return query.clone();
            }
        };

        let effective_variant = if mask.is_some() {
            AttentionVariant::Baseline
        } else {
            self.attention_variant
        };
        let sliding_window = match effective_variant {
            AttentionVariant::SlidingWindow { window_size } => Some(window_size),
            _ => None,
        };

        let out = match effective_variant {
            AttentionVariant::Baseline | AttentionVariant::SlidingWindow { .. } => {
                let k2t = k2.permute(vec![0, 2, 1]);
                let qk = q2.batched_matmul(&k2t);
                let scale = 1.0f32 / (head_dim as f32).sqrt();
                let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1][..]), scale), false);
                let mut scaled_logits = qk.mul(&scalar_tensor);
                if let Some(window_size) = sliding_window {
                    let mut window_mask_arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(
                        &[b * self.num_heads, q_seq, kv_seq][..],
                    ));
                    for i in 0..(b * self.num_heads) {
                        for r in 0..q_seq {
                            for c2 in 0..kv_seq {
                                if c2.abs_diff(r) > window_size {
                                    window_mask_arr[[i, r, c2]] = -1e9_f32;
                                }
                            }
                        }
                    }
                    let window_mask_t =
                        Tensor::new(window_mask_arr.into_dyn(), false);
                    scaled_logits = scaled_logits.add(&window_mask_t);
                }
                if let Some(m) = mask {
                    scaled_logits = scaled_logits.add(m);
                }
                let attn = scaled_logits.softmax(2);
                attn.batched_matmul(&v2)
            }
            AttentionVariant::FlashRef => {
                let flash = FlashAttentionRef::new(head_dim);
                Tensor::apply(Arc::new(flash), &[q2.clone(), k2.clone(), v2.clone()][..])
            }
            AttentionVariant::Chunked { chunk_size } => {
                let op = ChunkedAttention::new(head_dim, chunk_size);
                Tensor::apply(Arc::new(op), &[q2.clone(), k2.clone(), v2.clone()][..])
            }
        };

        let out2 = match out.reshape(vec![b, self.num_heads, q_seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MHA.forward_cross: reshape out failed: {}", e);
                return query.clone();
            }
        };
        let out3 = out2.permute(vec![0, 2, 1, 3]);
        let out4 = match out3.reshape(vec![b, q_seq, self.d_model]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MHA.forward_cross: reshape out4 failed: {}", e);
                return query.clone();
            }
        };
        self.linear_o.forward(&out4)
    }

    pub fn forward_with_paged_cache(
        &self,
        x: &Tensor,
        cache: &crate::nn::PagedKVCache,
        seq_ids: &[u64],
    ) -> Tensor {
        // 1. Projections
        let q = self.linear_q.forward(x);
        let k_new = self.linear_k.forward(x);
        let v_new = self.linear_v.forward(x);

        let b = q.lock().storage.shape()[0];
        let head_dim = self.d_model / self.num_heads;
        let k_reshaped = k_new.reshape(vec![b, 1, self.num_heads, head_dim]).unwrap();
        let v_reshaped = v_new.reshape(vec![b, 1, self.num_heads, head_dim]).unwrap();

        let k_arr = k_reshaped.to_f32_array();
        let v_arr = v_reshaped.to_f32_array();

        for i in 0..b {
            let k_slice = k_arr.index_axis(ndarray::Axis(0), i).to_owned();
            let v_slice = v_arr.index_axis(ndarray::Axis(0), i).to_owned();

            let k_t = Tensor::new(k_slice.into_dyn(), false);
            let v_t = Tensor::new(v_slice.into_dyn(), false);

            cache.reshape_and_cache(&k_t, &v_t, seq_ids[i]);
        }

        let q_view = q.reshape(vec![b, self.num_heads, head_dim]).unwrap();

        let attn_out = crate::nn::paged_attention::paged_attention(
            &q_view,
            cache,
            seq_ids,
            1.0 / (head_dim as f32).sqrt(),
            self.num_heads,
            head_dim,
        );

        let out_flat = attn_out.reshape(vec![b, 1, self.d_model]).unwrap();

        // 3. Output projection
        self.linear_o.forward(&out_flat)
    }

    /// Forward with distance matrix integrating NL-OOB distances as additional attention bias.
    /// `dist` may be 2D (seq x seq) or 3D (batch x seq x seq).
    pub fn forward_with_distance(&self, x: &Tensor, dist: &Tensor) -> Tensor {
        // debugging prints
        println!("[MHA] enter forward_with_distance");
        let shape = x.lock().storage.shape().to_vec();
        println!("[MHA] x shape {:?}", shape);
        if shape.len() != 3 {
            println!("[MHA] exit early: input not 3D");
            return x.clone();
        }
        let b = shape[0];
        let seq = shape[1];
        let dist_shape = dist.lock().storage.shape().to_vec();
        println!("[MHA] dist shape {:?}", dist_shape);
        let okay = if dist_shape == [seq, seq] {
            true
        } else if dist_shape.len() == 3
            && dist_shape[0] == b
            && dist_shape[1] == seq
            && dist_shape[2] == seq
        {
            true
        } else {
            false
        };
        if !okay {
            println!("[MHA] mismatch -> returning independent copy");
            // create a deep copy instead of cloning Arc so caller can lock both
            // tensor and original simultaneously without deadlock.
            let arr = x.lock().storage.to_f32_array();
            let requires = x.lock().requires_grad;
            return Tensor::new(arr, requires);
        }
        println!("[MHA] shapes ok, proceeding to forward_with_causal");
        // guard prints when returning
        println!("[MHA] exit forward_with_distance normally");
        self.forward_with_causal(x, false, None, Some(dist))
    }

    /// Debug: return intermediate tensors for inspection
    pub fn forward_debug(
        &self,
        x: &Tensor,
        causal: bool,
        causal_offset: Option<usize>,
        distance: Option<&Tensor>,
    ) -> HashMap<String, Tensor> {
        let mut out = HashMap::new();
        // q/k/v pre
        let mut q = self.linear_q.forward(x);
        let mut k = self.linear_k.forward(x);
        let v = self.linear_v.forward(x);
        out.insert("q_pre".to_string(), q.clone());
        out.insert("k_pre".to_string(), k.clone());
        out.insert("v_pre".to_string(), v.clone());
        // Apply RoPE if configured
        if self.use_rope {
            let offset = causal_offset.unwrap_or(0);
            q = q.rope(self.num_heads, self.rope_theta, self.rope_scale, offset);
            k = k.rope(self.kv_heads, self.rope_theta, self.rope_scale, offset);
        }
        out.insert("q_rope".to_string(), q.clone());
        out.insert("k_rope".to_string(), k.clone());
        // reshape and prepare batched matmul
        let shape = q.lock().storage.shape().to_vec();
        if shape.len() != 3 {
            return out;
        }
        let b = shape[0];
        let seq = shape[1];
        let head_dim = self.d_model / self.num_heads;
        // reshape into (b*num_heads, seq, head_dim)
        let q = match q.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t
                .permute(vec![0, 2, 1, 3])
                .reshape(vec![b * self.num_heads, seq, head_dim])
                .unwrap_or_else(|_| q.clone()),
            Err(_) => q.clone(),
        };
        let k = match k.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t
                .permute(vec![0, 2, 1, 3])
                .reshape(vec![b * self.num_heads, seq, head_dim])
                .unwrap_or_else(|_| k.clone()),
            Err(_) => k.clone(),
        };
        let v = match v.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t
                .permute(vec![0, 2, 1, 3])
                .reshape(vec![b * self.num_heads, seq, head_dim])
                .unwrap_or_else(|_| v.clone()),
            Err(_) => v.clone(),
        };
        let k2t = k.permute(vec![0, 2, 1]);
        let qk = q.batched_matmul(&k2t);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1][..]), scale), false);
        let scaled = qk.mul(&scalar_tensor);
        out.insert("scaled_logits".to_string(), scaled.clone());
        let mut scaled_logits_final = scaled.clone();
        // Apply ALiBi if present
        if self.use_alibi {
            let slopes_vec = if let Some(s) = &self.alibi_slopes {
                s.clone()
            } else {
                compute_alibi_slopes(self.num_heads)
            };
            let mut bias_arr =
                ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq][..]));
            for batch in 0..b {
                for h in 0..self.num_heads {
                    let slope = slopes_vec[h];
                    for i in 0..seq {
                        for j in 0..seq {
                            let dist_val = (j as isize - i as isize) as f32;
                            bias_arr[[batch * self.num_heads + h, i, j]] = -slope * dist_val;
                        }
                    }
                }
            }
            let bias_t = Tensor::new(bias_arr, false);
            scaled_logits_final = scaled_logits_final.add(&bias_t);
        }

        // Apply NL-OOB distance bias if provided
        if let Some(dist) = distance {
            if let Some(cfg) = self.nl_oob_config {
                if let Some(slopes_param) = &self.slopes {
                    let dist_arr = dist.to_f32_array();
                    let dist_shape = dist_arr.shape().to_vec();
                    // distance bias calculation mirroring forward_with_caching
                    if dist_shape == [seq, seq]
                        || (dist_shape.len() == 3
                            && dist_shape[0] == b
                            && dist_shape[1] == seq
                            && dist_shape[2] == seq)
                    {
                        let mut fdist = if dist_shape.len() == 2 {
                            let raw: Vec<f32> = dist_arr.iter().cloned().collect();
                            Array::from_shape_vec((1, 1, seq, seq), raw)
                                .unwrap_or_else(|_| Array::zeros((1, 1, seq, seq)))
                        } else {
                            let raw: Vec<f32> = dist_arr.iter().cloned().collect();
                            Array::from_shape_vec((b, 1, seq, seq), raw)
                                .unwrap_or_else(|_| Array::zeros((b, 1, seq, seq)))
                        };

                        if cfg == BiasFunction::Logarithmic {
                            fdist = fdist.mapv(|v| (v + 1.0f32).ln());
                        } else {
                            fdist = fdist.mapv(|v| v * v);
                        }

                        let fdist_t = Tensor::new(fdist.into_dyn(), false);
                        let nl_bias = slopes_param.mul(&fdist_t);
                        // reshape nl_bias to (b*num_heads, seq, seq) for broadcast add
                        let nl_bias_flat = nl_bias
                            .reshape(vec![b * self.num_heads, seq, seq])
                            .unwrap_or(nl_bias);
                        scaled_logits_final = scaled_logits_final.sub(&nl_bias_flat);
                    }
                }
            }
        }
        // causal mask
        if causal {
            let mut mask_arr =
                ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq][..]));
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
            let mask_t = Tensor::new(mask_arr, false);
            scaled_logits_final = scaled_logits_final.add(&mask_t);
        }
        out.insert(
            "scaled_logits_final".to_string(),
            scaled_logits_final.clone(),
        );
        let attn = scaled_logits_final.softmax(2);
        out.insert("attn_probs".to_string(), attn.clone());
        let attn_out = attn.batched_matmul(&v);
        // reshape back to (b, seq, d_model)
        let out2 = match attn_out.reshape(vec![b, self.num_heads, seq, head_dim]) {
            Ok(t) => t.permute(vec![0, 2, 1, 3]),
            Err(_) => attn_out.clone(),
        };
        let out4 = match out2.reshape(vec![b, seq, self.d_model]) {
            Ok(t) => t,
            Err(_) => attn_out.clone(),
        };
        out.insert("attn_out".to_string(), out4.clone());
        out
    }

    pub fn parameters_impl(&self) -> Vec<Tensor> {
        let mut p = self.linear_q.parameters();
        p.extend(self.linear_k.parameters());
        p.extend(self.linear_v.parameters());
        p.extend(self.linear_o.parameters());
        if let Some(s) = &self.slopes {
            p.push(s.clone());
        }
        p
    }
    pub fn named_parameters_impl(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(
            self.linear_q
                .named_parameters(&format!("{}.q_proj", prefix)),
        );
        out.extend(
            self.linear_k
                .named_parameters(&format!("{}.k_proj", prefix)),
        );
        out.extend(
            self.linear_v
                .named_parameters(&format!("{}.v_proj", prefix)),
        );
        out.extend(
            self.linear_o
                .named_parameters(&format!("{}.o_proj", prefix)),
        );
        if let Some(s) = &self.slopes {
            out.push((format!("{}.nl_oob.slopes", prefix), s.clone()));
        }
        out
    }
    pub fn load_state_dict_impl(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.linear_q
            .load_state_dict(state, &format!("{}.linear_q", prefix))?;
        // Handle key/value projection formats used in some checkpoints where
        // k/v are stored as [kv_heads * head_dim, d_model]. Expand them to
        // full [d_model, d_model] by repeating kv groups when kv_heads < num_heads.
        let key_k = format!("{}.mha.linear_k.weight", prefix);
        if let Some(k_t) = state.get(&key_k) {
            let arr = k_t.lock().storage.to_f32_array();
            if arr.ndim() == 2 {
                let rows = arr.shape()[0];
                let cols = arr.shape()[1];
                if cols == self.d_model && rows != self.d_model {
                    let head_dim = self.d_model / self.num_heads;
                    let expected_k_rows = self.kv_heads * head_dim;
                    if rows == expected_k_rows && self.num_heads.is_multiple_of(self.kv_heads) {
                        // reshape to [kv_heads, head_dim, d_model]
                        if let Ok(arr3) = arr.clone().into_dimensionality::<ndarray::Ix3>() {
                            // arr3 shape should be (kv_heads, head_dim, d_model)
                            let repeat = self.num_heads / self.kv_heads;
                            let mut expanded = Vec::with_capacity(self.num_heads * head_dim * cols);
                            for i in 0..self.kv_heads {
                                let sub = arr3.index_axis(ndarray::Axis(0), i);
                                for _r in 0..repeat {
                                    for v in sub.iter() {
                                        expanded.push(*v);
                                    }
                                }
                            }
                            if let Ok(exp_arr) = Array::from_shape_vec(
                                ndarray::IxDyn(&[self.num_heads * head_dim, cols][..]),
                                expanded,
                            ) {
                                if let Some(lk) = self.linear_k.as_f32_mut() {
                                    lk.weight =
                                        Tensor::new(exp_arr.into_dyn(), false);
                                }
                            }
                        } else {
                            // fall back to manual reshape if needed
                            if let Ok(arr2) = arr.clone().into_dimensionality::<ndarray::Ix2>() {
                                let mut expanded =
                                    Vec::with_capacity(self.num_heads * head_dim * cols);
                                // treat arr2 as (kv_heads, head_dim*cols)
                                for i in 0..self.kv_heads {
                                    let start = i * head_dim;
                                    for _r in 0..(self.num_heads / self.kv_heads) {
                                        for r in 0..head_dim {
                                            for c in 0..cols {
                                                expanded.push(arr2[[start + r, c]]);
                                            }
                                        }
                                    }
                                }
                                if let Ok(exp_arr) = Array::from_shape_vec(
                                    ndarray::IxDyn(&[self.num_heads * head_dim, cols][..]),
                                    expanded,
                                ) {
                                    if let Some(lk) = self.linear_k.as_f32_mut() {
                                        lk.weight =
                                            Tensor::new(exp_arr.into_dyn(), false);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // load v similarly
        let key_v = format!("{}.mha.linear_v.weight", prefix);
        if let Some(v_t) = state.get(&key_v) {
            let arr = v_t.lock().storage.to_f32_array();
            if arr.ndim() == 2 {
                let rows = arr.shape()[0];
                let cols = arr.shape()[1];
                if cols == self.d_model && rows != self.d_model {
                    let head_dim = self.d_model / self.num_heads;
                    let expected_v_rows = self.kv_heads * head_dim;
                    if rows == expected_v_rows && self.num_heads.is_multiple_of(self.kv_heads) {
                        if let Ok(arr3) = arr.clone().into_dimensionality::<ndarray::Ix3>() {
                            let repeat = self.num_heads / self.kv_heads;
                            let mut expanded = Vec::with_capacity(self.num_heads * head_dim * cols);
                            for i in 0..self.kv_heads {
                                let sub = arr3.index_axis(ndarray::Axis(0), i);
                                for _r in 0..repeat {
                                    for v in sub.iter() {
                                        expanded.push(*v);
                                    }
                                }
                            }
                            if let Ok(exp_arr) = Array::from_shape_vec(
                                ndarray::IxDyn(&[self.num_heads * head_dim, cols][..]),
                                expanded,
                            ) {
                                if let Some(lv) = self.linear_v.as_f32_mut() {
                                    lv.weight =
                                        Tensor::new(exp_arr.into_dyn(), false);
                                }
                            }
                        } else {
                            // fall back to manual reshape if needed
                            if let Ok(arr2) = arr.clone().into_dimensionality::<ndarray::Ix2>() {
                                let mut expanded =
                                    Vec::with_capacity(self.num_heads * head_dim * cols);
                                // treat arr2 as (kv_heads, head_dim*cols)
                                for i in 0..self.kv_heads {
                                    let start = i * head_dim;
                                    for _r in 0..(self.num_heads / self.kv_heads) {
                                        for r in 0..head_dim {
                                            for c in 0..cols {
                                                expanded.push(arr2[[start + r, c]]);
                                            }
                                        }
                                    }
                                }
                                if let Ok(exp_arr) = Array::from_shape_vec(
                                    ndarray::IxDyn(&[self.num_heads * head_dim, cols][..]),
                                    expanded,
                                ) {
                                    if let Some(lv) = self.linear_v.as_f32_mut() {
                                        lv.weight =
                                            Tensor::new(exp_arr.into_dyn(), false);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Finally, allow default loading to overwrite anything else
        self.linear_k
            .load_state_dict(state, &format!("{}.k_proj", prefix))?;
        if let Some(lk) = self.linear_k.as_f32_mut() {
            let shape = lk.weight.lock().storage.shape().to_vec();
            log::debug!("MHA.load_state_dict: k_proj loaded shape={:?}", shape);
            if shape.len() == 2 && shape[0] != self.d_model && shape[1] == self.d_model {
                let arr = lk.weight.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                lk.weight = Tensor::new(arr_t.into_dyn(), false);
                log::debug!(
                    "MHA.load_state_dict: k_proj transposed to shape={:?}",
                    lk.weight.lock().storage.shape()
                );
            }
        }

        self.linear_v
            .load_state_dict(state, &format!("{}.v_proj", prefix))?;
        if let Some(lv) = self.linear_v.as_f32_mut() {
            let shape = lv.weight.lock().storage.shape().to_vec();
            log::debug!("MHA.load_state_dict: v_proj loaded shape={:?}", shape);
            if shape.len() == 2 && shape[0] != self.d_model && shape[1] == self.d_model {
                let arr = lv.weight.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                lv.weight = Tensor::new(arr_t.into_dyn(), false);
                log::debug!(
                    "MHA.load_state_dict: v_proj transposed to shape={:?}",
                    lv.weight.lock().storage.shape()
                );
            }
        }

        self.linear_o
            .load_state_dict(state, &format!("{}.o_proj", prefix))?;
        if let Some(lo) = self.linear_o.as_f32_mut() {
            let shape = lo.weight.lock().storage.shape().to_vec();
            log::debug!("MHA.load_state_dict: o_proj loaded shape={:?}", shape);
            if shape.len() == 2 && shape[0] != self.d_model && shape[1] == self.d_model {
                let arr = lo.weight.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                lo.weight = Tensor::new(arr_t.into_dyn(), false);
                log::debug!(
                    "MHA.load_state_dict: o_proj transposed to shape={:?}",
                    lo.weight.lock().storage.shape()
                );
            }
        }
        Ok(())
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        self.parameters_impl()
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.named_parameters_impl(prefix)
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
            if arr.ndim() == 1 && !arr.is_empty() {
                if let Ok(vec1) = arr.into_dimensionality::<ndarray::Ix1>() {
                    let v = vec1[0];
                    if v == 1.0 {
                        self.nl_oob_config = Some(BiasFunction::Gaussian);
                    } else {
                        self.nl_oob_config = Some(BiasFunction::Logarithmic);
                    }
                } else {
                    log::error!("MultiHeadAttention load_state_dict: nl_oob.config had unexpected shape, skipping");
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

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_impl(input)
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
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
pub struct TransformerBlock {
    pub mha: MultiHeadAttention,
    pub linear1: LinearLayer,
    pub linear2: LinearLayer,
    pub causal: bool,
    // Per-layer KV cache for incremental decoding (packed storage)
    pub kv_cache: Option<crate::nn::KVCache>,
    // Llama-style pre-norm mode uses RMSNorm; store gamma parameters when enabled
    pub llama_style: bool,
    pub rms_attn_gamma: Option<Tensor>,
    pub rms_ffn_gamma: Option<Tensor>,
}
impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "TransformerBlock::new: d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }
        Ok(TransformerBlock {
            mha: MultiHeadAttention::new(d_model, num_heads),
            linear1: LinearLayer::new_f32(d_model, d_ff, true),
            linear2: LinearLayer::new_f32(d_ff, d_model, true),
            causal: false,
            kv_cache: None,
            llama_style: false,
            rms_attn_gamma: None,
            rms_ffn_gamma: None,
        })
    }
    pub fn new_with_kv_and_rope(config: TransformerConfig) -> Result<Self, String> {
        if !config.d_model.is_multiple_of(config.num_heads) {
            return Err(format!("TransformerBlock::new_with_kv_and_rope: d_model ({}) must be divisible by num_heads ({})", config.d_model, config.num_heads));
        }
        if !config.num_heads.is_multiple_of(config.kv_heads) {
            return Err(format!("TransformerBlock::new_with_kv_and_rope: num_heads ({}) must be divisible by kv_heads ({})", config.num_heads, config.kv_heads));
        }
        Ok(TransformerBlock {
            mha: MultiHeadAttention::new_with_kv_and_rope(
                config.d_model,
                config.num_heads,
                config.kv_heads,
                config.use_rope,
                config.rope_theta,
                config.rope_scale,
                config.bias,
            ),
            linear1: LinearLayer::new_f32(config.d_model, config.d_ff, true),
            linear2: LinearLayer::new_f32(config.d_ff, config.d_model, true),
            causal: true,
            kv_cache: None,
            llama_style: false,
            rms_attn_gamma: None,
            rms_ffn_gamma: None,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    pub kv_heads: usize,
    pub use_rope: bool,
    pub rope_theta: f32,
    pub rope_scale: f32,
    pub bias: bool,
}

impl TransformerBlock {
    pub fn new_with_nl_oob(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        config: BiasFunction,
        max_scale: f32,
    ) -> Result<Self, String> {
        let mut t = TransformerBlock::new_with_kv_and_rope(TransformerConfig {
            d_model,
            d_ff,
            num_heads,
            kv_heads: num_heads,
            use_rope: false,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            bias: true,
        })?;
        t.mha = MultiHeadAttention::new_with_nl_oob(d_model, num_heads, config, max_scale);
        Ok(t)
    }

    /// Create a Llama-style TransformerBlock.
    /// Defaults:
    /// - `bias`: whether to include biases in linear layers. Set to `false` for Llama-style biasless dense layers.
    /// - `use_rope`: apply RoPE to q/k during attention.
    pub fn new_llama_style(config: TransformerConfig) -> Result<Self, String> {
        // linear1 must output 2*d_ff for SwiGLU splitting
        if !config.d_model.is_multiple_of(config.num_heads) {
            return Err(format!("TransformerBlock::new_llama_style: d_model ({}) must be divisible by num_heads ({})", config.d_model, config.num_heads));
        }
        if !config.num_heads.is_multiple_of(config.kv_heads) {
            return Err(format!("TransformerBlock::new_llama_style: num_heads ({}) must be divisible by kv_heads ({})", config.num_heads, config.kv_heads));
        }
        let linear1 = LinearLayer::new_f32(config.d_model, config.d_ff * 2, config.bias);
        let linear2 = LinearLayer::new_f32(config.d_ff, config.d_model, config.bias);
        let gamma_attn = Tensor::new(
            Array::from_elem(IxDyn(&[config.d_model][..]), 1.0f32),
            true,
        );
        let gamma_ffn = Tensor::new(
            Array::from_elem(IxDyn(&[config.d_model][..]), 1.0f32),
            true,
        );
        Ok(TransformerBlock {
            mha: MultiHeadAttention::new_with_kv_and_rope(
                config.d_model,
                config.num_heads,
                config.kv_heads,
                config.use_rope,
                config.rope_theta,
                config.rope_scale,
                config.bias,
            ),
            linear1,
            linear2,
            causal: true,
            llama_style: true,
            kv_cache: None,
            rms_attn_gamma: Some(gamma_attn),
            rms_ffn_gamma: Some(gamma_ffn),
        })
    }
    pub fn new_decoder(d_model: usize, d_ff: usize, num_heads: usize) -> Result<Self, String> {
        let mut t = TransformerBlock::new(d_model, d_ff, num_heads)?;
        t.causal = true;
        Ok(t)
    }

    /// Accessor: mutable reference to the optional per-layer KV cache
    pub fn kv_cache_mut(&mut self) -> &mut Option<crate::nn::KVCache> {
        &mut self.kv_cache
    }

    /// Set the per-layer KV cache to the provided cache
    pub fn set_kv_cache(&mut self, cache: crate::nn::KVCache) {
        self.kv_cache = Some(cache);
    }

    /// Clear any per-layer KV cache
    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    /// Return an owned clone of the KV cache if present
    pub fn kv_cache_clone(&self) -> Option<crate::nn::KVCache> {
        self.kv_cache.clone()
    }

    /// Truncate the per-layer KV cache by n tokens from the end
    pub fn truncate_kv_cache(&mut self, n: usize) {
        if let Some(cache) = &mut self.kv_cache {
            cache.truncate(n);
        }
    }

    /// Forward a single token through this block using the per-layer KV cache.
    ///
    /// This is the canonical incremental-decoding path: it normalizes the input,
    /// runs attention against the cached keys/values, appends the new key/value pair,
    /// then applies the FFN branch with residual connections.
    ///
    /// # Arguments
    /// * `x` — Input tensor of shape `[batch, 1, d_model]` (single token)
    /// * `causal_offset` — Optional offset for multimodal contexts (image-token count)
    ///
    /// # Returns
    /// Output tensor of shape `[batch, 1, d_model]`
    pub fn forward_single_token(
        &mut self,
        x: &Tensor,
        causal_offset: Option<usize>,
    ) -> Result<Tensor, String> {
        // Validate input shape: must be [batch, 1, d_model]
        let shape = x.lock().storage.shape();
        if shape.len() != 3 || shape[1] != 1 {
            return Err(format!(
                "forward_single_token expects [batch, 1, d_model], got {:?}",
                shape
            ));
        }

        if self.llama_style {
            // Pre-norm RMSNorm -> Attention (with KV cache) -> Residual -> Pre-norm RMSNorm -> SwiGLU FFN
            let gamma_attn = match self.rms_attn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = shape[2];
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            let x_norm = x.rmsnorm(&gamma_attn, 2, 1e-5);

            // Use KV cache for incremental decoding
            let attn_out = if let Some(kvc) = self.kv_cache.as_mut() {
                self.mha.forward_with_caching(
                    &x_norm,
                    true,
                    causal_offset,
                    Some(kvc),
                    None,
                    None,
                )
            } else {
                // Fallback: no cache, compute attention fresh
                self.mha.forward_with_causal(&x_norm, true, causal_offset, None)
            };

            let x2 = x.add(&attn_out);

            let gamma_ffn = match self.rms_ffn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x2.lock().storage.shape()[2];
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            let x2_norm = x2.rmsnorm(&gamma_ffn, 2, 1e-5);
            let ff = self.linear1.forward(&x2_norm).swiglu();
            let ff = self.linear2.forward(&ff);

            Ok(x2.add(&ff))
        } else {
            // Standard post-norm: Attention (with KV cache) -> Residual -> LayerNorm -> FFN -> Residual
            let attn_out = if let Some(kvc) = self.kv_cache.as_mut() {
                self.mha.forward_with_caching(
                    x,
                    true,
                    causal_offset,
                    Some(kvc),
                    None,
                    None,
                )
            } else {
                self.mha.forward_with_causal(x, true, causal_offset, None)
            };

            let x2 = x.add(&attn_out);
            let dim = shape[2];
            let gamma = Tensor::new(Array::ones(IxDyn(&[dim][..])), true);
            let beta = Tensor::new(Array::zeros(IxDyn(&[dim][..])), true);
            let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);

            let ff = self.linear1.forward(&x2norm).relu();
            let ff = self.linear2.forward(&ff);

            Ok(x2.add(&ff))
        }
    }

    /// Initialize the per-layer KV cache for a given sequence length.
    ///
    /// This pre-allocates packed storage so that subsequent `forward_single_token` calls
    /// can append without reallocation overhead.
    pub fn init_kv_cache_for_seq_len(&mut self, seq_len: usize) -> Result<(), String> {
        if seq_len == 0 {
            return Err("init_kv_cache_for_seq_len: seq_len must be > 0".to_string());
        }

        let batch = 1; // Single sequence for inference
        let head_dim = self.mha.d_model / self.mha.num_heads;

        // Create empty packed tensors: [batch, seq_len, d_model] for keys/values
        let k_init = Tensor::new(
            Array::zeros(IxDyn(&[batch, seq_len, self.mha.d_model][..])),
            false,
        );
        let v_init = Tensor::new(
            Array::zeros(IxDyn(&[batch, seq_len, self.mha.d_model][..])),
            false,
        );

        let mut cache = crate::nn::KVCache::new();
        cache.set_packed(k_init, v_init);
        self.kv_cache = Some(cache);

        log::info!(
            "TransformerBlock: initialized KV cache for seq_len={}",
            seq_len
        );
        Ok(())
    }

    /// Clear the per-layer KV cache and reset to empty state.
    pub fn reset_kv_cache(&mut self) {
        if let Some(ref mut cache) = self.kv_cache {
            // Keep packed storage but zero it out by truncating all tokens
            cache.truncate(cache.seq_len());
        }
    }

    /// Non-mutating forward of the block which does not touch or populate per-layer KV cache.
    /// This is used for full-batch encoder/decoder forward passes where cache mutation is not desired.
    pub fn forward_block_no_cache(&self, x: &Tensor) -> Tensor {
        if self.llama_style {
            let gamma_attn = match self.rms_attn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_attn_gamma; using default ones tensor");
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            let x_norm = x.rmsnorm(&gamma_attn, 2, 1e-5);
            let attn_out = self
                .mha
                .forward_with_causal(&x_norm, self.causal, None, None);
            let x2 = x.add(&attn_out);
            let gamma_ffn = match self.rms_ffn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x2.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_ffn_gamma; using default ones tensor");
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            let x2_norm = x2.rmsnorm(&gamma_ffn, 2, 1e-5);
            let ff = self.linear1.forward(&x2_norm).swiglu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        } else {
            let attn_out = self.mha.forward_with_causal(x, self.causal, None, None);
            let x2 = x.add(&attn_out);
            let dim = x.lock().storage.shape()[2];
            let gamma = Tensor::new(Array::ones(IxDyn(&[dim][..])), true);
            let beta = Tensor::new(Array::zeros(IxDyn(&[dim][..])), true);
            let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
            let ff = self.linear1.forward(&x2norm).relu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        }
    }

    pub fn forward_block(&mut self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        if self.llama_style {
            // Pre-norm RMSNorm -> Attention -> Residual -> Pre-norm RMSNorm -> SwiGLU FFN
            let gamma_attn = match self.rms_attn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_attn_gamma; using default ones tensor");
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            // RMSNorm along the last axis
            let x_norm = x.rmsnorm(&gamma_attn, 2, 1e-5);
            // Use per-layer KV cache if present, otherwise fallback to causal no-cache path
            let attn_out = if let Some(kvc) = self.kv_cache.as_mut() {
                self.mha
                    .forward_with_caching(&x_norm, self.causal, None, Some(kvc), mask, None)
            } else {
                self.mha
                    .forward_with_causal(&x_norm, self.causal, None, None)
            };

            let x2 = x.add(&attn_out);
            let gamma_ffn = match self.rms_ffn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x2.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_ffn_gamma; using default ones tensor");
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            let x2_norm = x2.rmsnorm(&gamma_ffn, 2, 1e-5);
            // linear1 outputs 2*d_ff, SwiGLU will split it to produce d_ff activation
            let ff = self.linear1.forward(&x2_norm).swiglu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        } else {
            let attn_out = if let Some(kvc) = self.kv_cache.as_mut() {
                self.mha
                    .forward_with_caching(x, self.causal, None, Some(kvc), mask, None)
            } else {
                self.mha.forward_with_causal(x, self.causal, None, None)
            };
            let x2 = x.add(&attn_out);
            let dim = x.lock().storage.shape()[2];
            let gamma = Tensor::new(Array::ones(IxDyn(&[dim][..])), true);
            let beta = Tensor::new(Array::zeros(IxDyn(&[dim][..])), true);
            let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
            let ff = self.linear1.forward(&x2norm).relu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        }
    }

    /// Debug helper: return intermediate tensors from the block for inspection
    pub fn forward_block_debug(&self, x: &Tensor) -> HashMap<String, Tensor> {
        let mut out = HashMap::new();
        if self.llama_style {
            let gamma_attn = match self.rms_attn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x.lock().storage.shape()[2];
                    log::error!(
                        "llama_style missing rms_attn_gamma in debug; using default ones tensor"
                    );
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            let x_norm = x.rmsnorm(&gamma_attn, 2, 1e-5);
            out.insert("x_norm".to_string(), x_norm.clone());
            let mut attn_map = self.mha.forward_debug(&x_norm, self.causal, None, None);
            out.extend(attn_map.drain());
            let attn_out = match out.get("attn_out") {
                Some(a) => a.clone(),
                None => {
                    log::error!("forward_block_debug: attn_out missing from attention map; using zeros tensor");
                    let shape = x.lock().storage.shape().to_vec();
                    Tensor::new(Array::zeros(IxDyn(&shape)), false)
                }
            };
            let x_after = if x.lock().storage.shape() == attn_out.lock().storage.shape() {
                x.add(&attn_out)
            } else {
                x.clone()
            };
            out.insert("x_after_attn".to_string(), x_after.clone());
            let x2 = x_after;
            let gamma_ffn = match self.rms_ffn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x2.lock().storage.shape()[2];
                    log::error!(
                        "llama_style missing rms_ffn_gamma in debug; using default ones tensor"
                    );
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            let x2_norm = x2.rmsnorm(&gamma_ffn, 2, 1e-5);
            out.insert("x2_norm".to_string(), x2_norm.clone());
            let ff_lin1 = self.linear1.forward(&x2_norm);
            out.insert("ff_lin1".to_string(), ff_lin1.clone());
            let ff_swiglu = ff_lin1.swiglu();
            out.insert("ff_swiglu".to_string(), ff_swiglu.clone());
            let ff_lin2 = self.linear2.forward(&ff_swiglu);
            out.insert("ff_out".to_string(), ff_lin2.clone());
            out.insert("output".to_string(), x2.add(&ff_lin2));
        } else {
            let mut attn_map = self.mha.forward_debug(x, self.causal, None, None);
            out.extend(attn_map.drain());
            let attn_out = match out.get("attn_out") {
                Some(a) => a.clone(),
                None => {
                    log::error!("forward_block_debug: attn_out missing; using zeros tensor");
                    let shape = x.lock().storage.shape().to_vec();
                    Tensor::new(Array::zeros(IxDyn(&shape)), false)
                }
            };
            let x_after = if x.lock().storage.shape() == attn_out.lock().storage.shape() {
                x.add(&attn_out)
            } else {
                x.clone()
            };
            let x2 = x_after.clone();
            out.insert("x_after_attn".to_string(), x2.clone());
            let dim = x.lock().storage.shape()[2];
            let gamma = Tensor::new(Array::ones(IxDyn(&[dim][..])), true);
            let beta = Tensor::new(Array::zeros(IxDyn(&[dim][..])), true);
            let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
            out.insert("x2_norm".to_string(), x2norm.clone());
            let ff_lin1 = self.linear1.forward(&x2norm).relu();
            out.insert("ff_lin1".to_string(), ff_lin1.clone());
            let ff_lin2 = self.linear2.forward(&ff_lin1);
            out.insert("ff_out".to_string(), ff_lin2.clone());
            out.insert("output".to_string(), x2.add(&ff_lin2));
        }
        out
    }

    /// Backwards-compatible wrapper for older API that accepted a causal offset.
    pub fn forward_block_with_causal_offset(
        &mut self,
        x: &Tensor,
        causal_offset: Option<usize>,
    ) -> Tensor {
        if self.llama_style {
            let gamma_attn = match self.rms_attn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_attn_gamma; using default ones tensor");
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            let x_norm = x.rmsnorm(&gamma_attn, 2, 1e-5);
            let attn_out = if let Some(kvc) = self.kv_cache.as_mut() {
                self.mha.forward_with_caching(
                    &x_norm,
                    self.causal,
                    causal_offset,
                    Some(kvc),
                    None,
                    None,
                )
            } else {
                self.mha
                    .forward_with_causal(&x_norm, self.causal, causal_offset, None)
            };
            let x2 = x.add(&attn_out);
            let gamma_ffn = match self.rms_ffn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x2.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_ffn_gamma; using default ones tensor");
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            let x2_norm = x2.rmsnorm(&gamma_ffn, 2, 1e-5);
            let ff = self.linear1.forward(&x2_norm).swiglu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        } else {
            let attn_out = if let Some(kvc) = self.kv_cache.as_mut() {
                self.mha
                    .forward_with_caching(x, self.causal, causal_offset, Some(kvc), None, None)
            } else {
                self.mha
                    .forward_with_causal(x, self.causal, causal_offset, None)
            };
            let x2 = x.add(&attn_out);
            let dim = x.lock().storage.shape()[2];
            let gamma = Tensor::new(Array::ones(IxDyn(&[dim][..])), true);
            let beta = Tensor::new(Array::zeros(IxDyn(&[dim][..])), true);
            let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
            let ff = self.linear1.forward(&x2norm).relu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        }
    }
    pub fn forward_block_with_distance(&self, x: &Tensor, dist: &Tensor) -> Tensor {
        if self.llama_style {
            let gamma_attn = match self.rms_attn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_attn_gamma; using default ones tensor");
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            let x_norm = x.rmsnorm(&gamma_attn, 2, 1e-5);
            let attn_out = if let Some(_kvc) = self.kv_cache.as_ref() {
                // In forward_block_with_distance we don't assume we can mutate self.kv_cache easily if it's &self,
                // but MultiHeadAttention::forward_with_distance was also &self.
                // For now, call forward_with_caching with distance.
                self.mha
                    .forward_with_caching(&x_norm, self.causal, None, None, None, Some(dist))
            } else {
                self.mha
                    .forward_with_caching(&x_norm, self.causal, None, None, None, Some(dist))
            };
            let x2 = x.add(&attn_out);
            let gamma_ffn = match self.rms_ffn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x2.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_ffn_gamma; using default ones tensor");
                    Tensor::new(Array::ones(IxDyn(&[dim][..])), true)
                }
            };
            let x2_norm = x2.rmsnorm(&gamma_ffn, 2, 1e-5);
            let ff = self.linear1.forward(&x2_norm).swiglu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        } else {
            let attn_out =
                self.mha
                    .forward_with_caching(x, self.causal, None, None, None, Some(dist));
            let x2 = x.add(&attn_out);
            let dim = x.lock().storage.shape()[2];
            let gamma = Tensor::new(Array::ones(IxDyn(&[dim][..])), true);
            let beta = Tensor::new(Array::zeros(IxDyn(&[dim][..])), true);
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
        if let Some(g) = &self.rms_attn_gamma {
            p.push(g.clone());
        }
        if let Some(g) = &self.rms_ffn_gamma {
            p.push(g.clone());
        }
        p
    }
    pub fn named_parameters_impl(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(self.mha.named_parameters(&format!("{}.self_attn", prefix)));
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
            .load_state_dict(state, &format!("{}.self_attn", prefix))?;
        self.linear1
            .load_state_dict(state, &format!("{}.linear1", prefix))?;
        self.linear2
            .load_state_dict(state, &format!("{}.linear2", prefix))?;

        // LLaMA-style keys: input/post layernorm and MLP naming
        // input_layernorm.weight -> rms_attn_gamma
        let key_input_ln = format!("{}.input_layernorm.weight", prefix);
        if let Some(g) = state.get(&key_input_ln) {
            let mut glock = g.lock();
            glock.requires_grad = true;
            self.rms_attn_gamma = Some(g.clone());
        }
        // post_attention_layernorm.weight -> rms_ffn_gamma
        let key_post_ln = format!("{}.post_attention_layernorm.weight", prefix);
        if let Some(g) = state.get(&key_post_ln) {
            let mut glock = g.lock();
            glock.requires_grad = true;
            self.rms_ffn_gamma = Some(g.clone());
        }

        // MLP naming: gate_proj + down_proj -> linear1 weight (concat), up_proj -> linear2 weight
        let gate_key = format!("{}.mlp.gate_proj.weight", prefix);
        let down_key = format!("{}.mlp.down_proj.weight", prefix);
        if let (Some(gate_w), Some(down_w)) = (state.get(&gate_key), state.get(&down_key)) {
            // These are tensors from the state dict, so we can access storage directly
            let gate_arr = gate_w.lock().storage.to_f32_array();
            let down_arr = down_w.lock().storage.to_f32_array();
            // Determine how to concatenate respecting the existing linear1 weight shape
            if let Some(l1) = self.linear1.as_f32_mut() {
                let lin1_shape = l1.weight.lock().storage.shape().to_vec();
                if lin1_shape.len() == 2 {
                    let (r, c) = (lin1_shape[0], lin1_shape[1]);
                    // Case A: both have shape (r, x) and x+x == c -> concat on axis=1
                    if gate_arr.shape()[0] == r
                        && down_arr.shape()[0] == r
                        && gate_arr.shape()[1] + down_arr.shape()[1] == c
                    {
                        use ndarray::Axis;
                        let combined = match ndarray::concatenate(
                            Axis(1),
                            &[gate_arr.view(), down_arr.view()][..],
                        ) {
                            Ok(ca) => ca,
                            Err(e) => {
                                return Err(format!(
                                    "Failed to concatenate gate/down projections: {}",
                                    e
                                ))
                            }
                        };
                        l1.weight = Tensor::new(combined.into_dyn(), false);
                    } else if gate_arr.shape()[1] == r
                        && down_arr.shape()[1] == r
                        && gate_arr.shape()[0] + down_arr.shape()[0] == c
                    {
                        // Case B: inputs are transposed -> transpose both and concat
                        let ga_t = match gate_arr.into_dimensionality::<ndarray::Ix2>() {
                            Ok(m) => m.reversed_axes().into_dyn(),
                            Err(e) => return Err(format!("Unexpected gate_proj dim: {}", e)),
                        };
                        let da_t = match down_arr.into_dimensionality::<ndarray::Ix2>() {
                            Ok(m) => m.reversed_axes().into_dyn(),
                            Err(e) => return Err(format!("Unexpected down_proj dim: {}", e)),
                        };
                        use ndarray::Axis;
                        let combined =
                            match ndarray::concatenate(Axis(1), &[ga_t.view(), da_t.view()][..]) {
                                Ok(ca) => ca,
                                Err(e) => return Err(format!(
                                    "Failed to concatenate transposed gate/down projections: {}",
                                    e
                                )),
                            };
                        l1.weight = Tensor::new(combined.into_dyn(), false);
                    } else if gate_arr.shape()[1] == r
                        && down_arr.shape()[0] == r
                        && gate_arr.shape()[0] + down_arr.shape()[1] == c
                    {
                        // Case C: gate is transposed only; transpose gate and concat
                        let ga_t = match gate_arr.into_dimensionality::<ndarray::Ix2>() {
                            Ok(m) => m.reversed_axes().into_dyn(),
                            Err(e) => return Err(format!("Unexpected gate_proj dim: {}", e)),
                        };
                        use ndarray::Axis;
                        let combined = match ndarray::concatenate(
                            Axis(1),
                            &[ga_t.view(), down_arr.view()][..],
                        ) {
                            Ok(ca) => ca,
                            Err(e) => {
                                return Err(format!(
                                    "Failed to concatenate transposed gate/down projections: {}",
                                    e
                                ))
                            }
                        };
                        l1.weight = Tensor::new(combined.into_dyn(), false);
                    } else if gate_arr.shape()[0] == r
                        && down_arr.shape()[1] == r
                        && gate_arr.shape()[1] + down_arr.shape()[0] == c
                    {
                        // Case D: down_proj is transposed only; transpose down and concat
                        let da_t = match down_arr.into_dimensionality::<ndarray::Ix2>() {
                            Ok(m) => m.reversed_axes().into_dyn(),
                            Err(e) => return Err(format!("Unexpected down_proj dim: {}", e)),
                        };
                        use ndarray::Axis;
                        let combined = match ndarray::concatenate(
                            Axis(1),
                            &[gate_arr.view(), da_t.view()][..],
                        ) {
                            Ok(ca) => ca,
                            Err(e) => {
                                return Err(format!(
                                    "Failed to concatenate gate/down(transposed) projections: {}",
                                    e
                                ))
                            }
                        };
                        l1.weight = Tensor::new(combined.into_dyn(), false);
                    } else {
                        return Err(format!("Gate/down projections shapes incompatible: gate={:?} down={:?} expected lin1={:?}", gate_arr.shape(), down_arr.shape(), lin1_shape));
                    }
                }
            }
        }
        let up_key = format!("{}.mlp.up_proj.weight", prefix);
        if let Some(up_w) = state.get(&up_key) {
            if let Some(l2) = self.linear2.as_f32_mut() {
                l2.weight = up_w.clone();
            }
        }

        Ok(())
    }
}
impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_block_no_cache(input)
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
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
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
impl Module for EncoderDecoderTransformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut enc = input.clone();
        for blk in &self.encoder_blocks {
            enc = blk.forward_block_no_cache(&enc);
        }
        let mut dec = enc.clone();
        for blk in &self.decoder_blocks {
            dec = blk.forward_block_no_cache(&dec);
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

/// T5-style encoder-decoder wrapper with shared token embedding and explicit
/// decoder cross-attention blocks.
#[derive(Clone)]
pub struct T5EncoderDecoder {
    pub shared_embedding: Tensor,
    pub encoder_blocks: Vec<TransformerBlock>,
    pub decoder_blocks: Vec<TransformerBlock>,
    pub decoder_cross_attn: Vec<CrossAttention>,
    pub ln_gamma: Tensor,
    pub ln_beta: Tensor,
    pub lm_head: LinearLayer,
}

impl T5EncoderDecoder {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
    ) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "T5EncoderDecoder::new: d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }
        if !num_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "T5EncoderDecoder::new: num_heads ({}) must be divisible by kv_heads ({})",
                num_heads, kv_heads
            ));
        }

        let shared_embedding = Tensor::new(
            Array::zeros(IxDyn(&[vocab_size, d_model][..])),
            true,
        );

        let mut encoder_blocks = Vec::with_capacity(num_layers);
        let mut decoder_blocks = Vec::with_capacity(num_layers);
        let mut decoder_cross_attn = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            encoder_blocks.push(TransformerBlock::new(d_model, d_ff, num_heads)?);
            decoder_blocks.push(TransformerBlock::new_decoder(d_model, d_ff, num_heads)?);
            decoder_cross_attn.push(CrossAttention::new(
                d_model, num_heads, kv_heads, false, 10000.0, 1.0, true,
            )?);
        }

        let ln_gamma = Tensor::new(Array::ones(IxDyn(&[d_model][..])), true);
        let ln_beta = Tensor::new(Array::zeros(IxDyn(&[d_model][..])), true);
        let lm_head = LinearLayer::new_f32(d_model, vocab_size, false);

        Ok(Self {
            shared_embedding,
            encoder_blocks,
            decoder_blocks,
            decoder_cross_attn,
            ln_gamma,
            ln_beta,
            lm_head,
        })
    }

    pub fn forward_seq2seq(
        &self,
        encoder_input_ids: &Tensor,
        decoder_input_ids: &Tensor,
        encoder_mask: Option<&Tensor>,
        decoder_mask: Option<&Tensor>,
        cross_mask: Option<&Tensor>,
    ) -> Tensor {
        let enc_shape = encoder_input_ids.lock().storage.shape().to_vec();
        let dec_shape = decoder_input_ids.lock().storage.shape().to_vec();
        if enc_shape.len() != 2 || dec_shape.len() != 2 {
            log::error!(
                "T5EncoderDecoder.forward_seq2seq expects [batch, seq] ids, got enc={:?}, dec={:?}",
                enc_shape,
                dec_shape
            );
            return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
        }

        let mut enc = Tensor::embedding_lookup(&self.shared_embedding, encoder_input_ids);
        for blk in &self.encoder_blocks {
            enc = blk.forward_block_no_cache(&enc);
            if let Some(m) = encoder_mask {
                enc = enc.add(m);
            }
        }

        let mut dec = Tensor::embedding_lookup(&self.shared_embedding, decoder_input_ids);
        for (i, blk) in self.decoder_blocks.iter().enumerate() {
            dec = blk.forward_block_no_cache(&dec);
            if let Some(m) = decoder_mask {
                dec = dec.add(m);
            }
            if let Some(ca) = self.decoder_cross_attn.get(i) {
                let cross = ca.forward_cross(&dec, &enc, cross_mask);
                dec = dec.add(&cross);
            }
        }

        dec = dec.layer_norm(2, 1e-5, &self.ln_gamma, &self.ln_beta);
        self.lm_head.forward(&dec)
    }
}

impl Module for T5EncoderDecoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Compatibility forward: uses the same token ids for encoder and decoder paths.
        self.forward_seq2seq(input, input, None, None, None)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![
            self.shared_embedding.clone(),
            self.ln_gamma.clone(),
            self.ln_beta.clone(),
        ];
        for b in &self.encoder_blocks {
            p.extend(b.parameters());
        }
        for b in &self.decoder_blocks {
            p.extend(b.parameters());
        }
        for c in &self.decoder_cross_attn {
            p.extend(c.parameters());
        }
        p.extend(self.lm_head.parameters());
        p
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = vec![
            (
                format!("{}.shared_embedding.weight", prefix),
                self.shared_embedding.clone(),
            ),
            (format!("{}.ln_f.weight", prefix), self.ln_gamma.clone()),
            (format!("{}.ln_f.bias", prefix), self.ln_beta.clone()),
        ];
        for (i, b) in self.encoder_blocks.iter().enumerate() {
            out.extend(b.named_parameters(&format!("{}.encoder.blocks.{}", prefix, i)));
        }
        for (i, b) in self.decoder_blocks.iter().enumerate() {
            out.extend(b.named_parameters(&format!("{}.decoder.blocks.{}", prefix, i)));
        }
        for (i, c) in self.decoder_cross_attn.iter().enumerate() {
            out.extend(c.named_parameters(&format!("{}.decoder.cross_attn.{}", prefix, i)));
        }
        out.extend(
            self.lm_head
                .named_parameters(&format!("{}.lm_head", prefix)),
        );
        out
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        if let Some(t) = state.get(&format!("{}.shared_embedding.weight", prefix)) {
            self.shared_embedding = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.ln_f.weight", prefix)) {
            self.ln_gamma = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.ln_f.bias", prefix)) {
            self.ln_beta = t.clone();
        }
        for (i, b) in self.encoder_blocks.iter_mut().enumerate() {
            b.load_state_dict(state, &format!("{}.encoder.blocks.{}", prefix, i))?;
        }
        for (i, b) in self.decoder_blocks.iter_mut().enumerate() {
            b.load_state_dict(state, &format!("{}.decoder.blocks.{}", prefix, i))?;
        }
        for (i, c) in self.decoder_cross_attn.iter_mut().enumerate() {
            c.load_state_dict(state, &format!("{}.decoder.cross_attn.{}", prefix, i))?;
        }
        self.lm_head
            .load_state_dict(state, &format!("{}.lm_head", prefix))?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
pub struct Llama {
    pub embed_tokens: Tensor,
    pub layers: Vec<TransformerBlock>,
    pub norm: Tensor, // RMSNorm gamma
    pub lm_head: LinearLayer,
}

impl Llama {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
    ) -> Result<Self, String> {
        let embed_tokens = Tensor::new(
            Array::zeros(IxDyn(&[vocab_size, d_model][..])),
            true,
        );
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerBlock::new_llama_style(TransformerConfig {
                d_model,
                d_ff,
                num_heads,
                kv_heads,
                use_rope: true,
                bias: false,
                rope_theta: 10000.0,
                rope_scale: 1.0,
            })?);
        }
        let norm = Tensor::new(
            Array::from_elem(IxDyn(&[d_model][..]), 1.0f32),
            true,
        );
        let lm_head = LinearLayer::new_f32(d_model, vocab_size, false); // no bias for lm_head
        Ok(Llama {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }
    pub fn forward_with_mask(&mut self, input: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // input: [batch, seq] token ids OR [seq] for a single sequence
        let input_shape = input.lock().storage.shape().to_vec();
        let single_seq = input_shape.len() == 1;

        // Embedding lookup
        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        let xs = x.lock().storage.shape().to_vec();

        // If single sequence (no batch dim) -> reshape to [1, seq, d_model]
        if single_seq && xs.len() == 2 {
            let seq = xs[0];
            let dim = xs[1];
            x = match x.reshape(vec![1, seq, dim]) {
                Ok(t) => t,
                Err(e) => {
                    log::error!(
                        "Llama.forward_with_mask: failed to reshape embedding for single sequence: {}",
                        e
                    );
                    return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
                }
            };
        }

        // Iterate layers
        for layer in self.layers.iter_mut() {
            // We can catch unwind here if desired, similar to forward
            // But for mutable it's trickier with AssertUnwindSafe on mut reference?
            // std::panic::AssertUnwindSafe(layer) might imply shared Ref?
            // Logic: Just call directly for now to avoid complexity of UnwindSafe on &mut T
            x = layer.forward_block(&x, mask);
        }

        // RMSNorm
        x = x.rmsnorm(&self.norm, 2, 1e-5);
        let logits = self.lm_head.forward(&x);

        // If input was single sequence, remove the batch dim to return [seq, vocab]
        if single_seq {
            let ls = logits.lock().storage.shape().to_vec();
            if ls.len() == 3 && ls[0] == 1 {
                if let Ok(reshaped) = logits.reshape(vec![ls[1], ls[2]]) {
                    return reshaped;
                }
            }
        }
        logits
    }

    pub fn set_kv_cache(&mut self, use_cache: bool) {
        for layer in self.layers.iter_mut() {
            if use_cache {
                layer.set_kv_cache(crate::nn::KVCache::new());
            } else {
                layer.clear_kv_cache();
            }
        }
    }

    pub fn truncate_kv_cache(&mut self, n: usize) {
        for layer in self.layers.iter_mut() {
            layer.truncate_kv_cache(n);
        }
    }
}

impl Module for Llama {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, seq] token ids OR [seq] for a single sequence
        let input_shape = input.lock().storage.shape().to_vec();
        let single_seq = input_shape.len() == 1;
        log::debug!(
            "Llama.forward: input_shape={:?} single_seq={}",
            input_shape,
            single_seq
        );
        // Embedding lookup: will return [batch, seq, d_model] or [seq, d_model]
        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        let xs = x.lock().storage.shape().to_vec();
        log::debug!("Llama.forward: embedding output shape={:?}", xs);
        // If single sequence (no batch dim) -> reshape to [1, seq, d_model]
        if single_seq {
            if xs.len() == 2 {
                let seq = xs[0];
                let dim = xs[1];
                log::debug!("Llama.forward: attempting reshape to [1,{},{}]", seq, dim);
                x = match x.reshape(vec![1, seq, dim]) {
                    Ok(t) => {
                        log::debug!(
                            "Llama.forward: reshape succeeded, new shape={:?}",
                            t.lock().storage.shape()
                        );
                        t
                    }
                    Err(e) => {
                        log::error!(
                            "Llama.forward: failed to reshape embedding for single sequence: {}",
                            e
                        );
                        return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
                    }
                };
            } else {
                log::debug!(
                    "Llama.forward: single_seq flag true but embedding has ndim {}",
                    xs.len()
                );
            }
        }

        for (idx, layer) in self.layers.iter().enumerate() {
            log::debug!(
                "Llama.forward: before layer {} shape {:?}",
                idx,
                x.lock().storage.shape()
            );
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| layer.forward(&x)));
            match res {
                Ok(t) => {
                    x = t;
                    log::debug!(
                        "Llama.forward: after layer {} shape {:?}",
                        idx,
                        x.lock().storage.shape()
                    );
                }
                Err(e) => {
                    log::error!("Llama.forward: panic in layer {}: {:?}", idx, e);
                    return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
                }
            }
        }
        // RMSNorm
        x = x.rmsnorm(&self.norm, 2, 1e-5);
        let logits = self.lm_head.forward(&x);
        // If input was single sequence, remove the batch dim to return [seq, vocab]
        if single_seq {
            let lshape = logits.lock().storage.shape().to_vec();
            if lshape.len() == 3 && lshape[0] == 1 {
                let seq = lshape[1];
                let vocab = lshape[2];
                match logits.reshape(vec![seq, vocab]) {
                    Ok(t) => return t,
                    Err(e) => {
                        log::error!(
                            "Llama.forward: failed to reshape logits back to [seq,vocab]: {}",
                            e
                        );
                    }
                }
            }
        }
        logits
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.embed_tokens.clone(), self.norm.clone()];
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.lm_head.parameters());
        p
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = vec![
            (
                format!("{}.embed_tokens.weight", prefix),
                self.embed_tokens.clone(),
            ),
            (format!("{}.norm.weight", prefix), self.norm.clone()),
        ];
        for (i, layer) in self.layers.iter().enumerate() {
            out.extend(layer.named_parameters(&format!("{}.layers.{}", prefix, i)));
        }
        out.extend(
            self.lm_head
                .named_parameters(&format!("{}.lm_head", prefix)),
        );
        out
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        let embed_key = format!("{}.embed_tokens.weight", prefix);
        if let Some(t) = state.get(&embed_key) {
            self.embed_tokens = t.clone();
            // Fix transposed embeddings saved as [d_model, vocab] -> transpose to [vocab, d_model]
            let shape = self.embed_tokens.lock().storage.shape().to_vec();
            log::debug!(
                "Llama.load_state_dict: embed_tokens loaded shape={:?}",
                shape
            );
            if shape.len() == 2 {
                let check_shape = if let Some(lh) = self.lm_head.as_f32() {
                    shape[0] == lh.weight.lock().storage.shape()[0]
                } else {
                    false
                };

                if check_shape && shape[1] > 1 {
                    // If first dim equals d_model (lm_head rows) then transpose
                    let arr = self.embed_tokens.lock().storage.to_f32_array();
                    let arr_t = arr.reversed_axes();
                    self.embed_tokens = Tensor::new(arr_t.into_dyn(), false);
                    log::debug!(
                        "Llama.load_state_dict: transposed embed_tokens to shape={:?}",
                        self.embed_tokens.lock().storage.shape()
                    );
                }
            }
        }
        let norm_key = format!("{}.norm.weight", prefix);
        if let Some(t) = state.get(&norm_key) {
            self.norm = t.clone();
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(state, &format!("{}.layers.{}", prefix, i))?;
        }
        self.lm_head
            .load_state_dict(state, &format!("{}.lm_head", prefix))?;
        // If lm_head not present in the state dict, tie it to embed_tokens (transpose)
        let lm_key = format!("{}.lm_head.weight", prefix);
        if !state.contains_key(&lm_key) {
            // transpose embed_tokens [vocab, d_model] -> [d_model, vocab]
            let emb_arr = self.embed_tokens.lock().storage.to_f32_array();
            let emb_t = emb_arr.reversed_axes();
            if let Some(lh) = self.lm_head.as_f32_mut() {
                lh.weight = Tensor::new(emb_t.into_dyn(), false);
            }
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
pub struct Mistral {
    pub embed_tokens: Tensor,
    pub layers: Vec<TransformerBlock>,
    pub norm: Tensor, // RMSNorm gamma
    pub lm_head: LinearLayer,
    pub vocab_size: usize,
    pub sliding_window: usize,
}

impl Mistral {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
        sliding_window: usize,
    ) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "Mistral::new: d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }
        if !num_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "Mistral::new: num_heads ({}) must be divisible by kv_heads ({})",
                num_heads, kv_heads
            ));
        }

        let embed_tokens = Tensor::new(
            Array::zeros(IxDyn(&[vocab_size, d_model][..])),
            true,
        );
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let mut block = TransformerBlock::new_llama_style(TransformerConfig {
                d_model,
                d_ff,
                num_heads,
                kv_heads,
                use_rope: true,
                bias: false,
                rope_theta: 10000.0,
                rope_scale: 1.0,
            })?;
            // Configure sliding window attention
            block
                .mha
                .set_attention_variant(AttentionVariant::SlidingWindow {
                    window_size: sliding_window,
                });
            layers.push(block);
        }
        let norm = Tensor::new(
            Array::from_elem(IxDyn(&[d_model][..]), 1.0f32),
            true,
        );
        let lm_head = LinearLayer::new_f32(d_model, vocab_size, false);
        Ok(Mistral {
            embed_tokens,
            layers,
            norm,
            lm_head,
            vocab_size,
            sliding_window,
        })
    }

    pub fn forward_with_mask(&mut self, input: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let input_shape = input.lock().storage.shape().to_vec();
        let single_seq = input_shape.len() == 1;

        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        let xs = x.lock().storage.shape().to_vec();

        if single_seq && xs.len() == 2 {
            let seq = xs[0];
            let dim = xs[1];
            x = match x.reshape(vec![1, seq, dim]) {
                Ok(t) => t,
                Err(e) => {
                    log::error!(
                        "Mistral.forward_with_mask: failed to reshape embedding: {}",
                        e
                    );
                    return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
                }
            };
        }

        for layer in self.layers.iter_mut() {
            x = layer.forward_block(&x, mask);
        }

        x = x.rmsnorm(&self.norm, 2, 1e-5);
        let logits = self.lm_head.forward(&x);

        if single_seq {
            let ls = logits.lock().storage.shape().to_vec();
            if ls.len() == 3 && ls[0] == 1 {
                if let Ok(reshaped) = logits.reshape(vec![ls[1], ls[2]]) {
                    return reshaped;
                }
            }
        }
        logits
    }

    pub fn set_kv_cache(&mut self, use_cache: bool) {
        for layer in self.layers.iter_mut() {
            if use_cache {
                layer.set_kv_cache(crate::nn::KVCache::new());
            } else {
                layer.clear_kv_cache();
            }
        }
    }

    pub fn truncate_kv_cache(&mut self, n: usize) {
        for layer in self.layers.iter_mut() {
            layer.truncate_kv_cache(n);
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.embed_tokens.clone(), self.norm.clone()];
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.lm_head.parameters());
        p
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = vec![
            (
                format!("{}.embed_tokens.weight", prefix),
                self.embed_tokens.clone(),
            ),
            (format!("{}.norm.weight", prefix), self.norm.clone()),
        ];
        for (i, layer) in self.layers.iter().enumerate() {
            out.extend(layer.named_parameters(&format!("{}.layers.{}", prefix, i)));
        }
        out.extend(
            self.lm_head
                .named_parameters(&format!("{}.lm_head", prefix)),
        );
        out
    }

    pub fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        let embed_key = format!("{}.model.embed_tokens.weight", prefix);
        if let Some(t) = state.get(&embed_key) {
            self.embed_tokens = t.clone();
            let shape = self.embed_tokens.lock().storage.shape().to_vec();
            if shape.len() == 2 && shape[1] > 1 {
                let arr = self.embed_tokens.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                self.embed_tokens = Tensor::new(arr_t.into_dyn(), false);
            }
        }
        let norm_key = format!("{}.model.norm.weight", prefix);
        if let Some(t) = state.get(&norm_key) {
            self.norm = t.clone();
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(state, &format!("{}.model.layers.{}", prefix, i))?;
        }
        let lm_key = format!("{}.lm_head.weight", prefix);
        if !state.contains_key(&lm_key) {
            let emb_arr = self.embed_tokens.lock().storage.to_f32_array();
            let emb_t = emb_arr.reversed_axes();
            if let Some(lh) = self.lm_head.as_f32_mut() {
                lh.weight = Tensor::new(emb_t.into_dyn(), false);
            }
        } else if let Some(lh) = state.get(&lm_key) {
            if let Some(lh_layer) = self.lm_head.as_f32_mut() {
                lh_layer.weight = lh.clone();
            }
        }
        Ok(())
    }
}

impl Module for Mistral {
    fn forward(&self, input: &Tensor) -> Tensor {
        let input_shape = input.lock().storage.shape().to_vec();
        let single_seq = input_shape.len() == 1;
        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        let xs = x.lock().storage.shape().to_vec();
        if single_seq && xs.len() == 2 {
            let seq = xs[0];
            let dim = xs[1];
            x = match x.reshape(vec![1, seq, dim]) {
                Ok(t) => t,
                Err(_) => return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false),
            };
        }
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x = x.rmsnorm(&self.norm, 2, 1e-5);
        self.lm_head.forward(&x)
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
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Phi model architecture (Microsoft).
/// Key differences from Llama:
/// - Uses phi-style rotary embedding scaling
/// - GELU activation in FFN (not SwiGLU)
/// - Different attention pattern with multi-query attention support
#[derive(Clone)]
pub struct Phi {
    pub embed_tokens: Tensor,
    pub layers: Vec<TransformerBlock>,
    pub norm: Tensor,
    pub lm_head: LinearLayer,
    pub final_bias: bool,
    pub vocab_size: usize,
}

impl Phi {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
        final_bias: bool,
    ) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "Phi::new: d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }
        if !num_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "Phi::new: num_heads ({}) must be divisible by kv_heads ({})",
                num_heads, kv_heads
            ));
        }

        let embed_tokens = Tensor::new(
            Array::zeros(IxDyn(&[vocab_size, d_model][..])),
            true,
        );
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerBlock::new_llama_style(TransformerConfig {
                d_model,
                d_ff,
                num_heads,
                kv_heads,
                use_rope: true,
                bias: true,
                rope_theta: 10000.0,
                rope_scale: 1.0,
            })?);
        }
        let norm = Tensor::new(
            Array::from_elem(IxDyn(&[d_model][..]), 1.0f32),
            true,
        );
        let lm_head = LinearLayer::new_f32(d_model, vocab_size, final_bias);
        Ok(Phi {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_bias,
            vocab_size,
        })
    }

    pub fn forward_with_mask(&mut self, input: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let input_shape = input.lock().storage.shape().to_vec();
        let single_seq = input_shape.len() == 1;

        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        let xs = x.lock().storage.shape().to_vec();

        if single_seq && xs.len() == 2 {
            let seq = xs[0];
            let dim = xs[1];
            x = match x.reshape(vec![1, seq, dim]) {
                Ok(t) => t,
                Err(e) => {
                    log::error!("Phi.forward_with_mask: reshape failed: {}", e);
                    return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
                }
            };
        }

        for layer in self.layers.iter_mut() {
            x = layer.forward_block(&x, mask);
        }

        x = x.rmsnorm(&self.norm, 2, 1e-5);
        let logits = self.lm_head.forward(&x);

        if single_seq {
            let ls = logits.lock().storage.shape().to_vec();
            if ls.len() == 3 && ls[0] == 1 {
                if let Ok(reshaped) = logits.reshape(vec![ls[1], ls[2]]) {
                    return reshaped;
                }
            }
        }
        logits
    }

    pub fn set_kv_cache(&mut self, use_cache: bool) {
        for layer in self.layers.iter_mut() {
            if use_cache {
                layer.set_kv_cache(crate::nn::KVCache::new());
            } else {
                layer.clear_kv_cache();
            }
        }
    }

    pub fn truncate_kv_cache(&mut self, n: usize) {
        for layer in self.layers.iter_mut() {
            layer.truncate_kv_cache(n);
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.embed_tokens.clone(), self.norm.clone()];
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.lm_head.parameters());
        p
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = vec![
            (
                format!("{}.model.embed_tokens.weight", prefix),
                self.embed_tokens.clone(),
            ),
            (format!("{}.model.norm.weight", prefix), self.norm.clone()),
        ];
        for (i, layer) in self.layers.iter().enumerate() {
            out.extend(layer.named_parameters(&format!("{}.model.layers.{}", prefix, i)));
        }
        out.extend(
            self.lm_head
                .named_parameters(&format!("{}.lm_head", prefix)),
        );
        out
    }

    pub fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        let embed_key = format!("{}.model.embed_tokens.weight", prefix);
        if let Some(t) = state.get(&embed_key) {
            self.embed_tokens = t.clone();
            let shape = self.embed_tokens.lock().storage.shape().to_vec();
            if shape.len() == 2 && shape[1] > 1 {
                let arr = self.embed_tokens.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                self.embed_tokens = Tensor::new(arr_t.into_dyn(), false);
            }
        }
        let norm_key = format!("{}.model.norm.weight", prefix);
        if let Some(t) = state.get(&norm_key) {
            self.norm = t.clone();
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(state, &format!("{}.model.layers.{}", prefix, i))?;
        }
        let lm_key = format!("{}.lm_head.weight", prefix);
        if !state.contains_key(&lm_key) {
            let emb_arr = self.embed_tokens.lock().storage.to_f32_array();
            let emb_t = emb_arr.reversed_axes();
            if let Some(lh) = self.lm_head.as_f32_mut() {
                lh.weight = Tensor::new(emb_t.into_dyn(), false);
            }
        } else if let Some(lh) = state.get(&lm_key) {
            if let Some(lh_layer) = self.lm_head.as_f32_mut() {
                lh_layer.weight = lh.clone();
            }
        }
        Ok(())
    }
}

impl Module for Phi {
    fn forward(&self, input: &Tensor) -> Tensor {
        let input_shape = input.lock().storage.shape().to_vec();
        let single_seq = input_shape.len() == 1;
        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        let xs = x.lock().storage.shape().to_vec();
        if single_seq && xs.len() == 2 {
            let seq = xs[0];
            let dim = xs[1];
            x = match x.reshape(vec![1, seq, dim]) {
                Ok(t) => t,
                Err(_) => return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false),
            };
        }
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x = x.rmsnorm(&self.norm, 2, 1e-5);
        self.lm_head.forward(&x)
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
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Qwen model architecture (Alibaba).
/// Key differences:
/// - Uses Qwen-style rotary embedding with partial rotation
/// - SwiGLU activation in FFN
/// - RMSNorm with pre-norm
/// - Supports both GQA and multi-query attention
#[derive(Clone)]
pub struct Qwen {
    pub embed_tokens: Tensor,
    pub layers: Vec<TransformerBlock>,
    pub norm: Tensor,
    pub lm_head: LinearLayer,
    pub vocab_size: usize,
    pub rotary_dim: usize,
}

impl Qwen {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
        rotary_dim: usize,
    ) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "Qwen::new: d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }
        if !num_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "Qwen::new: num_heads ({}) must be divisible by kv_heads ({})",
                num_heads, kv_heads
            ));
        }

        let embed_tokens = Tensor::new(
            Array::zeros(IxDyn(&[vocab_size, d_model][..])),
            true,
        );
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let mut block = TransformerBlock::new_llama_style(TransformerConfig {
                d_model,
                d_ff,
                num_heads,
                kv_heads,
                use_rope: true,
                bias: true,
                rope_theta: 10000.0,
                rope_scale: 1.0,
            })?;
            // Qwen uses partial rotary embedding
            block.mha.rope_scale = 1.0;
            layers.push(block);
        }
        let norm = Tensor::new(
            Array::from_elem(IxDyn(&[d_model][..]), 1.0f32),
            true,
        );
        let lm_head = LinearLayer::new_f32(d_model, vocab_size, true);
        Ok(Qwen {
            embed_tokens,
            layers,
            norm,
            lm_head,
            vocab_size,
            rotary_dim,
        })
    }

    pub fn forward_with_mask(&mut self, input: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let input_shape = input.lock().storage.shape().to_vec();
        let single_seq = input_shape.len() == 1;

        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        let xs = x.lock().storage.shape().to_vec();

        if single_seq && xs.len() == 2 {
            let seq = xs[0];
            let dim = xs[1];
            x = match x.reshape(vec![1, seq, dim]) {
                Ok(t) => t,
                Err(e) => {
                    log::error!("Qwen.forward_with_mask: reshape failed: {}", e);
                    return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
                }
            };
        }

        for layer in self.layers.iter_mut() {
            x = layer.forward_block(&x, mask);
        }

        x = x.rmsnorm(&self.norm, 2, 1e-5);
        let logits = self.lm_head.forward(&x);

        if single_seq {
            let ls = logits.lock().storage.shape().to_vec();
            if ls.len() == 3 && ls[0] == 1 {
                if let Ok(reshaped) = logits.reshape(vec![ls[1], ls[2]]) {
                    return reshaped;
                }
            }
        }
        logits
    }

    pub fn set_kv_cache(&mut self, use_cache: bool) {
        for layer in self.layers.iter_mut() {
            if use_cache {
                layer.set_kv_cache(crate::nn::KVCache::new());
            } else {
                layer.clear_kv_cache();
            }
        }
    }

    pub fn truncate_kv_cache(&mut self, n: usize) {
        for layer in self.layers.iter_mut() {
            layer.truncate_kv_cache(n);
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.embed_tokens.clone(), self.norm.clone()];
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.lm_head.parameters());
        p
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = vec![
            (
                format!("{}.model.embed_tokens.weight", prefix),
                self.embed_tokens.clone(),
            ),
            (format!("{}.model.norm.weight", prefix), self.norm.clone()),
        ];
        for (i, layer) in self.layers.iter().enumerate() {
            out.extend(layer.named_parameters(&format!("{}.model.layers.{}", prefix, i)));
        }
        out.extend(
            self.lm_head
                .named_parameters(&format!("{}.lm_head", prefix)),
        );
        out
    }

    pub fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        let embed_key = format!("{}.model.embed_tokens.weight", prefix);
        if let Some(t) = state.get(&embed_key) {
            self.embed_tokens = t.clone();
            let shape = self.embed_tokens.lock().storage.shape().to_vec();
            if shape.len() == 2 && shape[1] > 1 {
                let arr = self.embed_tokens.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                self.embed_tokens = Tensor::new(arr_t.into_dyn(), false);
            }
        }
        let norm_key = format!("{}.model.norm.weight", prefix);
        if let Some(t) = state.get(&norm_key) {
            self.norm = t.clone();
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(state, &format!("{}.model.layers.{}", prefix, i))?;
        }
        let lm_key = format!("{}.lm_head.weight", prefix);
        if !state.contains_key(&lm_key) {
            let emb_arr = self.embed_tokens.lock().storage.to_f32_array();
            let emb_t = emb_arr.reversed_axes();
            if let Some(lh) = self.lm_head.as_f32_mut() {
                lh.weight = Tensor::new(emb_t.into_dyn(), false);
            }
        } else if let Some(lh) = state.get(&lm_key) {
            if let Some(lh_layer) = self.lm_head.as_f32_mut() {
                lh_layer.weight = lh.clone();
            }
        }
        Ok(())
    }
}

impl Module for Qwen {
    fn forward(&self, input: &Tensor) -> Tensor {
        let input_shape = input.lock().storage.shape().to_vec();
        let single_seq = input_shape.len() == 1;
        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        let xs = x.lock().storage.shape().to_vec();
        if single_seq && xs.len() == 2 {
            let seq = xs[0];
            let dim = xs[1];
            x = match x.reshape(vec![1, seq, dim]) {
                Ok(t) => t,
                Err(_) => return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false),
            };
        }
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x = x.rmsnorm(&self.norm, 2, 1e-5);
        self.lm_head.forward(&x)
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
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Gemma model architecture (Google).
/// Key differences:
/// - Uses RMSNorm without bias terms
/// - SwiGLU activation in FFN
/// - Pre-norm architecture
/// - Uses gemma-style rotary embedding
#[derive(Clone)]
pub struct Gemma {
    pub embed_tokens: Tensor,
    pub layers: Vec<TransformerBlock>,
    pub norm: Tensor,
    pub lm_head: LinearLayer,
    pub vocab_size: usize,
    pub embedding_multiplier: f32,
}

impl Gemma {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
        embedding_multiplier: f32,
    ) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "Gemma::new: d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }
        if !num_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "Gemma::new: num_heads ({}) must be divisible by kv_heads ({})",
                num_heads, kv_heads
            ));
        }

        let embed_tokens = Tensor::new(
            Array::zeros(IxDyn(&[vocab_size, d_model][..])),
            true,
        );
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let block = TransformerBlock::new_llama_style(TransformerConfig {
                d_model,
                d_ff,
                num_heads,
                kv_heads,
                use_rope: true,
                bias: false,
                rope_theta: 10000.0,
                rope_scale: 1.0,
            })?;
            layers.push(block);
        }
        let norm = Tensor::new(
            Array::from_elem(IxDyn(&[d_model][..]), 1.0f32),
            true,
        );
        let lm_head = LinearLayer::new_f32(d_model, vocab_size, false);
        Ok(Gemma {
            embed_tokens,
            layers,
            norm,
            lm_head,
            vocab_size,
            embedding_multiplier,
        })
    }

    pub fn forward_with_mask(&mut self, input: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let input_shape = input.lock().storage.shape().to_vec();
        let single_seq = input_shape.len() == 1;

        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        // Gemma scales embeddings by sqrt(d_model)
        let scale = (self.embedding_multiplier * self.d_model() as f32).sqrt();
        x = x.mul(&Tensor::new(
            Array::from_elem(IxDyn(&[1]), scale),
            false,
        ));
        let xs = x.lock().storage.shape().to_vec();

        if single_seq && xs.len() == 2 {
            let seq = xs[0];
            let dim = xs[1];
            x = match x.reshape(vec![1, seq, dim]) {
                Ok(t) => t,
                Err(e) => {
                    log::error!("Gemma.forward_with_mask: reshape failed: {}", e);
                    return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
                }
            };
        }

        for layer in self.layers.iter_mut() {
            x = layer.forward_block(&x, mask);
        }

        x = x.rmsnorm(&self.norm, 2, 1e-5);
        let logits = self.lm_head.forward(&x);

        if single_seq {
            let ls = logits.lock().storage.shape().to_vec();
            if ls.len() == 3 && ls[0] == 1 {
                if let Ok(reshaped) = logits.reshape(vec![ls[1], ls[2]]) {
                    return reshaped;
                }
            }
        }
        logits
    }

    pub fn d_model(&self) -> usize {
        self.norm.lock().storage.shape()[0]
    }

    pub fn set_kv_cache(&mut self, use_cache: bool) {
        for layer in self.layers.iter_mut() {
            if use_cache {
                layer.set_kv_cache(crate::nn::KVCache::new());
            } else {
                layer.clear_kv_cache();
            }
        }
    }

    pub fn truncate_kv_cache(&mut self, n: usize) {
        for layer in self.layers.iter_mut() {
            layer.truncate_kv_cache(n);
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.embed_tokens.clone(), self.norm.clone()];
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.lm_head.parameters());
        p
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = vec![
            (
                format!("{}.model.embed_tokens.weight", prefix),
                self.embed_tokens.clone(),
            ),
            (format!("{}.model.norm.weight", prefix), self.norm.clone()),
        ];
        for (i, layer) in self.layers.iter().enumerate() {
            out.extend(layer.named_parameters(&format!("{}.model.layers.{}", prefix, i)));
        }
        out.extend(
            self.lm_head
                .named_parameters(&format!("{}.lm_head", prefix)),
        );
        out
    }

    pub fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        let embed_key = format!("{}.model.embed_tokens.weight", prefix);
        if let Some(t) = state.get(&embed_key) {
            self.embed_tokens = t.clone();
            let shape = self.embed_tokens.lock().storage.shape().to_vec();
            if shape.len() == 2 && shape[1] > 1 {
                let arr = self.embed_tokens.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                self.embed_tokens = Tensor::new(arr_t.into_dyn(), false);
            }
        }
        let norm_key = format!("{}.model.norm.weight", prefix);
        if let Some(t) = state.get(&norm_key) {
            self.norm = t.clone();
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(state, &format!("{}.model.layers.{}", prefix, i))?;
        }
        let lm_key = format!("{}.lm_head.weight", prefix);
        if !state.contains_key(&lm_key) {
            let emb_arr = self.embed_tokens.lock().storage.to_f32_array();
            let emb_t = emb_arr.reversed_axes();
            if let Some(lh) = self.lm_head.as_f32_mut() {
                lh.weight = Tensor::new(emb_t.into_dyn(), false);
            }
        } else if let Some(lh) = state.get(&lm_key) {
            if let Some(lh_layer) = self.lm_head.as_f32_mut() {
                lh_layer.weight = lh.clone();
            }
        }
        Ok(())
    }
}

impl Module for Gemma {
    fn forward(&self, input: &Tensor) -> Tensor {
        let input_shape = input.lock().storage.shape().to_vec();
        let single_seq = input_shape.len() == 1;
        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        let scale = (self.embedding_multiplier * self.d_model() as f32).sqrt();
        x = x.mul(&Tensor::new(
            Array::from_elem(IxDyn(&[1]), scale),
            false,
        ));
        let xs = x.lock().storage.shape().to_vec();
        if single_seq && xs.len() == 2 {
            let seq = xs[0];
            let dim = xs[1];
            x = match x.reshape(vec![1, seq, dim]) {
                Ok(t) => t,
                Err(_) => return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false),
            };
        }
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x = x.rmsnorm(&self.norm, 2, 1e-5);
        self.lm_head.forward(&x)
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
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
pub struct GPTDecoder {
    pub token_embedding: Tensor,
    pub position_embedding: Tensor,
    pub blocks: Vec<TransformerBlock>,
    pub ln_gamma: Tensor,
    pub ln_beta: Tensor,
    pub lm_head: LinearLayer,
    pub max_seq_len: usize,
}

impl GPTDecoder {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        d_ff: usize,
        num_heads: usize,
        max_seq_len: usize,
    ) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "GPTDecoder::new: d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }
        let token_embedding = Tensor::new(
            Array::zeros(IxDyn(&[vocab_size, d_model][..])),
            true,
        );
        let position_embedding = Tensor::new(
            Array::zeros(IxDyn(&[max_seq_len, d_model][..])),
            true,
        );
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new_decoder(d_model, d_ff, num_heads)?);
        }
        let ln_gamma = Tensor::new(Array::ones(IxDyn(&[d_model][..])), true);
        let ln_beta = Tensor::new(Array::zeros(IxDyn(&[d_model][..])), true);
        let lm_head = LinearLayer::new_f32(d_model, vocab_size, true);

        Ok(Self {
            token_embedding,
            position_embedding,
            blocks,
            ln_gamma,
            ln_beta,
            lm_head,
            max_seq_len,
        })
    }

    fn position_ids(batch: usize, seq: usize) -> Tensor {
        let mut pos = Vec::with_capacity(batch * seq);
        for _ in 0..batch {
            for i in 0..seq {
                pos.push(i as f32);
            }
        }
        let pos_arr = Array::from_shape_vec((batch, seq), pos)
            .unwrap_or_else(|_| Array::zeros((batch, seq)))
            .into_dyn();
        Tensor::new(pos_arr, false)
    }

    pub fn forward_with_mask(&mut self, input_ids: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let shape = input_ids.lock().storage.shape().to_vec();
        if shape.len() != 2 {
            log::error!(
                "GPTDecoder.forward_with_mask: expected input [batch, seq], got {:?}",
                shape
            );
            return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
        }
        let batch = shape[0];
        let seq = shape[1];
        if seq > self.max_seq_len {
            log::error!(
                "GPTDecoder.forward_with_mask: seq length {} exceeds max_seq_len {}",
                seq,
                self.max_seq_len
            );
            return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
        }

        let tok = Tensor::embedding_lookup(&self.token_embedding, input_ids);
        let pos_ids = Self::position_ids(batch, seq);
        let pos = Tensor::embedding_lookup(&self.position_embedding, &pos_ids);
        let mut x = tok.add(&pos);

        for blk in self.blocks.iter_mut() {
            x = blk.forward_block(&x, mask);
        }

        x = x.layer_norm(2, 1e-5, &self.ln_gamma, &self.ln_beta);
        self.lm_head.forward(&x)
    }
}

impl Module for GPTDecoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.lock().storage.shape().to_vec();
        if shape.len() != 2 {
            log::error!(
                "GPTDecoder.forward: expected input [batch, seq], got {:?}",
                shape
            );
            return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
        }
        let batch = shape[0];
        let seq = shape[1];
        if seq > self.max_seq_len {
            log::error!(
                "GPTDecoder.forward: seq length {} exceeds max_seq_len {}",
                seq,
                self.max_seq_len
            );
            return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
        }

        let tok = Tensor::embedding_lookup(&self.token_embedding, input);
        let pos_ids = Self::position_ids(batch, seq);
        let pos = Tensor::embedding_lookup(&self.position_embedding, &pos_ids);
        let mut x = tok.add(&pos);

        for blk in &self.blocks {
            x = blk.forward_block_no_cache(&x);
        }

        x = x.layer_norm(2, 1e-5, &self.ln_gamma, &self.ln_beta);
        self.lm_head.forward(&x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![
            self.token_embedding.clone(),
            self.position_embedding.clone(),
            self.ln_gamma.clone(),
            self.ln_beta.clone(),
        ];
        for blk in &self.blocks {
            p.extend(blk.parameters());
        }
        p.extend(self.lm_head.parameters());
        p
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = vec![
            (
                format!("{}.token_embedding.weight", prefix),
                self.token_embedding.clone(),
            ),
            (
                format!("{}.position_embedding.weight", prefix),
                self.position_embedding.clone(),
            ),
            (format!("{}.ln_f.weight", prefix), self.ln_gamma.clone()),
            (format!("{}.ln_f.bias", prefix), self.ln_beta.clone()),
        ];
        for (i, blk) in self.blocks.iter().enumerate() {
            out.extend(blk.named_parameters(&format!("{}.blocks.{}", prefix, i)));
        }
        out.extend(
            self.lm_head
                .named_parameters(&format!("{}.lm_head", prefix)),
        );
        out
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        if let Some(t) = state.get(&format!("{}.token_embedding.weight", prefix)) {
            self.token_embedding = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.position_embedding.weight", prefix)) {
            self.position_embedding = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.ln_f.weight", prefix)) {
            self.ln_gamma = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.ln_f.bias", prefix)) {
            self.ln_beta = t.clone();
        }
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            blk.load_state_dict(state, &format!("{}.blocks.{}", prefix, i))?;
        }
        self.lm_head
            .load_state_dict(state, &format!("{}.lm_head", prefix))?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
pub struct BERTEncoder {
    pub token_embedding: Tensor,
    pub position_embedding: Tensor,
    pub token_type_embedding: Tensor,
    pub blocks: Vec<TransformerBlock>,
    pub ln_gamma: Tensor,
    pub ln_beta: Tensor,
    pub pooler: LinearLayer,
    pub max_seq_len: usize,
}

impl BERTEncoder {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        d_ff: usize,
        num_heads: usize,
        max_seq_len: usize,
    ) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "BERTEncoder::new: d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }

        let token_embedding = Tensor::new(
            Array::zeros(IxDyn(&[vocab_size, d_model][..])),
            true,
        );
        let position_embedding = Tensor::new(
            Array::zeros(IxDyn(&[max_seq_len, d_model][..])),
            true,
        );
        let token_type_embedding =
            Tensor::new(Array::zeros(IxDyn(&[2, d_model][..])), true);
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(d_model, d_ff, num_heads)?);
        }
        let ln_gamma = Tensor::new(Array::ones(IxDyn(&[d_model][..])), true);
        let ln_beta = Tensor::new(Array::zeros(IxDyn(&[d_model][..])), true);
        let pooler = LinearLayer::new_f32(d_model, d_model, true);

        Ok(Self {
            token_embedding,
            position_embedding,
            token_type_embedding,
            blocks,
            ln_gamma,
            ln_beta,
            pooler,
            max_seq_len,
        })
    }

    fn position_ids(batch: usize, seq: usize) -> Tensor {
        let mut pos = Vec::with_capacity(batch * seq);
        for _ in 0..batch {
            for i in 0..seq {
                pos.push(i as f32);
            }
        }
        let pos_arr = Array::from_shape_vec((batch, seq), pos)
            .unwrap_or_else(|_| Array::zeros((batch, seq)))
            .into_dyn();
        Tensor::new(pos_arr, false)
    }

    pub fn forward_with_token_type(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Tensor {
        let shape = input_ids.lock().storage.shape().to_vec();
        if shape.len() != 2 {
            log::error!(
                "BERTEncoder.forward_with_token_type: expected input [batch, seq], got {:?}",
                shape
            );
            return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
        }
        let batch = shape[0];
        let seq = shape[1];
        if seq > self.max_seq_len {
            log::error!(
                "BERTEncoder.forward_with_token_type: seq length {} exceeds max_seq_len {}",
                seq,
                self.max_seq_len
            );
            return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
        }

        let tok = Tensor::embedding_lookup(&self.token_embedding, input_ids);
        let pos_ids = Self::position_ids(batch, seq);
        let pos = Tensor::embedding_lookup(&self.position_embedding, &pos_ids);

        let type_emb = if let Some(tt) = token_type_ids {
            Tensor::embedding_lookup(&self.token_type_embedding, tt)
        } else {
            let zero_type = Tensor::new(Array::zeros(IxDyn(&[batch, seq][..])), false);
            Tensor::embedding_lookup(&self.token_type_embedding, &zero_type)
        };

        let mut x = tok.add(&pos).add(&type_emb);

        for blk in &self.blocks {
            x = blk.forward_block_no_cache(&x);
        }

        if let Some(m) = mask {
            x = x.add(m);
        }

        x.layer_norm(2, 1e-5, &self.ln_gamma, &self.ln_beta)
    }

    /// Returns pooled output similar to BERT pooler: tanh(W * hidden_state_of_cls).
    pub fn pooled_output(&self, encoded: &Tensor) -> Tensor {
        let shape = encoded.lock().storage.shape().to_vec();
        if shape.len() != 3 || shape[1] == 0 {
            return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false);
        }
        let b = shape[0];
        let d = shape[2];
        let cls = Tensor::apply(
            Arc::new(crate::ops::Slice::new(1, 0, 1)),
            std::slice::from_ref(encoded),
        );
        let cls = match cls.reshape(vec![b, d]) {
            Ok(t) => t,
            Err(_) => return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0][..])), false),
        };
        self.pooler.forward(&cls).tanh()
    }
}

impl Module for BERTEncoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_with_token_type(input, None, None)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![
            self.token_embedding.clone(),
            self.position_embedding.clone(),
            self.token_type_embedding.clone(),
            self.ln_gamma.clone(),
            self.ln_beta.clone(),
        ];
        for blk in &self.blocks {
            p.extend(blk.parameters());
        }
        p.extend(self.pooler.parameters());
        p
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = vec![
            (
                format!("{}.token_embedding.weight", prefix),
                self.token_embedding.clone(),
            ),
            (
                format!("{}.position_embedding.weight", prefix),
                self.position_embedding.clone(),
            ),
            (
                format!("{}.token_type_embedding.weight", prefix),
                self.token_type_embedding.clone(),
            ),
            (format!("{}.ln.weight", prefix), self.ln_gamma.clone()),
            (format!("{}.ln.bias", prefix), self.ln_beta.clone()),
        ];
        for (i, blk) in self.blocks.iter().enumerate() {
            out.extend(blk.named_parameters(&format!("{}.layers.{}", prefix, i)));
        }
        out.extend(self.pooler.named_parameters(&format!("{}.pooler", prefix)));
        out
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        if let Some(t) = state.get(&format!("{}.token_embedding.weight", prefix)) {
            self.token_embedding = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.position_embedding.weight", prefix)) {
            self.position_embedding = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.token_type_embedding.weight", prefix)) {
            self.token_type_embedding = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.ln.weight", prefix)) {
            self.ln_gamma = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.ln.bias", prefix)) {
            self.ln_beta = t.clone();
        }
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            blk.load_state_dict(state, &format!("{}.layers.{}", prefix, i))?;
        }
        self.pooler
            .load_state_dict(state, &format!("{}.pooler", prefix))?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
