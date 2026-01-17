// Canonical clean transformer module.
#![allow(non_snake_case)]
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
    // RoPE base frequency (theta) for rotary embeddings
    pub rope_theta: f32,
    pub rope_scale: f32,
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
            linear_q: Linear::new(d_model, d_model, bias),
            linear_k: Linear::new(d_model, kv_dim, bias),
            linear_v: Linear::new(d_model, kv_dim, bias),
            linear_o: Linear::new(d_model, d_model, bias),
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
        let mut s = MultiHeadAttention::new_with_kv_and_rope(
            d_model, num_heads, num_heads, false, 10000.0, 1.0, true,
        );
        // create slopes as a per-head parameter shaped (1, num_heads, 1, 1)
        let arr =
            match ndarray::Array::from_shape_vec((1, num_heads, 1, 1), vec![1.0f32; num_heads]) {
                Ok(a) => a.into_dyn(),
                Err(e) => {
                    log::error!(
                        "MultiHeadAttention new_with_nl_oob: failed to construct slopes array: {}",
                        e
                    );
                    ndarray::Array::from_elem(IxDyn(&[1, num_heads, 1, 1]), 1.0f32)
                }
            };
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
        // Backward-compatible wrapper: no KV cache
        self.forward_with_caching(x, causal, causal_offset, None)
    }

    pub fn forward_with_caching(
        &self,
        x: &Tensor,
        causal: bool,
        causal_offset: Option<usize>,
        kv_cache: Option<&mut crate::nn::KVCache>,
    ) -> Tensor {
        // Compute q and the new k/v chunk for the current input x, handling transposed weights as needed
        let mut q = self.linear_q.forward(x);
        // new k chunk
        let mut new_k = {
            let shape_w = self.linear_k.weight.lock().storage.shape().to_vec();
            if shape_w.len() == 2 && shape_w[0] != self.d_model && shape_w[1] == self.d_model {
                log::debug!("MHA.forward_with_caching: detected transposed k_proj weight shape {:?}, fixing on-the-fly", shape_w);
                let arr = self.linear_k.weight.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                let w_fixed = crate::tensor::Tensor::new(arr_t.into_dyn(), false);
                let shape_x = x.lock().storage.shape().to_vec();
                let b = shape_x[0];
                let seq = shape_x[1];
                let last = shape_x[2];
                let batch = b * seq;
                let reshaped = match x.reshape(vec![batch, last]) {
                    Ok(t) => t,
                    Err(e) => {
                        log::error!(
                            "MHA.forward_with_caching: failed to reshape input for k matmul: {}",
                            e
                        );
                        return x.clone();
                    }
                };
                let out2 = reshaped.matmul(&w_fixed);
                let out_shape = vec![b, seq, w_fixed.lock().storage.shape()[1]];
                match out2.reshape(out_shape) {
                    Ok(t) => t,
                    Err(e) => {
                        log::error!(
                            "MHA.forward_with_caching: failed to reshape k output: {}",
                            e
                        );
                        return x.clone();
                    }
                }
            } else {
                self.linear_k.forward(x)
            }
        };
        // new v chunk
        let new_v = {
            let shape_w = self.linear_v.weight.lock().storage.shape().to_vec();
            if shape_w.len() == 2 && shape_w[0] != self.d_model && shape_w[1] == self.d_model {
                log::debug!("MHA.forward_with_caching: detected transposed v_proj weight shape {:?}, fixing on-the-fly", shape_w);
                let arr = self.linear_v.weight.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                let w_fixed = crate::tensor::Tensor::new(arr_t.into_dyn(), false);
                let shape_x = x.lock().storage.shape().to_vec();
                let b = shape_x[0];
                let seq = shape_x[1];
                let last = shape_x[2];
                let batch = b * seq;
                let reshaped = match x.reshape(vec![batch, last]) {
                    Ok(t) => t,
                    Err(e) => {
                        log::error!(
                            "MHA.forward_with_caching: failed to reshape input for v matmul: {}",
                            e
                        );
                        return x.clone();
                    }
                };
                let out2 = reshaped.matmul(&w_fixed);
                let out_shape = vec![b, seq, w_fixed.lock().storage.shape()[1]];
                match out2.reshape(out_shape) {
                    Ok(t) => t,
                    Err(e) => {
                        log::error!(
                            "MHA.forward_with_caching: failed to reshape v output: {}",
                            e
                        );
                        return x.clone();
                    }
                }
            } else {
                self.linear_v.forward(x)
            }
        };

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
        let (k_total, v_total) = if let Some(kvc) = kv_cache {
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

        let shape_q = q.lock().storage.shape();
        if shape_q.len() != 3 {
            log::debug!(
                "MHA.forward_with_caching: q expected 3D tensor, got {:?}",
                shape_q
            );
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
                let mut new =
                    ndarray::ArrayD::<f32>::zeros(IxDyn(&[b, self.num_heads, kv_seq, head_dim]));
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
                let mut new =
                    ndarray::ArrayD::<f32>::zeros(IxDyn(&[b, self.num_heads, kv_seq, head_dim]));
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
                    // bias shape: (b*num_heads, q_seq, kv_seq)
                    let mut bias_arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[
                        b * self.num_heads,
                        q_seq,
                        kv_seq,
                    ]));
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
                    let bias_t = crate::tensor::Tensor::new(bias_arr.into_dyn(), false);
                    scaled_logits = scaled_logits.add(&bias_t);
                }
                if let Some(rb) = &self.relative_bias {
                    let shape = rb.lock().storage.shape();
                    // accept shapes (1, q_seq, kv_seq) or (num_heads, q_seq, kv_seq)
                    if (shape.len() == 3 && shape[1] == q_seq && shape[2] == kv_seq)
                        && (shape[0] == 1 || shape[0] == self.num_heads)
                    {
                        scaled_logits = scaled_logits.add(rb);
                    }
                }
                if causal {
                    // mask shape: (b*num_heads, q_seq, kv_seq)
                    let mut mask_arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[
                        b * self.num_heads,
                        q_seq,
                        kv_seq,
                    ]));
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
                    let mask_t = crate::tensor::Tensor::new(mask_arr.into_dyn(), false);
                    log::debug!(
                        "Causal mask applied: q_seq={}, kv_seq={}, new_start={}, causal_offset={:?}",
                        q_seq, kv_seq, new_start, causal_offset
                    );
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
        let out2 = match out.reshape(vec![b, self.num_heads, q_seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MultiHeadAttention forward: reshape out to (b, num_heads, q_seq, head_dim) failed: {}", e);
                return x.clone();
            }
        };
        let out3 = out2.permute(vec![0, 2, 1, 3]);
        let out4 = match out3.reshape(vec![b, q_seq, self.d_model]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("MultiHeadAttention forward: reshape out after permute to (b, q_seq, d_model) failed: {}", e);
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
        log::debug!(
            "MHA forward: shapes q={:?} k={:?} v={:?}, b={}, seq={}, d_model={}, num_heads={}, kv_heads={}, head_dim={} ",
            q.lock().storage.shape(),
            k.lock().storage.shape(),
            v.lock().storage.shape(),
            b,
            seq,
            self.d_model,
            self.num_heads,
            self.kv_heads,
            head_dim
        );
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

        // Reshape keys/values with kv_heads support. k/v may have shape (b, seq, kv_heads * head_dim)
        let k2 = match k.reshape(vec![b * self.kv_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };
        let v2 = match v.reshape(vec![b * self.kv_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };
        // If kv_heads < num_heads, expand by repeating each kv head group
        let k2 = if self.kv_heads != self.num_heads {
            let repeat = self.num_heads / self.kv_heads;
            // Convert to ndarray, tile along first axis
            let arr = k2.to_f32_array();
            let mut new =
                ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, head_dim]));
            for i in 0..(b * self.kv_heads) {
                let src = arr.index_axis(ndarray::Axis(0), i).to_owned();
                for r in 0..repeat {
                    let dest_idx = i * repeat + r;
                    new.index_axis_mut(ndarray::Axis(0), dest_idx).assign(&src);
                }
            }
            let t = Tensor::new(new.into_dyn(), false);
            log::debug!("Expanded k2 shape: {:?}", t.lock().storage.shape());
            t
        } else {
            log::debug!(
                "No k expansion needed, k2 shape: {:?}",
                k2.lock().storage.shape()
            );
            k2
        };
        let v2 = if self.kv_heads != self.num_heads {
            let repeat = self.num_heads / self.kv_heads;
            let arr = v2.to_f32_array();
            let mut new =
                ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, head_dim]));
            for i in 0..(b * self.kv_heads) {
                let src = arr.index_axis(ndarray::Axis(0), i).to_owned();
                for r in 0..repeat {
                    let dest_idx = i * repeat + r;
                    new.index_axis_mut(ndarray::Axis(0), dest_idx).assign(&src);
                }
            }
            let t = Tensor::new(new.into_dyn(), false);
            log::debug!("Expanded v2 shape: {:?}", t.lock().storage.shape());
            t
        } else {
            log::debug!(
                "No v expansion needed, v2 shape: {:?}",
                v2.lock().storage.shape()
            );
            v2
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
        // Ensure slopes exist before proceeding
        let slopes_t = if let Some(slopes_param) = &self.slopes {
            slopes_param.clone()
        } else {
            log::error!("MultiHeadAttention forward_with_distance: slopes parameter missing");
            return x.clone();
        };
        // Compute NL-OOB bias efficiently using ndarray and a single Tensor multiply so gradient flows to slopes.
        let bias4 = if let Some(cfg) = self.nl_oob_config {
            // compute f(dist) as ndarray:
            let mut fdist = if dist_shape.len() == 2 {
                // shape (seq, seq) -> expand to (1, 1, seq, seq)
                let arr = match ndarray::Array::from_shape_vec(
                    (1, 1, seq, seq),
                    dist_arr.iter().cloned().collect(),
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        log::error!("MultiHeadAttention forward_with_distance: failed to construct 2D fdist array: {}", e);
                        return x.clone();
                    }
                };
                arr
            } else {
                // shape (b, seq, seq) -> expand to (b, 1, seq, seq)
                let raw: Vec<f32> = dist_arr.iter().cloned().collect();
                let arr = match ndarray::Array::from_shape_vec((b, 1, seq, seq), raw) {
                    Ok(a) => a,
                    Err(e) => {
                        log::error!("MultiHeadAttention forward_with_distance: failed to construct 3D fdist array: {}", e);
                        return x.clone();
                    }
                };
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
            slopes_t.mul(&fdist_t)
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

    /// Debug: return intermediate tensors for inspection
    pub fn forward_debug(
        &self,
        x: &Tensor,
        causal: bool,
        causal_offset: Option<usize>,
    ) -> std::collections::HashMap<String, Tensor> {
        let mut out = std::collections::HashMap::new();
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
        let shape = q.lock().storage.shape();
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
        let scalar_tensor = Tensor::new(ndarray::Array::from_elem(IxDyn(&[1]), scale), false);
        let scaled = qk.mul(&scalar_tensor);
        out.insert("scaled_logits".to_string(), scaled.clone());
        // shaped logits (b, num_heads, seq, seq)
        let scaled_logits4 = match scaled.reshape(vec![b, self.num_heads, seq, seq]) {
            Ok(t) => t,
            Err(_) => scaled.clone(),
        };
        let mut scaled_logits_final = scaled_logits4.clone();
        // Apply ALiBi if present
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
            scaled_logits_final = scaled_logits_final.add(&bias_t);
        }
        // causal mask
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
                            if let Ok(exp_arr) = ndarray::Array::from_shape_vec(
                                ndarray::IxDyn(&[self.num_heads * head_dim, cols]),
                                expanded,
                            ) {
                                self.linear_k.weight =
                                    crate::tensor::Tensor::new(exp_arr.into_dyn(), false);
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
                                if let Ok(exp_arr) = ndarray::Array::from_shape_vec(
                                    ndarray::IxDyn(&[self.num_heads * head_dim, cols]),
                                    expanded,
                                ) {
                                    self.linear_k.weight =
                                        crate::tensor::Tensor::new(exp_arr.into_dyn(), false);
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
                            if let Ok(exp_arr) = ndarray::Array::from_shape_vec(
                                ndarray::IxDyn(&[self.num_heads * head_dim, cols]),
                                expanded,
                            ) {
                                self.linear_v.weight =
                                    crate::tensor::Tensor::new(exp_arr.into_dyn(), false);
                            }
                        } else {
                            if let Ok(arr2) = arr.clone().into_dimensionality::<ndarray::Ix2>() {
                                let mut expanded =
                                    Vec::with_capacity(self.num_heads * head_dim * cols);
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
                                if let Ok(exp_arr) = ndarray::Array::from_shape_vec(
                                    ndarray::IxDyn(&[self.num_heads * head_dim, cols]),
                                    expanded,
                                ) {
                                    self.linear_v.weight =
                                        crate::tensor::Tensor::new(exp_arr.into_dyn(), false);
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
        {
            let shape = self.linear_k.weight.lock().storage.shape().to_vec();
            eprintln!("MHA.load_state_dict: k_proj loaded shape={:?}", shape);
            if shape.len() == 2 && shape[0] != self.d_model && shape[1] == self.d_model {
                let arr = self.linear_k.weight.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                self.linear_k.weight = Tensor::new(arr_t.into_dyn(), false);
                eprintln!(
                    "MHA.load_state_dict: k_proj transposed to shape={:?}",
                    self.linear_k.weight.lock().storage.shape()
                );
            }
        }
        self.linear_v
            .load_state_dict(state, &format!("{}.v_proj", prefix))?;
        {
            let shape = self.linear_v.weight.lock().storage.shape().to_vec();
            eprintln!("MHA.load_state_dict: v_proj loaded shape={:?}", shape);
            if shape.len() == 2 && shape[0] != self.d_model && shape[1] == self.d_model {
                let arr = self.linear_v.weight.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                self.linear_v.weight = Tensor::new(arr_t.into_dyn(), false);
                eprintln!(
                    "MHA.load_state_dict: v_proj transposed to shape={:?}",
                    self.linear_v.weight.lock().storage.shape()
                );
            }
        }
        self.linear_o
            .load_state_dict(state, &format!("{}.o_proj", prefix))?;
        {
            let shape = self.linear_o.weight.lock().storage.shape().to_vec();
            eprintln!("MHA.load_state_dict: o_proj loaded shape={:?}", shape);
            if shape.len() == 2 && shape[0] != self.d_model && shape[1] == self.d_model {
                let arr = self.linear_o.weight.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                self.linear_o.weight = Tensor::new(arr_t.into_dyn(), false);
                eprintln!(
                    "MHA.load_state_dict: o_proj transposed to shape={:?}",
                    self.linear_o.weight.lock().storage.shape()
                );
            }
        }
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

impl crate::nn::Module for MultiHeadAttention {
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
    pub linear1: Linear,
    pub linear2: Linear,
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
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
            causal: false,
            kv_cache: None,
            llama_style: false,
            rms_attn_gamma: None,
            rms_ffn_gamma: None,
        })
    }
    pub fn new_with_kv_and_rope(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
        use_rope: bool,
        rope_theta: f32,
        rope_scale: f32,
        bias: bool,
    ) -> Result<Self, String> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!("TransformerBlock::new_with_kv_and_rope: d_model ({}) must be divisible by num_heads ({})", d_model, num_heads));
        }
        if !num_heads.is_multiple_of(kv_heads) {
            return Err(format!("TransformerBlock::new_with_kv_and_rope: num_heads ({}) must be divisible by kv_heads ({})", num_heads, kv_heads));
        }
        Ok(TransformerBlock {
            mha: MultiHeadAttention::new_with_kv_and_rope(
                d_model, num_heads, kv_heads, use_rope, rope_theta, rope_scale, bias,
            ),
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
            causal: false,
            kv_cache: None,
            llama_style: false,
            rms_attn_gamma: None,
            rms_ffn_gamma: None,
        })
    }
    pub fn new_with_nl_oob(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        config: BiasFunction,
        max_scale: f32,
    ) -> Result<Self, String> {
        let mut t = TransformerBlock::new_with_kv_and_rope(
            d_model, d_ff, num_heads, num_heads, false, 10000.0, 1.0, true,
        )?;
        t.mha = MultiHeadAttention::new_with_nl_oob(d_model, num_heads, config, max_scale);
        Ok(t)
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
        rope_theta: f32,
        rope_scale: f32,
    ) -> Result<Self, String> {
        // linear1 must output 2*d_ff for SwiGLU splitting
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!("TransformerBlock::new_llama_style: d_model ({}) must be divisible by num_heads ({})", d_model, num_heads));
        }
        if !num_heads.is_multiple_of(kv_heads) {
            return Err(format!("TransformerBlock::new_llama_style: num_heads ({}) must be divisible by kv_heads ({})", num_heads, kv_heads));
        }
        let linear1 = Linear::new(d_model, d_ff * 2, bias);
        let linear2 = Linear::new(d_ff, d_model, bias);
        let gamma_attn = Tensor::new(ndarray::Array::from_elem(IxDyn(&[d_model]), 1.0f32), true);
        let gamma_ffn = Tensor::new(ndarray::Array::from_elem(IxDyn(&[d_model]), 1.0f32), true);
        Ok(TransformerBlock {
            mha: MultiHeadAttention::new_with_kv_and_rope(
                d_model, num_heads, kv_heads, use_rope, rope_theta, rope_scale, bias,
            ),
            linear1,
            linear2,
            causal: false,
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

    /// Non-mutating forward of the block which does not touch or populate per-layer KV cache.
    /// This is used for full-batch encoder/decoder forward passes where cache mutation is not desired.
    pub fn forward_block_no_cache(&self, x: &Tensor) -> Tensor {
        if self.llama_style {
            let gamma_attn = match self.rms_attn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_attn_gamma; using default ones tensor");
                    Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true)
                }
            };
            let x_norm = x.rmsnorm(&gamma_attn, 2, 1e-5);
            let attn_out = self.mha.forward_with_causal(&x_norm, self.causal, None);
            let x2 = x.add(&attn_out);
            let gamma_ffn = match self.rms_ffn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x2.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_ffn_gamma; using default ones tensor");
                    Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true)
                }
            };
            let x2_norm = x2.rmsnorm(&gamma_ffn, 2, 1e-5);
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

    pub fn forward_block_impl(&mut self, x: &Tensor) -> Tensor {
        if self.llama_style {
            // Pre-norm RMSNorm -> Attention -> Residual -> Pre-norm RMSNorm -> SwiGLU FFN
            let gamma_attn = match self.rms_attn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_attn_gamma; using default ones tensor");
                    Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true)
                }
            };
            // RMSNorm along the last axis
            let x_norm = x.rmsnorm(&gamma_attn, 2, 1e-5);
            // Use per-layer KV cache if present, otherwise fallback to causal no-cache path
            let attn_out = if let Some(kvc) = self.kv_cache.as_mut() {
                self.mha
                    .forward_with_caching(&x_norm, self.causal, None, Some(kvc))
            } else {
                self.mha.forward_with_causal(&x_norm, self.causal, None)
            };

            let x2 = x.add(&attn_out);
            let gamma_ffn = match self.rms_ffn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x2.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_ffn_gamma; using default ones tensor");
                    Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true)
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
                    .forward_with_caching(x, self.causal, None, Some(kvc))
            } else {
                self.mha.forward_with_causal(x, self.causal, None)
            };
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
    /// Debug helper: return intermediate tensors from the block for inspection
    pub fn forward_block_debug(&self, x: &Tensor) -> std::collections::HashMap<String, Tensor> {
        let mut out = std::collections::HashMap::new();
        if self.llama_style {
            let gamma_attn = match self.rms_attn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x.lock().storage.shape()[2];
                    log::error!(
                        "llama_style missing rms_attn_gamma in debug; using default ones tensor"
                    );
                    Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true)
                }
            };
            let x_norm = x.rmsnorm(&gamma_attn, 2, 1e-5);
            out.insert("x_norm".to_string(), x_norm.clone());
            let mut attn_map = self.mha.forward_debug(&x_norm, self.causal, None);
            out.extend(attn_map.drain());
            let attn_out = match out.get("attn_out") {
                Some(a) => a.clone(),
                None => {
                    log::error!("forward_block_debug: attn_out missing from attention map; using zeros tensor");
                    let shape = x.lock().storage.shape().to_vec();
                    Tensor::new(ndarray::Array::zeros(IxDyn(&shape)), false)
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
                    Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true)
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
            let mut attn_map = self.mha.forward_debug(x, self.causal, None);
            out.extend(attn_map.drain());
            let attn_out = match out.get("attn_out") {
                Some(a) => a.clone(),
                None => {
                    log::error!("forward_block_debug: attn_out missing; using zeros tensor");
                    let shape = x.lock().storage.shape().to_vec();
                    Tensor::new(ndarray::Array::zeros(IxDyn(&shape)), false)
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
            let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
            let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
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

    /// Backwards-compatible wrapper for older tests expecting `forward_block` method name.
    pub fn forward_block(&mut self, x: &Tensor) -> Tensor {
        self.forward_block_impl(x)
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
                    Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true)
                }
            };
            let x_norm = x.rmsnorm(&gamma_attn, 2, 1e-5);
            let attn_out = if let Some(kvc) = self.kv_cache.as_mut() {
                self.mha
                    .forward_with_caching(&x_norm, self.causal, causal_offset, Some(kvc))
            } else {
                self.mha
                    .forward_with_causal(&x_norm, self.causal, causal_offset)
            };
            let x2 = x.add(&attn_out);
            let gamma_ffn = match self.rms_ffn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x2.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_ffn_gamma; using default ones tensor");
                    Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true)
                }
            };
            let x2_norm = x2.rmsnorm(&gamma_ffn, 2, 1e-5);
            let ff = self.linear1.forward(&x2_norm).swiglu();
            let ff = self.linear2.forward(&ff);
            x2.add(&ff)
        } else {
            let attn_out = if let Some(kvc) = self.kv_cache.as_mut() {
                self.mha
                    .forward_with_caching(x, self.causal, causal_offset, Some(kvc))
            } else {
                self.mha.forward_with_causal(x, self.causal, causal_offset)
            };
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
            let gamma_attn = match self.rms_attn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_attn_gamma; using default ones tensor");
                    Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true)
                }
            };
            let x_norm = x.rmsnorm(&gamma_attn, 2, 1e-5);
            let attn_out = self.mha.forward_with_distance(&x_norm, dist);
            let x2 = x.add(&attn_out);
            let gamma_ffn = match self.rms_ffn_gamma.as_ref() {
                Some(g) => g.clone(),
                None => {
                    let dim = x2.lock().storage.shape()[2];
                    log::error!("llama_style missing rms_ffn_gamma; using default ones tensor");
                    Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true)
                }
            };
            let x2_norm = x2.rmsnorm(&gamma_ffn, 2, 1e-5);
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
            let gate_arr = gate_w.lock().storage.to_f32_array();
            let down_arr = down_w.lock().storage.to_f32_array();
            // Determine how to concatenate respecting the existing linear1 weight shape
            let lin1_shape = self.linear1.weight.lock().storage.shape().to_vec();
            if lin1_shape.len() == 2 {
                let (r, c) = (lin1_shape[0], lin1_shape[1]);
                // Case A: both have shape (r, x) and x+x == c -> concat on axis=1
                if gate_arr.shape()[0] == r
                    && down_arr.shape()[0] == r
                    && gate_arr.shape()[1] + down_arr.shape()[1] == c
                {
                    use ndarray::Axis;
                    let combined =
                        match ndarray::concatenate(Axis(1), &[gate_arr.view(), down_arr.view()]) {
                            Ok(ca) => ca,
                            Err(e) => {
                                return Err(format!(
                                    "Failed to concatenate gate/down projections: {}",
                                    e
                                ))
                            }
                        };
                    self.linear1.weight = Tensor::new(combined.into_dyn(), false);
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
                    let combined = match ndarray::concatenate(Axis(1), &[ga_t.view(), da_t.view()])
                    {
                        Ok(ca) => ca,
                        Err(e) => {
                            return Err(format!(
                                "Failed to concatenate transposed gate/down projections: {}",
                                e
                            ))
                        }
                    };
                    self.linear1.weight = Tensor::new(combined.into_dyn(), false);
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
                    let combined =
                        match ndarray::concatenate(Axis(1), &[ga_t.view(), down_arr.view()]) {
                            Ok(ca) => ca,
                            Err(e) => {
                                return Err(format!(
                                    "Failed to concatenate transposed gate/down projections: {}",
                                    e
                                ))
                            }
                        };
                    self.linear1.weight = Tensor::new(combined.into_dyn(), false);
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
                    let combined =
                        match ndarray::concatenate(Axis(1), &[gate_arr.view(), da_t.view()]) {
                            Ok(ca) => ca,
                            Err(e) => {
                                return Err(format!(
                                    "Failed to concatenate gate/down(transposed) projections: {}",
                                    e
                                ))
                            }
                        };
                    self.linear1.weight = Tensor::new(combined.into_dyn(), false);
                } else {
                    return Err(format!("Gate/down projections shapes incompatible: gate={:?} down={:?} expected lin1={:?}", gate_arr.shape(), down_arr.shape(), lin1_shape));
                }
            }
        }
        let up_key = format!("{}.mlp.up_proj.weight", prefix);
        if let Some(up_w) = state.get(&up_key) {
            self.linear2.weight = up_w.clone();
        }

        Ok(())
    }
}
impl crate::nn::Module for TransformerBlock {
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
impl crate::nn::Module for EncoderDecoderTransformer {
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

#[derive(Clone)]
pub struct Llama {
    pub embed_tokens: Tensor,
    pub layers: Vec<TransformerBlock>,
    pub norm: Tensor, // RMSNorm gamma
    pub lm_head: Linear,
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
        let embed_tokens = Tensor::new(ndarray::Array::zeros(IxDyn(&[vocab_size, d_model])), true);
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerBlock::new_llama_style(
                d_model, d_ff, num_heads, kv_heads, true, false, 10000.0, 1.0,
            )?);
        }
        let norm = Tensor::new(ndarray::Array::from_elem(IxDyn(&[d_model]), 1.0f32), true);
        let lm_head = Linear::new(d_model, vocab_size, false); // no bias for lm_head
        Ok(Llama {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }
}

impl Module for Llama {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, seq] token ids OR [seq] for a single sequence
        let input_shape = input.lock().storage.shape().to_vec();
        let single_seq = input_shape.len() == 1;
        eprintln!(
            "Llama.forward: input_shape={:?} single_seq={}",
            input_shape, single_seq
        );
        // Embedding lookup: will return [batch, seq, d_model] or [seq, d_model]
        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        let xs = x.lock().storage.shape().to_vec();
        eprintln!("Llama.forward: embedding output shape={:?}", xs);
        // If single sequence (no batch dim) -> reshape to [1, seq, d_model]
        if single_seq {
            if xs.len() == 2 {
                let seq = xs[0];
                let dim = xs[1];
                eprintln!("Llama.forward: attempting reshape to [1,{},{}]", seq, dim);
                x = match x.reshape(vec![1, seq, dim]) {
                    Ok(t) => {
                        eprintln!(
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
                        return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0])), false);
                    }
                };
            } else {
                eprintln!(
                    "Llama.forward: single_seq flag true but embedding has ndim {}",
                    xs.len()
                );
            }
        }

        for (idx, layer) in self.layers.iter().enumerate() {
            eprintln!(
                "Llama.forward: before layer {} shape {:?}",
                idx,
                x.lock().storage.shape()
            );
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| layer.forward(&x)));
            match res {
                Ok(t) => {
                    x = t;
                    eprintln!(
                        "Llama.forward: after layer {} shape {:?}",
                        idx,
                        x.lock().storage.shape()
                    );
                }
                Err(e) => {
                    log::error!("Llama.forward: panic in layer {}: {:?}", idx, e);
                    return Tensor::new(ndarray::ArrayD::zeros(IxDyn(&[0])), false);
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
        state: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        let embed_key = format!("{}.embed_tokens.weight", prefix);
        if let Some(t) = state.get(&embed_key) {
            self.embed_tokens = t.clone();
            // Fix transposed embeddings saved as [d_model, vocab] -> transpose to [vocab, d_model]
            let shape = self.embed_tokens.lock().storage.shape().to_vec();
            eprintln!(
                "Llama.load_state_dict: embed_tokens loaded shape={:?}",
                shape
            );
            if shape.len() == 2
                && shape[0] == self.lm_head.weight.lock().storage.shape()[0]
                && shape[1] > 1
            {
                // If first dim equals d_model (lm_head rows) then transpose
                let arr = self.embed_tokens.lock().storage.to_f32_array();
                let arr_t = arr.reversed_axes();
                self.embed_tokens = Tensor::new(arr_t.into_dyn(), false);
                eprintln!(
                    "Llama.load_state_dict: transposed embed_tokens to shape={:?}",
                    self.embed_tokens.lock().storage.shape()
                );
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
            self.lm_head.weight = Tensor::new(emb_t.into_dyn(), false);
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
