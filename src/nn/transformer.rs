//! Canonical transformer implementation - MultiHeadAttention & TransformerBlock
use crate::nn::Linear;
use crate::nn::Module;
use crate::ops::{ChunkedAttention, FlashAttentionRef};
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use std::sync::Arc;

/// Bias functions supported by NL-OOB
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasFunction {
    Logarithmic, // ln(1 + d)
    Gaussian,    // d^2
}

/// Attention variant enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionVariant { Baseline, FlashRef, Chunked { chunk_size: usize } }

pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> { let mut slopes = Vec::with_capacity(n_heads); for i in 0..n_heads { slopes.push(2f32.powf(-(i as f32) / (n_heads as f32 + 0.0))); } slopes }

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
    // NL-OOB: optional bias type and learnable slopes
    pub nl_oob_config: Option<BiasFunction>,
    pub slopes: Option<Tensor>, // shape: [1, num_heads, 1, 1], requires_grad=true
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self { Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false) }
    pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
        assert!(d_model % num_heads == 0);
        assert!(num_heads % kv_heads == 0);
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
            slopes: None,
        }
    }
    /// Builder: create with NL-OOB enabled and initialized geometric slopes
    pub fn new_with_nl_oob(d_model: usize, num_heads: usize, bias_type: BiasFunction, max_scale: f32) -> Self {
        let mut mha = MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, num_heads, false);
        mha.nl_oob_config = Some(bias_type);
        mha.slopes = Some(MultiHeadAttention::initialize_geometric_slopes(num_heads, max_scale));
        mha
    }
    pub fn with_alibi(mut self) -> Self { self.use_alibi = true; self.alibi_slopes = Some(compute_alibi_slopes(self.num_heads)); self }
    pub fn set_attention_variant(&mut self, var: AttentionVariant) { self.attention_variant = var; }
    pub fn forward_impl(&self, x: &Tensor) -> Tensor {
        self.forward_with_causal(x, false, None)
    }

    /// Forward with optional causal masking. If `causal` is true, an upper-triangle mask
    /// (large negative values) is added to logits to prevent attention to future tokens.
    pub fn forward_with_causal(&self, x: &Tensor, causal: bool, causal_offset: Option<usize>) -> Tensor {
        let q = self.linear_q.forward(x);
        let k = self.linear_k.forward(x);
        let v = self.linear_v.forward(x);
        let shape = q.lock().storage.shape();
        if shape.len() != 3 { return x.clone(); }
        let b = shape[0];
        let seq = shape[1];
        let head_dim = self.d_model / self.num_heads;
        let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();
        let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();
        let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();
        let out = match self.attention_variant {
            AttentionVariant::Baseline => {
                let k2t = k2.permute(vec![0,2,1]);
                let qk = q2.batched_matmul(&k2t);
                let scale = 1.0f32 / (head_dim as f32).sqrt();
                let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
                let scaled = qk.mul(&scalar_tensor);
                let mut scaled_logits = scaled.clone();
                if self.use_alibi {
                    let slopes = if let Some(s) = &self.alibi_slopes { s.clone() } else { compute_alibi_slopes(self.num_heads) };
                    let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
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
                // Apply causal mask if set: create mask of shape [b*num_heads, seq, seq]
                if causal {
                    let mut mask_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                    for i in 0..(b * self.num_heads) {
                        for r in 0..seq {
                            for c2 in (r+1)..seq {
                                // If we have a causal_offset, only apply mask for text->text positions
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
        let out2 = out.reshape(vec![b, self.num_heads, seq, head_dim]).unwrap();
        let out3 = out2.permute(vec![0,2,1,3]);
        let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
        self.linear_o.forward(&out4)
    }
    /// Forward for cross-attention: compute q from x, k/v from kv.
    pub fn forward_cross(&self, x: &Tensor, kv: &Tensor) -> Tensor {
        let q = self.linear_q.forward(x);
        let k = self.linear_k.forward(kv);
        let v = self.linear_v.forward(kv);
        let shape = q.lock().storage.shape();
        if shape.len() != 3 { return x.clone(); }
        let b = shape[0];
        let seq = shape[1];
        let head_dim = self.d_model / self.num_heads;
        let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();
        let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();
        let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();
        // Reuse baseline attention path logic
        let k2t = k2.permute(vec![0,2,1]);
        let qk = q2.batched_matmul(&k2t);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let scalar_tensor = Tensor::new(ndarray::Array::from_elem(IxDyn(&[1]), scale), false);
        let scaled = qk.mul(&scalar_tensor);
        let attn = scaled.softmax(2);
        let out = attn.batched_matmul(&v2);
        let out2 = out.reshape(vec![b, self.num_heads, seq, head_dim]).unwrap();
        let out3 = out2.permute(vec![0,2,1,3]);
        let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
        self.linear_o.forward(&out4)
    }
    pub fn parameters_impl(&self) -> Vec<Tensor> {
        let mut p = self.linear_q.parameters();
        p.extend(self.linear_k.parameters());
        p.extend(self.linear_v.parameters());
        p.extend(self.linear_o.parameters());
        // Include NL-OOB slopes as parameters if present
        if let Some(s) = &self.slopes {
            p.push(s.clone());
        }
        p
    }
    pub fn named_parameters_impl(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix)));
        out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix)));
        out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix)));
        out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix)));
        if let Some(s) = &self.slopes {
            out.push((format!("{}.nl_oob.slopes", prefix), s.clone()));
        }
        out
    }
    pub fn load_state_dict_impl(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
        self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?;
        self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?;
        self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?;
        self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?;
        // Load NL-OOB config and slopes if present
        let config_key = format!("{}.nl_oob.config", prefix);
        if let Some(cfg) = state.get(&config_key) {
            // Expect cfg to be a scalar or 1D tensor containing 0 => Logarithmic, 1 => Gaussian
            let arr = cfg.lock().storage.to_f32_array();
            if arr.len() > 0 {
                let val = arr.as_slice().unwrap()[0];
                self.nl_oob_config = match val as i32 {
                    0 => Some(BiasFunction::Logarithmic),
                    1 => Some(BiasFunction::Gaussian),
                    _ => {
                        log::warn!("Unrecognized nl_oob.config {}, ignoring", val);
                        None
                    }
                };
            }
        }
        // Load slopes if present
        let s_key = format!("{}.nl_oob.slopes", prefix);
        if let Some(s) = state.get(&s_key) {
            self.slopes = Some(s.clone());
            // Ensure the slopes require grad flag is true (learnable)
            let mut lock = self.slopes.as_ref().unwrap().lock();
            lock.requires_grad = true;
        }
        Ok(())
    }

    /// Initialize geometric slopes as a learnable Tensor with shape [1, num_heads, 1, 1].
    /// Returns a Tensor with requires_grad=true.
    pub fn initialize_geometric_slopes(num_heads: usize, max_scale: f32) -> Tensor {
        // Protect against invalid inputs
        let min_scale = 0.5_f32;
        let mut slopes: Vec<f32> = Vec::with_capacity(num_heads);
        if num_heads == 1 {
            slopes.push(max_scale);
        } else {
            let ratio = (min_scale / max_scale).powf(1.0 / ((num_heads - 1) as f32));
            for h in 0..num_heads {
                slopes.push(max_scale * ratio.powf(h as f32));
            }
        }
        // Shape: [1, num_heads, 1, 1]
        let mut arr_vals = Vec::new();
        for v in slopes.iter() {
            arr_vals.push(*v);
        }
        // Build ndarray array of shape [1, num_heads, 1, 1]
        let arr = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1, num_heads, 1, 1]), arr_vals).unwrap().into_dyn();
        // Create a Tensor with requires_grad = true so slopes are learnable
        Tensor::new(arr, true)
    }

    /// Forward with a provided distance matrix for NL-OOB penalties.
    /// `distance_matrix` can be shape [seq, seq] or [b, seq, seq].
    pub fn forward_with_distance(&self, x: &Tensor, distance_matrix: &Tensor) -> Tensor {
        log::debug!("MultiHeadAttention::forward_with_distance start");
        // If NL-OOB not configured, fallback to vanilla forward
        if self.nl_oob_config.is_none() || self.slopes.is_none() {
            return self.forward_impl(x);
        }

        let q = self.linear_q.forward(x);
        let k = self.linear_k.forward(x);
        let v = self.linear_v.forward(x);
        let shape = q.lock().storage.shape();
        if shape.len() != 3 {
            log::error!("MultiHeadAttention::forward_with_distance expected 3D input");
            return x.clone();
        }
        let b = shape[0];
        let seq = shape[1];
        let head_dim = self.d_model / self.num_heads;
        let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();
        let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();
        let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();

        let k2t = k2.permute(vec![0,2,1]);
        let qk = q2.batched_matmul(&k2t);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
        let scaled = qk.mul(&scalar_tensor);

        log::debug!(
            "MultiHeadAttention::forward_with_distance computing phi: config={:?}",
            self.nl_oob_config
        );
        log::debug!("DEBUG MHA: computing phi: config={:?}", self.nl_oob_config);
        // Build phi(D) according to config
        let phi = match self.nl_oob_config.unwrap() {
            BiasFunction::Logarithmic => {
                // log(1 + d)
                let one = Tensor::new(ndarray::Array::from_elem(IxDyn(&[1]), 1.0), false);
                distance_matrix.add(&one).log()
            }
            BiasFunction::Gaussian => {
                // d^2
                distance_matrix.pow(2.0)
            }
        };

        log::debug!("MultiHeadAttention::forward_with_distance computed phi");
        log::debug!("DEBUG MHA: computed phi");
        // slopes shape: [1, num_heads, 1, 1]
        let slopes = self.slopes.as_ref().unwrap().clone();
        // Ensure slopes shaped correctly; reshape if necessary
        let slopes_ndim = {
            let lock = slopes.lock();
            lock.storage.shape().len()
        };
        let slopes_reshaped = match slopes_ndim {
            4 => slopes.clone(),
            _ => slopes.reshape(vec![1, self.num_heads, 1, 1]).unwrap(),
        };

        // Now compute penalty = - slopes * phi
        // phi shape either [seq, seq] or [b, seq, seq]; adapt to [b, num_heads, seq, seq]
        log::debug!("MultiHeadAttention::forward_with_distance computing penalty");
        log::debug!("DEBUG MHA: computing penalty");
        let penalty_full = {
            log::debug!("DEBUG MHA STEP: about to compute phi4 reshape");
            // Get phi shape without holding the lock across the match arms
            let phi_shape = {
                let lock = phi.lock();
                lock.storage.shape()
            };
            let phi_ndim = phi_shape.len();
            let phi_batch_dim = if phi_ndim >= 1 { phi_shape[0] } else { 0 };
            let mut phi4 = match phi_ndim {
                2 => phi.reshape(vec![1, 1, seq, seq]).unwrap(),
                3 => {
                    if phi_batch_dim != b {
                        log::error!("distance_matrix batch mismatch: expected batch size {} found {}", b, phi_batch_dim);
                        return x.clone();
                    }
                    phi.reshape(vec![b, 1, seq, seq]).unwrap()
                }
                _ => {
                    log::error!("distance_matrix must be 2D or 3D");
                    return x.clone();
                }
            };
            // If phi was 2D (global distances) and we have a batch dimension > 1, broadcast it across batch by multiplying with ones
            if phi_ndim == 2 && b > 1 {
                let ones = Tensor::new(ndarray::Array::from_elem(IxDyn(&[b, 1, seq, seq]), 1.0), false);
                phi4 = phi4.mul(&ones);
            }
            // Multiply slopes: slopes_reshaped has shape [1, num_heads, 1, 1], phi4 has [b, 1, seq, seq]
            // Avoid locking tensors inside the logging macro for long periods; obtain shapes and drop the locks
            let phi4_shape = {
                let lock = phi4.lock();
                lock.storage.shape()
            };
            let slopes_shape = {
                let lock = slopes_reshaped.lock();
                lock.storage.shape()
            };
            log::debug!("phi4 shape={:?}, slopes_reshaped shape={:?}", phi4_shape, slopes_shape);
            log::debug!("DEBUG MHA: phi4={:p}, slopes_reshaped={:p}", &phi4 as *const _, &slopes_reshaped as *const _);
            // Time the mul operation precisely
            log::debug!("DEBUG MHA: about to call mul for phi4.mul(slopes_reshaped) -> phi4={:p}, slopes_reshaped={:p}", &phi4 as *const _, &slopes_reshaped as *const _);
            let start = std::time::Instant::now();
            log::debug!("DEBUG MHA STEP: before phi4.mul -> phi4={:p}, slopes_reshaped={:p}", &phi4 as *const _, &slopes_reshaped as *const _);
            let prod = phi4.mul(&slopes_reshaped);
            log::debug!("DEBUG MHA STEP: after phi4.mul");
            let mul_dur = start.elapsed();
            let prod_shape = {
                let l = prod.lock();
                l.storage.shape()
            };
            log::debug!("prod shape after mul={:?}; mul_time_ms={:?}", prod_shape, mul_dur.as_millis());
            log::debug!("DEBUG MHA: prod shape after mul={:?}; mul_time_ms={:?}", prod_shape, mul_dur.as_millis());
            // prod is [b, num_heads, seq, seq]
            // convert to [b * num_heads, seq, seq]
            let start2 = std::time::Instant::now();
            log::debug!("DEBUG MHA STEP: before prod.reshape");
            let prod_reshaped = prod.reshape(vec![b * self.num_heads, seq, seq]).unwrap();
            log::debug!("DEBUG MHA STEP: after prod.reshape");
            let reshape_dur = start2.elapsed();
            let prod_reshaped_shape = { let l = prod_reshaped.lock(); l.storage.shape() };
            log::debug!("prod_reshaped shape after reshape={:?}; reshape_time_ms={:?}", prod_reshaped_shape, reshape_dur.as_millis());
            log::debug!("DEBUG MHA: prod_reshaped shape after reshape={:?}; reshape_time_ms={:?}", prod_reshaped_shape, reshape_dur.as_millis());
            // Negative penalty
            let neg_one = Tensor::new(ndarray::Array::from_elem(IxDyn(&[1]), -1.0), false);
            log::debug!("DEBUG MHA STEP: before neg multiply");
            let ret = prod_reshaped.mul(&neg_one);
            log::debug!("DEBUG MHA STEP: after neg multiply");
            ret
        };
        log::debug!("MultiHeadAttention::forward_with_distance computed penalty");

        let scaled_logits = scaled.add(&penalty_full);
        log::debug!("MultiHeadAttention::forward_with_distance added penalty to logits");
        let attn = scaled_logits.softmax(2);
        let out = attn.batched_matmul(&v2);
        let out2 = out.reshape(vec![b, self.num_heads, seq, head_dim]).unwrap();
        let out3 = out2.permute(vec![0,2,1,3]);
        let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
        self.linear_o.forward(&out4)
    }
}

impl crate::nn::Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor { self.forward_impl(input) }
    fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

pub struct TransformerBlock { pub mha: MultiHeadAttention, pub linear1: Linear, pub linear2: Linear, pub causal: bool }

impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self {
        TransformerBlock {
            mha: MultiHeadAttention::new(d_model, num_heads),
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
            causal: false,
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
        }
    }
    pub fn new_with_nl_oob(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        bias_type: BiasFunction,
        max_scale: f32,
    ) -> Self {
        TransformerBlock {
            mha: MultiHeadAttention::new_with_nl_oob(d_model, num_heads, bias_type, max_scale),
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
            causal: false,
        }
    }
    pub fn new_decoder(d_model: usize, d_ff: usize, num_heads: usize) -> Self {
        let mut t = TransformerBlock::new(d_model, d_ff, num_heads);
        t.causal = true;
        t
    }
    pub fn forward_block(&self, x: &Tensor) -> Tensor {
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
    /// Forward block that accepts a distance matrix for NL-OOB enabled MultiHeadAttention.
    pub fn forward_block_with_distance(&self, x: &Tensor, distance_matrix: &Tensor) -> Tensor {
        let attn_out = self.mha.forward_with_distance(x, distance_matrix);
        let x2 = x.add(&attn_out);
        let dim = x.lock().storage.shape()[2];
        let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
        let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
        let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
        let ff = self.linear1.forward(&x2norm).relu();
        let ff = self.linear2.forward(&ff);
        x2.add(&ff)
    }
    /// Forward block that uses cross-attention: query from x, key/value from kv
    pub fn forward_block_cross_attn(&self, x: &Tensor, kv: &Tensor) -> Tensor {
        let attn_out = self.mha.forward_cross(x, kv);
        let x2 = x.add(&attn_out);
        let dim = x.lock().storage.shape()[2];
        let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
        let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
        let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
        let ff = self.linear1.forward(&x2norm).relu();
        let ff = self.linear2.forward(&ff);
        x2.add(&ff)
    }
    pub fn parameters_impl(&self) -> Vec<Tensor> {
        let mut p = self.mha.parameters();
        p.extend(self.linear1.parameters());
        p.extend(self.linear2.parameters());
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
        Ok(())
    }
}

impl crate::nn::Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_block(input)
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
}

/// A simple encoder-decoder wrapper using pre-built TransformerBlocks
pub struct EncoderDecoderTransformer {
    pub encoder: Vec<TransformerBlock>,
    pub decoder: Vec<TransformerBlock>,
}

impl EncoderDecoderTransformer {
    pub fn new(encoder: Vec<TransformerBlock>, decoder: Vec<TransformerBlock>) -> Self {
        Self { encoder, decoder }
    }
    /// Encoder forward: sequentially runs encoder blocks
    pub fn encode(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        for b in &self.encoder {
            out = b.forward(&out);
        }
        out
    }
    /// Decoder forward: runs decoder blocks with cross-attention against encoder output
    pub fn decode(&self, x: &Tensor, encoder_output: &Tensor) -> Tensor {
        let mut out = x.clone();
        for b in &self.decoder {
            out = b.forward_block_cross_attn(&out, encoder_output);
        }
        out
    }
}

impl crate::nn::Module for EncoderDecoderTransformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        // For forward we treat input as encoder input and use a simple autoregressive decoder input that is the same
        let enc = self.encode(input);
        self.decode(input, &enc)
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.encoder.iter().flat_map(|b| b.parameters_impl()).chain(self.decoder.iter().flat_map(|b| b.parameters_impl())).collect()
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, b) in self.encoder.iter().enumerate() {
            out.extend(b.named_parameters_impl(&format!("{}.encoder.{}", prefix, i)));
        }
        for (i, b) in self.decoder.iter().enumerate() {
            out.extend(b.named_parameters_impl(&format!("{}.decoder.{}", prefix, i)));
        }
        out
    }
    fn load_state_dict(&mut self, _state: &HashMap<String, Tensor>, _prefix: &str) -> Result<(), String> {
        Err("EncoderDecoderTransformer::load_state_dict not implemented".to_string())
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
}