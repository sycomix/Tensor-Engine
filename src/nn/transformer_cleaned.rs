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

/// Compute simple ALiBi slopes
pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
    let mut slopes = Vec::with_capacity(n_heads);
    for i in 0..n_heads { let x = (i as f32) / (n_heads as f32 + 0.0f32); slopes.push(2f32.powf(-x)); }
    slopes
}

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
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self { Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false) }
    pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { MultiHeadAttention { linear_q: Linear::new(d_model, d_model, true), linear_k: Linear::new(d_model, d_model, true), linear_v: Linear::new(d_model, d_model, true), linear_o: Linear::new(d_model, d_model, true), num_heads, d_model, kv_heads, use_rope, use_alibi: false, alibi_slopes: None, relative_bias: None, attention_variant: AttentionVariant::Baseline } }
    pub fn with_alibi(mut self) -> Self { self.use_alibi = true; self.alibi_slopes = Some(compute_alibi_slopes(self.num_heads)); self }
    pub fn with_relative_bias(mut self, bias: Tensor) -> Self { self.relative_bias = Some(bias); self }
    pub fn set_attention_variant(&mut self, var: AttentionVariant) { self.attention_variant = var; }

    pub fn forward_impl(&self, x: &Tensor) -> Tensor {
        self.forward_with_causal(x, false, None)
    }

    pub fn forward_with_causal(&self, x: &Tensor, causal: bool, causal_offset: Option<usize>) -> Tensor {
        let q = self.linear_q.forward(x);
        let k = self.linear_k.forward(x);
        let v = self.linear_v.forward(x);
        let shape = q.lock().storage.shape(); if shape.len() != 3 { return x.clone(); }
        let b = shape[0]; let seq = shape[1]; let head_dim = self.d_model / self.num_heads;
        let q = match q.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape q to (b, seq, num_heads, head_dim) failed: {}", e); return x.clone(); }
        };
        let q = q.permute(vec![0,2,1,3]);
        let q2 = match q.reshape(vec![b*self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape q after permute failed: {}", e); return x.clone(); }
        };
        let k = match k.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape k to (b, seq, num_heads, head_dim) failed: {}", e); return x.clone(); }
        };
        let k = k.permute(vec![0,2,1,3]);
        let k2 = match k.reshape(vec![b*self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape k after permute failed: {}", e); return x.clone(); }
        };
        let v = match v.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape v to (b, seq, num_heads, head_dim) failed: {}", e); return x.clone(); }
        };
        let v = v.permute(vec![0,2,1,3]);
        let v2 = match v.reshape(vec![b*self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape v after permute failed: {}", e); return x.clone(); }
        };

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
                    for batch in 0..b { for h in 0..self.num_heads { let slope = slopes[h]; for i in 0..seq { for j in 0..seq { let dist = (j as isize - i as isize) as f32; bias_arr[[batch * self.num_heads + h, i, j]] = -slope * dist; }}}}
                    let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                    scaled_logits = scaled_logits.add(&bias_t);
                }
                if let Some(rb) = &self.relative_bias {
                    let shape = rb.lock().storage.shape(); if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
                }
                if causal {
                    let mut mask_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                    for i in 0..(b * self.num_heads) { for r in 0..seq { for c2 in (r+1)..seq { if let Some(offset) = causal_offset { let r_is_text = r >= offset; let c2_is_text = c2 >= offset; if r_is_text && c2_is_text { mask_arr[[i, r, c2]] = -1e9_f32; } } else { mask_arr[[i, r, c2]] = -1e9_f32; } }}}
                    let mask_t = crate::tensor::Tensor::new(mask_arr, false);
                    scaled_logits = scaled_logits.add(&mask_t);
                }
                let attn = scaled_logits.softmax(2);
                attn.batched_matmul(&v2)
            }
            AttentionVariant::FlashRef => { let flash = FlashAttentionRef::new(head_dim); Tensor::apply(Arc::new(flash), &[q2.clone(), k2.clone(), v2.clone()]) }
            AttentionVariant::Chunked { chunk_size } => { let op = ChunkedAttention::new(head_dim, chunk_size); Tensor::apply(Arc::new(op), &[q2.clone(), k2.clone(), v2.clone()]) }
        };
        let out2 = match out.reshape(vec![b, self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape out to (b, num_heads, seq, head_dim) failed: {}", e); return x.clone(); }
        };
        let out3 = out2.permute(vec![0,2,1,3]);
        let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape out after permute to (b, seq, d_model) failed: {}", e); return x.clone(); }
        };
        self.linear_o.forward(&out4)
    }

    pub fn parameters_impl(&self) -> Vec<Tensor> { let mut p = self.linear_q.parameters(); p.extend(self.linear_k.parameters()); p.extend(self.linear_v.parameters()); p.extend(self.linear_o.parameters()); p }
    pub fn named_parameters_impl(&self, prefix: &str) -> Vec<(String, Tensor)> { let mut out = Vec::new(); out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix))); out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix))); out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix))); out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix))); out }
    pub fn load_state_dict_impl(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?; self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?; self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?; self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?; Ok(()) }
}

impl crate::nn::Module for MultiHeadAttention { fn forward(&self, input: &Tensor) -> Tensor { self.forward_impl(input) } fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() } fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) } fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) } fn as_any(&self) -> &dyn std::any::Any { self } }

pub struct TransformerBlock { pub mha: MultiHeadAttention, pub linear1: Linear, pub linear2: Linear, pub causal: bool }
impl TransformerBlock { pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true), causal: false } } pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { TransformerBlock { mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true), causal: false } } pub fn new_decoder(d_model: usize, d_ff: usize, num_heads: usize) -> Self { let mut t = TransformerBlock::new(d_model, d_ff, num_heads); t.causal = true; t } pub fn forward_block_impl(&self, x: &Tensor) -> Tensor { let attn_out = self.mha.forward_with_causal(x, self.causal, None); let x2 = x.add(&attn_out); let dim = x.lock().storage.shape()[2]; let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true); let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true); let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta); let ff = self.linear1.forward(&x2norm).relu(); let ff = self.linear2.forward(&ff); x2.add(&ff) }
impl crate::nn::Module for TransformerBlock { fn forward(&self, input: &Tensor) -> Tensor { self.forward_block_impl(input) } fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() } fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) } fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) } fn as_any(&self) -> &dyn std::any::Any { self } }
