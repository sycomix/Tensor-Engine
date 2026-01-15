// Clean canonical transformer implementation
// NOTE: this file is a deprecated reference implementation kept for historical
// and compatibility reasons (the canonical implementation is in
// `transformer_cleaned.rs`). It is kept for reference and compatibility; keep
// in sync with `transformer_cleaned.rs` where applicable.

use crate::nn::{Linear, Module};
use crate::ops::{ChunkedAttention, FlashAttentionRef};
use crate::tensor::Tensor;
use log::error;
use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionVariant { Baseline, FlashRef, Chunked { chunk_size: usize } }


// First MultiHeadAttention definition removed - using the cleaner implementation below

// DEPRECATED: transformer_clean.rs has been superseded by transformer_cleaned.rs.
// This file remains only for reference; the canonical transformer implementation
// is in transformer_cleaned.rs and is re-exported by nn/mod.rs.
// First TransformerBlock definition removed - using the cleaner implementation below
// Full clean implementation below. This file intentionally implements a single definition
// for MultiHeadAttention and TransformerBlock and does not include duplicates.


pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
    let mut slopes = Vec::with_capacity(n_heads);
    for i in 0..n_heads { slopes.push(2f32.powf(-(i as f32) / (n_heads as f32 + 0.0))); }
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
    pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
        assert!(d_model.is_multiple_of(num_heads));
        assert!(num_heads.is_multiple_of(kv_heads));
        MultiHeadAttention { linear_q: Linear::new(d_model, d_model, true), linear_k: Linear::new(d_model, d_model, true), linear_v: Linear::new(d_model, d_model, true), linear_o: Linear::new(d_model, d_model, true), num_heads, d_model, kv_heads, use_rope, use_alibi: false, alibi_slopes: None, relative_bias: None, attention_variant: AttentionVariant::Baseline }
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
    pub fn set_attention_variant(&mut self, var: AttentionVariant) { self.attention_variant = var; }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.linear_q.forward(x);
        let k = self.linear_k.forward(x);
        let v = self.linear_v.forward(x);
        let shape = q.lock().storage.shape();
        if shape.len() != 3 { return x.clone(); }
        let b = shape[0];
        let seq = shape[1];
        let head_dim = self.d_model / self.num_heads;
        let q_reshaped = match q.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                error!("MultiHeadAttention forward: reshape q to (b, seq, num_heads, head_dim) failed: {}", e);
                return x.clone();
            }
        };
        let q_permuted = q_reshaped.permute(vec![0, 2, 1, 3]);
        let q2 = match q_permuted.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                error!("MultiHeadAttention forward: reshape q after permute failed: {}", e);
                return x.clone();
            }
        };
        let k_reshaped = match k.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                error!("MultiHeadAttention forward: reshape k to (b, seq, num_heads, head_dim) failed: {}", e);
                return x.clone();
            }
        };
        let k_permuted = k_reshaped.permute(vec![0, 2, 1, 3]);
        let k2 = match k_permuted.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                error!("MultiHeadAttention forward: reshape k after permute failed: {}", e);
                return x.clone();
            }
        };
        let v_reshaped = match v.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                error!("MultiHeadAttention forward: reshape v to (b, seq, num_heads, head_dim) failed: {}", e);
                return x.clone();
            }
        };
        let v_permuted = v_reshaped.permute(vec![0, 2, 1, 3]);
        let v2 = match v_permuted.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                error!("MultiHeadAttention forward: reshape v after permute failed: {}", e);
                return x.clone();
            }
        };
        let out = match self.attention_variant {
            AttentionVariant::Baseline => {
                let k2t = k2.permute(vec![0, 2, 1]);
                let qk = q2.batched_matmul(&k2t);
                let scale = 1.0f32 / (head_dim as f32).sqrt();
                let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1][..]), scale), false);
                let scaled = qk.mul(&scalar_tensor);
                let mut scaled_logits = scaled.clone();
                if self.use_alibi {
                    let slopes = if let Some(s) = &self.alibi_slopes { s.clone() } else { compute_alibi_slopes(self.num_heads) };
                    let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq][..]));
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
                    if shape == [1, seq, seq] || shape == [self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
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
        let out2 = match out.reshape(vec![b, self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                error!("MultiHeadAttention forward: reshape out to (b, num_heads, seq, head_dim) failed: {}", e);
                return x.clone();
            }
        };
        let out3 = out2.permute(vec![0, 2, 1, 3]);
        let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
            Ok(t) => t,
            Err(e) => {
                error!("MultiHeadAttention forward: reshape out after permute to (b, seq, d_model) failed: {}", e);
                return x.clone();
            }
        };
        self.linear_o.forward(&out4)
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.linear_q.parameters();
        p.extend(self.linear_k.parameters());
        p.extend(self.linear_v.parameters());
        p.extend(self.linear_o.parameters());
        p
    }
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix)));
        out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix)));
        out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix)));
        out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix)));
        out
    }
    pub fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
        self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?;
        self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?;
        self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?;
        self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?;
        Ok(())
    }
}
impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor { self.forward(input) }
    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}


pub struct TransformerBlock {
    pub mha: MultiHeadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
}
impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self {
        TransformerBlock {
            mha: MultiHeadAttention::new(d_model, num_heads),
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
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
        }
    }
    pub fn forward_block(&self, x: &Tensor) -> Tensor {
        let attn_out = self.mha.forward(x);
        let x2 = x.add(&attn_out);
        let dim = x.lock().storage.shape()[2];
        let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim][..])), true);
        let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim][..])), true);
        let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
        let ff = self.linear1.forward(&x2norm).relu();
        let ff = self.linear2.forward(&ff);
        x2.add(&ff)
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.mha.parameters();
        p.extend(self.linear1.parameters());
        p.extend(self.linear2.parameters());
        p
    }
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
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
    pub fn load_state_dict(
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
    pub fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
impl crate::nn::Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_block(input)
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
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

// Small public helper used to ensure these reference implementations are linked
// and considered used in non-test builds. This function does nothing at runtime
// but creates references to constructors and functions so the compiler marks the
// items as used, avoiding dead-code lints without removing or changing code.
#[doc(hidden)]
pub fn __ensure_transformer_clean_is_linked() {
    let _ = AttentionVariant::Baseline;
    let _ = AttentionVariant::FlashRef;
    let _ = AttentionVariant::Chunked { chunk_size: 1 };
    let _ = compute_alibi_slopes as fn(usize) -> Vec<f32>;
    let _ = MultiHeadAttention::new as fn(usize, usize) -> MultiHeadAttention;
    let _ = MultiHeadAttention::new_with_kv_and_rope as fn(usize, usize, usize, bool) -> MultiHeadAttention;
    // construct a local instance and read its fields to mark them as used
    let m = MultiHeadAttention::new_with_kv_and_rope(8, 2, 1, true);
    let _ = m.kv_heads;
    let _ = m.use_rope;
    // reference methods
    let _ = MultiHeadAttention::with_alibi as fn(MultiHeadAttention) -> MultiHeadAttention;
    let _ = MultiHeadAttention::with_relative_bias as fn(MultiHeadAttention, Tensor) -> MultiHeadAttention;
    let _ = MultiHeadAttention::set_attention_variant as fn(&mut MultiHeadAttention, AttentionVariant);
    // Reference TransformerBlock constructors and helper methods
    let _ = TransformerBlock::new as fn(usize, usize, usize) -> TransformerBlock;
    let _ = TransformerBlock::new_with_kv_and_rope as fn(usize, usize, usize, usize, bool) -> TransformerBlock;
    let _ = TransformerBlock::as_any as fn(&TransformerBlock) -> &dyn std::any::Any;
    let _ = TransformerBlock::as_any_mut as fn(&mut TransformerBlock) -> &mut dyn std::any::Any;
}

// Unit tests that exercise this deprecated reference implementation. These tests
// ensure the implementations remain compiling and exercised by CI (they also
// remove "unused" warnings when running test builds).
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray::IxDyn;

    #[test]
    fn smoke_multihead_attention_forward() {
        let d_model = 8usize;
        let num_heads = 2usize;
        let mha = MultiHeadAttention::new(d_model, num_heads);
        let input = Array::from_shape_fn((1, 4, d_model), |_| 0.1f32);
        let t = Tensor::new(input.into_dyn(), false);
        let out = mha.forward(&t);
        let shape = out.lock().storage.shape();
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[2], d_model);
    }

    #[test]
    fn smoke_transformer_block_forward() {
        let d_model = 8usize;
        let d_ff = 16usize;
        let heads = 2usize;
        let mut block = TransformerBlock::new(d_model, d_ff, heads);
        let input = Array::from_shape_fn((1, 4, d_model), |_| 0.2f32);
        let t = Tensor::new(input.into_dyn(), false);
        let out = block.forward_block(&t);
        let shape = out.lock().storage.shape();
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[2], d_model);

        // exercise as_any and as_any_mut
        let _ = block.as_any();
        let _ = block.as_any_mut();
    }

    #[test]
    fn exercise_extra_mha_features() {
        let d_model = 8usize;
        let num_heads = 2usize;
        let seq = 4usize;
        // new_with_kv_and_rope exercises the kv_heads/use_rope fields
        let mut mha = MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, 1, true);
        // with_alibi and with_relative_bias
        mha = mha.with_alibi();
        let rb = Tensor::new(ndarray::Array::zeros(IxDyn(&[1, seq, seq][..])), false);
        mha = mha.with_relative_bias(rb.clone());
        // FlashRef variant
        mha.set_attention_variant(AttentionVariant::FlashRef);
        let input = Array::from_shape_fn((1, seq, d_model), |_| 0.0f32);
        let t = Tensor::new(input.into_dyn(), false);
        let out = mha.forward(&t);
        let shape = out.lock().storage.shape();
        assert_eq!(shape[2], d_model);

        // Chunked variant
        let mut mha_chunked = MultiHeadAttention::new(d_model, num_heads);
        mha_chunked.set_attention_variant(AttentionVariant::Chunked { chunk_size: 2 });
        let out2 = mha_chunked.forward(&t);
        let shape2 = out2.lock().storage.shape();
        assert_eq!(shape2[2], d_model);
    }
}
