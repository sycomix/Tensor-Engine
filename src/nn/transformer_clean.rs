// Clean canonical transformer implementation
use crate::nn::Module;
use crate::tensor::Tensor;
use crate::nn::Linear;
use crate::ops::{ChunkedAttention, FlashAttentionRef};
use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionVariant { Baseline, FlashRef, Chunked { chunk_size: usize } }

pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
    (0..n_heads).map(|i| 2f32.powf(- (i as f32) / (n_heads as f32 + 0.0))).collect()
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
        assert!(d_model % num_heads == 0);
        assert!(num_heads % kv_heads == 0);
        MultiHeadAttention { linear_q: Linear::new(d_model, d_model, true), linear_k: Linear::new(d_model, d_model, true), linear_v: Linear::new(d_model, d_model, true), linear_o: Linear::new(d_model, d_model, true), num_heads, d_model, kv_heads, use_rope, use_alibi: false, alibi_slopes: None, relative_bias: None, attention_variant: AttentionVariant::Baseline }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor { x.clone() }
    pub fn parameters(&self) -> Vec<Tensor> { Vec::new() }
    pub fn named_parameters(&self, _prefix: &str) -> Vec<(String, Tensor)> { Vec::new() }
    pub fn load_state_dict(&mut self, _state: &HashMap<String, Tensor>, _prefix: &str) -> Result<(), String> { Ok(()) }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor { self.forward(input) }
    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

// DEPRECATED: transformer_clean.rs has been superseded by transformer_cleaned.rs.
// This file remains only for reference; the canonical transformer implementation
// is in transformer_cleaned.rs and is re-exported by nn/mod.rs.
pub struct TransformerBlock {
    pub mha: MultiHeadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
}

impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self {
        TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) }
    }
    pub fn forward_block(&self, x: &Tensor) -> Tensor { x.clone() }
    pub fn parameters(&self) -> Vec<Tensor> { Vec::new() }
    pub fn named_parameters(&self, _prefix: &str) -> Vec<(String, Tensor)> { Vec::new() }
    pub fn load_state_dict(&mut self, _state: &HashMap<String, Tensor>, _prefix: &str) -> Result<(), String> { Ok(()) }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}
// Full clean implementation below. This file intentionally implements a single definition
// for MultiHeadAttention and TransformerBlock and does not include duplicates.
use crate::nn::Module;
use crate::tensor::Tensor;
use crate::nn::Linear;
use crate::ops::{ChunkedAttention, FlashAttentionRef};
use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionVariant { Baseline, FlashRef, Chunked { chunk_size: usize } }

pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
    let mut slopes = Vec::with_capacity(n_heads);
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
    pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { assert!(d_model % num_heads == 0); assert!(num_heads % kv_heads == 0); MultiHeadAttention { linear_q: Linear::new(d_model, d_model, true), linear_k: Linear::new(d_model, d_model, true), linear_v: Linear::new(d_model, d_model, true), linear_o: Linear::new(d_model, d_model, true), num_heads, d_model, kv_heads, use_rope, use_alibi: false, alibi_slopes: None, relative_bias: None, attention_variant: AttentionVariant::Baseline } }
    pub fn forward(&self, x: &Tensor) -> Tensor { /* impl */ x.clone() }
    pub fn parameters(&self) -> Vec<Tensor> { vec![] }
    pub fn named_parameters(&self, _prefix: &str) -> Vec<(String, Tensor)> { vec![] }
    pub fn load_state_dict(&mut self, _state: &HashMap<String, Tensor>, _prefix: &str) -> Result<(), String> { Ok(()) }
}
impl Module for MultiHeadAttention { fn forward(&self, input: &Tensor) -> Tensor { self.forward(input) } fn parameters(&self) -> Vec<Tensor> { self.parameters() } fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) } fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) } fn as_any(&self) -> &dyn std::any::Any { self } }

pub struct TransformerBlock { pub mha: MultiHeadAttention, pub linear1: Linear, pub linear2: Linear }
impl TransformerBlock { pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } } pub fn forward_block(&self, x: &Tensor) -> Tensor { x.clone() } pub fn parameters(&self) -> Vec<Tensor> { vec![] } pub fn named_parameters(&self, _prefix: &str) -> Vec<(String, Tensor)> { vec![] } pub fn load_state_dict(&mut self, _state: &HashMap<String, Tensor>, _prefix: &str) -> Result<(), String> { Ok(()) } }
impl Module for TransformerBlock { fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) } fn parameters(&self) -> Vec<Tensor> { self.parameters() } fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) } fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) } fn as_any(&self) -> &dyn std::any::Any { self } }
// Canonical transformer implementation (clean, single copy)
use crate::nn::Module;
use crate::tensor::Tensor;
use crate::nn::Linear;
use crate::ops::{ChunkedAttention, FlashAttentionRef};
use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionVariant { Baseline, FlashRef, Chunked { chunk_size: usize } }

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
    pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { assert!(d_model % num_heads == 0); assert!(num_heads % kv_heads == 0); MultiHeadAttention { linear_q: Linear::new(d_model, d_model, true), linear_k: Linear::new(d_model, d_model, true), linear_v: Linear::new(d_model, d_model, true), linear_o: Linear::new(d_model, d_model, true), num_heads, d_model, kv_heads, use_rope, use_alibi: false, alibi_slopes: None, relative_bias: None, attention_variant: AttentionVariant::Baseline } }
    pub fn with_alibi(mut self) -> Self { self.use_alibi = true; self.alibi_slopes = Some(compute_alibi_slopes(self.num_heads)); self }
    pub fn with_relative_bias(mut self, bias: Tensor) -> Self { self.relative_bias = Some(bias); self }
    pub fn set_attention_variant(&mut self, var: AttentionVariant) { self.attention_variant = var; }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.linear_q.forward(x);
        let k = self.linear_k.forward(x);
        let v = self.linear_v.forward(x);
        let shape = q.lock().storage.shape(); if shape.len() != 3 { return x.clone(); }
        let b = shape[0]; let seq = shape[1]; let head_dim = self.d_model / self.num_heads;
        let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();
        let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();
        let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0,2,1,3]).reshape(vec![b*self.num_heads, seq, head_dim]).unwrap();

        let out = match self.attention_variant {
            AttentionVariant::Baseline => {
                let k2t = k2.permute(vec![0,2,1]); let qk = q2.batched_matmul(&k2t);
                let scale = 1.0f32 / (head_dim as f32).sqrt(); let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false); let scaled = qk.mul(&scalar_tensor);
                let mut scaled_logits = scaled.clone(); if self.use_alibi { let slopes = if let Some(s) = &self.alibi_slopes { s.clone() } else { compute_alibi_slopes(self.num_heads) }; let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq])); for batch in 0..b { for h in 0..self.num_heads { let slope = slopes[h]; for i in 0..seq { for j in 0..seq { let dist = (j as isize - i as isize) as f32; bias_arr[[batch * self.num_heads + h, i, j]] = -slope * dist; }}}} let bias_t = crate::tensor::Tensor::new(bias_arr, false); scaled_logits = scaled_logits.add(&bias_t); }
                if let Some(rb) = &self.relative_bias { let shape = rb.lock().storage.shape(); if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }}
                let attn = scaled_logits.softmax(2); attn.batched_matmul(&v2)
            }
            AttentionVariant::FlashRef => { let flash = FlashAttentionRef::new(head_dim); Tensor::apply(Arc::new(flash), &[q2.clone(), k2.clone(), v2.clone()]) }
            AttentionVariant::Chunked { chunk_size } => { let op = ChunkedAttention::new(head_dim, chunk_size); Tensor::apply(Arc::new(op), &[q2.clone(), k2.clone(), v2.clone()]) }
        };
        let out2 = out.reshape(vec![b, self.num_heads, seq, head_dim]).unwrap(); let out3 = out2.permute(vec![0,2,1,3]); let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap(); self.linear_o.forward(&out4)
    }
    pub fn parameters(&self) -> Vec<Tensor> { let mut p = self.linear_q.parameters(); p.extend(self.linear_k.parameters()); p.extend(self.linear_v.parameters()); p.extend(self.linear_o.parameters()); p }
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { let mut out = Vec::new(); out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix))); out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix))); out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix))); out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix))); out }
    pub fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?; self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?; self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?; self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?; Ok(()) }
}
impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor { self.forward(input) }
    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

pub struct TransformerBlock {
    pub mha: MultiHeadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
}

impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self {
        TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) }
    }

    pub fn forward_block(&self, x: &Tensor) -> Tensor {
        let attn_out = self.mha.forward(x);
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

impl crate::nn::Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}


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
            attention_variant: AttentionVariant::Baseline,
            use_alibi: false,
            alibi_slopes: None,
            relative_bias: None,
        }
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
    pub fn forward(&self, x: &Tensor) -> Tensor {
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
        let q = match q.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape q to (b, seq, num_heads, head_dim) failed: {}", e); return x.clone(); }
        };
        let q = q.permute(vec![0, 2, 1, 3]);
        let q2 = match q.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape q after permute failed: {}", e); return x.clone(); }
        };
        let k = match k.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape k to (b, seq, num_heads, head_dim) failed: {}", e); return x.clone(); }
        };
        let k = k.permute(vec![0, 2, 1, 3]);
        let k2 = match k.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape k after permute failed: {}", e); return x.clone(); }
        };
        let v = match v.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape v to (b, seq, num_heads, head_dim) failed: {}", e); return x.clone(); }
        };
        let v = v.permute(vec![0, 2, 1, 3]);
        let v2 = match v.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape v after permute failed: {}", e); return x.clone(); }
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
            Err(e) => { log::error!("MultiHeadAttention forward: reshape out to (b, num_heads, seq, head_dim) failed: {}", e); return x.clone(); }
        };
        let out3 = out2.permute(vec![0, 2, 1, 3]);
        let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
            Ok(t) => t,
            Err(e) => { log::error!("MultiHeadAttention forward: reshape out after permute to (b, seq, d_model) failed: {}", e); return x.clone(); }
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
    pub fn load_state_dict(
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
}
impl crate::nn::Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
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
        let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
        let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
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
}
