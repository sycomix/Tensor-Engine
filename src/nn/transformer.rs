use crate::nn::Linear;
// Minimal transformer — single canonical implementation.
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

/// Compute simple ALiBi slopes; geometric decay-style.
pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
    let mut slopes = Vec::with_capacity(n_heads);
    for i in 0..n_heads {
        let x = (i as f32) / (n_heads as f32 + 0.0);
        slopes.push(2f32.powf(-x));
    }
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
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
    }
    pub fn new_with_kv_and_rope(
        d_model: usize,
        num_heads: usize,
        kv_heads: usize,
        use_rope: bool,
    ) -> Self {
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
        let q2 = q
            .reshape(vec![b, seq, self.num_heads, head_dim])
            .unwrap()
            .permute(vec![0, 2, 1, 3])
            .reshape(vec![b * self.num_heads, seq, head_dim])
            .unwrap();
        let k2 = k
            .reshape(vec![b, seq, self.num_heads, head_dim])
            .unwrap()
            .permute(vec![0, 2, 1, 3])
            .reshape(vec![b * self.num_heads, seq, head_dim])
            .unwrap();
        let v2 = v
            .reshape(vec![b, seq, self.num_heads, head_dim])
            .unwrap()
            .permute(vec![0, 2, 1, 3])
            .reshape(vec![b * self.num_heads, seq, head_dim])
            .unwrap();

        let out = match self.attention_variant {
            AttentionVariant::Baseline => {
                let k2t = k2.permute(vec![0, 2, 1]);
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

        let out2 = out
            .reshape(vec![b, self.num_heads, seq, head_dim])
            .unwrap();
        let out3 = out2.permute(vec![0, 2, 1, 3]);
        let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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

/// TransformerBlock: attention + residual + layernorm + feedforward + residual
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
        log::info!(
            "TransformerBlock forward: input shape {:?}",
            x.lock().storage.shape()
        );
        let attn_out = self.mha.forward(x);
        let x2 = x.add(&attn_out);
        // LayerNorm not exposed as module; use Tensor::layer_norm with axis=-1
        let dim = x.lock().storage.shape()[2];
        // create gamma and beta as ones/zeros (affine) for now
        let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
        let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
        let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
        let ff = self.linear1.forward(&x2norm).relu();
        let ff = self.linear2.forward(&ff);
        log::debug!(
            "TransformerBlock forward complete: output shape {:?}",
            x2.lock().storage.shape()
        );
        x2.add(&ff)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.mha.parameters();
        p.extend(self.linear1.parameters());
        p.extend(self.linear2.parameters());
        p
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(self.mha.named_parameters(&format!("{}.mha", prefix)));
        out.extend(self.linear1.named_parameters(&format!("{}.linear1", prefix)));
        out.extend(self.linear2.named_parameters(&format!("{}.linear2", prefix)));
        out
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.mha.load_state_dict(state, &format!("{}.mha", prefix))?;
        self.linear1.load_state_dict(state, &format!("{}.linear1", prefix))?;
        self.linear2.load_state_dict(state, &format!("{}.linear2", prefix))?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
use crate::nn::Linear;
// Minimal transformer placeholder (temporary)
// This file is a placeholder to ensure compilation while we finalize the canonical implementation.
use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::IxDyn;

pub struct TransformerModulePlaceholder {
    pub linear: Linear,
}

impl TransformerModulePlaceholder {
    pub fn new(d_model: usize) -> Self { TransformerModulePlaceholder { linear: Linear::new(d_model, d_model, true) } }
}

impl Module for TransformerModulePlaceholder {
    fn forward(&self, input: &Tensor) -> Tensor {
        // pass-through for placeholder
        input.clone()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![self.linear.weight.clone(), self.linear.bias.clone()] }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
//! Canonical transformer implementation — MultiHeadAttention & TransformerBlock
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

/// Compute simple ALiBi slopes; geometric decay-style.
pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
    let mut slopes = Vec::with_capacity(n_heads);
    for i in 0..n_heads {
        let x = (i as f32) / (n_heads as f32);
        slopes.push(2f32.powf(-x));
    }
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
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
    }
    pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
        assert!(d_model % num_heads == 0);
        assert!(num_heads % kv_heads == 0);
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

    pub fn forward_impl(&self, x: &Tensor) -> Tensor {
        let q = self.linear_q.forward(x);
        let k = self.linear_k.forward(x);
        let v = self.linear_v.forward(x);
        let shape = q.lock().storage.shape();
        if shape.len() != 3 { return x.clone(); }
        let b = shape[0];
        let seq = shape[1];
        let head_dim = self.d_model / self.num_heads;
        let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
        let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
        let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
        let out = match self.attention_variant {
            AttentionVariant::Baseline => {
                let k2t = k2.permute(vec![0, 2, 1]);
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
                    if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
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
        let out3 = out2.permute(vec![0, 2, 1, 3]);
        let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
        out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix)));
        out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix)));
        out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix)));
        out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix)));
        out
    }
    pub fn load_state_dict_impl(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
        self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?;
        self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?;
        self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?;
        self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?;
        Ok(())
    }
}
impl crate::nn::Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor { self.forward_impl(input) }
    fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

pub struct TransformerBlock {
    pub mha: MultiHeadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
}
impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
    pub fn forward_block_impl(&self, x: &Tensor) -> Tensor {
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
    impl crate::nn::Module for TransformerBlock {
        fn forward(&self, input: &Tensor) -> Tensor { self.forward_block_impl(input) }
        fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
        fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
        fn as_any(&self) -> &dyn std::any::Any { self }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    }
    // Canonical transformer implementation — MultiHeadAttention & TransformerBlock
    use crate::nn::Linear;
    use crate::nn::Module;
    use crate::ops::{ChunkedAttention, FlashAttentionRef};
    // Canonical transformer implementation — single copy only
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

    /// Compute simple ALiBi slopes; geometric decay-style.
    pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
        let mut slopes = Vec::with_capacity(n_heads);
        for i in 0..n_heads {
            // A simple geometric-style slope assignment; not necessarily identical to all literature
            let x = (i as f32) / (n_heads as f32 + 0.0);
            slopes.push(2f32.powf(-x));
        }
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
            assert!(d_model % num_heads == 0);
            assert!(num_heads % kv_heads == 0);
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

        pub fn forward_impl(&self, x: &Tensor) -> Tensor {
            let q = self.linear_q.forward(x);
            let k = self.linear_k.forward(x);
            let v = self.linear_v.forward(x);
            let shape = q.lock().storage.shape();
            if shape.len() != 3 { return x.clone(); }
            let b = shape[0];
            let seq = shape[1];
            let head_dim = self.d_model / self.num_heads;
            let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
            let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
            let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();

            let out = match self.attention_variant {
                AttentionVariant::Baseline => {
                    let k2t = k2.permute(vec![0, 2, 1]);
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
                        if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
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
            let out3 = out2.permute(vec![0, 2, 1, 3]);
            let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
            out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix)));
            out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix)));
            out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix)));
            out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix)));
            out
        }
        pub fn load_state_dict_impl(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
            self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?;
            self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?;
            self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?;
            self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?;
            Ok(())
        }
    }

    impl crate::nn::Module for MultiHeadAttention {
        fn forward(&self, input: &Tensor) -> Tensor { self.forward_impl(input) }
        fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
        fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
        fn as_any(&self) -> &dyn std::any::Any { self }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    }

    pub struct TransformerBlock {
        pub mha: MultiHeadAttention,
        pub linear1: Linear,
        pub linear2: Linear,
    }
    impl TransformerBlock {
        pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
        pub fn forward_block_impl(&self, x: &Tensor) -> Tensor {
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
        impl crate::nn::Module for TransformerBlock {
            fn forward(&self, input: &Tensor) -> Tensor { self.forward_block_impl(input) }
            fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
            fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
            fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
            fn as_any(&self) -> &dyn std::any::Any { self }
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        }
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
            pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
                assert!(d_model % num_heads == 0);
                assert!(num_heads % kv_heads == 0);
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
                let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();

                let out = match self.attention_variant {
                    AttentionVariant::Baseline => {
                        let k2t = k2.permute(vec![0, 2, 1]);
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
                let out3 = out2.permute(vec![0, 2, 1, 3]);
                let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
            pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
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
            impl Module for TransformerBlock {
                fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                fn as_any(&self) -> &dyn std::any::Any { self }
                fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
            }
            // Minimal canonical transformer implementation — single definition
            use crate::nn::Linear;
            use crate::nn::Module;
            use crate::ops::{ChunkedAttention, FlashAttentionRef};
            use crate::tensor::Tensor;
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
                pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
                    assert!(d_model % num_heads == 0);
                    assert!(num_heads % kv_heads == 0);
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
                    let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                    let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                    let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                    let out = match self.attention_variant {
                        AttentionVariant::Baseline => {
                            let k2t = k2.permute(vec![0, 2, 1]);
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
                    let out3 = out2.permute(vec![0, 2, 1, 3]);
                    let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
                pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
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
                impl Module for TransformerBlock {
                    fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                    fn as_any(&self) -> &dyn std::any::Any { self }
                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                }
                // Replaced corrupted, duplicated content with a single, compact, canonical implementation.
                use crate::nn::Linear;
                use crate::nn::Module;
                use crate::ops::{ChunkedAttention, FlashAttentionRef};
                use crate::tensor::Tensor;
                use ndarray::{Array, IxDyn};
                use std::collections::HashMap;
                use std::sync::Arc;

                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                pub enum AttentionVariant {
                    Baseline,
                    FlashRef,
                    Chunked { chunk_size: usize },
                }

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
                        assert!(d_model % num_heads == 0);
                        assert!(num_heads % kv_heads == 0);
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
                        let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                        let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                        let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();

                        let out = match self.attention_variant {
                            AttentionVariant::Baseline => {
                                let k2t = k2.permute(vec![0, 2, 1]);
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
                                    if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
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
                        let out3 = out2.permute(vec![0, 2, 1, 3]);
                        let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
                    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
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
                    impl Module for TransformerBlock {
                        fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                        fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                        fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                        fn as_any(&self) -> &dyn std::any::Any { self }
                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                    }
                    // Canonical transformer implementation — MultiHeadAttention & TransformerBlock
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

                    /// Compute simple ALiBi slopes; geometric decay-style.
                    pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                        let mut slopes = Vec::with_capacity(n_heads);
                        for i in 0..n_heads {
                            // A simple geometric-style slope assignment; not necessarily identical to all literature
                            let x = (i as f32) / (n_heads as f32 + 0.0);
                            slopes.push(2f32.powf(-x));
                        }
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
                        pub fn new(d_model: usize, num_heads: usize) -> Self {
                            Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
                        }

                        pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
                            assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
                            assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
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
                        pub fn set_attention_variant(&mut self, var: AttentionVariant) { self.attention_variant = var; }

                        /// Core forward impl that computes attention output.
                        pub fn forward_impl(&self, x: &Tensor) -> Tensor {
                            let q = self.linear_q.forward(x);
                            let k = self.linear_k.forward(x);
                            let v = self.linear_v.forward(x);

                            let shape = q.lock().storage.shape();
                            if shape.len() != 3 { return x.clone(); }
                            let b = shape[0];
                            let seq = shape[1];
                            let head_dim = self.d_model / self.num_heads;

                            // Reshape into [b*heads, seq, head_dim]
                            let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                            let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                            let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();

                            let out = match self.attention_variant {
                                AttentionVariant::Baseline => {
                                    let k2t = k2.permute(vec![0, 2, 1]);
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
                                        if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
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
                            let out3 = out2.permute(vec![0, 2, 1, 3]);
                            let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
                            out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix)));
                            out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix)));
                            out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix)));
                            out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix)));
                            out
                        }
                        pub fn load_state_dict_impl(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                            self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?;
                            self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?;
                            self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?;
                            self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?;
                            Ok(())
                        }
                    }

                    impl crate::nn::Module for MultiHeadAttention {
                        fn forward(&self, input: &Tensor) -> Tensor { self.forward_impl(input) }
                        fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
                        fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
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

                        pub fn forward_block_impl(&self, x: &Tensor) -> Tensor {
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

                        pub fn parameters_impl(&self) -> Vec<Tensor> {
                            let mut p = self.mha.parameters();
                            p.extend(self.linear1.parameters());
                            p.extend(self.linear2.parameters());
                            p
                        }
                        pub fn named_parameters_impl(&self, prefix: &str) -> Vec<(String, Tensor)> {
                            let mut out = Vec::new();
                            out.extend(self.mha.named_parameters(&format!("{}.mha", prefix)));
                            out.extend(self.linear1.named_parameters(&format!("{}.linear1", prefix)));
                            out.extend(self.linear2.named_parameters(&format!("{}.linear2", prefix)));
                            out
                        }
                        pub fn load_state_dict_impl(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                            self.mha.load_state_dict(state, &format!("{}.mha", prefix))?;
                            self.linear1.load_state_dict(state, &format!("{}.linear1", prefix))?;
                            self.linear2.load_state_dict(state, &format!("{}.linear2", prefix))?;
                            Ok(())
                        }
                    }

                    impl crate::nn::Module for TransformerBlock {
                        fn forward(&self, input: &Tensor) -> Tensor { self.forward_block_impl(input) }
                        fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
                        fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
                        fn as_any(&self) -> &dyn std::any::Any { self }
                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                    }
                    // Canonical transformer implementation — MultiHeadAttention & TransformerBlock
                    use crate::nn::Linear;
                    use crate::nn::Module;
                    use crate::ops::{ChunkedAttention, FlashAttentionRef};
                    use crate::tensor::Tensor;
                    use ndarray::{Array, IxDyn};
                    use std::collections::HashMap;
                    use std::sync::Arc;

                    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                    pub enum AttentionVariant { Baseline, FlashRef, Chunked { chunk_size: usize } }

                    pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                        let mut slopes = Vec::with_capacity(n_heads);
                        for i in 0..n_heads {
                            let x = (i as f32) / (n_heads as f32 + 0.0);
                            slopes.push(2f32.powf(-x));
                        }
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
                            assert!(d_model % num_heads == 0);
                            assert!(num_heads % kv_heads == 0);
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

                        pub fn forward_impl(&self, x: &Tensor) -> Tensor {
                            let q = self.linear_q.forward(x);
                            let k = self.linear_k.forward(x);
                            let v = self.linear_v.forward(x);
                            let shape = q.lock().storage.shape();
                            if shape.len() != 3 { return x.clone(); }
                            let b = shape[0];
                            let seq = shape[1];
                            let head_dim = self.d_model / self.num_heads;
                            let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                            let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                            let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                            let scale = 1.0f32 / (head_dim as f32).sqrt();
                            let out = match self.attention_variant {
                                AttentionVariant::Baseline => {
                                    let k2t = k2.permute(vec![0, 2, 1]);
                                    let qk = q2.batched_matmul(&k2t);
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
                                        if shape.len() == 2 {
                                            let max_range = (shape[1] - 1) / 2;
                                            let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                                            let rb_arr = rb.lock().storage.to_f32_array();
                                            let rb_view = rb_arr.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                                            for batch in 0..b {
                                                for h in 0..self.num_heads {
                                                    for i in 0..seq {
                                                        for j in 0..seq {
                                                            let rel = (j as isize - i as isize).max(-(max_range as isize)).min(max_range as isize) as isize;
                                                            let idx = (rel + max_range as isize) as usize;
                                                            bias_arr[[batch * self.num_heads + h, i, j]] = rb_view[[h, idx]];
                                                        }
                                                    }
                                                }
                                            }
                                            let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                            scaled_logits = scaled_logits.add(&bias_t);
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
                            let out2 = out.reshape(vec![b, self.num_heads, seq, head_dim]).unwrap();
                            let out3 = out2.permute(vec![0, 2, 1, 3]);
                            let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
                            out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix)));
                            out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix)));
                            out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix)));
                            out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix)));
                            out
                        }
                        pub fn load_state_dict_impl(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                            self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?;
                            self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?;
                            self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?;
                            self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?;
                            Ok(())
                        }
                    }
                    impl crate::nn::Module for MultiHeadAttention {
                        fn forward(&self, input: &Tensor) -> Tensor { self.forward_impl(input) }
                        fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
                        fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
                        fn as_any(&self) -> &dyn std::any::Any { self }
                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                    }

                    pub struct TransformerBlock {
                        pub mha: MultiHeadAttention,
                        pub linear1: Linear,
                        pub linear2: Linear,
                    }
                    impl TransformerBlock {
                        pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                        pub fn forward_block_impl(&self, x: &Tensor) -> Tensor {
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
                        impl crate::nn::Module for TransformerBlock {
                            fn forward(&self, input: &Tensor) -> Tensor { self.forward_block_impl(input) }
                            fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
                            fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
                            fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
                            fn as_any(&self) -> &dyn std::any::Any { self }
                            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                        }
                        // Canonical transformer module — MultiHeadAttention and TransformerBlock
                        // DEPRECATED: transformer.rs replaced by transformer_cleaned.rs
                        // This file is retained temporarily for reference; the canonical implementation
                        // is now provided in transformer_cleaned.rs which is re-exported as `transformer`.
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

                        /// Compute simple ALiBi slopes; geometric decay-style.
                        pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                            let mut slopes = Vec::with_capacity(n_heads);
                            for i in 0..n_heads {
                                let x = (i as f32) / (n_heads as f32 + 0.0);
                                slopes.push(2f32.powf(-x));
                            }
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
                            pub fn new(d_model: usize, num_heads: usize) -> Self {
                                Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
                            }

                            pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
                                assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
                                assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
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
                            pub fn set_attention_variant(&mut self, var: AttentionVariant) { self.attention_variant = var; }

                            /// Core forward impl that computes attention output.
                            pub fn forward_impl(&self, x: &Tensor) -> Tensor {
                                let q = self.linear_q.forward(x);
                                let k = self.linear_k.forward(x);
                                let v = self.linear_v.forward(x);

                                let shape = q.lock().storage.shape();
                                if shape.len() != 3 { return x.clone(); }
                                let b = shape[0];
                                let seq = shape[1];
                                let head_dim = self.d_model / self.num_heads;

                                // Reshape into [b*heads, seq, head_dim]
                                let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();

                                let out = match self.attention_variant {
                                    AttentionVariant::Baseline => {
                                        let k2t = k2.permute(vec![0, 2, 1]);
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
                                            if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
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
                                let out3 = out2.permute(vec![0, 2, 1, 3]);
                                let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
                                out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix)));
                                out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix)));
                                out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix)));
                                out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix)));
                                out
                            }
                            pub fn load_state_dict_impl(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                                self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?;
                                self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?;
                                self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?;
                                self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?;
                                Ok(())
                            }
                        }

                        impl crate::nn::Module for MultiHeadAttention {
                            fn forward(&self, input: &Tensor) -> Tensor { self.forward_impl(input) }
                            fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
                            fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
                            fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
                            fn as_any(&self) -> &dyn std::any::Any { self }
                            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                        }

                        pub struct TransformerBlock {
                            pub mha: MultiHeadAttention,
                            pub linear1: Linear,
                            pub linear2: Linear,
                        }
                        impl TransformerBlock {
                            pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                            pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { TransformerBlock { mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                            pub fn forward_block_impl(&self, x: &Tensor) -> Tensor {
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
                            impl crate::nn::Module for TransformerBlock {
                                fn forward(&self, input: &Tensor) -> Tensor { self.forward_block_impl(input) }
                                fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
                                fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
                                fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
                                fn as_any(&self) -> &dyn std::any::Any { self }
                                fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                            }
                            // Canonical MultiHeadAttention + TransformerBlock
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
                                for i in 0..n_heads {
                                    let x = (i as f32) / (n_heads as f32 + 0.0);
                                    slopes.push(2f32.powf(-x));
                                }
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
                                pub fn new(d_model: usize, num_heads: usize) -> Self {
                                    Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
                                }

                                pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
                                    assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
                                    assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
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
                                pub fn set_attention_variant(&mut self, var: AttentionVariant) { self.attention_variant = var; }

                                pub fn forward_impl(&self, x: &Tensor) -> Tensor {
                                    let q = self.linear_q.forward(x);
                                    let k = self.linear_k.forward(x);
                                    let v = self.linear_v.forward(x);
                                    let shape = q.lock().storage.shape();
                                    if shape.len() != 3 { return x.clone(); }
                                    let b = shape[0];
                                    let seq = shape[1];
                                    let head_dim = self.d_model / self.num_heads;
                                    let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                    let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                    let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();

                                    let out = match self.attention_variant {
                                        AttentionVariant::Baseline => {
                                            let k2t = k2.permute(vec![0, 2, 1]);
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
                                                if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
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
                                    let out3 = out2.permute(vec![0, 2, 1, 3]);
                                    let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
                                    out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix)));
                                    out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix)));
                                    out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix)));
                                    out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix)));
                                    out
                                }
                                pub fn load_state_dict_impl(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                                    self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?;
                                    self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?;
                                    self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?;
                                    self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?;
                                    Ok(())
                                }
                            }

                            impl crate::nn::Module for MultiHeadAttention {
                                fn forward(&self, input: &Tensor) -> Tensor { self.forward_impl(input) }
                                fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
                                fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
                                fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
                                fn as_any(&self) -> &dyn std::any::Any { self }
                                fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                            }

                            pub struct TransformerBlock {
                                pub mha: MultiHeadAttention,
                                pub linear1: Linear,
                                pub linear2: Linear,
                            }
                            impl TransformerBlock {
                                pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { TransformerBlock { mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                pub fn forward_block_impl(&self, x: &Tensor) -> Tensor {
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
                                impl crate::nn::Module for TransformerBlock {
                                    fn forward(&self, input: &Tensor) -> Tensor { self.forward_block_impl(input) }
                                    fn parameters(&self) -> Vec<Tensor> { self.parameters_impl() }
                                    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters_impl(prefix) }
                                    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict_impl(state, prefix) }
                                    fn as_any(&self) -> &dyn std::any::Any { self }
                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                }
                                // Canonical transformer module implementation
                                use crate::nn::Linear;
                                use crate::nn::Module;
                                use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                use crate::tensor::Tensor;
                                use ndarray::{Array, IxDyn};
                                use std::collections::HashMap;
                                use std::sync::Arc;

                                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                pub enum AttentionVariant { Baseline, FlashRef, Chunked { chunk_size: usize } }

                                pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                    let mut slopes = Vec::with_capacity(n_heads);
                                    for i in 0..n_heads {
                                        let x = (i as f32) / (n_heads as f32 + 0.0);
                                        slopes.push(2f32.powf(-x));
                                    }
                                    slopes
                                }

                                pub struct MultiHeadAttention {
                                    pub linear_q: Linear,
                                    pub linear_k: Linear,
                                    pub linear_v: Linear,
                                    pub linear_o: Linear,
                                    pub num_heads: usize,
                                    pub d_model: usize,
                                    pub use_alibi: bool,
                                    pub alibi_slopes: Option<Vec<f32>>,
                                    pub relative_bias: Option<Tensor>,
                                    pub attention_variant: AttentionVariant,
                                }

                                impl MultiHeadAttention {
                                    pub fn new(d_model: usize, num_heads: usize) -> Self { MultiHeadAttention { linear_q: Linear::new(d_model, d_model, true), linear_k: Linear::new(d_model, d_model, true), linear_v: Linear::new(d_model, d_model, true), linear_o: Linear::new(d_model, d_model, true), num_heads, d_model, use_alibi: false, alibi_slopes: None, relative_bias: None, attention_variant: AttentionVariant::Baseline } }
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
                                        let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                        let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                        let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                        let out = match self.attention_variant {
                                            AttentionVariant::Baseline => {
                                                let k2t = k2.permute(vec![0, 2, 1]);
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
                                                    if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
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
                                        let out3 = out2.permute(vec![0, 2, 1, 3]);
                                        let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
                                impl crate::nn::Module for MultiHeadAttention {
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
                                    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                    pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, _num_heads: usize, _kv_heads: usize, _use_rope: bool) -> Self { TransformerBlock::new(d_model, d_ff, _num_heads) }
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
                                    impl crate::nn::Module for TransformerBlock {
                                        fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                                        fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                        fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                        fn as_any(&self) -> &dyn std::any::Any { self }
                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                    } // Minimal Transformer implementation (single clean file).
                                    use crate::nn::Linear;
                                    use crate::nn::Module;
                                    use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                    use crate::tensor::Tensor;
                                    use ndarray::{Array, IxDyn};
                                    use std::collections::HashMap;
                                    use std::sync::Arc;

                                    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                    pub enum AttentionVariant { Baseline, FlashRef, Chunked { chunk_size: usize } }

                                    pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                        let mut slopes = Vec::with_capacity(n_heads);
                                        for i in 0..n_heads { slopes.push(2f32.powf(-(i as f32) / (n_heads as f32 + 1e-6))); }
                                        slopes
                                    }

                                    pub struct MultiHeadAttention {
                                        pub linear_q: Linear,
                                        pub linear_k: Linear,
                                        pub linear_v: Linear,
                                        pub linear_o: Linear,
                                        pub num_heads: usize,
                                        pub d_model: usize,
                                        pub use_alibi: bool,
                                        pub alibi_slopes: Option<Vec<f32>>,
                                        pub relative_bias: Option<Tensor>,
                                        pub attention_variant: AttentionVariant,
                                    }

                                    impl MultiHeadAttention {
                                        pub fn new(d_model: usize, num_heads: usize) -> Self {
                                            MultiHeadAttention {
                                                linear_q: Linear::new(d_model, d_model, true),
                                                linear_k: Linear::new(d_model, d_model, true),
                                                linear_v: Linear::new(d_model, d_model, true),
                                                linear_o: Linear::new(d_model, d_model, true),
                                                num_heads,
                                                d_model,
                                                use_alibi: false,
                                                alibi_slopes: None,
                                                relative_bias: None,
                                                attention_variant: AttentionVariant::Baseline,
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
                                            let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                            let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                            let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                            let out = match self.attention_variant {
                                                AttentionVariant::Baseline => {
                                                    let k2t = k2.permute(vec![0, 2, 1]);
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
                                                        if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
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
                                            let out3 = out2.permute(vec![0, 2, 1, 3]);
                                            let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
                                    impl crate::nn::Module for MultiHeadAttention {
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
                                        pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                        pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, _kv_heads: usize, _use_rope: bool) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
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
                                        impl crate::nn::Module for TransformerBlock {
                                            fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                                            fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                            fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                            fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                            fn as_any(&self) -> &dyn std::any::Any { self }
                                            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                        }
                                        // Simple clean transformer module
                                        use crate::nn::Linear;
                                        use crate::nn::Module;
                                        use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                        use crate::tensor::Tensor;
                                        use ndarray::{Array, IxDyn};
                                        use std::collections::HashMap;
                                        use std::sync::Arc;

                                        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                        pub enum AttentionVariant { Baseline, FlashRef, Chunked { chunk_size: usize } }

                                        pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                            let mut slopes = Vec::with_capacity(n_heads);
                                            for i in 0..n_heads {
                                                slopes.push(2f32.powf(-(i as f32) / (n_heads as f32 + 1e-6)));
                                            }
                                            slopes
                                        }

                                        pub struct MultiHeadAttention {
                                            pub linear_q: Linear,
                                            pub linear_k: Linear,
                                            pub linear_v: Linear,
                                            pub linear_o: Linear,
                                            pub num_heads: usize,
                                            pub d_model: usize,
                                            pub use_alibi: bool,
                                            pub alibi_slopes: Option<Vec<f32>>,
                                            pub relative_bias: Option<Tensor>,
                                            pub attention_variant: AttentionVariant,
                                        }

                                        impl MultiHeadAttention {
                                            pub fn new(d_model: usize, num_heads: usize) -> Self {
                                                MultiHeadAttention::new(d_model, num_heads)
                                            }
                                            // For simplicity the public methods are same as above
                                        }
                                        impl crate::nn::Module for MultiHeadAttention {
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
                                            pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                            pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
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
                                            impl crate::nn::Module for TransformerBlock {
                                                fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                                                fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                                fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                                fn as_any(&self) -> &dyn std::any::Any { self }
                                                fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                            }
                                            // Transformer module — single clean definition of MultiHeadAttention and TransformerBlock.
                                            use crate::nn::Linear;
                                            use crate::nn::Module;
                                            use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                            use crate::tensor::Tensor;
                                            use ndarray::{Array, IxDyn};
                                            use std::collections::HashMap;
                                            use std::sync::Arc;

                                            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                            pub enum AttentionVariant {
                                                Baseline,
                                                FlashRef,
                                                Chunked { chunk_size: usize },
                                            }

                                            /// Compute ALiBi slopes: geometric decay used to bias attention toward recent tokens.
                                            pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                                let mut slopes = Vec::with_capacity(n_heads);
                                                for i in 0..n_heads {
                                                    let x = (i as f32) / (n_heads as f32 + 0.0);
                                                    slopes.push(2f32.powf(-x));
                                                }
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
                                                        log::error!("MultiHeadAttention::forward expected input with 3 dims, got {:?}", shape);
                                                        return x.clone();
                                                    }
                                                    let b = shape[0];
                                                    let seq = shape[1];
                                                    let head_dim = self.d_model / self.num_heads;

                                                    let q2 = q
                                                        .reshape(vec![b, seq, self.num_heads, head_dim])
                                                        .unwrap()
                                                        .permute(vec![0, 2, 1, 3])
                                                        .reshape(vec![b * self.num_heads, seq, head_dim])
                                                        .unwrap();
                                                    let k2 = k
                                                        .reshape(vec![b, seq, self.num_heads, head_dim])
                                                        .unwrap()
                                                        .permute(vec![0, 2, 1, 3])
                                                        .reshape(vec![b * self.num_heads, seq, head_dim])
                                                        .unwrap();
                                                    let v2 = v
                                                        .reshape(vec![b, seq, self.num_heads, head_dim])
                                                        .unwrap()
                                                        .permute(vec![0, 2, 1, 3])
                                                        .reshape(vec![b * self.num_heads, seq, head_dim])
                                                        .unwrap();

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
                                                    let out3 = out2.permute(vec![0, 2, 1, 3]);
                                                    let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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

                                            impl crate::nn::Module for MultiHeadAttention {
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
                                                pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                                pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { TransformerBlock { mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
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
                                                impl crate::nn::Module for TransformerBlock {
                                                    fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                                                    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                                    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                                    fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }
                                                // Transformer module — single clean definition of MultiHeadAttention and TransformerBlock
                                                use crate::nn::Linear;
                                                use crate::nn::Module;
                                                use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                                use crate::tensor::Tensor;
                                                use ndarray::{Array, IxDyn};
                                                use std::collections::HashMap;
                                                use std::sync::Arc;

                                                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                                pub enum AttentionVariant { Baseline, FlashRef, Chunked { chunk_size: usize } }

                                                /// ALiBi slope generator: geometric decay. This matches expected behavior in ALiBi but
                                                /// is simplified compared to some published reference functions.
                                                pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                                    let mut slopes = Vec::with_capacity(n_heads);
                                                    for i in 0..n_heads {
                                                        let x = (i as f32) / (n_heads as f32);
                                                        slopes.push(2f32.powf(-x));
                                                    }
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
                                                        let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let out = match self.attention_variant {
                                                            AttentionVariant::Baseline => {
                                                                let k2t = k2.permute(vec![0, 2, 1]);
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
                                                                    if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
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
                                                        let out3 = out2.permute(vec![0, 2, 1, 3]);
                                                        let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
                                                impl crate::nn::Module for MultiHeadAttention {
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
                                                    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                                    pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { TransformerBlock { mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
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
                                                        out.extend(self.linear1.named_parameters(&format!("{}.linear1", prefix)));
                                                        out.extend(self.linear2.named_parameters(&format!("{}.linear2", prefix)));
                                                        out
                                                    }
                                                    pub fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                                                        self.mha.load_state_dict(state, &format!("{}.mha", prefix))?;
                                                        self.linear1.load_state_dict(state, &format!("{}.linear1", prefix))?;
                                                        self.linear2.load_state_dict(state, &format!("{}.linear2", prefix))?;
                                                        Ok(())
                                                    }
                                                    pub fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }
                                                impl crate::nn::Module for TransformerBlock {
                                                    fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                                                    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                                    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                                    fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }
                                                // Transformer module (single definition). Provides MultiHeadAttention and TransformerBlock.
                                                use crate::nn::Linear;
                                                use crate::nn::Module;
                                                use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                                use crate::tensor::Tensor;
                                                use ndarray::{Array, IxDyn};
                                                use std::collections::HashMap;
                                                use std::sync::Arc;

                                                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                                pub enum AttentionVariant { Baseline, FlashRef, Chunked { chunk_size: usize } }

                                                /// Compute ALiBi slopes: produce a monotonic decreasing sequence of positive floats
                                                /// This is a stable, commonly-used slope generator (not necessarily the exact HuggingFace impl,
                                                /// but compatible behavior-wise). If you want exact HF slopes, replace with the HF routine.
                                                pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                                    let mut slopes = Vec::with_capacity(n_heads);
                                                    for i in 0..n_heads {
                                                        // geometric progression in (0, 1]  — slope declines with head index
                                                        let x = (i as f32) / (n_heads as f32 + 0.0f32);
                                                        slopes.push(2f32.powf(-x));
                                                    }
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
                                                        let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let out = match self.attention_variant {
                                                            AttentionVariant::Baseline => {
                                                                let k2t = k2.permute(vec![0, 2, 1]);
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
                                                                    if shape == &[1, seq, seq] || shape == &[self.num_heads, seq, seq] { scaled_logits = scaled_logits.add(rb); }
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
                                                        let out3 = out2.permute(vec![0, 2, 1, 3]);
                                                        let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
                                                impl crate::nn::Module for MultiHeadAttention {
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
                                                    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                                    pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { TransformerBlock { mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
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
                                                        out.extend(self.linear1.named_parameters(&format!("{}.linear1", prefix)));
                                                        out.extend(self.linear2.named_parameters(&format!("{}.linear2", prefix)));
                                                        out
                                                    }
                                                    pub fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                                                        self.mha.load_state_dict(state, &format!("{}.mha", prefix))?;
                                                        self.linear1.load_state_dict(state, &format!("{}.linear1", prefix))?;
                                                        self.linear2.load_state_dict(state, &format!("{}.linear2", prefix))?;
                                                        Ok(())
                                                    }
                                                    pub fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }
                                                impl crate::nn::Module for TransformerBlock {
                                                    fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                                                    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                                    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                                    fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }
                                                // Minimal, single-definition transformer module.
                                                use crate::nn::Linear;
                                                use crate::nn::Module;
                                                use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                                use crate::tensor::Tensor;
                                                use ndarray::{Array, IxDyn};
                                                use std::sync::Arc;
                                                use std::collections::HashMap;

                                                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                                pub enum AttentionVariant {
                                                    Baseline,
                                                    FlashRef,
                                                    Chunked { chunk_size: usize },
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
                                                    pub attention_variant: AttentionVariant,
                                                }

                                                impl MultiHeadAttention {
                                                    pub fn new(d_model: usize, num_heads: usize) -> Self {
                                                        Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
                                                    }

                                                    pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
                                                        MultiHeadAttention { linear_q: Linear::new(d_model, d_model, true), linear_k: Linear::new(d_model, d_model, true), linear_v: Linear::new(d_model, d_model, true), linear_o: Linear::new(d_model, d_model, true), num_heads, d_model, kv_heads, use_rope, use_alibi: false, attention_variant: AttentionVariant::Baseline }
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
                                                        let q2 = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let k2 = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let v2 = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap().permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let out = match self.attention_variant {
                                                            AttentionVariant::Baseline => {
                                                                let k2t = k2.permute(vec![0, 2, 1]);
                                                                let qk = q2.batched_matmul(&k2t);
                                                                let scale = 1.0f32 / (head_dim as f32).sqrt();
                                                                let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
                                                                let scaled = qk.mul(&scalar_tensor);
                                                                let attn = scaled.softmax(2);
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
                                                        let out3 = out2.permute(vec![0, 2, 1, 3]);
                                                        let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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
                                                    pub fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }
                                                impl crate::nn::Module for MultiHeadAttention {
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
                                                    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
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
                                                        out.extend(self.linear1.named_parameters(&format!("{}.linear1", prefix)));
                                                        out.extend(self.linear2.named_parameters(&format!("{}.linear2", prefix)));
                                                        out
                                                    }
                                                    pub fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                                                        self.mha.load_state_dict(state, &format!("{}.mha", prefix))?;
                                                        self.linear1.load_state_dict(state, &format!("{}.linear1", prefix))?;
                                                        self.linear2.load_state_dict(state, &format!("{}.linear2", prefix))?;
                                                        Ok(())
                                                    }
                                                    pub fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }
                                                impl crate::nn::Module for TransformerBlock {
                                                    fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                                                    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                                    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                                    fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }
                                                use crate::nn::Linear;
                                                use crate::nn::Module;
                                                use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                                use crate::tensor::Tensor;
                                                use ndarray::{Array, IxDyn};
                                                use std::sync::Arc;
                                                use std::collections::HashMap;

                                                /// Helper: Compute simple ALiBi slopes
                                                pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                                    let mut slopes = Vec::with_capacity(n_heads);
                                                    for i in 0..n_heads {
                                                        let x = (i as f32) / (n_heads as f32);
                                                        slopes.push(2f32.powf(-x));
                                                    }
                                                    slopes
                                                }

                                                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                                pub enum AttentionVariant {
                                                    Baseline,
                                                    FlashRef,
                                                    Chunked { chunk_size: usize },
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
                                                    pub fn new(d_model: usize, num_heads: usize) -> Self {
                                                        Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
                                                    }

                                                    pub fn new_with_kv_and_rope(
                                                        d_model: usize,
                                                        num_heads: usize,
                                                        kv_heads: usize,
                                                        use_rope: bool,
                                                    ) -> Self {
                                                        assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
                                                        assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
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

                                                    /// Simplified forward that supports Baseline, FlashRef, and Chunked variants.
                                                    pub fn forward(&self, x: &Tensor) -> Tensor {
                                                        // Build Q, K, V
                                                        let q = self.linear_q.forward(x);
                                                        let k = self.linear_k.forward(x);
                                                        let v = self.linear_v.forward(x);

                                                        let shape = q.lock().storage.shape();
                                                        if shape.len() != 3 {
                                                            log::error!("MultiHeadAttention::forward requires [b, seq, d_model]");
                                                            return x.clone();
                                                        }
                                                        let b = shape[0];
                                                        let seq = shape[1];
                                                        let head_dim = self.d_model / self.num_heads;

                                                        // Reshape and permute into [b*heads, seq, head_dim]
                                                        let q_heads = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap();
                                                        let k_heads = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap();
                                                        let v_heads = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap();
                                                        let q2 = q_heads.permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let k2 = k_heads.permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let v2 = v_heads.permute(vec![0, 2, 1, 3]).reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();

                                                        let out = match self.attention_variant {
                                                            AttentionVariant::Baseline => {
                                                                // baseline: QK^T scaled + softmax then matmul V
                                                                let k2t = k2.permute(vec![0, 2, 1]);
                                                                let qk = q2.batched_matmul(&k2t);
                                                                let scale = 1.0f32 / (head_dim as f32).sqrt();
                                                                let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
                                                                let scaled = qk.mul(&scalar_tensor);
                                                                let mut scaled_logits = scaled.clone();
                                                                if self.use_alibi {
                                                                    if let Some(slopes) = &self.alibi_slopes {
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
                                                                        let bias_t = Tensor::new(bias_arr, false);
                                                                        scaled_logits = scaled_logits.add(&bias_t);
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

                                                        // Merge heads back
                                                        let out2 = out.reshape(vec![b, self.num_heads, seq, head_dim]).unwrap();
                                                        let out3 = out2.permute(vec![0, 2, 1, 3]);
                                                        let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
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

                                                    pub fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }

                                                impl crate::nn::Module for MultiHeadAttention {
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
                                                        TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) }
                                                    }
                                                    pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
                                                        TransformerBlock { mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) }
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
                                                        out.extend(self.linear1.named_parameters(&format!("{}.linear1", prefix)));
                                                        out.extend(self.linear2.named_parameters(&format!("{}.linear2", prefix)));
                                                        out
                                                    }
                                                    pub fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                                                        self.mha.load_state_dict(state, &format!("{}.mha", prefix))?;
                                                        self.linear1.load_state_dict(state, &format!("{}.linear1", prefix))?;
                                                        self.linear2.load_state_dict(state, &format!("{}.linear2", prefix))?;
                                                        Ok(())
                                                    }
                                                    pub fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }

                                                impl crate::nn::Module for TransformerBlock {
                                                    fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                                                    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                                    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                                    fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }

                                                use crate::nn::Linear;
                                                use crate::nn::Module;
                                                use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                                use crate::tensor::Tensor;
                                                use ndarray::{Array, IxDyn};
                                                use std::sync::Arc;
                                                use std::collections::HashMap;

                                                /// ALiBi slope computation helper
                                                pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                                    let mut slopes = Vec::with_capacity(n_heads);
                                                    for i in 0..n_heads {
                                                        let x = (i as f32) / (n_heads as f32);
                                                        slopes.push(2f32.powf(-x));
                                                    }
                                                    slopes
                                                }

                                                /// Attention variant selector
                                                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                                pub enum AttentionVariant {
                                                    Baseline,
                                                    FlashRef,
                                                    Chunked { chunk_size: usize },
                                                }

                                                /// MultiHeadAttention implemented using existing Linear layers and matmul.
                                                /// This is a reference implementation optimized for clarity and correct shapes.
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
                                                    pub fn new(d_model: usize, num_heads: usize) -> Self {
                                                        Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
                                                    }

                                                    pub fn new_with_kv_and_rope(
                                                        d_model: usize,
                                                        num_heads: usize,
                                                        kv_heads: usize,
                                                        use_rope: bool,
                                                    ) -> Self {
                                                        assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
                                                        assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
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

                                                    /// Forward attention. Input: [batch, seq, d_model]. Output: [batch, seq, d_model]
                                                    pub fn forward(&self, x: &Tensor) -> Tensor {
                                                        // Create Q, K, V
                                                        let q = self.linear_q.forward(x);
                                                        let k = self.linear_k.forward(x);
                                                        let v = self.linear_v.forward(x);

                                                        // reshape to [batch, seq, d_model]
                                                        let shape = q.lock().storage.shape();
                                                        if shape.len() != 3 {
                                                            log::error!(
                "MultiHeadAttention::forward expected input with 3 dims [batch, seq, d_model], got {:?}",
                shape
            );
                                                            return x.clone();
                                                        }
                                                        let b = shape[0];
                                                        let seq = shape[1];
                                                        let head_dim = self.d_model / self.num_heads;

                                                        // reshape to per-head
                                                        let q_heads = match q.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                            Ok(t) => t,
                                                            Err(e) => {
                                                                log::error!("reshape error q: {}", e);
                                                                return x.clone();
                                                            }
                                                        };
                                                        let mut k_heads = match k.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]) {
                                                            Ok(t) => t,
                                                            Err(e) => {
                                                                log::error!("reshape error k: {}", e);
                                                                return x.clone();
                                                            }
                                                        };
                                                        let mut v_heads = match v.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]) {
                                                            Ok(t) => t,
                                                            Err(e) => {
                                                                log::error!("reshape error v: {}", e);
                                                                return x.clone();
                                                            }
                                                        };

                                                        // expand kv if necessary
                                                        if self.kv_heads != self.num_heads {
                                                            let group_size = self.num_heads / self.kv_heads;
                                                            let kv_head_dim = self.d_model / self.kv_heads;
                                                            let new_head_dim = kv_head_dim / group_size; // should equal head_dim
                                                            k_heads = match k_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error k split: {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            v_heads = match v_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error v split: {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            k_heads = match k_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error k merge: {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            v_heads = match v_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error v merge: {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                        } else {
                                                            k_heads = match k_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error k reshape: {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            v_heads = match v_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error v reshape: {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                        }

                                                        let mut q_proc = q_heads;
                                                        let mut k_proc = k_heads;
                                                        let v_proc = v_heads;
                                                        if self.use_rope {
                                                            q_proc = q_proc.rope(self.num_heads);
                                                            k_proc = k_proc.rope(self.num_heads);
                                                        }

                                                        // permute to [b, num_heads, seq, head_dim]
                                                        let q_perm = q_proc.permute(vec![0, 2, 1, 3]);
                                                        let k_perm = k_proc.permute(vec![0, 2, 1, 3]);
                                                        let v_perm = v_proc.permute(vec![0, 2, 1, 3]);

                                                        let q2 = match q_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                            Ok(t) => t,
                                                            Err(e) => {
                                                                log::error!("reshape q2: {}", e);
                                                                return x.clone();
                                                            }
                                                        };
                                                        let k2 = match k_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                            Ok(t) => t,
                                                            Err(e) => {
                                                                log::error!("reshape k2: {}", e);
                                                                return x.clone();
                                                            }
                                                        };
                                                        let v2 = match v_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                            Ok(t) => t,
                                                            Err(e) => {
                                                                log::error!("reshape v2: {}", e);
                                                                return x.clone();
                                                            }
                                                        };

                                                        let scale = 1.0f32 / (head_dim as f32).sqrt();

                                                        // dispatch based on attention variant
                                                        let out = match self.attention_variant {
                                                            AttentionVariant::Baseline => {
                                                                let k2t = k2.permute(vec![0, 2, 1]);
                                                                let qk = q2.batched_matmul(&k2t);
                                                                let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
                                                                let mut scaled = qk.mul(&scalar_tensor);
                                                                // optionally add ALiBi
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
                                                                    let bias_t = Tensor::new(bias_arr, false);
                                                                    scaled = scaled.add(&bias_t);
                                                                }
                                                                // optionally add relative bias
                                                                if let Some(rb) = &self.relative_bias {
                                                                    let shape = rb.lock().storage.shape();
                                                                    if shape.len() == 2 {
                                                                        let max_range = (shape[1] - 1) / 2;
                                                                        let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                                                                        let rb_arr = rb.lock().storage.to_f32_array();
                                                                        let rb_view = rb_arr.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                                                                        for batch in 0..b {
                                                                            for h in 0..self.num_heads {
                                                                                for i in 0..seq {
                                                                                    for j in 0..seq {
                                                                                        let rel = (j as isize - i as isize).max(-(max_range as isize)).min(max_range as isize) as isize;
                                                                                        let idx = (rel + max_range as isize) as usize;
                                                                                        bias_arr[[batch * self.num_heads + h, i, j]] = rb_view[[h, idx]];
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                        let bias_t = Tensor::new(bias_arr, false);
                                                                        scaled = scaled.add(&bias_t);
                                                                    }
                                                                }
                                                                let attn = scaled.softmax(2);
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

                                                        // reshape out [b*num_heads, seq, head_dim] -> [b, num_heads, seq, head_dim]
                                                        let out2 = match out.reshape(vec![b, self.num_heads, seq, head_dim]) {
                                                            Ok(t) => t,
                                                            Err(e) => {
                                                                log::error!("reshape out2: {}", e);
                                                                return x.clone();
                                                            }
                                                        };
                                                        // permute back and merge heads
                                                        let out3 = out2.permute(vec![0, 2, 1, 3]);
                                                        let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
                                                            Ok(t) => t,
                                                            Err(e) => {
                                                                log::error!("reshape out4: {}", e);
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

                                                    pub fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }

                                                impl crate::nn::Module for MultiHeadAttention {
                                                    fn forward(&self, input: &Tensor) -> Tensor { self.forward(input) }
                                                    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                                    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                                    fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }

                                                /// TransformerBlock: attention + residual + layernorm + feedforward + residual
                                                pub struct TransformerBlock {
                                                    pub mha: MultiHeadAttention,
                                                    pub linear1: Linear,
                                                    pub linear2: Linear,
                                                }

                                                impl TransformerBlock {
                                                    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self {
                                                        TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) }
                                                    }
                                                    pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
                                                        TransformerBlock { mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) }
                                                    }
                                                    pub fn forward_block(&self, x: &Tensor) -> Tensor {
                                                        log::info!("TransformerBlock forward: input shape {:?}", x.lock().storage.shape());
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
                                                        out.extend(self.linear1.named_parameters(&format!("{}.linear1", prefix)));
                                                        out.extend(self.linear2.named_parameters(&format!("{}.linear2", prefix)));
                                                        out
                                                    }
                                                    pub fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                                                        self.mha.load_state_dict(state, &format!("{}.mha", prefix))?;
                                                        self.linear1.load_state_dict(state, &format!("{}.linear1", prefix))?;
                                                        self.linear2.load_state_dict(state, &format!("{}.linear2", prefix))?;
                                                        Ok(())
                                                    }
                                                    pub fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }

                                                impl crate::nn::Module for TransformerBlock {
                                                    fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                                                    fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                                    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                                    fn as_any(&self) -> &dyn std::any::Any { self }
                                                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                }

                                                use crate::nn::Linear;
                                                use crate::nn::Module;
                                                use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                                use crate::tensor::Tensor;
                                                use ndarray::{Array, IxDyn};
                                                use std::sync::Arc;

                                                pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                                    let mut slopes = Vec::with_capacity(n_heads);
                                                    for i in 0..n_heads {
                                                        let x = (i as f32) / (n_heads as f32);
                                                        slopes.push(2f32.powf(-x));
                                                    }
                                                    slopes
                                                }

                                                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                                pub enum AttentionVariant {
                                                    Baseline,
                                                    FlashRef,
                                                    Chunked { chunk_size: usize },
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
                                                    pub fn new(d_model: usize, num_heads: usize) -> Self {
                                                        Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
                                                    }

                                                    pub fn new_with_kv_and_rope(d_model: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self {
                                                        assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
                                                        assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
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
                                                    pub fn set_attention_variant(&mut self, var: AttentionVariant) { self.attention_variant = var; }

                                                    pub fn forward(&self, x: &Tensor) -> Tensor {
                                                        let q = self.linear_q.forward(x);
                                                        let k = self.linear_k.forward(x);
                                                        let v = self.linear_v.forward(x);
                                                        let shape = q.lock().storage.shape();
                                                        if shape.len() != 3 {
                                                            log::error!("MultiHeadAttention::forward expected input with 3 dims [batch, seq, d_model], got {:?}", shape);
                                                            return x.clone();
                                                        }
                                                        let b = shape[0];
                                                        let seq = shape[1];
                                                        let head_dim = self.d_model / self.num_heads;
                                                        let q_heads = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap();
                                                        let mut k_heads = k.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]).unwrap();
                                                        let mut v_heads = v.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]).unwrap();
                                                        if self.kv_heads != self.num_heads {
                                                            let group_size = self.num_heads / self.kv_heads;
                                                            let head_dim_kv = self.d_model / self.kv_heads;
                                                            let new_head_dim = head_dim_kv / group_size;
                                                            k_heads = k_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]).unwrap();
                                                            v_heads = v_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]).unwrap();
                                                            k_heads = k_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]).unwrap();
                                                            v_heads = v_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]).unwrap();
                                                        } else {
                                                            k_heads = k_heads.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap();
                                                            v_heads = v_heads.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap();
                                                        }
                                                        let mut q_proc = q_heads;
                                                        let mut k_proc = k_heads;
                                                        let v_proc = v_heads;
                                                        if self.use_rope {
                                                            q_proc = q_proc.rope(self.num_heads);
                                                            k_proc = k_proc.rope(self.num_heads);
                                                        }
                                                        let q_perm = q_proc.permute(vec![0, 2, 1, 3]);
                                                        let k_perm = k_proc.permute(vec![0, 2, 1, 3]);
                                                        let v_perm = v_proc.permute(vec![0, 2, 1, 3]);
                                                        let q2 = q_perm.reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let k2 = k_perm.reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let v2 = v_perm.reshape(vec![b * self.num_heads, seq, head_dim]).unwrap();
                                                        let scale = 1.0f32 / (head_dim as f32).sqrt();
                                                        let out = match self.attention_variant {
                                                            AttentionVariant::Baseline => {
                                                                let k2t = k2.permute(vec![0, 2, 1]);
                                                                let qk = q2.batched_matmul(&k2t);
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
                                                                    };
                                                                    let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                                                    scaled_logits = scaled_logits.add(&bias_t);
                                                                }
                                                                if let Some(rb) = &self.relative_bias {
                                                                    let shape = rb.lock().storage.shape();
                                                                    if shape.len() == 2 {
                                                                        let max_range = (shape[1] - 1) / 2;
                                                                        let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                                                                        let rb_arr = rb.lock().storage.to_f32_array();
                                                                        let rb_view = rb_arr.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                                                                        for batch in 0..b {
                                                                            for h in 0..self.num_heads {
                                                                                for i in 0..seq {
                                                                                    for j in 0..seq {
                                                                                        let rel = (j as isize - i as isize).max(-(max_range as isize)).min(max_range as isize) as isize;
                                                                                        let idx = (rel + max_range as isize) as usize;
                                                                                        bias_arr[[batch * self.num_heads + h, i, j]] = rb_view[[h, idx]];
                                                                                    }
                                                                                }
                                                                            }
                                                                        };
                                                                        let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                                                        scaled_logits = scaled_logits.add(&bias_t);
                                                                    }
                                                                    let attn = scaled_logits.softmax(2);
                                                                    attn.batched_matmul(&v2)
                                                                }, AttentionVariant::FlashRef => {
                                                                    let flash = FlashAttentionRef::new(head_dim);
                                                                    Tensor::apply(Arc::new(flash), &[q2.clone(), k2.clone(), v2.clone()])
                                                                }, AttentionVariant::Chunked { chunk_size } => {
                                                                    let op = ChunkedAttention::new(head_dim, chunk_size);
                                                                    Tensor::apply(Arc::new(op), &[q2.clone(), k2.clone(), v2.clone()])
                                                                }
                                                            };

                                                            let out2 = out.reshape(vec![b, self.num_heads, seq, head_dim]).unwrap();
                                                            let out3 = out2.permute(vec![0, 2, 1, 3]);
                                                            let out4 = out3.reshape(vec![b, seq, self.d_model]).unwrap();
                                                            self.linear_o.forward(&out4)
                                                        }

                                                        pub fn parameters(&self) -> Vec<Tensor> {
                                                            let mut p = self.linear_q.parameters();
                                                            p.extend(self.linear_k.parameters());
                                                            p.extend(self.linear_v.parameters());
                                                            p.extend(self.linear_o.parameters());
                                                            p
                                                        }

                                                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
                                                            let mut out = Vec::new();
                                                            out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix)));
                                                            out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix)));
                                                            out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix)));
                                                            out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix)));
                                                            out
                                                        }

                                                        fn load_state_dict(&mut self, state: &std::collections::HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                                                            self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?;
                                                            self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?;
                                                            self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?;
                                                            self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?;
                                                            Ok(())
                                                        }

                                                        fn as_any(&self) -> &dyn std::any::Any { self }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }

                                                    impl crate::nn::Module for MultiHeadAttention {
                                                        fn forward(&self, input: &Tensor) -> Tensor { self.forward(input) }
                                                        fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                                        fn load_state_dict(&mut self, state: &std::collections::HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                                        fn as_any(&self) -> &dyn std::any::Any { self }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }

                                                    pub struct TransformerBlock {
                                                        pub mha: MultiHeadAttention,
                                                        pub linear1: Linear,
                                                        pub linear2: Linear,
                                                    }

                                                    impl TransformerBlock {
                                                        pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                                        pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { TransformerBlock { mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                                        pub fn forward_block(&self, x: &Tensor) -> Tensor {
                                                            log::info!("TransformerBlock forward: input shape {:?}", x.lock().storage.shape());
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

                                                        impl crate::nn::Module for TransformerBlock {
                                                            fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                                                            fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                            fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
                                                                let mut out = Vec::new();
                                                                out.extend(self.mha.named_parameters(&format!("{}.mha", prefix)));
                                                                out.extend(self.linear1.named_parameters(&format!("{}.linear1", prefix)));
                                                                out.extend(self.linear2.named_parameters(&format!("{}.linear2", prefix)));
                                                                out
                                                            }
                                                            fn load_state_dict(&mut self, state: &std::collections::HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                                                                self.mha.load_state_dict(state, &format!("{}.mha", prefix))?;
                                                                self.linear1.load_state_dict(state, &format!("{}.linear1", prefix))?;
                                                                self.linear2.load_state_dict(state, &format!("{}.linear2", prefix))?;
                                                                Ok(())
                                                            }
                                                            fn as_any(&self) -> &dyn std::any::Any { self }
                                                            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                        }

                                                        pub fn parameters(&self) -> Vec<Tensor> {
                                                            let mut p = self.linear_q.parameters();
                                                            p.extend(self.linear_k.parameters());
                                                            p.extend(self.linear_v.parameters());
                                                            p.extend(self.linear_o.parameters());
                                                            p
                                                        }

                                                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
                                                            let mut out = Vec::new();
                                                            out.extend(self.linear_q.named_parameters(&format!("{}.linear_q", prefix)));
                                                            out.extend(self.linear_k.named_parameters(&format!("{}.linear_k", prefix)));
                                                            out.extend(self.linear_v.named_parameters(&format!("{}.linear_v", prefix)));
                                                            out.extend(self.linear_o.named_parameters(&format!("{}.linear_o", prefix)));
                                                            out
                                                        }

                                                        fn load_state_dict(&mut self, state: &std::collections::HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                                                            self.linear_q.load_state_dict(state, &format!("{}.linear_q", prefix))?;
                                                            self.linear_k.load_state_dict(state, &format!("{}.linear_k", prefix))?;
                                                            self.linear_v.load_state_dict(state, &format!("{}.linear_v", prefix))?;
                                                            self.linear_o.load_state_dict(state, &format!("{}.linear_o", prefix))?;
                                                            Ok(())
                                                        }

                                                        fn as_any(&self) -> &dyn std::any::Any { self }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }

                                                    impl crate::nn::Module for MultiHeadAttention {
                                                        fn forward(&self, input: &Tensor) -> Tensor { self.forward(input) }
                                                        fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> { self.named_parameters(prefix) }
                                                        fn load_state_dict(&mut self, state: &std::collections::HashMap<String, Tensor>, prefix: &str) -> Result<(), String> { self.load_state_dict(state, prefix) }
                                                        fn as_any(&self) -> &dyn std::any::Any { self }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }

                                                    pub struct TransformerBlock {
                                                        pub mha: MultiHeadAttention,
                                                        pub linear1: Linear,
                                                        pub linear2: Linear,
                                                    }

                                                    impl TransformerBlock {
                                                        pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self { TransformerBlock { mha: MultiHeadAttention::new(d_model, num_heads), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                                        pub fn new_with_kv_and_rope(d_model: usize, d_ff: usize, num_heads: usize, kv_heads: usize, use_rope: bool) -> Self { TransformerBlock { mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope), linear1: Linear::new(d_model, d_ff, true), linear2: Linear::new(d_ff, d_model, true) } }
                                                        pub fn forward_block(&self, x: &Tensor) -> Tensor {
                                                            log::info!("TransformerBlock forward: input shape {:?}", x.lock().storage.shape());
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
                                                    }

                                                    impl crate::nn::Module for TransformerBlock {
                                                        fn forward(&self, input: &Tensor) -> Tensor { self.forward_block(input) }
                                                        fn parameters(&self) -> Vec<Tensor> { self.parameters() }
                                                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
                                                            let mut out = Vec::new();
                                                            out.extend(self.mha.named_parameters(&format!("{}.mha", prefix)));
                                                            out.extend(self.linear1.named_parameters(&format!("{}.linear1", prefix)));
                                                            out.extend(self.linear2.named_parameters(&format!("{}.linear2", prefix)));
                                                            out
                                                        }
                                                        fn load_state_dict(&mut self, state: &std::collections::HashMap<String, Tensor>, prefix: &str) -> Result<(), String> {
                                                            self.mha.load_state_dict(state, &format!("{}.mha", prefix))?;
                                                            self.linear1.load_state_dict(state, &format!("{}.linear1", prefix))?;
                                                            self.linear2.load_state_dict(state, &format!("{}.linear2", prefix))?;
                                                            Ok(())
                                                        }
                                                        fn as_any(&self) -> &dyn std::any::Any { self }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }
                                                    use crate::nn::Linear;
                                                    use crate::nn::Module;
                                                    use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                                    use crate::tensor::Tensor;
                                                    use ndarray::{Array, IxDyn};
                                                    use std::sync::Arc;

                                                    // ALiBi slope computation helper
                                                    pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                                        // simple decreasing slopes: 2^(-i/num_heads)
                                                        let mut slopes = Vec::with_capacity(n_heads);
                                                        for i in 0..n_heads {
                                                            let x = (i as f32) / (n_heads as f32);
                                                            slopes.push(2f32.powf(-x));
                                                        }
                                                        slopes
                                                    }

                                                    /// Attention variant selector
                                                    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                                    pub enum AttentionVariant {
                                                        Baseline,
                                                        FlashRef,
                                                        Chunked { chunk_size: usize },
                                                    }

                                                    /// MultiHeadAttention implemented using existing Linear layers and matmul.
                                                    /// This is a reference implementation optimized for clarity and correct shapes.
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
                                                        pub fn new(d_model: usize, num_heads: usize) -> Self {
                                                            Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
                                                        }

                                                        pub fn new_with_kv_and_rope(
                                                            d_model: usize,
                                                            num_heads: usize,
                                                            kv_heads: usize,
                                                            use_rope: bool,
                                                        ) -> Self {
                                                            assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
                                                            assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
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

                                                        /// Forward attention. Input: [batch, seq, d_model]. Output: [batch, seq, d_model]
                                                        pub fn forward(&self, x: &Tensor) -> Tensor {
                                                            // Create Q, K, V
                                                            let q = self.linear_q.forward(x);
                                                            let k = self.linear_k.forward(x);
                                                            let v = self.linear_v.forward(x);

                                                            // reshape to [batch, seq, d_model]
                                                            let shape = q.lock().storage.shape();
                                                            if shape.len() != 3 {
                                                                log::error!(
                "MultiHeadAttention::forward expected input with 3 dims [batch, seq, d_model], got {:?}",
                shape
            );
                                                                return x.clone();
                                                            }
                                                            let b = shape[0];
                                                            let seq = shape[1];
                                                            let _d = self.d_model;

                                                            // compute head dims and reshape QKV to head layout
                                                            let head_dim = self.d_model / self.num_heads;
                                                            let q_heads = match q.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (q): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let mut k_heads = match k.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (k): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let mut v_heads = match v.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (v): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            // Expand kv if needed
                                                            if self.kv_heads != self.num_heads {
                                                                let group_size = self.num_heads / self.kv_heads;
                                                                let head_dim_kv = self.d_model / self.kv_heads;
                                                                let new_head_dim = head_dim_kv / group_size; // should equal head_dim
                                                                match k_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k split): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v split): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match k_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k merge): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v merge): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                            } else {
                                                                match k_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k reshape): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v reshape): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                            }

                                                            let mut q_proc = q_heads;
                                                            let mut k_proc = k_heads;
                                                            let v_proc = v_heads;
                                                            if self.use_rope {
                                                                q_proc = q_proc.rope(self.num_heads);
                                                                k_proc = k_proc.rope(self.num_heads);
                                                            }
                                                            // Permute to [b, num_heads, seq, head_dim]
                                                            let q_perm = q_proc.permute(vec![0, 2, 1, 3]);
                                                            let k_perm = k_proc.permute(vec![0, 2, 1, 3]);
                                                            let v_perm = v_proc.permute(vec![0, 2, 1, 3]);
                                                            let q2 = match q_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (q_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let k2 = match k_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (k_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let v2 = match v_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (v_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };

                                                            let scale = 1.0f32 / (head_dim as f32).sqrt();

                                                            // If we are using a specialized attention variant, dispatch to an op
                                                            let out = match self.attention_variant {
                                                                AttentionVariant::Baseline => {
                                                                    let k2t = k2.permute(vec![0, 2, 1]);
                                                                    let qk = q2.batched_matmul(&k2t);
                                                                    let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
                                                                    let scaled = qk.mul(&scalar_tensor);
                                                                    // Optionally add ALiBi or relative positional bias to scaled logits
                                                                    let mut scaled_logits = scaled.clone();
                                                                    if self.use_alibi {
                                                                        let slopes = if let Some(s) = &self.alibi_slopes {
                                                                            s.clone()
                                                                        } else {
                                                                            compute_alibi_slopes(self.num_heads)
                                                                        };
                                                                        let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                                                                        for batch in 0..b {
                                                                            for h in 0..self.num_heads {
                                                                                let slope = slopes[h];
                                                                                for i in 0..seq {
                                                                                    for j in 0..seq {
                                                                                        let dist = (j as isize - i as isize) as f32;
                                                                                        let val = -slope * dist;
                                                                                        bias_arr[[batch * self.num_heads + h, i, j]] = val;
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                        let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                                                        scaled_logits = scaled_logits.add(&bias_t);
                                                                    }
                                                                    if let Some(rb) = &self.relative_bias {
                                                                        let shape = rb.lock().storage.shape();
                                                                        if shape.len() == 2 {
                                                                            let max_range = (shape[1] - 1) / 2;
                                                                            let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                                                                            let rb_arr = rb.lock().storage.to_f32_array();
                                                                            let rb_view = rb_arr.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                                                                            for batch in 0..b {
                                                                                for h in 0..self.num_heads {
                                                                                    for i in 0..seq {
                                                                                        for j in 0..seq {
                                                                                            let rel = (j as isize - i as isize)
                                                                                                .max(-(max_range as isize))
                                                                                                .min(max_range as isize)
                                                                                                as isize;
                                                                                            let idx = (rel + max_range as isize) as usize;
                                                                                            bias_arr[[batch * self.num_heads + h, i, j]] = rb_view[[h, idx]];
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                            let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                                                            scaled_logits = scaled_logits.add(&bias_t);
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

                                                            // reshape out [b*num_heads, seq, head_dim] -> [b, num_heads, seq, head_dim]
                                                            let out2 = match out.reshape(vec![b, self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error during attention result reshape: {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            // permute back to [b, seq, num_heads, head_dim]
                                                            let out3 = out2.permute(vec![0, 2, 1, 3]);
                                                            // reshape to [b, seq, d_model]
                                                            let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error during attention final reshape: {}", e);
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

                                                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
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

                                                        fn load_state_dict(
                                                            &mut self,
                                                            state: &std::collections::HashMap<String, Tensor>,
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

                                                        fn as_any(&self) -> &dyn std::any::Any {
                                                            self
                                                        }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
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
                                                            state: &std::collections::HashMap<String, Tensor>,
                                                            prefix: &str,
                                                        ) -> Result<(), String> {
                                                            self.load_state_dict(state, prefix)
                                                        }
                                                        fn as_any(&self) -> &dyn std::any::Any {
                                                            self
                                                        }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }

                                                    /// TransformerBlock: attention + residual + layernorm + feedforward + residual
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
                                                            log::info!(
            "TransformerBlock forward: input shape {:?}",
            x.lock().storage.shape()
        );
                                                            let attn_out = self.mha.forward(x);
                                                            let x2 = x.add(&attn_out);
                                                            // LayerNorm not exposed as module; use Tensor::layer_norm with axis=-1
                                                            let dim = x.lock().storage.shape()[2];
                                                            // create gamma and beta as ones/zeros (affine) for now
                                                            let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
                                                            let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
                                                            let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
                                                            let ff = self.linear1.forward(&x2norm).relu();
                                                            let ff = self.linear2.forward(&ff);
                                                            log::debug!(
            "TransformerBlock forward complete: output shape {:?}",
            x2.lock().storage.shape()
        );
                                                            x2.add(&ff)
                                                        }

                                                        pub fn parameters(&self) -> Vec<Tensor> {
                                                            let mut p = self.mha.parameters();
                                                            p.extend(self.linear1.parameters());
                                                            p.extend(self.linear2.parameters());
                                                            p
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
                                                        fn load_state_dict(
                                                            &mut self,
                                                            state: &std::collections::HashMap<String, Tensor>,
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
                                                        fn as_any(&self) -> &dyn std::any::Any {
                                                            self
                                                        }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }
                                                    use crate::nn::Linear;
                                                    use crate::nn::Module;
                                                    use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                                    use crate::tensor::Tensor;
                                                    use ndarray::{Array, IxDyn};
                                                    use std::sync::Arc;

                                                    // ALiBi slope computation helper
                                                    pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                                        // simple decreasing slopes: 2^(-i/num_heads)
                                                        let mut slopes = Vec::with_capacity(n_heads);
                                                        for i in 0..n_heads {
                                                            let x = (i as f32) / (n_heads as f32);
                                                            slopes.push(2f32.powf(-x));
                                                        }
                                                        slopes
                                                    }

                                                    /// Attention variant selector
                                                    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                                    pub enum AttentionVariant {
                                                        Baseline,
                                                        FlashRef,
                                                        Chunked { chunk_size: usize },
                                                    }

                                                    /// MultiHeadAttention implemented using existing Linear layers and matmul.
                                                    /// This is a reference implementation optimized for clarity and correct shapes.
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
                                                        pub fn new(d_model: usize, num_heads: usize) -> Self {
                                                            Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
                                                        }

                                                        pub fn new_with_kv_and_rope(
                                                            d_model: usize,
                                                            num_heads: usize,
                                                            kv_heads: usize,
                                                            use_rope: bool,
                                                        ) -> Self {
                                                            assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
                                                            assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
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

                                                        /// Forward attention. Input: [batch, seq, d_model]. Output: [batch, seq, d_model]
                                                        pub fn forward(&self, x: &Tensor) -> Tensor {
                                                            // Create Q, K, V
                                                            let q = self.linear_q.forward(x);
                                                            let k = self.linear_k.forward(x);
                                                            let v = self.linear_v.forward(x);

                                                            // reshape to [batch, seq, d_model]
                                                            let shape = q.lock().storage.shape();
                                                            if shape.len() != 3 {
                                                                log::error!(
                "MultiHeadAttention::forward expected input with 3 dims [batch, seq, d_model], got {:?}",
                shape
            );
                                                                return x.clone();
                                                            }
                                                            let b = shape[0];
                                                            let seq = shape[1];
                                                            let _d = self.d_model;

                                                            // compute head dims and reshape QKV to head layout
                                                            let head_dim = self.d_model / self.num_heads;
                                                            let q_heads = match q.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (q): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let mut k_heads = match k.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (k): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let mut v_heads = match v.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (v): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            // Expand kv if needed
                                                            if self.kv_heads != self.num_heads {
                                                                let group_size = self.num_heads / self.kv_heads;
                                                                let head_dim_kv = self.d_model / self.kv_heads;
                                                                let new_head_dim = head_dim_kv / group_size; // should equal head_dim
                                                                match k_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k split): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v split): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match k_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k merge): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v merge): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                            } else {
                                                                match k_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k reshape): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v reshape): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                            }

                                                            let mut q_proc = q_heads;
                                                            let mut k_proc = k_heads;
                                                            let v_proc = v_heads;
                                                            if self.use_rope {
                                                                q_proc = q_proc.rope(self.num_heads);
                                                                k_proc = k_proc.rope(self.num_heads);
                                                            }
                                                            // Permute to [b, num_heads, seq, head_dim]
                                                            let q_perm = q_proc.permute(vec![0, 2, 1, 3]);
                                                            let k_perm = k_proc.permute(vec![0, 2, 1, 3]);
                                                            let v_perm = v_proc.permute(vec![0, 2, 1, 3]);
                                                            let q2 = match q_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (q_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let k2 = match k_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (k_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let v2 = match v_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (v_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };

                                                            let scale = 1.0f32 / (head_dim as f32).sqrt();

                                                            // If we are using a specialized attention variant, dispatch to an op
                                                            let out = match self.attention_variant {
                                                                AttentionVariant::Baseline => {
                                                                    let k2t = k2.permute(vec![0, 2, 1]);
                                                                    let qk = q2.batched_matmul(&k2t);
                                                                    let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
                                                                    let scaled = qk.mul(&scalar_tensor);
                                                                    // Optionally add ALiBi or relative positional bias to scaled logits
                                                                    let mut scaled_logits = scaled.clone();
                                                                    if self.use_alibi {
                                                                        let slopes = if let Some(s) = &self.alibi_slopes {
                                                                            s.clone()
                                                                        } else {
                                                                            compute_alibi_slopes(self.num_heads)
                                                                        };
                                                                        let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                                                                        for batch in 0..b {
                                                                            for h in 0..self.num_heads {
                                                                                let slope = slopes[h];
                                                                                for i in 0..seq {
                                                                                    for j in 0..seq {
                                                                                        let dist = (j as isize - i as isize) as f32;
                                                                                        let val = -slope * dist;
                                                                                        bias_arr[[batch * self.num_heads + h, i, j]] = val;
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                        let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                                                        scaled_logits = scaled_logits.add(&bias_t);
                                                                    }
                                                                    if let Some(rb) = &self.relative_bias {
                                                                        let shape = rb.lock().storage.shape();
                                                                        if shape.len() == 2 {
                                                                            let max_range = (shape[1] - 1) / 2;
                                                                            let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                                                                            let rb_arr = rb.lock().storage.to_f32_array();
                                                                            let rb_view = rb_arr.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                                                                            for batch in 0..b {
                                                                                for h in 0..self.num_heads {
                                                                                    for i in 0..seq {
                                                                                        for j in 0..seq {
                                                                                            let rel = (j as isize - i as isize)
                                                                                                .max(-(max_range as isize))
                                                                                                .min(max_range as isize)
                                                                                                as isize;
                                                                                            let idx = (rel + max_range as isize) as usize;
                                                                                            bias_arr[[batch * self.num_heads + h, i, j]] = rb_view[[h, idx]];
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                            let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                                                            scaled_logits = scaled_logits.add(&bias_t);
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

                                                            // reshape out [b*num_heads, seq, head_dim] -> [b, num_heads, seq, head_dim]
                                                            let out2 = match out.reshape(vec![b, self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error during attention result reshape: {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            // permute back to [b, seq, num_heads, head_dim]
                                                            let out3 = out2.permute(vec![0, 2, 1, 3]);
                                                            // reshape to [b, seq, d_model]
                                                            let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error during attention final reshape: {}", e);
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

                                                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
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

                                                        fn load_state_dict(
                                                            &mut self,
                                                            state: &std::collections::HashMap<String, Tensor>,
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

                                                        fn as_any(&self) -> &dyn std::any::Any {
                                                            self
                                                        }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
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
                                                            state: &std::collections::HashMap<String, Tensor>,
                                                            prefix: &str,
                                                        ) -> Result<(), String> {
                                                            self.load_state_dict(state, prefix)
                                                        }
                                                        fn as_any(&self) -> &dyn std::any::Any {
                                                            self
                                                        }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }

                                                    /// TransformerBlock: attention + residual + layernorm + feedforward + residual
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
                                                            log::info!(
            "TransformerBlock forward: input shape {:?}",
            x.lock().storage.shape()
        );
                                                            let attn_out = self.mha.forward(x);
                                                            let x2 = x.add(&attn_out);
                                                            // LayerNorm not exposed as module; use Tensor::layer_norm with axis=-1
                                                            let dim = x.lock().storage.shape()[2];
                                                            // create gamma and beta as ones/zeros (affine) for now
                                                            let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
                                                            let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
                                                            let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
                                                            let ff = self.linear1.forward(&x2norm).relu();
                                                            let ff = self.linear2.forward(&ff);
                                                            log::debug!(
            "TransformerBlock forward complete: output shape {:?}",
            x2.lock().storage.shape()
        );
                                                            x2.add(&ff)
                                                        }

                                                        pub fn parameters(&self) -> Vec<Tensor> {
                                                            let mut p = self.mha.parameters();
                                                            p.extend(self.linear1.parameters());
                                                            p.extend(self.linear2.parameters());
                                                            p
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
                                                        fn load_state_dict(
                                                            &mut self,
                                                            state: &std::collections::HashMap<String, Tensor>,
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
                                                        fn as_any(&self) -> &dyn std::any::Any {
                                                            self
                                                        }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }
                                                    use crate::nn::Linear;
                                                    use crate::nn::Module;
                                                    use crate::ops::{ChunkedAttention, FlashAttentionRef};
                                                    use crate::tensor::Tensor;
                                                    use ndarray::{Array, IxDyn};
                                                    use std::sync::Arc;

                                                    // ALiBi slope computation helper
                                                    pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                                        // simple decreasing slopes: 2^(-i/num_heads)
                                                        let mut slopes = Vec::with_capacity(n_heads);
                                                        for i in 0..n_heads {
                                                            let x = (i as f32) / (n_heads as f32);
                                                            slopes.push(2f32.powf(-x));
                                                        }
                                                        slopes
                                                    }

                                                    /// Attention variant selector
                                                    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                                    pub enum AttentionVariant {
                                                        Baseline,
                                                        FlashRef,
                                                        Chunked { chunk_size: usize },
                                                    }

                                                    /// MultiHeadAttention implemented using existing Linear layers and matmul.
                                                    /// This is a reference implementation optimized for clarity and correct shapes.
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
                                                        pub fn new(d_model: usize, num_heads: usize) -> Self {
                                                            Self::new_with_kv_and_rope(d_model, num_heads, num_heads, false)
                                                        }

                                                        pub fn new_with_kv_and_rope(
                                                            d_model: usize,
                                                            num_heads: usize,
                                                            kv_heads: usize,
                                                            use_rope: bool,
                                                        ) -> Self {
                                                            assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
                                                            assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
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

                                                        /// Forward attention. Input: [batch, seq, d_model]. Output: [batch, seq, d_model]
                                                        pub fn forward(&self, x: &Tensor) -> Tensor {
                                                            // Create Q, K, V
                                                            let q = self.linear_q.forward(x);
                                                            let k = self.linear_k.forward(x);
                                                            let v = self.linear_v.forward(x);

                                                            // reshape to [batch, seq, d_model]
                                                            let shape = q.lock().storage.shape();
                                                            if shape.len() != 3 {
                                                                log::error!(
                "MultiHeadAttention::forward expected input with 3 dims [batch, seq, d_model], got {:?}",
                shape
            );
                                                                return x.clone();
                                                            }
                                                            let b = shape[0];
                                                            let seq = shape[1];
                                                            let _d = self.d_model;

                                                            // compute head dims and reshape QKV to head layout
                                                            let head_dim = self.d_model / self.num_heads;
                                                            let q_heads = match q.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (q): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let mut k_heads = match k.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (k): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let mut v_heads = match v.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (v): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            // Expand kv if needed
                                                            if self.kv_heads != self.num_heads {
                                                                let group_size = self.num_heads / self.kv_heads;
                                                                let head_dim_kv = self.d_model / self.kv_heads;
                                                                let new_head_dim = head_dim_kv / group_size; // should equal head_dim
                                                                match k_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k split): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v split): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match k_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k merge): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v merge): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                            } else {
                                                                match k_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k reshape): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v reshape): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                            }

                                                            let mut q_proc = q_heads;
                                                            let mut k_proc = k_heads;
                                                            let v_proc = v_heads;
                                                            if self.use_rope {
                                                                q_proc = q_proc.rope(self.num_heads);
                                                                k_proc = k_proc.rope(self.num_heads);
                                                            }
                                                            // Permute to [b, num_heads, seq, head_dim]
                                                            let q_perm = q_proc.permute(vec![0, 2, 1, 3]);
                                                            let k_perm = k_proc.permute(vec![0, 2, 1, 3]);
                                                            let v_perm = v_proc.permute(vec![0, 2, 1, 3]);
                                                            let q2 = match q_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (q_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let k2 = match k_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (k_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let v2 = match v_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (v_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };

                                                            let scale = 1.0f32 / (head_dim as f32).sqrt();

                                                            // If we are using a specialized attention variant, dispatch to an op
                                                            let out = match self.attention_variant {
                                                                AttentionVariant::Baseline => {
                                                                    let k2t = k2.permute(vec![0, 2, 1]);
                                                                    let qk = q2.batched_matmul(&k2t);
                                                                    let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
                                                                    let scaled = qk.mul(&scalar_tensor);
                                                                    // Optionally add ALiBi or relative positional bias to scaled logits
                                                                    let mut scaled_logits = scaled.clone();
                                                                    if self.use_alibi {
                                                                        let slopes = if let Some(s) = &self.alibi_slopes {
                                                                            s.clone()
                                                                        } else {
                                                                            compute_alibi_slopes(self.num_heads)
                                                                        };
                                                                        let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                                                                        for batch in 0..b {
                                                                            for h in 0..self.num_heads {
                                                                                let slope = slopes[h];
                                                                                for i in 0..seq {
                                                                                    for j in 0..seq {
                                                                                        let dist = (j as isize - i as isize) as f32;
                                                                                        let val = -slope * dist;
                                                                                        bias_arr[[batch * self.num_heads + h, i, j]] = val;
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                        let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                                                        scaled_logits = scaled_logits.add(&bias_t);
                                                                    }
                                                                    if let Some(rb) = &self.relative_bias {
                                                                        let shape = rb.lock().storage.shape();
                                                                        if shape.len() == 2 {
                                                                            let max_range = (shape[1] - 1) / 2;
                                                                            let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                                                                            let rb_arr = rb.lock().storage.to_f32_array();
                                                                            let rb_view = rb_arr.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                                                                            for batch in 0..b {
                                                                                for h in 0..self.num_heads {
                                                                                    for i in 0..seq {
                                                                                        for j in 0..seq {
                                                                                            let rel = (j as isize - i as isize)
                                                                                                .max(-(max_range as isize))
                                                                                                .min(max_range as isize)
                                                                                                as isize;
                                                                                            let idx = (rel + max_range as isize) as usize;
                                                                                            bias_arr[[batch * self.num_heads + h, i, j]] = rb_view[[h, idx]];
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                            let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                                                            scaled_logits = scaled_logits.add(&bias_t);
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

                                                            // reshape out [b*num_heads, seq, head_dim] -> [b, num_heads, seq, head_dim]
                                                            let out2 = match out.reshape(vec![b, self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error during attention result reshape: {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            // permute back to [b, seq, num_heads, head_dim]
                                                            let out3 = out2.permute(vec![0, 2, 1, 3]);
                                                            // reshape to [b, seq, d_model]
                                                            let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error during attention final reshape: {}", e);
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

                                                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
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

                                                        fn load_state_dict(
                                                            &mut self,
                                                            state: &std::collections::HashMap<String, Tensor>,
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

                                                        fn as_any(&self) -> &dyn std::any::Any {
                                                            self
                                                        }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
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
                                                            state: &std::collections::HashMap<String, Tensor>,
                                                            prefix: &str,
                                                        ) -> Result<(), String> {
                                                            self.load_state_dict(state, prefix)
                                                        }
                                                        fn as_any(&self) -> &dyn std::any::Any {
                                                            self
                                                        }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }

                                                    /// TransformerBlock: attention + residual + layernorm + feedforward + residual
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
                                                            log::info!(
            "TransformerBlock forward: input shape {:?}",
            x.lock().storage.shape()
        );
                                                            let attn_out = self.mha.forward(x);
                                                            let x2 = x.add(&attn_out);
                                                            // LayerNorm not exposed as module; use Tensor::layer_norm with axis=-1
                                                            let dim = x.lock().storage.shape()[2];
                                                            // create gamma and beta as ones/zeros (affine) for now
                                                            let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
                                                            let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
                                                            let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
                                                            let ff = self.linear1.forward(&x2norm).relu();
                                                            let ff = self.linear2.forward(&ff);
                                                            log::debug!(
            "TransformerBlock forward complete: output shape {:?}",
            x2.lock().storage.shape()
        );
                                                            x2.add(&ff)
                                                        }

                                                        pub fn parameters(&self) -> Vec<Tensor> {
                                                            let mut p = self.mha.parameters();
                                                            p.extend(self.linear1.parameters());
                                                            p.extend(self.linear2.parameters());
                                                            p
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
                                                        fn load_state_dict(
                                                            &mut self,
                                                            state: &std::collections::HashMap<String, Tensor>,
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
                                                        fn as_any(&self) -> &dyn std::any::Any {
                                                            self
                                                        }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }
                                                    use crate::nn::Linear;
                                                    use crate::nn::Module;
                                                    use std::sync::Arc;
                                                    use crate::tensor::Tensor;
                                                    use ndarray::{Array, IxDyn};
                                                    // ALiBi slope computation helper
                                                    pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
                                                        // simple decreasing slopes: 2^(-i/num_heads)
                                                        let mut slopes = Vec::with_capacity(n_heads);
                                                        for i in 0..n_heads {
                                                            let x = (i as f32) / (n_heads as f32);
                                                            slopes.push(2f32.powf(-x));
                                                        }
                                                        slopes
                                                    }

                                                    /// Attention variant selector
                                                    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                                                    pub enum AttentionVariant {
                                                        Baseline,
                                                        FlashRef,
                                                        Chunked { chunk_size: usize },
                                                    }

                                                    impl crate::nn::Module for TransformerBlock {
                                                        fn forward(&self, input: &Tensor) -> Tensor {
                                                            self.forward_block(input)
                                                        }
                                                        fn parameters(&self) -> Vec<Tensor> {
                                                            self.parameters()
                                                        }
                                                        fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
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
                                                        fn load_state_dict(
                                                            &mut self,
                                                            state: &std::collections::HashMap<String, Tensor>,
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
                                                        fn as_any(&self) -> &dyn std::any::Any {
                                                            self
                                                        }
                                                        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
                                                    }

                                                    impl MultiHeadAttention {
                                                        pub fn forward(&self, x: &Tensor) -> Tensor {
                                                            // Create Q, K, V
                                                            let q = self.linear_q.forward(x);
                                                            let k = self.linear_k.forward(x);
                                                            let v = self.linear_v.forward(x);

                                                            // reshape to [batch, seq, d_model]
                                                            let shape = q.lock().storage.shape();
                                                            if shape.len() != 3 {
                                                                log::error!("MultiHeadAttention::forward expected input with 3 dims [batch, seq, d_model], got {:?}", shape);
                                                                return x.clone();
                                                            }
                                                            let b = shape[0];
                                                            let seq = shape[1];
                                                            let _d = self.d_model;

                                                            // compute head dims and reshape QKV to head layout
                                                            let head_dim = self.d_model / self.num_heads;
                                                            let q_heads = match q.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (q): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let mut k_heads = match k.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (k): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let mut v_heads = match v.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error in MultiHeadAttention::forward (v): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            // Expand kv if needed
                                                            if self.kv_heads != self.num_heads {
                                                                let group_size = self.num_heads / self.kv_heads;
                                                                let head_dim_kv = self.d_model / self.kv_heads;
                                                                let new_head_dim = head_dim_kv / group_size; // should equal head_dim
                                                                match k_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k split): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v split): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match k_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k merge): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.num_heads, new_head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v merge): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                            } else {
                                                                match k_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                    Ok(t) => k_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (k reshape): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                                match v_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                                                                    Ok(t) => v_heads = t,
                                                                    Err(e) => {
                                                                        log::error!("reshape error in MultiHeadAttention::forward (v reshape): {}", e);
                                                                        return x.clone();
                                                                    }
                                                                }
                                                            }

                                                            let mut q_proc = q_heads;
                                                            let mut k_proc = k_heads;
                                                            let v_proc = v_heads;
                                                            if self.use_rope {
                                                                q_proc = q_proc.rope(self.num_heads);
                                                                k_proc = k_proc.rope(self.num_heads);
                                                            }
                                                            // Permute to [b, num_heads, seq, head_dim]
                                                            let q_perm = q_proc.permute(vec![0, 2, 1, 3]);
                                                            let k_perm = k_proc.permute(vec![0, 2, 1, 3]);
                                                            let v_perm = v_proc.permute(vec![0, 2, 1, 3]);
                                                            let q2 = match q_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (q_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let k2 = match k_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (k_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            let v2 = match v_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error (v_heads to 3D): {}", e);
                                                                    return x.clone();
                                                                }
                                                            };

                                                            // attention scale
                                                            let scale = 1.0f32 / (head_dim as f32).sqrt();

                                                            // compute attn via variant
                                                            let out = match self.attention_variant {
                                                                AttentionVariant::Baseline => {
                                                                    let k2t = k2.permute(vec![0, 2, 1]);
                                                                    let qk = q2.batched_matmul(&k2t);
                                                                    let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
                                                                    let scaled = qk.mul(&scalar_tensor);
                                                                    let mut scaled_logits = scaled.clone();
                                                                    if self.use_alibi {
                                                                        let slopes = if let Some(s) = &self.alibi_slopes {
                                                                            s.clone()
                                                                        } else {
                                                                            compute_alibi_slopes(self.num_heads)
                                                                        };
                                                                        let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                                                                        for batch in 0..b {
                                                                            for h in 0..self.num_heads {
                                                                                let slope = slopes[h];
                                                                                for i in 0..seq {
                                                                                    for j in 0..seq {
                                                                                        let dist = (j as isize - i as isize) as f32;
                                                                                        let val = -slope * dist;
                                                                                        bias_arr[[batch * self.num_heads + h, i, j]] = val;
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                        let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                                                        scaled_logits = scaled_logits.add(&bias_t);
                                                                    }
                                                                    if let Some(rb) = &self.relative_bias {
                                                                        let shape = rb.lock().storage.shape();
                                                                        if shape.len() == 2 {
                                                                            let max_range = (shape[1] - 1) / 2;
                                                                            let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                                                                            let rb_arr = rb.lock().storage.to_f32_array();
                                                                            let rb_view = rb_arr.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                                                                            for batch in 0..b {
                                                                                for h in 0..self.num_heads {
                                                                                    for i in 0..seq {
                                                                                        for j in 0..seq {
                                                                                            let rel = (j as isize - i as isize)
                                                                                                .max(-(max_range as isize))
                                                                                                .min(max_range as isize) as isize;
                                                                                            let idx = (rel + max_range as isize) as usize;
                                                                                            bias_arr[[batch * self.num_heads + h, i, j]] = rb_view[[h, idx]];
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                            let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                                                            scaled_logits = scaled_logits.add(&bias_t);
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

                                                            // reshape out [b*num_heads, seq, head_dim] -> [b, num_heads, seq, head_dim]
                                                            let out2 = match out.reshape(vec![b, self.num_heads, seq, head_dim]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error during attention result reshape: {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            // permute back to [b, seq, num_heads, head_dim]
                                                            let out3 = out2.permute(vec![0, 2, 1, 3]);
                                                            // reshape to [b, seq, d_model]
                                                            let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
                                                                Ok(t) => t,
                                                                Err(e) => {
                                                                    log::error!("reshape error during attention final reshape: {}", e);
                                                                    return x.clone();
                                                                }
                                                            };
                                                            self.linear_o.forward(&out4)
                                                            // relative distance j-i
                                                            let dist = (j as isize - i as isize) as f32;
                                                            let val = -slope * dist;
                                                            bias_arr[[batch * self.num_heads + h, i, j]] = val;
                                                        }
                                                    }
                                                }
                                                //                            }
                                                let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                                scaled_logits = scaled_logits.add( & bias_t);
                                            }
                                            if let Some(rb) = & self .relative_bias {
                                            // rb shape [num_heads, 2*max+1]
                                            // create bias matrix for seq x seq
                                            let shape = rb.lock().storage.shape();
                                            if shape.len() == 2 {
                                            let max_range = (shape[1] - 1) / 2;
                                            let mut bias_arr = ndarray::ArrayD::< f32 >::zeros(IxDyn(& [b * self.num_heads, seq, seq]));
                                            let rb_arr = rb.lock().storage.to_f32_array();
                                            let rb_view = rb_arr.view().into_dimensionality::< ndarray::Ix2 > ().unwrap();
                                            for batch in 0..b {
                                            for h in 0..self.num_heads {
                                            for i in 0..seq {
                                            for j in 0..seq {
                                            let rel = (j as isize - i as isize)
                                            .max( - (max_range as isize))
                                            .min(max_range as isize);
                                            let idx = (rel + max_range as isize) as usize;
                                            bias_arr[[batch * self.num_heads + h, i, j]] = rb_view[[h, idx]];
                                            }
                                            }
                                            }
                                            }
                                            let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                                            scaled_logits = scaled_logits.add( & bias_t);
                                            }
                                        }
                                        let attn = scaled_logits.softmax(2);
                                        attn.batched_matmul( & v2)
                                    }
                                    AttentionVariant::FlashRef => {
                                    let flash = FlashAttentionRef::new(head_dim);
                                    Tensor::apply(Arc::new(flash), & [q2.clone(), k2.clone(), v2.clone()])
                                    }
                                    AttentionVariant::Chunked { chunk_size
                                } => {
                                let op = ChunkedAttention::new(head_dim, chunk_size);
                                Tensor::apply(Arc::new(op), & [q2.clone(), k2.clone(), v2.clone()])
                                }
                            };
                            log::error!("reshape error (v_heads to 3D): {}", e);
                            return x.clone();
                        }
                    };

                    // Compute attention per head using batched matmul
                    let k2t = k2.permute(vec![0, 2, 1]); // [batch, head_dim, seq]
                    let qk = q2.batched_matmul( & k2t);
                    let scale = 1.0f32 / (head_dim as f32).sqrt();
                    let scalar_tensor = Tensor::new(Array::from_elem(IxDyn( & [1]), scale), false);
                    let scaled = qk.mul( & scalar_tensor);
                    // Optionally add ALiBi or relative positional bias to scaled logits
                    let mut scaled_logits = scaled.clone();
                    if self .use_alibi {
                    // build alibi bias: shape [b*num_heads, seq, seq]
                    let slopes = if let Some(s) = & self .alibi_slopes {
                    s.clone()
                } else {
                compute_alibi_slopes( self.num_heads)
                };
                // create bias array
                let mut bias_arr =
                ndarray::ArrayD::<f32>::zeros(IxDyn( & [b * self .num_heads, seq, seq]));
                for batch in 0..b {
                for h in 0..self.num_heads {
                let slope = slopes[h];
                for i in 0..seq {
                for j in 0..seq {
                // relative distance j-i
                let dist = (j as isize - i as isize) as f32;
                let val = - slope * dist;
                bias_arr[[batch * self.num_heads + h, i, j]] = val;
                }
                }
                }
                }
                let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                scaled_logits = scaled_logits.add( & bias_t);
            }
            // Add relative bias if provided (shape expected [num_heads, range])
            if let Some(rb) = & self .relative_bias {
            // rb shape [num_heads, 2*max+1]
            // create bias matrix for seq x seq
            let shape = rb.lock().storage.shape();
            if shape.len() == 2 {
            let max_range = (shape[1] - 1) / 2;
            let mut bias_arr = ndarray::ArrayD::< f32 >::zeros(IxDyn(& [b * self.num_heads, seq, seq]));
            let rb_arr = rb.lock().storage.to_f32_array();
            let rb_view = rb_arr.view().into_dimensionality::< ndarray::Ix2 > ().unwrap();
            for batch in 0..b {
            for h in 0..self.num_heads {
            for i in 0..seq {
            for j in 0..seq {
            let rel = (j as isize - i as isize)
            .max( - (max_range as isize))
            .min(max_range as isize) as isize;
            let idx = (rel + max_range as isize) as usize;
            bias_arr[[batch * self.num_heads + h, i, j]] = rb_view[[h, idx]];
            }
            }
            }
            }
            let bias_t = crate::tensor::Tensor::new(bias_arr, false);
            scaled_logits = scaled_logits.add( & bias_t);
            }
        }
        // Continue: compute softmax
        let attn = scaled_logits.softmax(2);
        AttentionVariant::Baseline => {
        let k2t = k2.permute(vec ! [0, 2, 1]);
        let qk = q2.batched_matmul( & k2t);
        let scalar_tensor = Tensor::new(Array::from_elem(IxDyn( & [1]), scale), false);
        let scaled = qk.mul( & scalar_tensor);
        // Optionally add ALiBi or relative positional bias to scaled logits
        let mut scaled_logits = scaled.clone();
        if self.use_alibi {
        let slopes = if let Some(s) = & self.alibi_slopes {
        s.clone()
        } else {
        compute_alibi_slopes( self.num_heads)
        };
        let mut bias_arr = ndarray::ArrayD::<f32 >::zeros(IxDyn( & [b * self.num_heads, seq, seq]));
        for batch in 0..b {
        for h in 0..self.num_heads {
        let slope = slopes[h];
        for i in 0..seq {
        for j in 0..seq {
        let dist = (j as isize - i as isize) as f32;
        let val = - slope * dist;
        bias_arr[[batch * self.num_heads + h, i, j]] = val;
        }
        }
        }
        }
        let bias_t = crate::tensor::Tensor::new(bias_arr, false);
        scaled_logits = scaled_logits.add( & bias_t);
        }
        if let Some(rb) = & self.relative_bias {
        // rb shape [num_heads, 2*max+1]
        let shape = rb.lock().storage.shape();
        if shape.len() == 2 {
        let max_range = (shape[1] - 1) / 2;
        let mut bias_arr = ndarray::ArrayD::< f32 >::zeros(IxDyn(& [b * self.num_heads, seq, seq]));
        let rb_arr = rb.lock().storage.to_f32_array();
        let rb_view = rb_arr.view().into_dimensionality::< ndarray::Ix2 > ().unwrap();
        for batch in 0..b {
        for h in 0..self.num_heads {
        for i in 0..seq {
        for j in 0..seq {
        let rel = (j as isize - i as isize)
        .max( - (max_range as isize))
        .min(max_range as isize)
        as isize;
        let idx = (rel + max_range as isize) as usize;
        bias_arr[[batch * self.num_heads + h, i, j]] = rb_view[[h, idx]];
        }
        }
        }
        }
        let bias_t = crate::tensor::Tensor::new(bias_arr, false);
        scaled_logits = scaled_logits.add( & bias_t);
        }
        }
        let attn = scaled_logits.softmax(2);
        let out = attn.batched_matmul( & v2);
        out
        }
        AttentionVariant::FlashRef => {
        let flash = FlashAttentionRef::new(head_dim);
        Tensor::apply(Arc::new(flash), & [q2.clone(), k2.clone(), v2.clone()])
        }
        AttentionVariant::Chunked { chunk_size
    } => {
    let op = ChunkedAttention::new(head_dim, chunk_size);
    Tensor::apply(Arc::new(op), & [q2.clone(), k2.clone(), v2.clone()])
    }
};
let bias_t = crate::tensor::Tensor::new(bias_arr, false);
scaled_logits = scaled_logits.add( & bias_t);
}
}
let attn = scaled_logits.softmax(2);
let out = attn.batched_matmul( & v2);
// reshape out [b*num_heads, seq, head_dim] -> [b, num_heads, seq, head_dim]
let out2 = match out.reshape(vec![b, self.num_heads, seq, head_dim]) {
Ok(t) => t,
Err(e) => {
log::error ! ("reshape error during attention result reshape: {}", e);
return x.clone();
}
};
// permute back to [b, seq, num_heads, head_dim]
let out3 = out2.permute(vec![0, 2, 1, 3]);
// reshape to [b, seq, d_model]
let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
Ok(t) => t,
Err(e) => {
log::error ! ("reshape error during attention final reshape: {}", e);
return x.clone();
}
};
self .linear_o.forward( & out4)
}

pub fn parameters(&self) -> Vec<Tensor> {
    let mut p = self.linear_q.parameters();
    p.extend(self.linear_k.parameters());
    p.extend(self.linear_v.parameters());
    p.extend(self.linear_o.parameters());
    p
}

fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
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

fn load_state_dict(
    &mut self,
    state: &std::collections::HashMap<String, Tensor>,
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

fn as_any(&self) -> &dyn std::any::Any {
    self
}
fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
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
        state: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.load_state_dict(state, prefix)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// TransformerBlock: attention + residual + layernorm + feedforward + residual
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
        log::info!(
            "TransformerBlock forward: input shape {:?}",
            x.lock().storage.shape()
        );
        let attn_out = self.mha.forward(x);
        let x2 = x.add(&attn_out);
        // LayerNorm not exposed as module; use Tensor::layer_norm with axis=-1
        let dim = x.lock().storage.shape()[2];
        // create gamma and beta as ones/zeros (affine) for now
        let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
        let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
        let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
        let ff = self.linear1.forward(&x2norm).relu();
        let ff = self.linear2.forward(&ff);
        log::debug!(
            "TransformerBlock forward complete: output shape {:?}",
            x2.lock().storage.shape()
        );
        x2.add(&ff)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.mha.parameters();
        p.extend(self.linear1.parameters());
        p.extend(self.linear2.parameters());
        p
    }
}
