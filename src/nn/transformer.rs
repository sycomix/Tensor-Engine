// DEPRECATED: `transformer.rs` was replaced by `transformer_cleaned.rs` and kept for historical
// reference only. Do NOT add new code here; update `src/nn/transformer_cleaned.rs` instead.
// Minimal transformer — single canonical implementation.
use crate::nn::Linear;
use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};
use std::any::Any;
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

/// Compute simple ALiBi slopes; geometric decay-style.
pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
    let mut slopes = Vec::with_capacity(n_heads);
    for i in 0..n_heads {
        let x = (i as f32) / (n_heads as f32);
        slopes.push(2f32.powf(-x));
    }
    slopes
}

pub fn reshape_for_multihead(
    t: &Tensor,
    b: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor, String> {
    let r = t.reshape(vec![b, seq, num_heads, head_dim])?;
    let p = r.permute(vec![0, 2, 1, 3]);
    p.reshape(vec![b * num_heads, seq, head_dim])
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
        assert!(d_model.is_multiple_of(num_heads));
        assert!(num_heads.is_multiple_of(kv_heads));
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

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.linear_q.forward(x);
        let k = self.linear_k.forward(x);
        let v = self.linear_v.forward(x);

        let shape = q.lock().storage.shape();
        if shape.len() != 3 {
            return x.clone();
        }
        let (b, seq) = (shape[0], shape[1]);
        let head_dim = self.d_model / self.num_heads;

        let mut q_proc = q.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap();
        let mut k_proc = k.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap();
        let v_proc = v.reshape(vec![b, seq, self.num_heads, head_dim]).unwrap();

        if self.use_rope {
            q_proc = q_proc.rope(self.num_heads, 10000.0, 1.0, 0);
            k_proc = k_proc.rope(self.num_heads, 10000.0, 1.0, 0);
        }

        let q2 = q_proc
            .permute(vec![0, 2, 1, 3])
            .reshape(vec![b * self.num_heads, seq, head_dim])
            .unwrap();
        let k2 = k_proc
            .permute(vec![0, 2, 1, 3])
            .reshape(vec![b * self.num_heads, seq, head_dim])
            .unwrap();
        let v2 = v_proc
            .permute(vec![0, 2, 1, 3])
            .reshape(vec![b * self.num_heads, seq, head_dim])
            .unwrap();

        let out = match self.attention_variant {
            AttentionVariant::Baseline => {
                let k2t = k2.permute(vec![0, 2, 1]);
                let qk = q2.batched_matmul(&k2t);
                let scale = 1.0f32 / (head_dim as f32).sqrt();
                let scalar_t = Tensor::new(Array::from_elem(ndarray::IxDyn(&[1]), scale), false);
                let mut logits = qk.mul(&scalar_t);

                if self.use_alibi {
                    let slopes = self
                        .alibi_slopes
                        .as_ref()
                        .cloned()
                        .unwrap_or_else(|| compute_alibi_slopes(self.num_heads));
                    let mut bias_arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[
                        b * self.num_heads,
                        seq,
                        seq,
                    ]));
                    for bh in 0..(b * self.num_heads) {
                        let slope = slopes[bh % self.num_heads];
                        for i in 0..seq {
                            for j in 0..seq {
                                bias_arr[[bh, i, j]] =
                                    -slope * (i as isize - j as isize).abs() as f32;
                            }
                        }
                    }
                    logits = logits.add(&Tensor::new(bias_arr, false));
                }

                if let Some(rb) = &self.relative_bias {
                    logits = logits.add(rb);
                }

                logits.softmax(2).batched_matmul(&v2)
            }
            AttentionVariant::FlashRef => {
                let flash = crate::ops::FlashAttentionRef::new(head_dim);
                Tensor::apply(Arc::new(flash), &[q2, k2, v2])
            }
            AttentionVariant::Chunked { chunk_size } => {
                let op = crate::ops::ChunkedAttention::new(head_dim, chunk_size);
                Tensor::apply(Arc::new(op), &[q2, k2, v2])
            }
        };

        let merged = out
            .reshape(vec![b, self.num_heads, seq, head_dim])
            .unwrap()
            .permute(vec![0, 2, 1, 3])
            .reshape(vec![b, seq, self.d_model])
            .unwrap();

        self.linear_o.forward(&merged)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }
    fn parameters(&self) -> Vec<Tensor> {
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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
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
        let attn_out = self.mha.forward(x);
        let x2 = x.add(&attn_out);
        let dim = x.lock().storage.shape()[2];
        let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
        let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
        let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
        let ff = self.linear2.forward(&self.linear1.forward(&x2norm).relu());
        x2.add(&ff)
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_block(x)
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.mha.parameters();
        p.extend(self.linear1.parameters());
        p.extend(self.linear2.parameters());
        p
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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
