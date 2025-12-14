// DEPRECATED: This file was replaced by `transformer_cleaned.rs` and
// is preserved only for reference and migration. Do not re-export or use
// this file in new code. The canonical `transformer` module is
// `transformer_cleaned.rs` and is re-exported from `mod.rs`.
use crate::nn::Linear;
use crate::nn::Module;
use crate::ops::{ChunkedAttention, FlashAttentionRef};
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionVariant {
    Baseline,
    FlashRef,
    Chunked { chunk_size: usize },
}

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
    // Deprecated implementation preserved for reference.
}

// The rest of the file contains a complete MultiHeadAttention and TransformerBlock implementation
// preserved here for reference only and should not be included in builds.
