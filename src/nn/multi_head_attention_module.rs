//! Compatibility shim: consolidate MultiHeadAttention implementation to
//! `transformer_cleaned.rs` and re-export its public API here to preserve the
//! original module path while avoiding duplicate implementations.

pub use crate::nn::transformer_cleaned::{
    AttentionVariant,
    compute_alibi_slopes,
    MultiHeadAttention,
    TransformerBlock,
};

// Basic smoke tests to exercise the re-exported API and ensure the canonical
// implementation remains exercised by test builds.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Module; // bring the Module trait into scope so `.forward()` is available
    use ndarray::Array;

    #[test]
    fn smoke_mha_forward() {
        let d_model = 8usize;
        let num_heads = 2usize;
        let mha = MultiHeadAttention::new(d_model, num_heads);
        let input = Array::from_shape_fn((1, 4, d_model), |_| 0.1f32);
        let t = crate::tensor::Tensor::new(input.into_dyn(), false);
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
        let block = TransformerBlock::new(d_model, d_ff, heads)
            .expect("failed to create TransformerBlock");
        let input = Array::from_shape_fn((1, 4, d_model), |_| 0.2f32);
        let t = crate::tensor::Tensor::new(input.into_dyn(), false);
        let out = block.forward(&t);
        let shape = out.lock().storage.shape();
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[2], d_model);
    }
}

// Small const to reference re-exported items so the compiler marks them used
// and avoids "unused import" warnings in this module.
const _REEXPORTS_USED: () = {
    let _ = AttentionVariant::Baseline;
    let _ = compute_alibi_slopes as fn(usize) -> Vec<f32>;
    let _ = MultiHeadAttention::new as fn(usize, usize) -> MultiHeadAttention;
    let _ = TransformerBlock::new as fn(usize, usize, usize) -> Result<TransformerBlock, String>;
};

#[doc(hidden)]
pub fn __ensure_multi_head_reexports_linked() {
    let _ = AttentionVariant::Baseline;
    let _ = AttentionVariant::FlashRef;
    let _ = AttentionVariant::Chunked { chunk_size: 1 };
    let _ = compute_alibi_slopes as fn(usize) -> Vec<f32>;
    let _ = MultiHeadAttention::new as fn(usize, usize) -> MultiHeadAttention;
    let _ = TransformerBlock::new as fn(usize, usize, usize) -> Result<TransformerBlock, String>;
}
