#![cfg(feature = "hf_compat")]

use crate::tensor::Tensor;
use hf_compat::token_sampler::TokenSampler as HFTokenSampler;
use hf_compat::tokenizer::Tokenizer as HFTokenizer;
use hf_compat::Embedding as HFEmbedding;
use ndarray::Array;

/// Convert an `hf_compat::Embedding` into the main crate `Tensor`.
pub fn embedding_to_tensor(he: &HFEmbedding) -> Result<Tensor, String> {
    let rows = he.rows;
    let cols = he.cols;
    if he.data.len() != rows * cols {
        return Err("embedding data length mismatch".to_string());
    }
    let arr = match Array::from_shape_vec((rows, cols), he.data.clone()) {
        Ok(a) => a.into_dyn(),
        Err(e) => return Err(format!("ndarray reshape error: {}", e)),
    };
    Ok(Tensor::new(arr, false))
}

/// Sample from a `tensor_engine::Tensor` logits using an `hf_compat` TokenSampler.
/// The logits `Tensor` is expected to be shape `(vocab_size, 1)`.
pub fn sample_from_tensor(
    sampler: &HFTokenSampler,
    logits: &Tensor,
    tokenizer: &HFTokenizer,
    existing_tokens: &[usize],
) -> Result<(usize, f32), String> {
    let arr = logits.lock().storage.to_f32_array();
    let shape = arr.shape().to_vec();
    // Accept (vocab, 1) or (vocab,)
    let vocab_size: usize;
    let mut vec: Vec<f32> = Vec::new();
    if shape.len() == 2 && shape[1] == 1 {
        vocab_size = shape[0];
        for i in 0..vocab_size {
            vec.push(arr[[i, 0]]);
        }
    } else if shape.len() == 1 {
        vocab_size = shape[0];
        for i in 0..vocab_size {
            vec.push(arr[[i]]);
        }
    } else {
        return Err(format!("Unsupported logits shape: {:?}", shape));
    }
    Ok(sampler.sample_ids(&vec, vocab_size, tokenizer, existing_tokens))
}

/// Ergonomic wrapper around `hf_compat::token_sampler::TokenSampler` that exposes sampling directly on `Tensor`.
pub struct HfTokenSampler(pub hf_compat::token_sampler::TokenSampler);

impl HfTokenSampler {
    pub fn new() -> Self {
        HfTokenSampler(hf_compat::token_sampler::TokenSampler::new())
    }

    pub fn sample_from_tensor(&self, logits: &Tensor, tokenizer: &HFTokenizer, existing_tokens: &[usize]) -> Result<(usize, f32), String> {
        sample_from_tensor(&self.0, logits, tokenizer, existing_tokens)
    }
}
