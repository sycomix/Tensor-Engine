use super::attention_weights::compute_attention_weights_flat;
use rand::Rng;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Errors returned by strict self-attention execution.
#[derive(Debug, Clone, PartialEq)]
pub enum SelfAttentionError {
    EmptyInput,
    ZeroEmbeddingDim,
    RaggedInput,
    InvalidDropoutRate(f32),
}

impl Display for SelfAttentionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SelfAttentionError::EmptyInput => write!(f, "input sequence is empty"),
            SelfAttentionError::ZeroEmbeddingDim => write!(f, "embedding dimension must be > 0"),
            SelfAttentionError::RaggedInput => {
                write!(f, "all input embedding rows must have the same dimension")
            }
            SelfAttentionError::InvalidDropoutRate(p) => {
                write!(f, "dropout_rate must be in [0.0, 1.0); got {}", p)
            }
        }
    }
}

impl Error for SelfAttentionError {}

/// Parameter-free self-attention block.
///
/// This struct has no trainable parameters. It computes attention outputs from
/// input embeddings by:
/// 1) obtaining attention weights,
/// 2) optionally applying causal masking,
/// 3) multiplying weights by input embeddings.
#[derive(Debug, Clone, Copy)]
pub struct SelfAttention {
    causal: bool,
    dropout_rate: Option<f32>,
}

impl SelfAttention {
    /// Create a new self-attention module.
    ///
    /// If `causal` is true, future positions are masked out during `forward`.
    pub fn new(causal: bool) -> Self {
        Self {
            causal,
            dropout_rate: None,
        }
    }

    /// Set optional dropout probability for attention weights.
    ///
    /// `Some(p)` enables dropout with probability `p`, while `None` disables dropout.
    pub fn set_dropout_rate(&mut self, dropout_rate: Option<f32>) {
        self.dropout_rate = dropout_rate;
    }

    /// Returns the configured dropout probability.
    pub fn dropout_rate(&self) -> Option<f32> {
        self.dropout_rate
    }

    /// Enable or disable causal masking.
    pub fn set_causal(&mut self, causal: bool) {
        self.causal = causal;
    }

    /// Returns whether causal masking is enabled.
    pub fn causal(&self) -> bool {
        self.causal
    }

    /// Compute self-attention outputs.
    ///
    /// Input shape: `[seq_len][embedding_dim]`
    /// Output shape: `[seq_len][embedding_dim]`
    ///
    /// This method is ergonomic and returns an empty output on invalid input.
    /// For explicit error handling, use `try_forward`.
    ///
    /// Dropout is only applied when `training == true`.
    pub fn forward(&self, input: &[Vec<f32>], training: bool) -> Vec<Vec<f32>> {
        let mut rng = rand::rng();
        match self.try_forward_with_rng(input, training, &mut rng) {
            Ok(v) => v,
            Err(_) => Vec::new(),
        }
    }

    /// Strict self-attention forward pass with explicit error reporting.
    ///
    /// - Validates input shape.
    /// - Uses flattened attention weights for memory-efficient computation.
    /// - Applies optional causal masking with row re-normalization.
    /// - Applies optional dropout *after softmax/masking and before weighted sum*.
    pub fn try_forward(
        &self,
        input: &[Vec<f32>],
        training: bool,
    ) -> Result<Vec<Vec<f32>>, SelfAttentionError> {
        let mut rng = rand::rng();
        self.try_forward_with_rng(input, training, &mut rng)
    }

    /// Strict self-attention forward pass with explicit RNG injection.
    ///
    /// This variant enables reproducible training by allowing callers to pass a
    /// seeded RNG.
    pub fn try_forward_with_rng<R: Rng + ?Sized>(
        &self,
        input: &[Vec<f32>],
        training: bool,
        rng: &mut R,
    ) -> Result<Vec<Vec<f32>>, SelfAttentionError> {
        if input.is_empty() {
            return Err(SelfAttentionError::EmptyInput);
        }

        let seq_len = input.len();
        let embedding_dim = input[0].len();
        if embedding_dim == 0 {
            return Err(SelfAttentionError::ZeroEmbeddingDim);
        }

        if input.iter().any(|row| row.len() != embedding_dim) {
            return Err(SelfAttentionError::RaggedInput);
        }

        // Attention weights are produced in flattened row-major form:
        // weights[i * seq_len + j].
        let (mut weights, n) = compute_attention_weights_flat(input);
        if n != seq_len {
            return Err(SelfAttentionError::RaggedInput);
        }

        if self.causal {
            self.apply_causal_mask_in_place(&mut weights, seq_len);
        }

        // Dropout is applied on attention probabilities (after softmax and any mask)
        // and before the weighted-sum aggregation.
        self.apply_dropout_in_place_with_rng(&mut weights, training, rng)?;

        // Output[i][d] = sum_j weights[i,j] * input[j][d]
        let mut output = vec![vec![0.0_f32; embedding_dim]; seq_len];
        for i in 0..seq_len {
            let row_offset = i * seq_len;
            for j in 0..seq_len {
                let w = weights[row_offset + j];
                if w == 0.0 {
                    continue;
                }
                let src = &input[j];
                let dst = &mut output[i];
                for d in 0..embedding_dim {
                    dst[d] += w * src[d];
                }
            }
        }

        Ok(output)
    }

    #[allow(dead_code)]
    pub(crate) fn apply_dropout_in_place(
        &self,
        weights: &mut [f32],
        training: bool,
    ) -> Result<(), SelfAttentionError> {
        let mut rng = rand::rng();
        self.apply_dropout_in_place_with_rng(weights, training, &mut rng)
    }

    pub(crate) fn apply_dropout_in_place_with_rng<R: Rng + ?Sized>(
        &self,
        weights: &mut [f32],
        training: bool,
        rng: &mut R,
    ) -> Result<(), SelfAttentionError> {
        let Some(p) = self.dropout_rate else {
            return Ok(());
        };

        if !(0.0..1.0).contains(&p) {
            return Err(SelfAttentionError::InvalidDropoutRate(p));
        }

        if !training || p == 0.0 {
            return Ok(());
        }

        let keep_scale = 1.0 / (1.0 - p);
        for w in weights.iter_mut() {
            if rng.random::<f32>() < p {
                *w = 0.0;
            } else {
                *w *= keep_scale;
            }
        }

        Ok(())
    }

    fn apply_causal_mask_in_place(&self, weights: &mut [f32], seq_len: usize) {
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row_end = row_start + seq_len;
            let row = &mut weights[row_start..row_end];

            // Zero out future positions j > i.
            for j in (i + 1)..seq_len {
                row[j] = 0.0;
            }

            // Re-normalize row to keep a valid probability distribution.
            let sum: f32 = row.iter().copied().sum();
            if sum > 0.0 {
                for v in row {
                    *v /= sum;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn dropout_applies_and_preserves_expected_scale() {
        let mut sa = SelfAttention::new(false);
        sa.set_dropout_rate(Some(0.5));

        let mut weights = vec![1.0_f32; 4096];
        let mut rng = StdRng::seed_from_u64(7);

        sa.apply_dropout_in_place_with_rng(&mut weights, true, &mut rng)
            .expect("dropout should succeed");

        let zeros = weights.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0, "expected some dropped weights");

        let mean: f32 = weights.iter().sum::<f32>() / (weights.len() as f32);
        assert!(
            (mean - 1.0).abs() < 0.15,
            "dropout scaling drifted too far from expectation: {}",
            mean
        );
    }
}
