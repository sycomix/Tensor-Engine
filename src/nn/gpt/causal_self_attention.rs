use super::attention_weights::compute_attention_weights_flat;
use rand::Rng;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Errors for strict causal self-attention execution.
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

/// Compact, parameter-free causal self-attention block.
///
/// This struct has no trainable parameters and only performs attention-weight
/// computation, optional masking/dropout, and weighted aggregation.
#[derive(Debug, Clone, Copy)]
pub struct CausalSelfAttention {
    causal: bool,
    dropout_rate: Option<f32>,
}

impl CausalSelfAttention {
    /// Create a new attention block.
    ///
    /// `dropout_rate` is initialized as `None`.
    pub fn new(causal: bool) -> Self {
        Self {
            causal,
            dropout_rate: None,
        }
    }

    /// Enable or disable causal masking.
    pub fn set_causal(&mut self, causal: bool) {
        self.causal = causal;
    }

    /// Return whether causal masking is enabled.
    pub fn causal(&self) -> bool {
        self.causal
    }

    /// Set optional attention dropout probability.
    ///
    /// Use `Some(p)` to enable dropout, `None` to disable.
    pub fn set_dropout_rate(&mut self, dropout_rate: Option<f32>) {
        self.dropout_rate = dropout_rate;
    }

    /// Return the configured dropout probability.
    pub fn dropout_rate(&self) -> Option<f32> {
        self.dropout_rate
    }

    /// Compute self-attention output for one sequence.
    ///
    /// Input shape: `[seq_len][embedding_dim]`
    /// Output shape: `[seq_len][embedding_dim]`
    ///
    /// This ergonomic method returns an empty output on invalid input.
    /// Dropout is applied only when `training == true`.
    pub fn forward(&self, input: &[Vec<f32>], training: bool) -> Vec<Vec<f32>> {
        let mut rng = rand::rng();
        match self.try_forward_with_rng(input, training, &mut rng) {
            Ok(v) => v,
            Err(_) => Vec::new(),
        }
    }

    /// Strict forward pass with explicit error reporting.
    pub fn try_forward(
        &self,
        input: &[Vec<f32>],
        training: bool,
    ) -> Result<Vec<Vec<f32>>, SelfAttentionError> {
        let mut rng = rand::rng();
        self.try_forward_with_rng(input, training, &mut rng)
    }

    fn try_forward_with_rng<R: Rng + ?Sized>(
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

        // Compute flattened row-major attention weights: weights[i * seq_len + j].
        let (mut weights, n) = compute_attention_weights_flat(input);
        if n != seq_len {
            return Err(SelfAttentionError::RaggedInput);
        }

        if self.causal {
            self.apply_causal_mask_in_place(&mut weights, seq_len);
        }

        self.apply_dropout_in_place(&mut weights, training, rng)?;

        // Weighted sum: output[i, d] = sum_j weights[i, j] * input[j, d]
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

    fn apply_causal_mask_in_place(&self, weights: &mut [f32], seq_len: usize) {
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row_end = row_start + seq_len;
            let row = &mut weights[row_start..row_end];

            // Mask future positions (j > i).
            for j in (i + 1)..seq_len {
                row[j] = 0.0;
            }

            // Re-normalize each row after masking.
            let sum: f32 = row.iter().copied().sum();
            if sum > 0.0 {
                for v in row {
                    *v /= sum;
                }
            }
        }
    }

    fn apply_dropout_in_place<R: Rng + ?Sized>(
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

        // Inverted dropout on attention probabilities.
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
}
