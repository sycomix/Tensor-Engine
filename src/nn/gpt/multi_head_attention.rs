use super::causal_self_attention::{CausalSelfAttention, SelfAttentionError};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Errors for strict multi-head causal attention execution.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiHeadAttentionError {
    ZeroHeads,
    EmptyInput,
    RaggedInput,
    EmbeddingDimNotDivisible {
        embedding_dim: usize,
        num_heads: usize,
    },
    HeadError(SelfAttentionError),
}

impl Display for MultiHeadAttentionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MultiHeadAttentionError::ZeroHeads => write!(f, "num_heads must be > 0"),
            MultiHeadAttentionError::EmptyInput => write!(f, "input sequence is empty"),
            MultiHeadAttentionError::RaggedInput => {
                write!(f, "all input embedding rows must have the same dimension")
            }
            MultiHeadAttentionError::EmbeddingDimNotDivisible {
                embedding_dim,
                num_heads,
            } => write!(
                f,
                "embedding_dim {} must be divisible by num_heads {}",
                embedding_dim, num_heads
            ),
            MultiHeadAttentionError::HeadError(err) => write!(f, "head forward failed: {}", err),
        }
    }
}

impl Error for MultiHeadAttentionError {}

/// Parameter-free multi-head causal self-attention.
///
/// Each head runs an independent `CausalSelfAttention` over a disjoint slice of
/// the embedding dimension. Head outputs are concatenated back together.
#[derive(Debug, Clone)]
pub struct MultiHeadCausalAttention {
    pub heads: Vec<CausalSelfAttention>,
    pub num_heads: usize,
    pub dropout_rate: Option<f32>,
}

impl MultiHeadCausalAttention {
    /// Create a new multi-head attention module with `num_heads` heads.
    ///
    /// If `num_heads == 0`, this creates an empty module; `forward` will return
    /// an empty output, while `try_forward` returns a `ZeroHeads` error.
    pub fn new(num_heads: usize, causal: bool) -> Self {
        let mut heads = Vec::with_capacity(num_heads);
        for _ in 0..num_heads {
            heads.push(CausalSelfAttention::new(causal));
        }

        Self {
            heads,
            num_heads,
            dropout_rate: None,
        }
    }

    /// Set causal masking mode for all heads.
    pub fn set_causal_all(&mut self, causal: bool) {
        for head in &mut self.heads {
            head.set_causal(causal);
        }
    }

    /// Set dropout probability for all heads.
    pub fn set_dropout_rate_all(&mut self, dropout_rate: Option<f32>) {
        self.dropout_rate = dropout_rate;
        for head in &mut self.heads {
            head.set_dropout_rate(dropout_rate);
        }
    }

    /// Ergonomic single-sequence forward pass.
    ///
    /// Input shape: `[seq_len][embedding_dim]`
    /// Output shape: `[seq_len][embedding_dim]`
    ///
    /// Returns empty output on invalid input. For strict errors, use `try_forward`.
    pub fn forward(&self, input: &[Vec<f32>], training: bool) -> Vec<Vec<f32>> {
        self.try_forward(input, training).unwrap_or_default()
    }

    /// Strict single-sequence forward pass with explicit error propagation.
    pub fn try_forward(
        &self,
        input: &[Vec<f32>],
        training: bool,
    ) -> Result<Vec<Vec<f32>>, MultiHeadAttentionError> {
        if self.num_heads == 0 || self.heads.is_empty() {
            return Err(MultiHeadAttentionError::ZeroHeads);
        }
        if input.is_empty() {
            return Err(MultiHeadAttentionError::EmptyInput);
        }

        let seq_len = input.len();
        let embedding_dim = input[0].len();
        if embedding_dim == 0 {
            return Err(MultiHeadAttentionError::RaggedInput);
        }
        if input.iter().any(|row| row.len() != embedding_dim) {
            return Err(MultiHeadAttentionError::RaggedInput);
        }
        if !embedding_dim.is_multiple_of(self.num_heads) {
            return Err(MultiHeadAttentionError::EmbeddingDimNotDivisible {
                embedding_dim,
                num_heads: self.num_heads,
            });
        }

        let head_dim = embedding_dim / self.num_heads;

        // Run each head on its own embedding slice.
        let mut head_outputs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.num_heads);
        for head_index in 0..self.num_heads {
            let start = head_index * head_dim;
            let end = start + head_dim;

            let mut head_input = Vec::with_capacity(seq_len);
            for row in input {
                head_input.push(row[start..end].to_vec());
            }

            let head_out = self.heads[head_index]
                .try_forward(&head_input, training)
                .map_err(MultiHeadAttentionError::HeadError)?;
            head_outputs.push(head_out);
        }

        // Concatenate per-head outputs along the embedding dimension.
        let mut output = vec![vec![0.0_f32; embedding_dim]; seq_len];
        for i in 0..seq_len {
            let mut write_offset = 0usize;
            for head_out in &head_outputs {
                let src = &head_out[i];
                let dst = &mut output[i][write_offset..write_offset + head_dim];
                dst.copy_from_slice(src);
                write_offset += head_dim;
            }
        }

        Ok(output)
    }

    /// Strict incremental forward pass for the newest token.
    ///
    /// `past_input` is the cached sequence entering this attention module and
    /// `query` is the current token representation. The returned row matches
    /// the final row from `try_forward([past_input..., query])` when dropout is
    /// disabled, while avoiding recomputation for all earlier rows.
    pub fn try_forward_last(
        &self,
        past_input: &[Vec<f32>],
        query: &[f32],
        training: bool,
    ) -> Result<Vec<f32>, MultiHeadAttentionError> {
        if self.num_heads == 0 || self.heads.is_empty() {
            return Err(MultiHeadAttentionError::ZeroHeads);
        }
        if query.is_empty() {
            return Err(MultiHeadAttentionError::RaggedInput);
        }

        let embedding_dim = query.len();
        if past_input.iter().any(|row| row.len() != embedding_dim) {
            return Err(MultiHeadAttentionError::RaggedInput);
        }
        if !embedding_dim.is_multiple_of(self.num_heads) {
            return Err(MultiHeadAttentionError::EmbeddingDimNotDivisible {
                embedding_dim,
                num_heads: self.num_heads,
            });
        }

        let head_dim = embedding_dim / self.num_heads;
        let mut output = vec![0.0_f32; embedding_dim];

        for head_index in 0..self.num_heads {
            let start = head_index * head_dim;
            let end = start + head_dim;

            let head_out = self.heads[head_index]
                .try_forward_last_range(past_input, query, start, end, training)
                .map_err(MultiHeadAttentionError::HeadError)?;
            output[start..end].copy_from_slice(&head_out);
        }

        Ok(output)
    }

    /// Strict incremental forward pass for a contiguous cached sequence.
    ///
    /// `past_input_flat` is row-major `[past_len][embedding_dim]`.
    pub fn try_forward_last_flat(
        &self,
        past_input_flat: &[f32],
        past_len: usize,
        embedding_dim: usize,
        query: &[f32],
        training: bool,
    ) -> Result<Vec<f32>, MultiHeadAttentionError> {
        if self.num_heads == 0 || self.heads.is_empty() {
            return Err(MultiHeadAttentionError::ZeroHeads);
        }
        if query.len() != embedding_dim || embedding_dim == 0 {
            return Err(MultiHeadAttentionError::RaggedInput);
        }
        if past_input_flat.len() != past_len.saturating_mul(embedding_dim) {
            return Err(MultiHeadAttentionError::RaggedInput);
        }
        if !embedding_dim.is_multiple_of(self.num_heads) {
            return Err(MultiHeadAttentionError::EmbeddingDimNotDivisible {
                embedding_dim,
                num_heads: self.num_heads,
            });
        }

        let head_dim = embedding_dim / self.num_heads;
        let mut output = vec![0.0_f32; embedding_dim];
        let mut weights = Vec::with_capacity(past_len + 1);
        for head_index in 0..self.num_heads {
            let start = head_index * head_dim;
            let end = start + head_dim;
            self.heads[head_index]
                .try_forward_last_range_flat_into(
                    past_input_flat,
                    past_len,
                    embedding_dim,
                    query,
                    start,
                    end,
                    training,
                    &mut weights,
                    &mut output[start..end],
                )
                .map_err(MultiHeadAttentionError::HeadError)?;
        }

        Ok(output)
    }

    /// Ergonomic batch forward pass.
    ///
    /// Input shape: `[batch_size][seq_len][embedding_dim]`
    /// Output shape: `[batch_size][seq_len][embedding_dim]`
    ///
    /// Returns empty output on invalid input. For strict errors, use `try_forward_batch`.
    pub fn forward_batch(&self, input: &[Vec<Vec<f32>>], training: bool) -> Vec<Vec<Vec<f32>>> {
        self.try_forward_batch(input, training).unwrap_or_default()
    }

    /// Strict batch forward pass with explicit error propagation.
    pub fn try_forward_batch(
        &self,
        input: &[Vec<Vec<f32>>],
        training: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>, MultiHeadAttentionError> {
        if input.is_empty() {
            return Err(MultiHeadAttentionError::EmptyInput);
        }

        let mut batch_out = Vec::with_capacity(input.len());
        for seq in input {
            batch_out.push(self.try_forward(seq, training)?);
        }

        Ok(batch_out)
    }
}
