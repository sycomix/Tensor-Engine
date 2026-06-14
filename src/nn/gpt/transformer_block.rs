use super::feed_forward::{FeedForward, FeedForwardWorkspace};
use super::layer_norm::LayerNorm;
use super::multi_head_attention::{MultiHeadAttentionError, MultiHeadCausalAttention};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Errors for strict transformer block execution.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerBlockError {
    EmptyInput,
    RaggedInput,
    AttentionFailed(MultiHeadAttentionError),
    ResidualShapeMismatch,
    Norm1Failed,
    FeedForwardFailed,
    Norm2Failed,
    EmptySequenceInBatch { index: usize },
    SequenceFailed { index: usize, source: Box<TransformerBlockError> },
}

impl Display for TransformerBlockError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformerBlockError::EmptyInput => write!(f, "input sequence is empty"),
            TransformerBlockError::RaggedInput => write!(
                f,
                "all input rows must have dimension equal to input_dim"
            ),
            TransformerBlockError::AttentionFailed(err) => {
                write!(f, "attention forward failed: {}", err)
            }
            TransformerBlockError::ResidualShapeMismatch => {
                write!(f, "residual add shape mismatch")
            }
            TransformerBlockError::Norm1Failed => write!(f, "first layer-norm failed"),
            TransformerBlockError::FeedForwardFailed => write!(f, "feed-forward forward failed"),
            TransformerBlockError::Norm2Failed => write!(f, "second layer-norm failed"),
            TransformerBlockError::EmptySequenceInBatch { index } => {
                write!(f, "batch sequence at index {} is empty", index)
            }
            TransformerBlockError::SequenceFailed { index, source } => {
                write!(f, "batch sequence at index {} failed: {}", index, source)
            }
        }
    }
}

impl Error for TransformerBlockError {}

/// Transformer block composed of:
/// - Multi-head causal attention
/// - Feed-forward network
/// - Two layer-normalization layers
///
/// Residual flow:
/// 1) `x1 = LayerNorm1(input + Attention(input))`
/// 2) `x2 = LayerNorm2(x1 + FeedForward(x1))`
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    attention: MultiHeadCausalAttention,
    feed_forward: FeedForward<f32>,
    norm1: LayerNorm<f32>,
    norm2: LayerNorm<f32>,
    input_dim: usize,
    hidden_dim: usize,
}

impl TransformerBlock {
    /// Create a new transformer block.
    ///
    /// - `input_dim`: embedding/model dimension.
    /// - `hidden_dim`: feed-forward hidden dimension.
    /// - `num_heads`: number of attention heads.
    pub fn new(input_dim: usize, hidden_dim: usize, num_heads: usize) -> Self {
        Self {
            attention: MultiHeadCausalAttention::new(num_heads, true),
            feed_forward: FeedForward::<f32>::new(input_dim, hidden_dim),
            norm1: LayerNorm::<f32>::new(input_dim, None),
            norm2: LayerNorm::<f32>::new(input_dim, None),
            input_dim,
            hidden_dim,
        }
    }

    /// Forward pass for a single sequence.
    ///
    /// Input shape: `[seq_len][input_dim]`
    /// Output shape: `[seq_len][input_dim]`
    ///
    /// Returns empty output on invalid input or shape mismatch.
    pub fn forward(&self, input: &[Vec<f32>], training: bool) -> Vec<Vec<f32>> {
        match self.try_forward(input, training) {
            Ok(v) => v,
            Err(_) => Vec::new(),
        }
    }

    /// Strict forward pass for a single sequence with explicit error propagation.
    pub fn try_forward(
        &self,
        input: &[Vec<f32>],
        training: bool,
    ) -> Result<Vec<Vec<f32>>, TransformerBlockError> {
        let mut workspace = FeedForwardWorkspace::<f32>::new(
            self.hidden_dim,
            self.input_dim,
        );
        self.try_forward_with_workspace(input, &mut workspace, training)
    }

    /// Forward pass for a batch of sequences.
    ///
    /// Input shape: `[batch][seq_len][input_dim]`
    /// Output shape: `[batch][seq_len][input_dim]`
    ///
    /// Returns empty output on invalid input or shape mismatch.
    /// Reuses one feed-forward workspace across sequences for efficiency.
    pub fn forward_batch(&self, input: &[Vec<Vec<f32>>], training: bool) -> Vec<Vec<Vec<f32>>> {
        match self.try_forward_batch(input, training) {
            Ok(v) => v,
            Err(_) => Vec::new(),
        }
    }

    /// Strict forward pass for a batch with explicit error propagation.
    ///
    /// Note: one workspace is intentionally reused across sequences for serial
    /// execution efficiency; parallel callers should use per-sequence workspaces.
    pub fn try_forward_batch(
        &self,
        input: &[Vec<Vec<f32>>],
        training: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>, TransformerBlockError> {
        if input.is_empty() {
            return Err(TransformerBlockError::EmptyInput);
        }

        let mut workspace = FeedForwardWorkspace::<f32>::new(
            self.hidden_dim,
            self.input_dim,
        );

        let mut out = Vec::with_capacity(input.len());
        for (index, seq) in input.iter().enumerate() {
            if seq.is_empty() {
                return Err(TransformerBlockError::EmptySequenceInBatch { index });
            }

            match self.try_forward_with_workspace(seq, &mut workspace, training) {
                Ok(y) => out.push(y),
                Err(source) => {
                    return Err(TransformerBlockError::SequenceFailed {
                        index,
                        source: Box::new(source),
                    })
                }
            }
        }

        Ok(out)
    }

    fn try_forward_with_workspace(
        &self,
        input: &[Vec<f32>],
        workspace: &mut FeedForwardWorkspace<f32>,
        training: bool,
    ) -> Result<Vec<Vec<f32>>, TransformerBlockError> {
        if input.is_empty() {
            return Err(TransformerBlockError::EmptyInput);
        }
        if input
            .iter()
            .any(|row| row.len() != self.input_dim)
        {
            return Err(TransformerBlockError::RaggedInput);
        }

        // Attention branch.
        let attn_out = self
            .attention
            .try_forward(input, training)
            .map_err(TransformerBlockError::AttentionFailed)?;
        if attn_out.is_empty() || !same_shape(input, &attn_out) {
            return Err(TransformerBlockError::ResidualShapeMismatch);
        }

        // Residual + norm 1: input + attention
        let residual1 = add_matrices(input, &attn_out)
            .ok_or(TransformerBlockError::ResidualShapeMismatch)?;
        let norm1_out = self.norm1.forward_batch(&residual1);
        if norm1_out.is_empty() {
            return Err(TransformerBlockError::Norm1Failed);
        }

        // Feed-forward branch.
        let ff_out = self
            .feed_forward
            .forward_into_training(&norm1_out, workspace, training);
        if ff_out.is_empty() || !same_shape(&norm1_out, &ff_out) {
            return Err(TransformerBlockError::FeedForwardFailed);
        }

        // Residual + norm 2: norm1 + feed-forward
        let residual2 = add_matrices(&norm1_out, &ff_out)
            .ok_or(TransformerBlockError::ResidualShapeMismatch)?;
        let out = self.norm2.forward_batch(&residual2);
        if out.is_empty() {
            return Err(TransformerBlockError::Norm2Failed);
        }
        Ok(out)
    }
}

fn same_shape(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| x.len() == y.len())
}

fn add_matrices(a: &[Vec<f32>], b: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
    if !same_shape(a, b) {
        return None;
    }

    let mut out = Vec::with_capacity(a.len());
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        let mut row_out = Vec::with_capacity(row_a.len());
        for (x, y) in row_a.iter().zip(row_b.iter()) {
            row_out.push(x + y);
        }
        out.push(row_out);
    }

    Some(out)
}
