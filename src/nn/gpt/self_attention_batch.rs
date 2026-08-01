use super::attention_weights::compute_attention_weights_flat;
use super::self_attention::{SelfAttention, SelfAttentionError};
use rand::Rng;

/// Backend metadata for batched self-attention execution.
///
/// This keeps the API aligned with potential future GPU execution without
/// requiring device kernels today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelfAttentionBatchBackend {
    Cpu,
    Gpu { device_id: usize },
}

/// Batched extension methods for parameter-free self-attention.
///
/// This implementation processes each sequence independently in the batch,
/// reusing flattened attention weights for better memory locality.
impl SelfAttention {
    /// Compute self-attention outputs for a batch of sequences.
    ///
    /// Input shape: `[batch_size][seq_len][embedding_dim]`
    /// Output shape: `[batch_size][seq_len][embedding_dim]`
    ///
    /// This ergonomic API returns an empty output if any sequence is invalid.
    /// For explicit error handling, use `try_forward_batch`.
    ///
    /// Dropout is only applied when `training == true`.
    pub fn forward_batch(&self, input: &[Vec<Vec<f32>>], training: bool) -> Vec<Vec<Vec<f32>>> {
        let mut rng = rand::rng();
        self.try_forward_batch_with_rng(input, training, &mut rng)
            .unwrap_or_default()
    }

    /// Compute batched self-attention outputs with backend metadata.
    ///
    /// Currently both variants execute the CPU path; GPU is reserved for future
    /// integration.
    pub fn forward_batch_with_backend(
        &self,
        input: &[Vec<Vec<f32>>],
        training: bool,
        _backend: SelfAttentionBatchBackend,
    ) -> Vec<Vec<Vec<f32>>> {
        self.forward_batch(input, training)
    }

    /// Strict batched self-attention forward pass with explicit error reporting.
    ///
    /// - Validates each sequence independently.
    /// - Uses flattened row-major attention weights per sequence to improve memory efficiency.
    /// - Applies causal masking per sequence when `self.causal()` is enabled.
    pub fn try_forward_batch(
        &self,
        input: &[Vec<Vec<f32>>],
        training: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>, SelfAttentionError> {
        let mut rng = rand::rng();
        self.try_forward_batch_with_rng(input, training, &mut rng)
    }

    /// Strict batched self-attention with explicit RNG injection for reproducibility.
    pub fn try_forward_batch_with_rng<R: Rng + ?Sized>(
        &self,
        input: &[Vec<Vec<f32>>],
        training: bool,
        rng: &mut R,
    ) -> Result<Vec<Vec<Vec<f32>>>, SelfAttentionError> {
        let mut weight_workspace = Vec::new();
        self.try_forward_batch_with_workspace_and_rng(input, training, &mut weight_workspace, rng)
    }

    /// Strict batched forward pass that reuses a caller-owned flattened weight buffer.
    ///
    /// This minimizes allocation overhead for large batches by storing all
    /// sequence attention matrices in one contiguous row-major workspace.
    pub fn try_forward_batch_with_workspace(
        &self,
        input: &[Vec<Vec<f32>>],
        training: bool,
        weight_workspace: &mut Vec<f32>,
    ) -> Result<Vec<Vec<Vec<f32>>>, SelfAttentionError> {
        let mut rng = rand::rng();
        self.try_forward_batch_with_workspace_and_rng(input, training, weight_workspace, &mut rng)
    }

    /// Strict batched self-attention with caller workspace + RNG injection.
    pub fn try_forward_batch_with_workspace_and_rng<R: Rng + ?Sized>(
        &self,
        input: &[Vec<Vec<f32>>],
        training: bool,
        weight_workspace: &mut Vec<f32>,
        rng: &mut R,
    ) -> Result<Vec<Vec<Vec<f32>>>, SelfAttentionError> {
        if input.is_empty() {
            return Err(SelfAttentionError::EmptyInput);
        }

        let mut seq_meta: Vec<(usize, usize, usize)> = Vec::with_capacity(input.len());
        // (weight_offset, seq_len, embedding_dim)

        weight_workspace.clear();

        for sequence in input {
            if sequence.is_empty() {
                return Err(SelfAttentionError::EmptyInput);
            }

            let seq_len = sequence.len();
            let embedding_dim = sequence[0].len();
            if embedding_dim == 0 {
                return Err(SelfAttentionError::ZeroEmbeddingDim);
            }
            if sequence.iter().any(|row| row.len() != embedding_dim) {
                return Err(SelfAttentionError::RaggedInput);
            }

            // Flattened row-major attention matrix:
            // weights[i * seq_len + j].
            // This avoids nested matrix allocations and improves cache locality.
            let (mut weights, n) = compute_attention_weights_flat(sequence);
            if n != seq_len {
                return Err(SelfAttentionError::RaggedInput);
            }

            if self.causal() {
                // Causal masking is applied independently per sequence:
                // for each row i, set all future columns j > i to zero,
                // then re-normalize the row to preserve probability mass.
                apply_causal_mask_in_place(&mut weights, seq_len);
            }

            // Apply dropout on attention probabilities after softmax/mask and
            // before the weighted sum.
            self.apply_dropout_in_place_with_rng(&mut weights, training, rng)?;

            let offset = weight_workspace.len();
            weight_workspace.extend_from_slice(&weights);
            seq_meta.push((offset, seq_len, embedding_dim));
        }

        let mut batch_output: Vec<Vec<Vec<f32>>> = Vec::with_capacity(input.len());

        for (batch_idx, sequence) in input.iter().enumerate() {
            let (offset, seq_len, embedding_dim) = seq_meta[batch_idx];
            let weights = &weight_workspace[offset..offset + seq_len * seq_len];

            // Output[i][d] = sum_j weights[i,j] * sequence[j][d]
            let mut sequence_output = vec![vec![0.0_f32; embedding_dim]; seq_len];
            for i in 0..seq_len {
                let row_offset = i * seq_len;
                let out_row = &mut sequence_output[i];

                for j in 0..seq_len {
                    let w = weights[row_offset + j];
                    if w == 0.0 {
                        continue;
                    }

                    let src = &sequence[j];
                    for d in 0..embedding_dim {
                        out_row[d] += w * src[d];
                    }
                }
            }

            batch_output.push(sequence_output);
        }

        Ok(batch_output)
    }
}

fn apply_causal_mask_in_place(weights: &mut [f32], seq_len: usize) {
    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row_end = row_start + seq_len;
        let row = &mut weights[row_start..row_end];

        for j in (i + 1)..seq_len {
            row[j] = 0.0;
        }

        let sum: f32 = row.iter().copied().sum();
        if sum > 0.0 {
            for v in row {
                *v /= sum;
            }
        }
    }
}
