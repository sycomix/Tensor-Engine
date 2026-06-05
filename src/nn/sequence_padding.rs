//! Sequence padding and masking utilities.
//!
//! Provides operations for padding sequences to uniform length and creating
//! attention masks for transformer models.

use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// Padding mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PaddingMode {
    /// Pad to the end of the sequence
    Post,
    /// Pad to the beginning of the sequence
    Pre,
}

/// Sequence padding configuration.
#[derive(Clone, Debug)]
pub struct PadConfig {
    /// Padding mode (pre or post)
    pub mode: PaddingMode,
    /// Padding value (typically 0)
    pub pad_value: f32,
    /// Target length (if None, pad to max in batch)
    pub target_length: Option<usize>,
    /// Whether to truncate sequences that exceed target length
    pub truncate: bool,
}

impl Default for PadConfig {
    fn default() -> Self {
        PadConfig {
            mode: PaddingMode::Post,
            pad_value: 0.0,
            target_length: None,
            truncate: false,
        }
    }
}

/// Pad a batch of sequences to uniform length.
///
/// # Arguments
/// * `sequences` - Vector of 1D tensors representing sequences
/// * `config` - Padding configuration
///
/// # Returns
/// Padded tensor of shape [batch_size, max_seq_len]
pub fn pad_sequences(sequences: &[Tensor], config: &PadConfig) -> Tensor {
    if sequences.is_empty() {
        return Tensor::zeros(&[0, 0]);
    }

    // Determine target length
    let target_len = if let Some(len) = config.target_length {
        len
    } else {
        sequences
            .iter()
            .map(|s| s.lock().storage.shape()[0])
            .max()
            .unwrap_or(0)
    };

    let batch_size = sequences.len();
    let mut padded = ArrayD::<f32>::zeros(IxDyn(&[batch_size, target_len][..]));

    for (batch_idx, seq) in sequences.iter().enumerate() {
        let seq_data = seq.lock().storage.to_f32_array();
        let seq_len = seq_data.len();

        if config.truncate && seq_len > target_len {
            // Truncate to target length
            for i in 0..target_len {
                padded[[batch_idx, i]] = seq_data[i];
            }
        } else {
            match config.mode {
                PaddingMode::Post => {
                    // Pad at the end
                    for i in 0..seq_len.min(target_len) {
                        padded[[batch_idx, i]] = seq_data[i];
                    }
                }
                PaddingMode::Pre => {
                    // Pad at the beginning
                    let offset = target_len.saturating_sub(seq_len);
                    for i in 0..seq_len {
                        padded[[batch_idx, i + offset]] = seq_data[i];
                    }
                }
            }
        }
    }

    Tensor::new(padded, false)
}

/// Create an attention mask for padded sequences.
///
/// # Arguments
/// * `padded_sequences` - Padded tensor of shape [batch_size, seq_len]
/// * `pad_value` - The padding value (typically 0)
///
/// # Returns
/// Attention mask of shape [batch_size, seq_len] with 1 for real tokens, 0 for padding
pub fn create_attention_mask(padded_sequences: &Tensor, pad_value: f32) -> Tensor {
    let data = padded_sequences.lock().storage.to_f32_array();
    let shape = data.shape().to_vec();

    let (batch_size, seq_len) = (shape[0], shape[1]);
    let mut mask = ArrayD::<f32>::zeros(IxDyn(&[batch_size, seq_len][..]));

    for b in 0..batch_size {
        for t in 0..seq_len {
            if data[[b, t]] != pad_value {
                mask[[b, t]] = 1.0;
            }
        }
    }

    Tensor::new(mask, false)
}

/// Create a causal (causality) mask for self-attention.
///
/// # Arguments
/// * `seq_len` - Sequence length
///
/// # Returns
/// Causal mask of shape [seq_len, seq_len] where mask[i,j] = 1 if j <= i, else 0
pub fn create_causal_mask(seq_len: usize) -> Tensor {
    let mut mask = ArrayD::<f32>::zeros(IxDyn(&[seq_len, seq_len][..]));

    for i in 0..seq_len {
        for j in 0..seq_len {
            if j <= i {
                mask[[i, j]] = 1.0;
            }
        }
    }

    Tensor::new(mask, false)
}

/// Create a combined attention mask (padding + causal).
///
/// # Arguments
/// * `padded_sequences` - Padded tensor of shape [batch_size, seq_len]
/// * `pad_value` - The padding value
///
/// # Returns
/// Combined mask of shape [batch_size, seq_len, seq_len]
pub fn create_combined_mask(padded_sequences: &Tensor, pad_value: f32) -> Tensor {
    let data = padded_sequences.lock().storage.to_f32_array();
    let shape = data.shape().to_vec();

    let (batch_size, seq_len) = (shape[0], shape[1]);
    let mut mask = ArrayD::<f32>::zeros(IxDyn(&[batch_size, seq_len, seq_len][..]));

    for b in 0..batch_size {
        for i in 0..seq_len {
            for j in 0..seq_len {
                // Causal mask: j <= i
                let causal = if j <= i { 1.0 } else { f32::NEG_INFINITY };
                // Padding mask: token is not padding
                let padding = if data[[b, i]] != pad_value && data[[b, j]] != pad_value {
                    1.0
                } else {
                    f32::NEG_INFINITY
                };

                // Combined: minimum of causal and padding (both must be valid)
                mask[[b, i, j]] = causal.min(padding);
            }
        }
    }

    Tensor::new(mask, false)
}

/// Pad a batch of 2D tensors (e.g., [batch, seq, features]).
///
/// # Arguments
/// * `sequences` - Vector of 2D tensors
/// * `config` - Padding configuration
///
/// # Returns
/// Padded tensor of shape [batch_size, max_seq_len, features]
pub fn pad_2d_sequences(sequences: &[Tensor], config: &PadConfig) -> Tensor {
    if sequences.is_empty() {
        return Tensor::zeros(&[0, 0, 0]);
    }

    // Determine target length
    let target_len = if let Some(len) = config.target_length {
        len
    } else {
        sequences
            .iter()
            .map(|s| s.lock().storage.shape()[0])
            .max()
            .unwrap_or(0)
    };

    // Get feature dimension from first sequence
    let features = sequences[0]
        .lock()
        .storage
        .shape()
        .get(1)
        .copied()
        .unwrap_or(0);

    let batch_size = sequences.len();
    let mut padded = ArrayD::<f32>::zeros(IxDyn(&[batch_size, target_len, features][..]));

    for (batch_idx, seq) in sequences.iter().enumerate() {
        let seq_data = seq.lock().storage.to_f32_array();
        let seq_len = seq_data.shape()[0];

        if config.truncate && seq_len > target_len {
            for i in 0..target_len {
                for f in 0..features {
                    padded[[batch_idx, i, f]] = seq_data[[i, f]];
                }
            }
        } else {
            match config.mode {
                PaddingMode::Post => {
                    for i in 0..seq_len.min(target_len) {
                        for f in 0..features {
                            padded[[batch_idx, i, f]] = seq_data[[i, f]];
                        }
                    }
                }
                PaddingMode::Pre => {
                    let offset = target_len.saturating_sub(seq_len);
                    for i in 0..seq_len {
                        for f in 0..features {
                            padded[[batch_idx, i + offset, f]] = seq_data[[i, f]];
                        }
                    }
                }
            }
        }
    }

    Tensor::new(padded, false)
}

/// Create a key padding mask for variable-length sequences.
///
/// # Arguments
/// * `sequences` - Vector of 1D tensors
/// * `pad_value` - The padding value
///
/// # Returns
/// Mask of shape [batch_size, max_seq_len] with 0 for real tokens, 1 for padding
pub fn create_key_padding_mask(sequences: &[Tensor], pad_value: f32) -> Tensor {
    let max_len = sequences
        .iter()
        .map(|s| s.lock().storage.shape()[0])
        .max()
        .unwrap_or(0);

    let batch_size = sequences.len();
    let mut mask = ArrayD::<f32>::zeros(IxDyn(&[batch_size, max_len][..]));

    for (batch_idx, seq) in sequences.iter().enumerate() {
        let seq_data = seq.lock().storage.to_f32_array();
        for t in 0..max_len {
            if t < seq_data.len() && seq_data[t] == pad_value {
                mask[[batch_idx, t]] = 1.0;
            }
        }
    }

    Tensor::new(mask, false)
}

/// Pad and mask a batch of sequences for transformer input.
///
/// # Arguments
/// * `sequences` - Vector of 1D tensors
/// * `config` - Padding configuration
///
/// # Returns
/// Tuple of (padded_sequences, attention_mask, causal_mask)
pub fn pad_and_mask(sequences: &[Tensor], config: &PadConfig) -> (Tensor, Tensor, Tensor) {
    let padded = pad_sequences(sequences, config);
    let pad_value = config.pad_value;
    let attn_mask = create_attention_mask(&padded, pad_value);
    let seq_len = padded.lock().storage.shape()[1];
    let causal_mask = create_causal_mask(seq_len);

    (padded, attn_mask, causal_mask)
}

#[cfg(test)]
mod sequence_padding_tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_pad_sequences_post() {
        let seq1 = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
            false,
        );
        let seq2 = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[5]), vec![4.0, 5.0, 6.0, 7.0, 8.0]).unwrap(),
            false,
        );

        let config = PadConfig {
            mode: PaddingMode::Post,
            ..PadConfig::default()
        };

        let padded = pad_sequences(&[seq1, seq2], &config);
        let data = padded.lock().storage.to_f32_array();

        assert_eq!(data[[0, 0]], 1.0);
        assert_eq!(data[[0, 1]], 2.0);
        assert_eq!(data[[0, 2]], 3.0);
        assert_eq!(data[[0, 3]], 0.0); // padding
        assert_eq!(data[[0, 4]], 0.0); // padding
        assert_eq!(data[[1, 0]], 4.0);
        assert_eq!(data[[1, 4]], 8.0);
    }

    #[test]
    fn test_pad_sequences_pre() {
        let seq1 = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
            false,
        );
        let seq2 = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[5]), vec![4.0, 5.0, 6.0, 7.0, 8.0]).unwrap(),
            false,
        );

        let config = PadConfig {
            mode: PaddingMode::Pre,
            ..PadConfig::default()
        };

        let padded = pad_sequences(&[seq1, seq2], &config);
        let data = padded.lock().storage.to_f32_array();

        assert_eq!(data[[0, 0]], 0.0); // padding
        assert_eq!(data[[0, 1]], 0.0); // padding
        assert_eq!(data[[0, 2]], 1.0);
        assert_eq!(data[[0, 3]], 2.0);
        assert_eq!(data[[0, 4]], 3.0);
    }

    #[test]
    fn test_create_attention_mask() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0];
        let padded = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[2, 5]), data).unwrap(), false);

        let mask = create_attention_mask(&padded, 0.0);
        let mask_data = mask.lock().storage.to_f32_array();

        assert_eq!(mask_data[[0, 0]], 1.0);
        assert_eq!(mask_data[[0, 1]], 1.0);
        assert_eq!(mask_data[[0, 2]], 1.0);
        assert_eq!(mask_data[[0, 3]], 0.0); // padding
        assert_eq!(mask_data[[0, 4]], 0.0); // padding
        assert_eq!(mask_data[[1, 0]], 1.0);
        assert_eq!(mask_data[[1, 1]], 1.0);
        assert_eq!(mask_data[[1, 2]], 0.0); // padding
    }

    #[test]
    fn test_create_causal_mask() {
        let mask = create_causal_mask(4);
        let data = mask.lock().storage.to_f32_array();

        // Lower triangular matrix
        assert_eq!(data[[0, 0]], 1.0);
        assert_eq!(data[[0, 1]], 0.0);
        assert_eq!(data[[0, 2]], 0.0);
        assert_eq!(data[[0, 3]], 0.0);

        assert_eq!(data[[1, 0]], 1.0);
        assert_eq!(data[[1, 1]], 1.0);
        assert_eq!(data[[1, 2]], 0.0);
        assert_eq!(data[[1, 3]], 0.0);

        assert_eq!(data[[2, 0]], 1.0);
        assert_eq!(data[[2, 1]], 1.0);
        assert_eq!(data[[2, 2]], 1.0);
        assert_eq!(data[[2, 3]], 0.0);

        assert_eq!(data[[3, 0]], 1.0);
        assert_eq!(data[[3, 1]], 1.0);
        assert_eq!(data[[3, 2]], 1.0);
        assert_eq!(data[[3, 3]], 1.0);
    }

    #[test]
    fn test_pad_and_mask() {
        let seq1 = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
            false,
        );
        let seq2 = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[5]), vec![4.0, 5.0, 6.0, 7.0, 8.0]).unwrap(),
            false,
        );

        let config = PadConfig::default();
        let (padded, attn_mask, causal_mask) = pad_and_mask(&[seq1, seq2], &config);

        let padded_data = padded.lock().storage.to_f32_array();
        assert_eq!(padded_data.shape(), &[2, 5]);

        let attn_data = attn_mask.lock().storage.to_f32_array();
        assert_eq!(attn_data.shape(), &[2, 5]);

        let causal_data = causal_mask.lock().storage.to_f32_array();
        assert_eq!(causal_data.shape(), &[5, 5]);
    }

    #[test]
    fn test_pad_2d_sequences() {
        let seq1 = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap(),
            false,
        );
        let seq2 = Tensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[3, 4]),
                vec![
                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                ],
            )
            .unwrap(),
            false,
        );

        let config = PadConfig::default();
        let padded = pad_2d_sequences(&[seq1, seq2], &config);
        let data = padded.lock().storage.to_f32_array();

        assert_eq!(data.shape(), &[2, 3, 4]);
        assert_eq!(data[[0, 0, 0]], 1.0);
        assert_eq!(data[[0, 1, 3]], 4.0);
        assert_eq!(data[[1, 2, 3]], 20.0);
    }

    #[test]
    fn test_create_key_padding_mask() {
        let seq1 = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 0.0]).unwrap(),
            false,
        );
        let seq2 = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[5]), vec![4.0, 5.0, 6.0, 0.0, 0.0]).unwrap(),
            false,
        );

        let mask = create_key_padding_mask(&[seq1, seq2], 0.0);
        let data = mask.lock().storage.to_f32_array();

        assert_eq!(data[[0, 0]], 0.0);
        assert_eq!(data[[0, 1]], 0.0);
        assert_eq!(data[[0, 2]], 1.0); // padding
        assert_eq!(data[[1, 0]], 0.0);
        assert_eq!(data[[1, 1]], 0.0);
        assert_eq!(data[[1, 2]], 0.0);
        assert_eq!(data[[1, 3]], 1.0); // padding
        assert_eq!(data[[1, 4]], 1.0); // padding
    }

    #[test]
    fn test_pad_with_truncation() {
        let seq1 = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[10]), (0..10).map(|i| i as f32).collect()).unwrap(),
            false,
        );
        let seq2 = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[5]), (10..15).map(|i| i as f32).collect()).unwrap(),
            false,
        );

        let config = PadConfig {
            truncate: true,
            target_length: Some(7),
            ..PadConfig::default()
        };

        let padded = pad_sequences(&[seq1, seq2], &config);
        let data = padded.lock().storage.to_f32_array();

        assert_eq!(data.shape(), &[2, 7]);
        // seq1 truncated to first 7 elements
        assert_eq!(data[[0, 0]], 0.0);
        assert_eq!(data[[0, 6]], 6.0);
        // seq2 padded to 7
        assert_eq!(data[[1, 0]], 10.0);
        assert_eq!(data[[1, 4]], 14.0);
        assert_eq!(data[[1, 5]], 0.0); // padding
        assert_eq!(data[[1, 6]], 0.0); // padding
    }
}
