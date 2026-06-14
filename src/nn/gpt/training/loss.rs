use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum CrossEntropyError {
	EmptyInput,
	EmptySequenceInBatch { index: usize },
	RaggedBatchLogits {
		index: usize,
		expected_len: usize,
		found_len: usize,
	},
	RaggedBatchTargets {
		index: usize,
		expected_len: usize,
		found_len: usize,
	},
	LengthMismatch { logits_len: usize, targets_len: usize },
	EmptyVocab { step: usize },
	TargetOutOfRange {
		step: usize,
		target: u32,
		vocab_size: usize,
	},
	NonFiniteLogit { step: usize, class: usize },
}

impl Display for CrossEntropyError {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		match self {
			CrossEntropyError::EmptyInput => write!(f, "inputs must be non-empty"),
			CrossEntropyError::EmptySequenceInBatch { index } => {
				write!(f, "empty sequence in batch at index {}", index)
			}
			CrossEntropyError::RaggedBatchLogits {
				index,
				expected_len,
				found_len,
			} => write!(
				f,
				"ragged batch logits at index {}: expected seq_len {}, found {}",
				index, expected_len, found_len
			),
			CrossEntropyError::RaggedBatchTargets {
				index,
				expected_len,
				found_len,
			} => write!(
				f,
				"ragged batch targets at index {}: expected seq_len {}, found {}",
				index, expected_len, found_len
			),
			CrossEntropyError::LengthMismatch {
				logits_len,
				targets_len,
			} => write!(
				f,
				"length mismatch: logits_len={} targets_len={}",
				logits_len, targets_len
			),
			CrossEntropyError::EmptyVocab { step } => {
				write!(f, "empty vocabulary row at step {}", step)
			}
			CrossEntropyError::TargetOutOfRange {
				step,
				target,
				vocab_size,
			} => write!(
				f,
				"target {} out of range for vocab size {} at step {}",
				target, vocab_size, step
			),
			CrossEntropyError::NonFiniteLogit { step, class } => {
				write!(f, "non-finite logit at step {}, class {}", step, class)
			}
		}
	}
}

impl Error for CrossEntropyError {}

/// Numerically stable mean cross-entropy for next-token prediction.
///
/// - `logits` shape: `[time][vocab]`
/// - `targets` shape: `[time]`, where each entry is the correct token index.
///
/// Uses the log-sum-exp trick:
/// `NLL = log(sum_j exp(logit_j)) - logit_target`
/// with `log(sum_j exp(logit_j)) = m + log(sum_j exp(logit_j - m))`, `m=max(logits)`.
pub fn try_next_token_cross_entropy(
	logits: &[Vec<f32>],
	targets: &[u32],
) -> Result<f32, CrossEntropyError> {
	if logits.is_empty() || targets.is_empty() {
		return Err(CrossEntropyError::EmptyInput);
	}
	if logits.len() != targets.len() {
		return Err(CrossEntropyError::LengthMismatch {
			logits_len: logits.len(),
			targets_len: targets.len(),
		});
	}

	let mut total_loss = 0.0_f32;

	for (step, (row, &target)) in logits.iter().zip(targets.iter()).enumerate() {
		if row.is_empty() {
			return Err(CrossEntropyError::EmptyVocab { step });
		}

		let mut max_logit = f32::NEG_INFINITY;
		for (class, &v) in row.iter().enumerate() {
			if !v.is_finite() {
				return Err(CrossEntropyError::NonFiniteLogit { step, class });
			}
			if v > max_logit {
				max_logit = v;
			}
		}

		let vocab_size = row.len();
		let target_idx = target as usize;
		if target_idx >= vocab_size {
			return Err(CrossEntropyError::TargetOutOfRange {
				step,
				target,
				vocab_size,
			});
		}

		let sum_exp: f32 = row.iter().map(|&v| (v - max_logit).exp()).sum();

		let log_denom = max_logit + sum_exp.ln();
		let nll = log_denom - row[target_idx];
		total_loss += nll;
	}

	Ok(total_loss / logits.len() as f32)
}

/// Ergonomic wrapper that returns `f32::NAN` on invalid input.
pub fn next_token_cross_entropy(logits: &[Vec<f32>], targets: &[u32]) -> f32 {
	match try_next_token_cross_entropy(logits, targets) {
		Ok(v) => v,
		Err(_) => f32::NAN,
	}
}

/// Numerically stable batch mean cross-entropy for next-token prediction.
///
/// - `batch_logits` shape: `[batch][time][vocab]`
/// - `batch_targets` shape: `[batch][time]`
///
/// Returns token-weighted mean loss across the whole batch:
/// `sum_i (loss_i * seq_len_i) / sum_i seq_len_i`.
pub fn try_next_token_cross_entropy_batch(
	batch_logits: &[Vec<Vec<f32>>],
	batch_targets: &[Vec<u32>],
) -> Result<f32, CrossEntropyError> {
	if batch_logits.is_empty() || batch_targets.is_empty() {
		return Err(CrossEntropyError::EmptyInput);
	}
	if batch_logits.len() != batch_targets.len() {
		return Err(CrossEntropyError::LengthMismatch {
			logits_len: batch_logits.len(),
			targets_len: batch_targets.len(),
		});
	}

	let expected_logits_seq_len = batch_logits[0].len();
	let expected_targets_seq_len = batch_targets[0].len();

	for (index, seq_logits) in batch_logits.iter().enumerate() {
		if seq_logits.len() != expected_logits_seq_len {
			return Err(CrossEntropyError::RaggedBatchLogits {
				index,
				expected_len: expected_logits_seq_len,
				found_len: seq_logits.len(),
			});
		}
	}

	for (index, seq_targets) in batch_targets.iter().enumerate() {
		if seq_targets.len() != expected_targets_seq_len {
			return Err(CrossEntropyError::RaggedBatchTargets {
				index,
				expected_len: expected_targets_seq_len,
				found_len: seq_targets.len(),
			});
		}
	}

	let mut total_weighted_loss = 0.0_f32;
	let mut total_tokens = 0usize;

	for (index, (seq_logits, seq_targets)) in batch_logits
		.iter()
		.zip(batch_targets.iter())
		.enumerate()
	{
		let seq_len = seq_logits.len();
		if seq_len == 0 || seq_targets.is_empty() {
			return Err(CrossEntropyError::EmptySequenceInBatch { index });
		}

		let seq_loss = try_next_token_cross_entropy(seq_logits, seq_targets)?;
		total_weighted_loss += seq_loss * seq_len as f32;
		total_tokens += seq_len;
	}

	Ok(total_weighted_loss / total_tokens as f32)
}

/// Ergonomic batch wrapper that returns `f32::NAN` on invalid input.
pub fn next_token_cross_entropy_batch(
	batch_logits: &[Vec<Vec<f32>>],
	batch_targets: &[Vec<u32>],
) -> f32 {
	match try_next_token_cross_entropy_batch(batch_logits, batch_targets) {
		Ok(v) => v,
		Err(_) => f32::NAN,
	}
}

#[cfg(test)]
mod tests {
	use super::{
		next_token_cross_entropy,
		try_next_token_cross_entropy,
		try_next_token_cross_entropy_batch,
		CrossEntropyError,
	};

	#[test]
	fn stable_for_large_logits() {
		let logits = vec![vec![1000.0, 1001.0, 1002.0], vec![5000.0, 4999.0, 4998.0]];
		let targets = vec![2, 0];

		let loss = try_next_token_cross_entropy(&logits, &targets).unwrap();
		assert!(loss.is_finite());
		assert!(loss >= 0.0);
	}

	#[test]
	fn invariant_to_constant_shift() {
		let logits_a = vec![vec![0.2, -0.1, 1.4], vec![2.0, -1.0, 0.3]];
		let logits_b = vec![vec![100.2, 99.9, 101.4], vec![1002.0, 999.0, 1000.3]];
		let targets = vec![2, 0];

		let la = next_token_cross_entropy(&logits_a, &targets);
		let lb = next_token_cross_entropy(&logits_b, &targets);

		let diff = (la - lb).abs();
		assert!(diff < 2e-5, "losses differ: la={} lb={} diff={}", la, lb, diff);
	}

	#[test]
	fn target_out_of_range_reports_error() {
		let logits = vec![vec![0.1, 0.2, 0.3]];
		let targets = vec![5];

		let err = try_next_token_cross_entropy(&logits, &targets).unwrap_err();
		assert_eq!(
			err,
			CrossEntropyError::TargetOutOfRange {
				step: 0,
				target: 5,
				vocab_size: 3,
			}
		);
	}

	#[test]
	fn nan_logit_reports_error() {
		let logits = vec![vec![0.1, f32::NAN, 0.3]];
		let targets = vec![0];

		let err = try_next_token_cross_entropy(&logits, &targets).unwrap_err();
		assert_eq!(err, CrossEntropyError::NonFiniteLogit { step: 0, class: 1 });
	}

	#[test]
	fn batch_ragged_sequence_lengths_report_length_mismatch() {
		let batch_logits = vec![
			vec![vec![0.1, 0.2], vec![0.3, 0.4]],
			vec![vec![0.5, 0.6], vec![0.7, 0.8], vec![0.9, 1.0]],
		];
		let batch_targets = vec![vec![1, 0], vec![1, 0]];

		let err = try_next_token_cross_entropy_batch(&batch_logits, &batch_targets).unwrap_err();
		assert_eq!(
			err,
			CrossEntropyError::RaggedBatchLogits {
				index: 1,
				expected_len: 2,
				found_len: 3,
			}
		);
	}
}
