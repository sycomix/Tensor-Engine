/// Compute parameter-free self-attention weights from input embeddings.
///
/// Given input embeddings shaped `[seq_len][embedding_dim]`, this computes:
/// 1) scaled dot-product scores:
///    `scores[i][j] = dot(input[i], input[j]) / sqrt(embedding_dim)`
/// 2) row-wise softmax over `j` to produce attention probabilities.
///
/// Returns a matrix of shape `[seq_len][seq_len]`.
pub fn compute_attention_weights(input: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let (flat, seq_len) = compute_attention_weights_flat(input);
    let mut out = vec![vec![0.0_f32; seq_len]; seq_len];

    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row_end = row_start + seq_len;
        out[i].copy_from_slice(&flat[row_start..row_end]);
    }

    out
}

/// Compute parameter-free self-attention weights and return a flattened row-major matrix.
///
/// Output shape is logically `[seq_len][seq_len]` and stored as:
/// `flat[i * seq_len + j]`.
///
/// This layout improves cache locality and is easier to pass into batched/GPU-style paths.
pub fn compute_attention_weights_flat(input: &[Vec<f32>]) -> (Vec<f32>, usize) {
    let seq_len = input.len();
    if seq_len == 0 {
        return (Vec::new(), 0);
    }

    let embedding_dim = input[0].len();
    if embedding_dim == 0 {
        return (vec![0.0_f32; seq_len * seq_len], seq_len);
    }

    for row in input {
        assert!(
            row.len() == embedding_dim,
            "all embedding vectors must have the same dimension"
        );
    }

    let scale = (embedding_dim as f32).sqrt();

    // Compute scores and softmax in-place row by row in a flattened output buffer.
    // This avoids allocating an intermediate score matrix (`Vec<Vec<f32>>`) and
    // improves locality for large sequences.
    let mut attention = vec![0.0_f32; seq_len * seq_len];

    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row_end = row_start + seq_len;
        let row = &mut attention[row_start..row_end];

        // 1) Scaled dot-product scores.
        for j in 0..seq_len {
            let dot = input[i]
                .iter()
                .zip(input[j].iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
            row[j] = dot / scale;
        }

        // 2) Row-wise softmax with numerical stabilization:
        //    softmax(x_k) = exp(x_k - max_x) / sum_j exp(x_j - max_x)
        let max_val = row
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0_f32;
        for j in 0..seq_len {
            let e = (row[j] - max_val).exp();
            row[j] = e;
            sum_exp += e;
        }

        if sum_exp > 0.0 {
            for j in 0..seq_len {
                row[j] /= sum_exp;
            }
        }
    }

    (attention, seq_len)
}

/// Compute attention weights for a batch of sequences.
///
/// Input shape: `[batch][seq_len][embedding_dim]` (seq_len may vary per batch item).
/// Output shape: `[batch][seq_len][seq_len]` as nested vectors.
pub fn compute_attention_weights_batched(batch: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<f32>>> {
    batch
        .iter()
        .map(|seq| compute_attention_weights(seq))
        .collect()
}

/// Compute flattened attention weights for a batch of sequences.
///
/// Each batch item returns `(flat, seq_len)`, where `flat` is row-major and
/// corresponds to a logical `[seq_len][seq_len]` matrix.
pub fn compute_attention_weights_flat_batched(
    batch: &[Vec<Vec<f32>>],
) -> Vec<(Vec<f32>, usize)> {
    batch
        .iter()
        .map(|seq| compute_attention_weights_flat(seq))
        .collect()
}
