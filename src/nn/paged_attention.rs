use crate::nn::paged_kv_cache::PagedKVCache;
use crate::tensor::Tensor;
use ndarray::Axis;
use rayon::prelude::*;

/// Compute attention using Paged KV Cache.
///
/// Use Blockwise Attention to avoid materializing the full K/V tensors.
///
/// # Arguments
/// * `query` - [batch_size, num_heads, head_dim] (for decoding, seq_len=1 usually)
/// * `cache` - The PagedKVCache containing K/V blocks.
/// * `seq_ids` - The sequence IDs corresponding to each batch index.
/// * `scale` - Softmax scaling factor (1 / sqrt(head_dim)).
/// * `num_heads` - Number of attention heads.
/// * `head_dim` - Dimension of each head.
pub fn paged_attention(
    query: &Tensor,
    cache: &PagedKVCache,
    seq_ids: &[u64],
    scale: f32,
    num_heads: usize,
    head_dim: usize,
) -> Tensor {
    // 1. Prepare inputs
    // We assume query is [batch, num_heads, head_dim].
    // If it's [batch, 1, num_heads, head_dim], we squeeze it.
    let q_in = query.to_f32_array();
    let q_shape = q_in.shape();

    // Validate shape
    // Expected: [batch, num_heads, head_dim]
    // If [batch, seq=1, num_heads, head_dim], handle it.
    let (batch_size, q_len) = if q_shape.len() == 4 {
        (q_shape[0], q_shape[1])
    } else {
        (q_shape[0], 1)
    };

    if q_len != 1 {
        log::warn!(
            "paged_attention: q_len != 1 ({}), specialized for decoding only.",
            q_len
        );
    }

    // 2. Parallel processing per sequence (batch)
    // Output: [batch, num_heads, head_dim]
    let mut output_arr = ndarray::Array3::<f32>::zeros((batch_size, num_heads, head_dim));

    // We can parallelize over batch dimension
    let chunk_size = num_heads * head_dim;

    if let Some(slice) = output_arr.as_slice_mut() {
        slice
            .par_chunks_mut(chunk_size)
            .zip(seq_ids.par_iter())
            .enumerate()
            .for_each(|(batch_idx, (out_flat, &seq_id))| {
                let mut out_slice =
                    ndarray::ArrayViewMut2::from_shape((num_heads, head_dim), out_flat).expect("paged_attn");

                // Get query for this batch: [num_heads, head_dim]
                let q_batch = q_in.index_axis(Axis(0), batch_idx);
                let q_head = if q_shape.len() == 4 {
                    q_batch.index_axis(Axis(0), 0)
                } else {
                    q_batch
                };

                // Retrieve metadata
                let seqs = cache.sequences.lock().expect("paged_attn");
                let meta = if let Some(m) = seqs.get(&seq_id) {
                    m
                } else {
                    return; // Early exit if seq not found
                };

                let context_len = meta.context_len;
                let block_size = cache.config.block_size;
                let num_logical_blocks = (context_len + block_size - 1) / block_size;

                // Collect physical block IDs
                let phys_block_ids: Vec<usize> = (0..num_logical_blocks)
                    .map(|i| *meta.logical_to_physical.get(&i).unwrap())
                    .collect();
                drop(seqs);

                // SCORE PHASE
                let mut scores = ndarray::Array2::<f32>::zeros((num_heads, context_len));
                let engine = cache.engine.lock().expect("paged_attn");

                for (i, &phys_idx) in phys_block_ids.iter().enumerate() {
                    let block = engine.used_blocks.get(&phys_idx).expect("paged_attn");
                    let block_data_lock = block.data.lock();
                    let block_arr = block_data_lock.storage.to_f32_array();
                    // Shape: [2, num_heads, block_size, head_dim]

                    let k_block = block_arr.slice(ndarray::s![0, .., .., ..]);

                    let match_len = if i == num_logical_blocks - 1 {
                        (context_len - 1) % block_size + 1
                    } else {
                        block_size
                    };

                    for h in 0..num_heads {
                        let q_h = q_head.index_axis(Axis(0), h);
                        let k_b_h = k_block.slice(ndarray::s![h, 0..match_len, ..]);

                        // Manual score computation to avoid recursion
                        let mut score_chunk = ndarray::Array1::<f32>::zeros(match_len);
                        for pos in 0..match_len {
                            let k_vec = k_b_h.slice(ndarray::s![pos, ..]);
                            // Manual dot product
                            let mut val = 0.0;
                            for d in 0..head_dim {
                                val += q_h[d] * k_vec[d];
                            }
                            score_chunk[pos] = val;
                        }

                        let start_pos = i * block_size;
                        let mut dest_slice =
                            scores.slice_mut(ndarray::s![h, start_pos..start_pos + match_len]);
                        dest_slice.assign(&(&score_chunk * scale));
                    }
                }
                drop(engine);

                // Softmax per head
                for h in 0..num_heads {
                    let mut head_scores = scores.slice_mut(ndarray::s![h, ..]);
                    let max_val = head_scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_sum = head_scores
                        .map_mut(|x| {
                            *x = (*x - max_val).exp();
                            *x
                        })
                        .sum();
                    head_scores.map_mut(|x| *x /= exp_sum);
                }

                // ACCUMULATE PHASE
                let engine = cache.engine.lock().expect("paged_attn");

                for (i, &phys_idx) in phys_block_ids.iter().enumerate() {
                    let block = engine.used_blocks.get(&phys_idx).expect("paged_attn");
                    let block_data_lock = block.data.lock();
                    let block_arr = block_data_lock.storage.to_f32_array();
                    let v_block = block_arr.slice(ndarray::s![1, .., .., ..]);

                    let match_len = if i == num_logical_blocks - 1 {
                        (context_len - 1) % block_size + 1
                    } else {
                        block_size
                    };

                    for h in 0..num_heads {
                        let prob_chunk = scores
                            .slice(ndarray::s![h, i * block_size..i * block_size + match_len]);
                        let v_b_h = v_block.slice(ndarray::s![h, 0..match_len, ..]);

                        // Manual accumulation to avoid recursion
                        let mut weighted_v = ndarray::Array1::<f32>::zeros(head_dim);
                        for pos in 0..match_len {
                            let prob = prob_chunk[pos];
                            let v_vec = v_b_h.slice(ndarray::s![pos, ..]);
                            // weighted_v += prob * v_vec
                            for d in 0..head_dim {
                                weighted_v[d] += prob * v_vec[d];
                            }
                        }

                        let mut out_h = out_slice.index_axis_mut(Axis(0), h);
                        out_h.zip_mut_with(&weighted_v, |a, &b| *a += b);
                    }
                }
            });
    }

    Tensor::new(output_arr.into_dyn(), false)
}
