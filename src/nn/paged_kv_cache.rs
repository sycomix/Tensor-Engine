//! Paged KV Cache for efficient LLM inference.
//!
//! This module implements a paged memory management system for Key-Value caches,
//! inspired by vLLM. It allows non-contiguous memory allocation for KV blocks,
//! reducing fragmentation and enabling efficient request batching.

use crate::dtype::TensorStorage;
use crate::tensor::Tensor;
use ndarray::s;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

/// Logical block index.
pub type LogicalBlockIdx = usize;
/// Physical block index.
pub type PhysicalBlockIdx = usize;

/// Configuration for the Paged KV Cache.
#[derive(Debug, Clone)]
pub struct PagedCacheConfig {
    pub block_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub dtype: crate::dtype::DType,
    pub device: String, // "cpu" or "cuda:x" etc.
}

/// A single physical block of memory storing KV pairs.
/// Shape: [num_layers, 2, num_heads, block_size, head_dim]
/// OR simpler: [2, num_heads, block_size, head_dim] per layer?
/// Usually we allocate one huge block pool and slice it.
/// For simplicity in this reference impl, `PhysicalBlock` wraps a Tensor of shape:
/// [2, num_heads, block_size, head_dim] (Key and Value)
#[derive(Clone)]
pub struct PhysicalBlock {
    pub data: Tensor,
}

impl PhysicalBlock {
    pub fn new(
        num_heads: usize,
        block_size: usize,
        head_dim: usize,
        dtype: crate::dtype::DType,
    ) -> Self {
        let shape = vec![2, num_heads, block_size, head_dim]; // 2 for K and V
                                                              // Initialize with zeros or empty
        let data =
            Tensor::new_with_dtype(ndarray::ArrayD::zeros(ndarray::IxDyn(&shape)), false, dtype);
        PhysicalBlock { data }
    }
}

/// Block Engine manages the mapping between logical and physical blocks.
pub struct BlockEngine {
    pub free_blocks: Vec<PhysicalBlockIdx>,
    pub used_blocks: HashMap<PhysicalBlockIdx, PhysicalBlock>,
    #[allow(dead_code)]
    pub block_size: usize,
    #[allow(dead_code)]
    pub num_heads: usize,
    #[allow(dead_code)]
    pub head_dim: usize,
    #[allow(dead_code)]
    pub dtype: crate::dtype::DType,
}

impl BlockEngine {
    pub fn new(config: PagedCacheConfig, num_blocks: usize) -> Self {
        let mut free_blocks = Vec::with_capacity(num_blocks);
        let mut used_blocks = HashMap::with_capacity(num_blocks);

        // Pre-allocate blocks
        for i in 0..num_blocks {
            let block = PhysicalBlock::new(
                config.num_heads,
                config.block_size,
                config.head_dim,
                config.dtype,
            );
            used_blocks.insert(i, block);
            free_blocks.push(i);
        }

        BlockEngine {
            free_blocks,
            used_blocks,
            block_size: config.block_size,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            dtype: config.dtype,
        }
    }

    pub fn allocate(&mut self) -> Option<PhysicalBlockIdx> {
        self.free_blocks.pop()
    }

    pub fn free(&mut self, idx: PhysicalBlockIdx) {
        self.free_blocks.push(idx);
    }

    pub fn get_block(&self, idx: PhysicalBlockIdx) -> Option<&PhysicalBlock> {
        self.used_blocks.get(&idx)
    }
}

/// Sequence Metadata for Paged Attention.
pub struct SequenceMetadata {
    pub seq_id: u64,
    pub logical_to_physical: HashMap<LogicalBlockIdx, PhysicalBlockIdx>,
    pub context_len: usize,
}

impl SequenceMetadata {
    pub fn new(seq_id: u64) -> Self {
        SequenceMetadata {
            seq_id,
            logical_to_physical: HashMap::new(),
            context_len: 0,
        }
    }
}

/// Paged KV Cache Manager.
pub struct PagedKVCache {
    pub engine: Mutex<BlockEngine>,
    pub sequences: Mutex<HashMap<u64, SequenceMetadata>>,
    pub config: PagedCacheConfig,
}

impl PagedKVCache {
    pub fn new(config: PagedCacheConfig, num_blocks: usize) -> Self {
        PagedKVCache {
            engine: Mutex::new(BlockEngine::new(config.clone(), num_blocks)),
            sequences: Mutex::new(HashMap::new()),
            config,
        }
    }

    pub fn add_sequence(&self, seq_id: u64) {
        let mut seqs = self.sequences.lock().unwrap();
        seqs.insert(seq_id, SequenceMetadata::new(seq_id));
    }

    pub fn remove_sequence(&self, seq_id: u64) {
        let mut seqs = self.sequences.lock().unwrap();
        if let Some(meta) = seqs.remove(&seq_id) {
            let mut engine = self.engine.lock().unwrap();
            for (_, phys_idx) in meta.logical_to_physical {
                engine.free(phys_idx);
            }
        }
    }

    /// Reshape and cache new tokens.
    ///
    /// `key`: [num_tokens, num_heads, head_dim]
    /// `value`: [num_tokens, num_heads, head_dim]
    /// `slot_mapping`: [num_tokens] - maps each token to (block_idx, offset) implicitly or explicitly?
    /// vLLM uses a slot mapping tensor. Here we can compute it if we know context_len.
    pub fn reshape_and_cache(&self, key: &Tensor, value: &Tensor, seq_id: u64) {
        // 1. Setup & Allocation (Sequential logic, fast)
        // Extract data upfront to avoid lock contention during parallel copy.
        // NOTE: This involves a copy (to_f32_array), but it allows us to use par_iter.
        // For very large batches, specific zero-copy handling (e.g. unsafe pointers) could be faster
        // but this is significantly faster than the previous element-wise loop.
        let (k_data, v_data) = { (key.to_f32_array(), value.to_f32_array()) };

        let num_tokens = k_data.shape()[0];

        // Plan the copy operations
        // We collect all block updates needed.
        let mut ops: Vec<(PhysicalBlock, usize, usize, usize)> =
            Vec::with_capacity(num_tokens / self.config.block_size + 2);
        // Tuple: (Block, src_token_idx_start, dst_block_offset, len_in_tokens)

        {
            let mut seqs = self.sequences.lock().unwrap();
            let meta = seqs.get_mut(&seq_id).expect("Sequence not found");
            let mut engine = self.engine.lock().unwrap();

            let mut current_token = 0;
            while current_token < num_tokens {
                let logical_pos = meta.context_len + current_token;
                let logical_block_idx = logical_pos / self.config.block_size;
                let block_offset = logical_pos % self.config.block_size;

                let remaining_in_block = self.config.block_size - block_offset;
                let tokens_to_copy = remaining_in_block.min(num_tokens - current_token);

                // Allocate if needed
                let phys_idx = *meta
                    .logical_to_physical
                    .entry(logical_block_idx)
                    .or_insert_with(|| engine.allocate().expect("OOM: No free blocks"));

                // Clone the block reference (cheap Arc clone) to pass to thread
                let block = engine.used_blocks.get(&phys_idx).unwrap().clone();
                ops.push((block, current_token, block_offset, tokens_to_copy));

                current_token += tokens_to_copy;
            }

            meta.context_len += num_tokens;
        } // Release locks

        // 2. Execution (Parallel)
        // Grouping by block allows us to process distinct blocks in parallel.
        // Since we write to unique offsets per batch in a paged/continuous batching setting,
        // we assume no two ops write to the same location, but they might write to the same block.
        // Locking per block handles safety. `par_iter` scales with CPU cores.
        ops.par_iter()
            .for_each(|(block, src_idx, dst_offset, len)| {
                let mut guard = block.data.lock();

                // In-place mutation of the storage to avoid costly copy-update-copy cycle
                if let TensorStorage::F32(ref mut block_arr) = guard.storage {
                    // Dimensions:
                    // k_data: [num_tokens, num_heads, head_dim]
                    // block_arr: [2, num_heads, block_size, head_dim]

                    // Source slice: [len, num_heads, head_dim]
                    let k_src = k_data.slice(s![*src_idx..*src_idx + *len, .., ..]);
                    let v_src = v_data.slice(s![*src_idx..*src_idx + *len, .., ..]);

                    // Permute source to [num_heads, len, head_dim] to match block layout
                    // (optimization: this is a metadata view operation, cheap)
                    let k_src_perm = k_src.permuted_axes([1, 0, 2]);
                    let v_src_perm = v_src.permuted_axes([1, 0, 2]);

                    // Target slice for K (index 0)
                    let mut k_dst =
                        block_arr.slice_mut(s![0, .., *dst_offset..*dst_offset + *len, ..]);
                    k_dst.assign(&k_src_perm);

                    // Target slice for V (index 1)
                    let mut v_dst =
                        block_arr.slice_mut(s![1, .., *dst_offset..*dst_offset + *len, ..]);
                    v_dst.assign(&v_src_perm);
                } else {
                    // Fallback or error for quantized/other types (future work)
                    log::warn!("reshape_and_cache: Non-F32 storage not optimized yet");
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::tensor::Tensor;

    #[test]
    fn test_block_engine_allocation() {
        let config = PagedCacheConfig {
            block_size: 16,
            num_layers: 1,
            num_heads: 4,
            head_dim: 32,
            dtype: DType::F32,
            device: "cpu".to_string(),
        };

        let mut engine = BlockEngine::new(config, 10);

        let b1 = engine.allocate();
        assert!(b1.is_some());

        let b2 = engine.allocate();
        assert!(b2.is_some());
        assert_ne!(b1, b2);

        engine.free(b1.unwrap());
        let b3 = engine.allocate();
        // It might return b1 again depending on implementation (LIFO/FIFO)
        assert!(b3.is_some());
    }

    #[test]
    fn test_paged_kv_cache_reshape_and_cache() {
        let config = PagedCacheConfig {
            block_size: 4,
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
            dtype: DType::F32,
            device: "cpu".to_string(),
        };

        // Cache with 4 blocks of size 4 -> total capacity 16 tokens
        let cache = PagedKVCache::new(config.clone(), 4);
        let seq_id = 1;

        cache.add_sequence(seq_id);

        // Create dummy K, V
        // 6 tokens, 1 head, 2 dim
        let k_data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let v_data: Vec<f32> = (0..12).map(|x| x as f32 * 10.0).collect();

        let k = Tensor::new(
            ndarray::Array::from_shape_vec(ndarray::IxDyn(&[6, 1, 2][..]), k_data.clone()).unwrap(),
            false,
        );
        let v = Tensor::new(
            ndarray::Array::from_shape_vec(ndarray::IxDyn(&[6, 1, 2][..]), v_data.clone()).unwrap(),
            false,
        );

        cache.reshape_and_cache(&k, &v, seq_id);

        // Access internal state via private fields (allowed in child mod tests)
        // Need to traverse: PagedKVCache -> engine (Mutex), sequences (Mutex)

        let seqs = cache.sequences.lock().unwrap();
        let meta = seqs.get(&seq_id).unwrap();
        assert_eq!(meta.context_len, 6);

        // 6 tokens with block_size 4:
        // Tokens 0-3 in first block (logical 0)
        // Tokens 4-5 in second block (logical 1)
        assert!(meta.logical_to_physical.contains_key(&0));
        assert!(meta.logical_to_physical.contains_key(&1));

        let phys0 = meta.logical_to_physical[&0];
        let engine = cache.engine.lock().unwrap();
        let block0 = engine.used_blocks.get(&phys0).unwrap();

        let b0_data = block0.data.to_f32_array();

        // Verify Content
        // K is at index 0, V at index 1 in block shape [2, num_heads, block_size, head_dim]
        // K[0] -> k_data[0] = 0.0 -> b0[0, 0, 0, 0]
        assert_eq!(b0_data[[0, 0, 0, 0]], 0.0);
        assert_eq!(b0_data[[0, 0, 0, 1]], 1.0);

        // K[3] -> k_data[3] -> (3*2, 3*2+1) = (6, 7) -> b0[0, 0, 3, 0]
        assert_eq!(b0_data[[0, 0, 3, 0]], 6.0);
        assert_eq!(b0_data[[0, 0, 3, 1]], 7.0);

        // V[0] -> v_data[0] = 0.0 -> b0[1, 0, 0, 0]
        assert_eq!(b0_data[[1, 0, 0, 0]], 0.0);
        // V[1] -> v_data[1] = 10.0 -> b0[1, 0, 0, 1]
        assert_eq!(b0_data[[1, 0, 0, 1]], 10.0);
    }
}
