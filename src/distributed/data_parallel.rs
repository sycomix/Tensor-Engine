//! Data Parallel Training
//!
//! This module provides data parallelism utilities for distributed training,
//! including batch sharding and model replication.

use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis, IxDyn};
#[allow(unused_imports)]
use std::sync::Arc;

use super::all_reduce::{AllReduce, ReduceOp};
use super::context::DistributedContext;

/// A sharded batch of data distributed across ranks.
#[derive(Clone)]
pub struct ShardedBatch {
    /// The local shard of data for this rank.
    pub local_data: Tensor,
    /// The local shard of labels for this rank.
    pub local_labels: Option<Tensor>,
    /// The original batch size before sharding.
    pub original_batch_size: usize,
    /// The rank that owns this shard.
    pub rank: usize,
    /// Total number of shards.
    pub world_size: usize,
}

impl ShardedBatch {
    /// Get the local batch size for this shard.
    pub fn local_batch_size(&self) -> usize {
        let shape = self.local_data.lock().storage.shape();
        if shape.is_empty() {
            0
        } else {
            shape[0]
        }
    }

    /// Check if this is validly sharded.
    pub fn is_valid(&self) -> bool {
        self.local_batch_size() > 0
    }
}

/// Data parallel wrapper for models.
///
/// Automatically shards input data and synchronizes gradients during training.
pub struct DataParallel<M> {
    /// The model to parallelize.
    model: M,
    /// The distributed context.
    ctx: DistributedContext,
    /// All-reduce handler for gradient synchronization.
    all_reduce: AllReduce,
    /// Whether to synchronize gradients automatically.
    sync_gradients: bool,
}

impl<M: Clone> DataParallel<M> {
    /// Create a new data parallel wrapper.
    pub fn new(model: M, ctx: DistributedContext) -> Self {
        let all_reduce = AllReduce::new(ctx.clone());
        Self {
            model,
            ctx,
            all_reduce,
            sync_gradients: true,
        }
    }

    /// Disable automatic gradient synchronization.
    pub fn no_sync(mut self) -> Self {
        self.sync_gradients = false;
        self
    }

    /// Enable automatic gradient synchronization.
    pub fn with_sync(mut self) -> Self {
        self.sync_gradients = true;
        self
    }

    /// Get a reference to the underlying model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get a mutable reference to the underlying model.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get the distributed context.
    pub fn context(&self) -> &DistributedContext {
        &self.ctx
    }

    /// Shard a batch of data across ranks.
    ///
    /// The input tensor's first dimension (batch) is split evenly.
    pub fn shard_batch(&self, data: &Tensor, labels: Option<&Tensor>) -> ShardedBatch {
        let shape = data.lock().storage.shape();
        let batch_size = if shape.is_empty() { 1 } else { shape[0] };

        let world_size = self.ctx.world_size();
        let rank = self.ctx.rank();

        // Calculate shard boundaries
        let shard_size = batch_size / world_size;
        let remainder = batch_size % world_size;

        // Distribute remainder to first ranks
        let start = rank * shard_size + rank.min(remainder);
        let local_size = shard_size + if rank < remainder { 1 } else { 0 };
        let end = start + local_size;

        log::debug!(
            "Rank {}: sharding batch {} -> [{}, {})",
            rank,
            batch_size,
            start,
            end
        );

        // Extract local shard
        let local_data = slice_tensor_batch(data, start, end);
        let local_labels = labels.map(|l| slice_tensor_batch(l, start, end));

        ShardedBatch {
            local_data,
            local_labels,
            original_batch_size: batch_size,
            rank,
            world_size,
        }
    }

    /// Synchronize gradients across all ranks using all-reduce.
    pub fn sync_gradients(&self, parameters: &[Tensor]) {
        if !self.sync_gradients || self.ctx.world_size() == 1 {
            return;
        }

        log::debug!(
            "DataParallel: synchronizing {} parameters",
            parameters.len()
        );

        self.all_reduce
            .reduce_gradients_inplace(parameters, ReduceOp::Mean);
    }

    /// Replicate a tensor to all ranks.
    ///
    /// Master broadcasts the tensor; all ranks receive identical copies.
    pub fn replicate(&self, tensor: &Tensor) -> Tensor {
        if self.ctx.world_size() == 1 {
            return tensor.clone();
        }

        // Master broadcasts, all ranks receive
        self.all_reduce
            .broadcast(Some(tensor), 0)
            .unwrap_or_else(|| tensor.clone())
    }
}

/// Slice a tensor along the batch dimension (axis 0).
fn slice_tensor_batch(tensor: &Tensor, start: usize, end: usize) -> Tensor {
    let data = tensor.lock().storage.to_f32_array();
    let shape = data.shape();

    if shape.is_empty() || start >= end || end > shape[0] {
        // Return empty tensor or clone for edge cases
        return Tensor::new(ArrayD::zeros(IxDyn(&[])), tensor.lock().requires_grad);
    }

    // Slice along first axis
    let sliced = data.slice_axis(Axis(0), ndarray::Slice::from(start..end));
    Tensor::new(sliced.to_owned(), tensor.lock().requires_grad)
}

/// Gather sharded tensors back to full batch.
#[allow(dead_code)]
pub fn gather_shards(shards: Vec<Tensor>) -> Tensor {
    if shards.is_empty() {
        return Tensor::new(ArrayD::zeros(IxDyn(&[])), false);
    }

    if shards.len() == 1 {
        return shards.into_iter().next().expect("Non-empty vec");
    }

    // Concatenate along batch dimension
    let arrays: Vec<ArrayD<f32>> = shards
        .iter()
        .map(|t| t.lock().storage.to_f32_array())
        .collect();

    // Get the views and concatenate
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    let concatenated = ndarray::concatenate(Axis(0), &views).expect("Failed to concatenate shards");

    Tensor::new(concatenated, false)
}

/// Calculate the effective batch size per rank.
#[allow(dead_code)]
pub fn local_batch_size(global_batch_size: usize, world_size: usize, rank: usize) -> usize {
    let base_size = global_batch_size / world_size;
    let remainder = global_batch_size % world_size;
    base_size + if rank < remainder { 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use crate::distributed::DistributedContext;
    use crate::tensor::Tensor;
    use ndarray::{ArrayD, IxDyn};

    fn create_batch_tensor(batch_size: usize, features: usize) -> Tensor {
        let data: Vec<f32> = (0..batch_size * features).map(|i| i as f32).collect();
        let arr =
            ArrayD::from_shape_vec(IxDyn(&[batch_size, features]), data).expect("Valid shape");
        Tensor::new(arr, false)
    }

    #[test]
    fn test_shard_batch_single_rank() {
        let ctx = DistributedContext::single();
        let model = (); // Dummy model
        let dp = DataParallel::new(model, ctx);

        let batch = create_batch_tensor(8, 4);
        let shard = dp.shard_batch(&batch, None);

        assert_eq!(shard.local_batch_size(), 8);
        assert_eq!(shard.original_batch_size, 8);
        assert_eq!(shard.rank, 0);
    }

    #[test]
    fn test_local_batch_size_calculation() {
        // 10 samples across 4 ranks
        assert_eq!(local_batch_size(10, 4, 0), 3); // 2 + 1 extra
        assert_eq!(local_batch_size(10, 4, 1), 3); // 2 + 1 extra
        assert_eq!(local_batch_size(10, 4, 2), 2);
        assert_eq!(local_batch_size(10, 4, 3), 2);

        // Even split
        assert_eq!(local_batch_size(12, 4, 0), 3);
        assert_eq!(local_batch_size(12, 4, 3), 3);
    }

    #[test]
    fn test_slice_tensor_batch() {
        let batch = create_batch_tensor(8, 4);
        let sliced = slice_tensor_batch(&batch, 2, 5);

        let shape = sliced.lock().storage.shape();
        assert_eq!(shape, vec![3, 4]);
    }

    #[test]
    fn test_gather_shards() {
        let shard1 = create_batch_tensor(2, 4);
        let shard2 = create_batch_tensor(3, 4);

        let gathered = gather_shards(vec![shard1, shard2]);
        let shape = gathered.lock().storage.shape();

        assert_eq!(shape, vec![5, 4]);
    }

    #[test]
    fn test_sharded_batch_validity() {
        let ctx = DistributedContext::single();
        let dp = DataParallel::new((), ctx);

        let batch = create_batch_tensor(4, 2);
        let shard = dp.shard_batch(&batch, None);

        assert!(shard.is_valid());
    }
}
