//! All-Reduce Operations for Gradient Synchronization
//!
//! This module provides gradient synchronization primitives for distributed training.
//! Uses shared memory state for inter-rank communication.

use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
#[allow(unused_imports)]
use std::sync::Arc;

use super::context::DistributedContext;

/// Reduction operation to apply during all-reduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum all values.
    Sum,
    /// Average all values.
    Mean,
    /// Take minimum value.
    Min,
    /// Take maximum value.
    Max,
    /// Product of all values.
    Product,
}

impl ReduceOp {
    /// Apply the reduction operation to two f32 values.
    pub fn apply(&self, a: f32, b: f32) -> f32 {
        match self {
            ReduceOp::Sum => a + b,
            ReduceOp::Mean => (a + b) / 2.0,
            ReduceOp::Min => a.min(b),
            ReduceOp::Max => a.max(b),
            ReduceOp::Product => a * b,
        }
    }

    /// Get the identity value for this operation.
    pub fn identity(&self) -> f32 {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => 0.0,
            ReduceOp::Min => f32::INFINITY,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Product => 1.0,
        }
    }

    /// Apply reduction across a slice of values.
    pub fn reduce_slice(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return self.identity();
        }
        values
            .iter()
            .skip(1)
            .fold(values[0], |acc, &x| self.apply(acc, x))
    }
}

/// All-reduce operation handler.
///
/// Provides methods for synchronizing gradients across all ranks using
/// shared memory communication.
pub struct AllReduce {
    ctx: DistributedContext,
}

impl AllReduce {
    /// Create a new all-reduce handler for the given context.
    pub fn new(ctx: DistributedContext) -> Self {
        Self { ctx }
    }

    /// Perform an all-reduce operation on the given tensor.
    ///
    /// For single-process, returns the input unchanged.
    /// For multi-rank, aggregates values across all ranks via shared memory.
    pub fn reduce(&self, tensor: &Tensor, op: ReduceOp) -> Tensor {
        let world_size = self.ctx.world_size();
        if world_size == 1 {
            return tensor.clone();
        }

        log::debug!(
            "AllReduce: rank {} performing {:?} on tensor",
            self.ctx.rank(),
            op
        );

        let data = tensor.lock().storage.to_f32_array();
        let requires_grad = tensor.lock().requires_grad;

        // Serialize tensor data and share via context
        let flat_data: Vec<f32> = data.iter().cloned().collect();
        let bytes = f32_slice_to_bytes(&flat_data);

        // Store this rank's contribution
        let key = format!("allreduce_rank_{}", self.ctx.rank());
        self.ctx.put(&key, bytes);

        // Synchronize
        self.ctx.barrier();

        // Collect all contributions
        let mut contributions: Vec<Vec<f32>> = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            let peer_key = format!("allreduce_rank_{}", rank);
            if let Some(peer_bytes) = self.ctx.get(&peer_key) {
                contributions.push(bytes_to_f32_slice(&peer_bytes));
            }
        }

        // Apply reduction
        let mut result = flat_data.clone();
        for (i, val) in result.iter_mut().enumerate() {
            let values: Vec<f32> = contributions
                .iter()
                .filter_map(|c| c.get(i).copied())
                .collect();

            if !values.is_empty() {
                *val = match op {
                    ReduceOp::Sum => values.iter().sum(),
                    ReduceOp::Mean => values.iter().sum::<f32>() / values.len() as f32,
                    ReduceOp::Min => values.iter().cloned().fold(f32::INFINITY, f32::min),
                    ReduceOp::Max => values.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    ReduceOp::Product => values.iter().product(),
                };
            }
        }

        // Reconstruct tensor
        let result_array = ArrayD::from_shape_vec(IxDyn(data.shape()), result)
            .expect("Shape mismatch in all-reduce");

        Tensor::new(result_array, requires_grad)
    }

    /// Perform in-place all-reduce on tensor gradients.
    pub fn reduce_gradients_inplace(&self, tensors: &[Tensor], op: ReduceOp) {
        let world_size = self.ctx.world_size();
        if world_size == 1 {
            return;
        }

        log::debug!(
            "AllReduce: synchronizing {} tensor gradients with {:?}",
            tensors.len(),
            op
        );

        // Flatten all gradients
        let mut all_grads = flatten_gradients(tensors);

        // Store this rank's gradients
        let bytes = f32_slice_to_bytes(&all_grads);
        let key = format!("grad_rank_{}", self.ctx.rank());
        self.ctx.put(&key, bytes);

        // Synchronize
        self.ctx.barrier();

        // Collect from all ranks and reduce
        let mut contributions: Vec<Vec<f32>> = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            let peer_key = format!("grad_rank_{}", rank);
            if let Some(peer_bytes) = self.ctx.get(&peer_key) {
                contributions.push(bytes_to_f32_slice(&peer_bytes));
            }
        }

        // Apply reduction element-wise
        for (i, val) in all_grads.iter_mut().enumerate() {
            let values: Vec<f32> = contributions
                .iter()
                .filter_map(|c| c.get(i).copied())
                .collect();

            if !values.is_empty() {
                *val = match op {
                    ReduceOp::Sum => values.iter().sum(),
                    ReduceOp::Mean => values.iter().sum::<f32>() / values.len() as f32,
                    ReduceOp::Min => values.iter().cloned().fold(f32::INFINITY, f32::min),
                    ReduceOp::Max => values.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    ReduceOp::Product => values.iter().product(),
                };
            }
        }

        // Unflatten back into tensors
        unflatten_gradients(&all_grads, tensors);
    }

    /// All-gather operation - collect tensors from all ranks.
    ///
    /// Each rank contributes its tensor, all ranks receive all tensors.
    pub fn all_gather(&self, tensor: &Tensor) -> Vec<Tensor> {
        let world_size = self.ctx.world_size();
        if world_size == 1 {
            return vec![tensor.clone()];
        }

        log::debug!(
            "AllGather: rank {} gathering tensor from {} ranks",
            self.ctx.rank(),
            world_size
        );

        let data = tensor.lock().storage.to_f32_array();
        let shape: Vec<usize> = data.shape().to_vec();
        let flat_data: Vec<f32> = data.iter().cloned().collect();

        // Share tensor data
        let key = format!("gather_rank_{}", self.ctx.rank());
        self.ctx.put(&key, f32_slice_to_bytes(&flat_data));

        // Synchronize
        self.ctx.barrier();

        // Collect from all ranks
        let mut result = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            let peer_key = format!("gather_rank_{}", rank);
            if let Some(peer_bytes) = self.ctx.get(&peer_key) {
                let peer_data = bytes_to_f32_slice(&peer_bytes);
                let peer_array = ArrayD::from_shape_vec(IxDyn(&shape), peer_data)
                    .expect("Shape mismatch in all-gather");
                result.push(Tensor::new(peer_array, false));
            }
        }

        result
    }

    /// Scatter operation - distribute tensor chunks to ranks.
    ///
    /// Master rank splits tensors; each rank receives its chunk.
    pub fn scatter(&self, tensors: Option<Vec<Tensor>>) -> Option<Tensor> {
        let world_size = self.ctx.world_size();
        if world_size == 1 {
            return tensors.and_then(|t| t.into_iter().next());
        }

        let rank = self.ctx.rank();

        if self.ctx.is_master() {
            // Master shares all chunks
            if let Some(ref chunks) = tensors {
                for (i, chunk) in chunks.iter().enumerate() {
                    let data = chunk.lock().storage.to_f32_array();
                    let flat: Vec<f32> = data.iter().cloned().collect();
                    let key = format!("scatter_chunk_{}", i);
                    self.ctx.put(&key, f32_slice_to_bytes(&flat));
                }
            }
        }

        // Synchronize
        self.ctx.barrier();

        // Each rank retrieves its chunk
        let chunk_key = format!("scatter_chunk_{}", rank);
        self.ctx.get(&chunk_key).map(|bytes| {
            let data = bytes_to_f32_slice(&bytes);
            // Since we don't know the shape, make it 1D
            let array = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data)
                .expect("Shape mismatch in scatter");
            Tensor::new(array, false)
        })
    }

    /// Reduce operation to a single rank (typically master).
    pub fn reduce_to_rank(&self, tensor: &Tensor, op: ReduceOp, dst_rank: usize) -> Option<Tensor> {
        let world_size = self.ctx.world_size();
        if world_size == 1 {
            return Some(tensor.clone());
        }

        // All ranks participate in reduction
        let reduced = self.reduce(tensor, op);

        // Only destination rank returns the result
        if self.ctx.rank() == dst_rank {
            Some(reduced)
        } else {
            None
        }
    }

    /// Broadcast a tensor from the source rank to all ranks.
    pub fn broadcast(&self, tensor: Option<&Tensor>, src_rank: usize) -> Option<Tensor> {
        let world_size = self.ctx.world_size();
        if world_size == 1 {
            return tensor.cloned();
        }

        if self.ctx.rank() == src_rank {
            // Source rank shares tensor
            if let Some(t) = tensor {
                let data = t.lock().storage.to_f32_array();
                let flat: Vec<f32> = data.iter().cloned().collect();
                let shape: Vec<usize> = data.shape().to_vec();

                self.ctx.put("broadcast_data", f32_slice_to_bytes(&flat));
                self.ctx.put("broadcast_shape", shape_to_bytes(&shape));
            }
        }

        // Synchronize
        self.ctx.barrier();

        // All ranks receive
        let data_bytes = self.ctx.get("broadcast_data")?;
        let shape_bytes = self.ctx.get("broadcast_shape")?;

        let data = bytes_to_f32_slice(&data_bytes);
        let shape = bytes_to_shape(&shape_bytes);

        let array =
            ArrayD::from_shape_vec(IxDyn(&shape), data).expect("Shape mismatch in broadcast");
        Some(Tensor::new(array, false))
    }
}

// Serialization helpers

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn shape_to_bytes(shape: &[usize]) -> Vec<u8> {
    shape
        .iter()
        .flat_map(|s| (*s as u64).to_le_bytes())
        .collect()
}

fn bytes_to_shape(bytes: &[u8]) -> Vec<usize> {
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as usize
        })
        .collect()
}

/// Helper function to compute total gradient size for buffer allocation.
#[allow(dead_code)]
pub fn compute_gradient_buffer_size(tensors: &[Tensor]) -> usize {
    tensors
        .iter()
        .map(|t| {
            let guard = t.lock();
            guard.grad.as_ref().map(|g| g.len()).unwrap_or(0)
        })
        .sum()
}

/// Flatten gradients into a single buffer for efficient communication.
pub fn flatten_gradients(tensors: &[Tensor]) -> Vec<f32> {
    let mut buffer = Vec::new();
    for tensor in tensors {
        let guard = tensor.lock();
        if let Some(ref grad) = guard.grad {
            buffer.extend(grad.iter().cloned());
        }
    }
    buffer
}

/// Unflatten a gradient buffer back into tensors.
pub fn unflatten_gradients(buffer: &[f32], tensors: &[Tensor]) {
    let mut offset = 0;
    for tensor in tensors {
        let mut guard = tensor.lock();
        if let Some(ref mut grad) = guard.grad {
            let len = grad.len();
            if offset + len <= buffer.len() {
                for (i, val) in grad.iter_mut().enumerate() {
                    *val = buffer[offset + i];
                }
            }
            offset += len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bytes_to_f32_slice, bytes_to_shape, f32_slice_to_bytes, flatten_gradients, shape_to_bytes,
        unflatten_gradients, AllReduce, ReduceOp,
    };
    use crate::distributed::DistributedContext;
    use crate::tensor::Tensor;
    use ndarray::{ArrayD, IxDyn};

    fn create_test_tensor(shape: &[usize], val: f32) -> Tensor {
        let data = ArrayD::from_elem(IxDyn(shape), val);
        Tensor::new(data, true)
    }

    #[test]
    fn test_reduce_op_apply() {
        assert_eq!(ReduceOp::Sum.apply(2.0, 3.0), 5.0);
        assert_eq!(ReduceOp::Min.apply(2.0, 3.0), 2.0);
        assert_eq!(ReduceOp::Max.apply(2.0, 3.0), 3.0);
        assert_eq!(ReduceOp::Product.apply(2.0, 3.0), 6.0);
    }

    #[test]
    fn test_reduce_slice() {
        assert_eq!(ReduceOp::Sum.reduce_slice(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(ReduceOp::Min.reduce_slice(&[3.0, 1.0, 2.0]), 1.0);
        assert_eq!(ReduceOp::Max.reduce_slice(&[1.0, 3.0, 2.0]), 3.0);
    }

    #[test]
    fn test_single_process_reduce() {
        let ctx = DistributedContext::single();
        let all_reduce = AllReduce::new(ctx);

        let t = create_test_tensor(&[2, 3][..], 1.0);
        let result = all_reduce.reduce(&t, ReduceOp::Sum);

        let result_data = result.lock().storage.to_f32_array();
        assert!(result_data.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_all_gather_single() {
        let ctx = DistributedContext::single();
        let all_reduce = AllReduce::new(ctx);

        let t = create_test_tensor(&[2, 2][..], 1.0);
        let gathered = all_reduce.all_gather(&t);

        assert_eq!(gathered.len(), 1);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let data = vec![1.0f32, 2.5, -3.25, 0.0];
        let bytes = f32_slice_to_bytes(&data);
        let restored = bytes_to_f32_slice(&bytes);

        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_shape_serialization() {
        let shape = vec![2, 3, 4];
        let bytes = shape_to_bytes(&shape);
        let restored = bytes_to_shape(&bytes);
        assert_eq!(shape, restored);
    }

    #[test]
    fn test_flatten_unflatten() {
        let t1 = create_test_tensor(&[2, 2][..], 1.0);
        let t2 = create_test_tensor(&[3][..], 2.0);

        {
            let mut g1 = t1.lock();
            g1.grad = Some(ArrayD::from_elem(IxDyn(&[2, 2][..]), 0.5));
        }
        {
            let mut g2 = t2.lock();
            g2.grad = Some(ArrayD::from_elem(IxDyn(&[3][..]), 1.5));
        }

        let buffer = flatten_gradients(&[t1.clone(), t2.clone()]);
        assert_eq!(buffer.len(), 4 + 3);

        {
            let mut g1 = t1.lock();
            g1.grad = Some(ArrayD::zeros(IxDyn(&[2, 2])));
        }
        {
            let mut g2 = t2.lock();
            g2.grad = Some(ArrayD::zeros(IxDyn(&[3])));
        }

        unflatten_gradients(&buffer, &[t1.clone(), t2.clone()]);

        let g1_val = t1.lock().grad.as_ref().unwrap().iter().next().cloned();
        assert_eq!(g1_val, Some(0.5));
    }
}
