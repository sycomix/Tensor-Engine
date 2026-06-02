//! Weight pruning for model compression.
//!
//! Implements magnitude-based pruning, iterative pruning, and structured pruning
//! to reduce model size and inference latency.

use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis, IxDyn};
use rand::Rng;

/// Pruning method
#[derive(Clone, Debug, PartialEq)]
pub enum PruningMethod {
    /// Magnitude-based: prune weights with smallest absolute values
    Magnitude,
    /// Random pruning
    Random,
    /// Structured pruning: prune entire channels/heads
    Structured { axis: usize },
    /// N:M sparse pruning (N non-zeros in every M weights)
    SparseNtoM { n: usize, m: usize },
}

/// Pruning configuration
#[derive(Clone)]
pub struct PruningConfig {
    /// Method to use for pruning
    pub method: PruningMethod,
    /// Target sparsity (0.0 to 1.0)
    pub sparsity: f32,
    /// Minimum magnitude threshold (overrides sparsity if set)
    pub min_magnitude: Option<f32>,
    /// Whether to apply pruning iteratively
    pub iterative: bool,
    /// Number of pruning iterations
    pub num_iterations: usize,
    /// Sparsity increase per iteration
    pub sparsity_per_iteration: f32,
}

impl Default for PruningConfig {
    fn default() -> Self {
        PruningConfig {
            method: PruningMethod::Magnitude,
            sparsity: 0.5,
            min_magnitude: None,
            iterative: false,
            num_iterations: 10,
            sparsity_per_iteration: 0.05,
        }
    }
}

/// Pruner applies weight pruning to a tensor.
pub struct Pruner {
    config: PruningConfig,
}

impl Pruner {
    /// Create a new pruner with the given configuration.
    pub fn new(config: PruningConfig) -> Self {
        Pruner { config }
    }

    /// Create a magnitude-based pruner with target sparsity.
    pub fn new_magnitude(sparsity: f32) -> Self {
        Pruner {
            config: PruningConfig {
                method: PruningMethod::Magnitude,
                sparsity,
                ..Default::default()
            },
        }
    }

    /// Create an iterative magnitude pruner.
    pub fn new_iterative(sparsity: f32, num_iterations: usize) -> Self {
        Pruner {
            config: PruningConfig {
                method: PruningMethod::Magnitude,
                sparsity,
                iterative: true,
                num_iterations,
                sparsity_per_iteration: sparsity / num_iterations as f32,
                ..Default::default()
            },
        }
    }

    /// Create an N:M sparse pruner.
    pub fn new_sparse_n_to_m(n: usize, m: usize) -> Self {
        Pruner {
            config: PruningConfig {
                method: PruningMethod::SparseNtoM { n, m },
                sparsity: 1.0 - n as f32 / m as f32,
                ..Default::default()
            },
        }
    }

    /// Apply pruning to a tensor.
    pub fn prune(&self, weights: &Tensor) -> Tensor {
        let arr = weights.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();

        match self.config.method {
            PruningMethod::Magnitude => self.prune_magnitude(&arr, &shape),
            PruningMethod::Random => self.prune_random(&arr, &shape),
            PruningMethod::Structured { axis } => self.prune_structured(&arr, &shape, axis),
            PruningMethod::SparseNtoM { n, m } => self.prune_sparse_n_to_m(&arr, &shape, n, m),
        }
    }

    /// Magnitude-based pruning: set smallest absolute values to zero.
    fn prune_magnitude(&self, arr: &ArrayD<f32>, shape: &[usize]) -> Tensor {
        let sparsity = self.config.sparsity.clamp(0.0, 1.0);
        let total = arr.len();
        let num_zeros = (total as f32 * sparsity) as usize;

        if num_zeros == 0 {
            return Tensor::new(arr.clone().into_dyn(), false);
        }

        // Find threshold: sort absolute values and find the threshold at sparsity
        let abs_vals: Vec<f32> = arr.iter().map(|v| v.abs()).collect();
        let mut sorted_abs = abs_vals.clone();
        sorted_abs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let threshold = if num_zeros < total {
            sorted_abs[num_zeros]
        } else {
            0.0
        };

        // Apply threshold
        let pruned: Vec<f32> = arr
            .iter()
            .map(|&v| if v.abs() < threshold { 0.0 } else { v })
            .collect();

        Tensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), pruned).unwrap_or_else(|_| ArrayD::zeros(IxDyn(shape))),
            false,
        )
    }

    /// Random pruning: randomly set weights to zero.
    fn prune_random(&self, arr: &ArrayD<f32>, shape: &[usize]) -> Tensor {
        let sparsity = self.config.sparsity.clamp(0.0, 1.0);
        let total = arr.len();
        let num_zeros = (total as f32 * sparsity) as usize;

        if num_zeros == 0 {
            return Tensor::new(arr.clone().into_dyn(), false);
        }

        let mut pruned = arr.to_vec();
        let mut indices: Vec<usize> = (0..total).collect();
        // Fisher-Yates shuffle for first num_zeros elements
        let mut rng = rand::rng();
        for i in 0..num_zeros {
            let j = i + rng.random_range(0..(total - i));
            indices.swap(i, j);
        }

        for &idx in &indices[..num_zeros] {
            pruned[idx] = 0.0;
        }

        Tensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), pruned).unwrap_or_else(|_| ArrayD::zeros(IxDyn(shape))),
            false,
        )
    }

    /// Structured pruning: prune entire channels/heads along specified axis.
    fn prune_structured(&self, arr: &ArrayD<f32>, shape: &[usize], axis: usize) -> Tensor {
        let sparsity = self.config.sparsity.clamp(0.0, 1.0);
        let ndim = shape.len();

        if axis >= ndim {
            return Tensor::new(arr.clone().into_dyn(), false);
        }

        let axis_len = shape[axis];
        let num_to_prune = (axis_len as f32 * sparsity) as usize;

        if num_to_prune == 0 || num_to_prune >= axis_len {
            return Tensor::new(ArrayD::zeros(IxDyn(shape)), false);
        }

        // Compute L2 norm along all other axes for each element along pruning axis
        let mut norms = vec![0.0f32; axis_len];
        for i in 0..axis_len {
            let mut slice = arr.index_axis(Axis(axis), i);
            norms[i] = slice.iter().map(|v| v * v).sum::<f32>().sqrt();
        }

        // Find indices with smallest norms
        let mut indexed_norms: Vec<(f32, usize)> = norms.into_iter().enumerate().map(|(i, n)| (n, i)).collect();
        indexed_norms.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let prune_indices: std::collections::HashSet<usize> =
            indexed_norms[..num_to_prune].iter().map(|&(_, idx)| idx).collect();

        // Apply pruning
        let mut pruned = arr.clone();
        for i in 0..axis_len {
            if prune_indices.contains(&i) {
                pruned.index_axis_mut(Axis(axis), i).fill(0.0);
            }
        }

        Tensor::new(pruned.into_dyn(), false)
    }

    /// N:M sparse pruning: within each block of M weights, keep only N largest.
    fn prune_sparse_n_to_m(&self, arr: &ArrayD<f32>, shape: &[usize], n: usize, m: usize) -> Tensor {
        let total = arr.len();
        let num_blocks = total / m;
        let mut pruned = arr.clone();

        for block in 0..num_blocks {
            let start = block * m;
            let end = start + m;

            // Get block values with indices
            let block_vals: Vec<(f32, usize)> = (start..end)
                .map(|i| (pruned[i].abs(), i))
                .collect();

            // Sort by absolute value descending
            let mut sorted = block_vals;
            sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Keep top N, zero out the rest
            for (i, &(_, idx)) in sorted.iter().enumerate() {
                if i >= n {
                    pruned[idx] = 0.0;
                }
            }
        }

        Tensor::new(pruned.into_dyn(), false)
    }

    /// Iteratively prune a tensor over multiple steps.
    pub fn prune_iterative(&self, weights: &Tensor) -> Tensor {
        if !self.config.iterative {
            return self.prune(weights);
        }

        let mut current = weights.clone();
        let mut current_sparsity = 0.0f32;

        for _ in 0..self.config.num_iterations {
            let target_sparsity = (current_sparsity + self.config.sparsity_per_iteration)
                .min(self.config.sparsity);

            // Create a pruner with current sparsity
            let pruner = Pruner::new(PruningConfig {
                method: self.config.method.clone(),
                sparsity: target_sparsity,
                ..Default::default()
            });

            current = pruner.prune(&current);
            current_sparsity = target_sparsity;
        }

        current
    }

    /// Compute sparsity of a tensor (fraction of zero elements).
    pub fn compute_sparsity(&self, weights: &Tensor) -> f32 {
        let arr = weights.lock().storage.to_f32_array();
        let total = arr.len();
        if total == 0 {
            return 0.0;
        }
        let zeros = arr.iter().filter(|&&v| v == 0.0).count();
        zeros as f32 / total as f32
    }
}

/// Structured pruning for attention heads.
pub struct HeadPruner {
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of heads to keep
    pub heads_to_keep: usize,
}

impl HeadPruner {
    /// Create a new head pruner.
    pub fn new(num_heads: usize, keep_ratio: f32) -> Self {
        let heads_to_keep = (num_heads as f32 * keep_ratio) as usize;
        HeadPruner {
            num_heads,
            heads_to_keep: heads_to_keep.max(1),
        }
    }

    /// Get the indices of heads to keep.
    pub fn get_keep_indices(&self) -> Vec<usize> {
        (0..self.heads_to_keep).collect()
    }

    /// Get the indices of heads to prune.
    pub fn get_prune_indices(&self) -> Vec<usize> {
        (self.heads_to_keep..self.num_heads).collect()
    }

    /// Check if a head index should be kept.
    pub fn should_keep(&self, head_idx: usize) -> bool {
        head_idx < self.heads_to_keep
    }
}

/// Channel pruning for convolutional layers.
pub struct ChannelPruner {
    /// Number of channels
    pub num_channels: usize,
    /// Number of channels to keep
    pub channels_to_keep: usize,
}

impl ChannelPruner {
    /// Create a new channel pruner.
    pub fn new(num_channels: usize, keep_ratio: f32) -> Self {
        let channels_to_keep = (num_channels as f32 * keep_ratio) as usize;
        ChannelPruner {
            num_channels,
            channels_to_keep: channels_to_keep.max(1),
        }
    }

    /// Get the indices of channels to keep.
    pub fn get_keep_indices(&self) -> Vec<usize> {
        (0..self.channels_to_keep).collect()
    }

    /// Check if a channel index should be kept.
    pub fn should_keep(&self, channel_idx: usize) -> bool {
        channel_idx < self.channels_to_keep
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_magnitude_pruning() {
        let data: Vec<f32> = vec![0.1, 0.5, 0.9, -0.3, 0.7];
        let weights = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[5]), data).unwrap(), false);
        let pruner = Pruner::new_magnitude(0.4); // 40% sparsity = 2 zeros
        let pruned = pruner.prune(&weights);
        let pruned_arr = pruned.lock().storage.to_f32_array();
        let zeros = pruned_arr.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zeros, 2);
    }

    #[test]
    fn test_sparse_n_to_m() {
        // 2:4 sparse: in every 4 weights, keep 2 largest
        let data: Vec<f32> = vec![0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6];
        let weights = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[8]), data).unwrap(), false);
        let pruner = Pruner::new_sparse_n_to_m(2, 4);
        let pruned = pruner.prune(&weights);
        let pruned_arr = pruned.lock().storage.to_f32_array();

        // First block: [0.1, 0.9, 0.3, 0.7] -> keep 0.9, 0.7 -> [0.0, 0.9, 0.0, 0.7]
        assert_eq!(pruned_arr[0], 0.0);
        assert!((pruned_arr[1] - 0.9).abs() < 1e-6);
        assert_eq!(pruned_arr[2], 0.0);
        assert!((pruned_arr[3] - 0.7).abs() < 1e-6);

        // Second block: [0.2, 0.8, 0.4, 0.6] -> keep 0.8, 0.6 -> [0.0, 0.8, 0.0, 0.6]
        assert_eq!(pruned_arr[4], 0.0);
        assert!((pruned_arr[5] - 0.8).abs() < 1e-6);
        assert_eq!(pruned_arr[6], 0.0);
        assert!((pruned_arr[7] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_compute_sparsity() {
        let data: Vec<f32> = vec![0.0, 0.5, 0.0, 0.7, 0.0];
        let weights = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[5]), data).unwrap(), false);
        let pruner = Pruner::new_magnitude(0.0);
        let sparsity = pruner.compute_sparsity(&weights);
        assert!((sparsity - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_head_pruner() {
        let pruner = HeadPruner::new(8, 0.5); // Keep 4 heads
        assert_eq!(pruner.heads_to_keep, 4);
        assert_eq!(pruner.get_keep_indices(), vec![0, 1, 2, 3]);
        assert_eq!(pruner.get_prune_indices(), vec![4, 5, 6, 7]);
        assert!(pruner.should_keep(3));
        assert!(!pruner.should_keep(4));
    }

    #[test]
    fn test_channel_pruner() {
        let pruner = ChannelPruner::new(64, 0.25); // Keep 16 channels
        assert_eq!(pruner.channels_to_keep, 16);
        assert!(pruner.should_keep(15));
        assert!(!pruner.should_keep(16));
    }
}
