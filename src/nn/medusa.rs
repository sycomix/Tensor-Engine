//! Medusa Heads for efficient speculative decoding.
//!
//! Medusa enables efficient LLM inference by training small auxiliary heads
//! to predict multiple future tokens in parallel, significantly reducing latency.
//!
//! Reference: Medusa: Simple LLM Inference Acceleration Framework with Multiple Decomposed Heads
//! (Chen et al., 2024)

use crate::dtype::TensorStorage;
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;
use std::any::Any;

/// Medusa head configuration.
#[derive(Clone, Debug)]
pub struct MedusaHeadConfig {
    /// Hidden dimension of the model
    pub hidden_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of lookahead tokens to predict
    pub num_lookahead: usize,
    /// Number of medusa heads (depth of tree)
    pub num_heads: usize,
    /// Width per head (beam width)
    pub width_per_head: usize,
}

impl MedusaHeadConfig {
    /// Create a new Medusa configuration.
    pub fn new(hidden_dim: usize, vocab_size: usize, num_lookahead: usize) -> Self {
        MedusaHeadConfig {
            hidden_dim,
            vocab_size,
            num_lookahead,
            num_heads: 1,
            width_per_head: 4,
        }
    }

    /// Set the number of Medusa heads.
    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    /// Set the beam width per head.
    pub fn with_width_per_head(mut self, width: usize) -> Self {
        self.width_per_head = width;
        self
    }
}

/// Single Medusa head for predicting future tokens.
pub struct MedusaHead {
    /// Linear projection layers forming a tree
    linear_layers: Vec<Vec<Linear>>,
    config: MedusaHeadConfig,
}

impl MedusaHead {
    /// Create a new Medusa head.
    ///
    /// Creates a tree of linear layers where:
    /// - First layer: projects from hidden_dim to vocab_size
    /// - Subsequent layers: form branching paths for different token predictions
    pub fn new(config: MedusaHeadConfig) -> Self {
        let mut linear_layers = Vec::new();

        // First layer: main output head
        let layer_0 = vec![Linear::new(config.hidden_dim, config.vocab_size, true)];
        linear_layers.push(layer_0);

        // Additional layers for lookahead tokens
        for depth in 1..config.num_lookahead {
            let mut layer = Vec::new();

            // Create parallel branches for diverse predictions
            for _ in 0..config.width_per_head.pow(depth as u32) {
                let linear = Linear::new(config.hidden_dim, config.vocab_size, true);
                layer.push(linear);
            }
            linear_layers.push(layer);
        }

        MedusaHead {
            linear_layers,
            config,
        }
    }

    /// Forward pass through Medusa head.
    ///
    /// # Arguments
    /// * `hidden_states` - Input hidden states [batch, seq_len, hidden_dim]
    ///
    /// # Returns
    /// Logits for multiple lookahead positions
    pub fn forward(&self, hidden_states: &Tensor) -> Vec<Tensor> {
        let mut outputs = Vec::new();

        // Get the last token's hidden state
        // Since we don't have direct slicing, we'll use the full hidden states
        // and let the linear layer handle it
        let main_logits = self.linear_layers[0][0].forward(hidden_states);
        outputs.push(main_logits);

        // Lookahead predictions
        for depth in 1..self.config.num_lookahead.min(self.linear_layers.len()) {
            let num_branches = self
                .config
                .width_per_head
                .pow(depth as u32)
                .min(self.linear_layers[depth].len());

            for branch_idx in 0..num_branches {
                let logits = self.linear_layers[depth][branch_idx].forward(hidden_states);
                outputs.push(logits);
            }
        }

        outputs
    }

    /// Get the tree structure parameters.
    pub fn tree_structure(&self) -> Vec<Vec<usize>> {
        vec![
            vec![1], // Root has 1 branch
            (0..self.config.width_per_head).map(|i| 1 + i).collect(),
        ]
    }

    /// Get the linear layers (for parameter access).
    pub fn layers(&self) -> &[Vec<Linear>] {
        &self.linear_layers
    }

    /// Mutable access to linear layers (for parameter updates).
    pub fn layers_mut(&mut self) -> &mut [Vec<Linear>] {
        &mut self.linear_layers
    }
}

impl Module for MedusaHead {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Use the main head for single forward pass
        self.linear_layers[0][0].forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer_group in &self.linear_layers {
            for linear in layer_group {
                params.extend(linear.parameters());
            }
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        for layer_group in &mut self.linear_layers {
            for linear in layer_group {
                linear.set_training(training);
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Medusa inference engine for speculative decoding.
pub struct MedusaInference {
    medusa_head: MedusaHead,
    config: MedusaHeadConfig,
}

impl MedusaInference {
    /// Create a new Medusa inference engine.
    pub fn new(config: MedusaHeadConfig) -> Self {
        MedusaInference {
            medusa_head: MedusaHead::new(config.clone()),
            config,
        }
    }

    /// Perform speculative decoding step.
    ///
    /// Predicts multiple lookahead tokens using Medusa head and returns
    /// the most likely token paths for verification.
    pub fn speculative_decode(&self, hidden_states: &Tensor, temperature: f32) -> Vec<usize> {
        let logits = self.medusa_head.forward(hidden_states);

        // Get top-1 prediction from each head
        let mut predictions = Vec::new();

        for head_logits in logits {
            let pred_token = self.sample_from_logits(&head_logits, temperature);
            predictions.push(pred_token);
        }

        predictions
    }

    /// Sample token from logits with temperature scaling.
    fn sample_from_logits(&self, logits: &Tensor, temperature: f32) -> usize {
        let lock = logits.lock();

        // Get raw logits array
        match &lock.storage {
            TensorStorage::F32(arr) => {
                let data: Vec<f32> = arr.iter().copied().collect();
                let mut max_idx = 0;
                let mut max_val = data.first().copied().unwrap_or(0.0) / temperature;

                for (i, &val) in data.iter().enumerate().skip(1) {
                    let scaled_val = val / temperature;
                    if scaled_val > max_val {
                        max_val = scaled_val;
                        max_idx = i;
                    }
                }
                max_idx
            }
            _ => {
                let arr = lock.storage.to_f32_array();
                let data: Vec<f32> = arr.iter().copied().collect();
                let mut max_idx = 0;
                let mut max_val = data.first().copied().unwrap_or(0.0) / temperature;

                for (i, &val) in data.iter().enumerate().skip(1) {
                    let scaled_val = val / temperature;
                    if scaled_val > max_val {
                        max_val = scaled_val;
                        max_idx = i;
                    }
                }
                max_idx
            }
        }
    }

    /// Get Medusa head for training.
    pub fn head(&self) -> &MedusaHead {
        &self.medusa_head
    }

    /// Get mutable Medusa head.
    pub fn head_mut(&mut self) -> &mut MedusaHead {
        &mut self.medusa_head
    }

    /// Get configuration.
    pub fn config(&self) -> &MedusaHeadConfig {
        &self.config
    }
}

#[cfg(test)]
mod medusa_tests {
    use super::*;

    #[test]
    fn test_medusa_head_config() {
        let config = MedusaHeadConfig::new(768, 32000, 4)
            .with_num_heads(2)
            .with_width_per_head(4);

        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.num_lookahead, 4);
        assert_eq!(config.num_heads, 2);
        assert_eq!(config.width_per_head, 4);
    }

    #[test]
    fn test_medusa_head_creation() {
        let config = MedusaHeadConfig::new(768, 32000, 2);
        let medusa = MedusaHead::new(config);

        // Check that we have the right number of layer groups
        let layers = medusa.layers();
        assert!(layers.len() > 0);

        // First layer should have 1 linear
        assert_eq!(layers[0].len(), 1);
    }

    #[test]
    fn test_medusa_inference_creation() {
        let config = MedusaHeadConfig::new(768, 32000, 3);
        let medusa_inf = MedusaInference::new(config);

        assert_eq!(medusa_inf.config().hidden_dim, 768);
        assert_eq!(medusa_inf.config().vocab_size, 32000);
    }

    #[test]
    fn test_medusa_tree_structure() {
        let config = MedusaHeadConfig::new(768, 32000, 2).with_width_per_head(4);
        let medusa = MedusaHead::new(config);
        let tree = medusa.tree_structure();

        // Root should have 1
        assert_eq!(tree[0][0], 1);

        // Second level should have width branches
        assert_eq!(tree[1].len(), 4);
    }

    #[test]
    fn test_medusa_head_forward_shape() {
        let config = MedusaHeadConfig::new(768, 32000, 2);
        let medusa = MedusaHead::new(config);

        // Create dummy hidden states [batch=1, seq_len=10, hidden_dim=768]
        let hidden_states = Tensor::randn(vec![1, 10, 768]);
        let outputs = medusa.forward(&hidden_states);

        // Should have at least 1 output (from first layer)
        assert!(outputs.len() >= 1);
    }

    #[test]
    fn test_medusa_head_module_trait() {
        let config = MedusaHeadConfig::new(768, 32000, 2);
        let medusa = MedusaHead::new(config);

        let params = medusa.parameters();
        // Should have parameters from all linear layers
        assert!(!params.is_empty());
    }

    #[test]
    fn test_medusa_inference_decode() {
        let config = MedusaHeadConfig::new(768, 32000, 2);
        let medusa = MedusaInference::new(config);

        let hidden_states = Tensor::randn(vec![1, 10, 768]);
        let predictions = medusa.speculative_decode(&hidden_states, 1.0);

        // Should have predictions
        assert!(!predictions.is_empty());

        // All predictions should be valid token indices
        for pred in predictions {
            assert!(pred < 32000);
        }
    }
}
