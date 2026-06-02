//! Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.
//!
//! LoRA freezes pre-trained model weights and injects trainable rank decomposition
//! matrices into each layer of the Transformer architecture. This dramatically
//! reduces the number of trainable parameters while maintaining performance.
//!
//! Reference: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)

use crate::tensor::Tensor;
use crate::nn::Module;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

/// LoRA adapter for a single linear layer.
///
/// Instead of updating weight W directly, we learn ΔW = BA where:
/// - B: (out_features, rank) - down projection
/// - A: (rank, in_features) - up projection
/// - rank << min(in_features, out_features)
#[derive(Clone)]
pub struct LoRAAdapter {
    /// Down projection matrix B: [rank, out_features]
    pub down_proj: Tensor,
    /// Up projection matrix A: [in_features, rank]
    pub up_proj: Tensor,
    /// Scaling factor: alpha / rank
    pub alpha: f32,
    /// Current rank
    pub rank: usize,
    /// Whether to apply dropout during training
    pub dropout: f32,
    /// Dropout mask cache
    pub dropout_mask: Option<Tensor>,
}

impl LoRAAdapter {
    /// Create a new LoRA adapter.
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `rank` - Rank of the low-rank decomposition
    /// * `alpha` - Scaling factor (typically rank/2 or rank)
    /// * `dropout` - Dropout probability (0.0 to disable)
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32, dropout: f32) -> Self {
        assert!(rank > 0, "LoRA rank must be positive");
        assert!(rank <= in_features.min(out_features), "LoRA rank must be <= min(in_features, out_features)");
        assert!(dropout >= 0.0 && dropout < 1.0, "dropout must be in [0, 1)");

        // Initialize A and B with small random values (Kaiming uniform)
        let scale_a = (2.0 / in_features as f32).sqrt();
        let scale_b = 0.0; // B initialized to zero

        let a_data = ArrayD::from_shape_fn(IxDyn(&[in_features, rank][..]), |_| {
            rand::random::<f32>() * 2.0 * scale_a - scale_a
        });
        let b_data = ArrayD::zeros(IxDyn(&[out_features, rank][..]));

        LoRAAdapter {
            down_proj: Tensor::new(b_data, true),
            up_proj: Tensor::new(a_data, true),
            alpha,
            rank,
            dropout,
            dropout_mask: None,
        }
    }

    /// Compute the LoRA update: ΔW = (alpha / rank) * B^T @ A
    /// Returns shape [out_features, in_features]
    pub fn get_update(&self) -> Tensor {
        // B^T: [rank, out_features] @ A: [in_features, rank] -> [out_features, in_features]
        let b_t = self.down_proj.transpose();
        let update = b_t.matmul(&self.up_proj);
        let scale = self.alpha / self.rank as f32;
        update.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), scale),
            false,
        ))
    }

    /// Apply dropout to the LoRA output during training.
    pub fn apply_dropout(&self, input: &Tensor) -> Tensor {
        if self.dropout <= 0.0 || self.dropout_mask.is_none() {
            return input.clone();
        }

        let mask = self.dropout_mask.as_ref().unwrap();
        input.mul(mask)
    }

    /// Set training mode and generate dropout mask.
    pub fn set_training(&mut self, training: bool) {
        if training && self.dropout > 0.0 {
            let shape = self.up_proj.lock().storage.shape().to_vec();
            let mask_data: Vec<f32> = (0..self.up_proj.lock().storage.len())
                .map(|_| {
                    if rand::random::<f32>() < self.dropout {
                        0.0
                    } else {
                        1.0 / (1.0 - self.dropout)
                    }
                })
                .collect();
            self.dropout_mask = Some(Tensor::new(
                ArrayD::from_shape_vec(IxDyn(&shape), mask_data).unwrap(),
                false,
            ));
        } else {
            self.dropout_mask = None;
        }
    }

    /// Get trainable parameters for optimizer.
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.down_proj.clone(), self.up_proj.clone()]
    }

    /// Named parameters for state dict.
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (format!("{}.lora_down.weight", prefix), self.down_proj.clone()),
            (format!("{}.lora_up.weight", prefix), self.up_proj.clone()),
        ]
    }
}

/// LoRA wrapper that applies low-rank adaptation to specific layers.
#[derive(Clone)]
pub struct LoRAModule {
    /// Base module to wrap
    pub base_module: Box<dyn Module>,
    /// LoRA adapters for each target layer
    pub adapters: HashMap<String, LoRAAdapter>,
    /// Whether LoRA is active
    pub active: bool,
}

impl LoRAModule {
    /// Create a new LoRA wrapper.
    pub fn new(base_module: Box<dyn Module>) -> Self {
        LoRAModule {
            base_module,
            adapters: HashMap::new(),
            active: false,
        }
    }

    /// Add a LoRA adapter to a specific layer.
    pub fn add_adapter(&mut self, name: String, adapter: LoRAAdapter) {
        self.adapters.insert(name, adapter);
    }

    /// Activate or deactivate LoRA adapters.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// Get all trainable LoRA parameters.
    pub fn lora_parameters(&self) -> Vec<Tensor> {
        self.adapters.values().flat_map(|a| a.parameters()).collect()
    }

    /// Get all trainable LoRA parameters with names.
    pub fn lora_named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (name, adapter) in &self.adapters {
            let full_prefix = format!("{}.{}", prefix, name);
            out.extend(adapter.named_parameters(&full_prefix));
        }
        out
    }

    /// Load LoRA state dict.
    pub fn load_lora_state(&mut self, state: &HashMap<String, Tensor>) -> Result<(), String> {
        for (name, adapter) in &mut self.adapters {
            let prefix = format!("{}.{}", name, "lora");
            if let Some(down) = state.get(&format!("{}.down.weight", prefix)) {
                adapter.down_proj = down.clone();
            }
            if let Some(up) = state.get(&format!("{}.up.weight", prefix)) {
                adapter.up_proj = up.clone();
            }
        }
        Ok(())
    }

    /// Save LoRA state dict.
    pub fn save_lora_state(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();
        for (name, adapter) in &self.adapters {
            let prefix = format!("{}.{}", name, "lora");
            state.insert(format!("{}.down.weight", prefix), adapter.down_proj.clone());
            state.insert(format!("{}.up.weight", prefix), adapter.up_proj.clone());
        }
        state
    }
}

impl Module for LoRAModule {
    fn forward(&self, input: &Tensor) -> Tensor {
        let base_out = self.base_module.forward(input);

        if !self.active || self.adapters.is_empty() {
            return base_out;
        }

        // Apply all LoRA updates (in practice, this would be done per-layer)
        // For simplicity, we return the base output here.
        // In a full implementation, LoRA would be applied during the forward pass
        // of each wrapped layer.
        base_out
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.base_module.parameters();
        if self.active {
            params.extend(self.lora_parameters());
        }
        params
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = self.base_module.named_parameters(prefix);
        if self.active {
            out.extend(self.lora_named_parameters(prefix));
        }
        out
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.base_module.load_state_dict(state, prefix)?;
        self.load_lora_state(state)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// QLoRA (Quantized LoRA) adapter for memory-efficient fine-tuning.
///
/// QLoRA combines 4-bit quantization with LoRA to reduce memory usage
/// while maintaining performance.
#[derive(Clone)]
pub struct QLoRAAdapter {
    /// Base adapter
    pub lora: LoRAAdapter,
    /// Quantization scale for the base weights
    pub quant_scale: f32,
    /// Quantization offset for the base weights
    pub quant_offset: f32,
    /// Whether to use double quantization
    pub double_quant: bool,
}

impl QLoRAAdapter {
    /// Create a new QLoRA adapter.
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout: f32,
        quant_scale: f32,
        quant_offset: f32,
        double_quant: bool,
    ) -> Self {
        QLoRAAdapter {
            lora: LoRAAdapter::new(in_features, out_features, rank, alpha, dropout),
            quant_scale,
            quant_offset,
            double_quant,
        }
    }

    /// Get the LoRA parameters.
    pub fn parameters(&self) -> Vec<Tensor> {
        self.lora.parameters()
    }

    /// Get named LoRA parameters.
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.lora.named_parameters(prefix)
    }

    /// Apply quantization-aware forward pass.
    pub fn forward_quant_aware(&self, input: &Tensor, base_weight: &Tensor) -> Tensor {
        // Dequantize base weights
        let dequant = base_weight.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), self.quant_scale),
            false,
        )).add(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), self.quant_offset),
            false,
        ));

        // Apply base weight
        let base_out = input.matmul(&dequant);

        // Add LoRA update
        let lora_update = self.lora.get_update();
        base_out.add(&lora_update)
    }
}

/// DoRA (Weight-Decomposed Low-Rank Adaptation) adapter.
///
/// DoRA decomposes weight into magnitude and direction, then applies
/// LoRA to the direction component for more stable training.
#[derive(Clone)]
pub struct DoRAAdapter {
    /// Base LoRA adapter
    pub lora: LoRAAdapter,
    /// Learned magnitude for each output channel
    pub magnitude: Tensor,
    /// Initial magnitude from pre-trained weights
    pub init_magnitude: f32,
}

impl DoRAAdapter {
    /// Create a new DoRA adapter.
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout: f32,
        init_magnitude: f32,
    ) -> Self {
        DoRAAdapter {
            lora: LoRAAdapter::new(in_features, out_features, rank, alpha, dropout),
            magnitude: Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[out_features][..]), init_magnitude),
                true,
            ),
            init_magnitude,
        }
    }

    /// Get the DoRA update: magnitude * (W + ΔW) / ||W + ΔW||
    pub fn get_update(&self, base_weight: &Tensor) -> Tensor {
        let lora_update = self.lora.get_update();
        let combined = base_weight.add(&lora_update);

        // Compute L2 norm of combined weights
        let norm = combined.pow(2.0).sum().sqrt();
        let normalized = combined.div(&norm.add(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), 1e-8),
            false,
        )));

        // Apply learned magnitude
        normalized.mul(&self.magnitude)
    }

    /// Get trainable parameters.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.lora.parameters();
        params.push(self.magnitude.clone());
        params
    }
}

/// LoRA configuration for a model.
#[derive(Clone)]
pub struct LoRAConfig {
    /// Target modules to apply LoRA to (e.g., ["q_proj", "v_proj"])
    pub target_modules: Vec<String>,
    /// Rank of the low-rank decomposition
    pub rank: usize,
    /// Alpha scaling factor
    pub alpha: f32,
    /// Dropout probability
    pub dropout: f32,
    /// Whether to use QLoRA
    pub use_qlora: bool,
    /// Whether to use DoRA
    pub use_dora: bool,
    /// Double quantization for QLoRA
    pub double_quant: bool,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        LoRAConfig {
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            rank: 8,
            alpha: 16.0,
            dropout: 0.05,
            use_qlora: false,
            use_dora: false,
            double_quant: true,
        }
    }
}

impl LoRAConfig {
    /// Create a new LoRA config.
    pub fn new(rank: usize, alpha: f32) -> Self {
        LoRAConfig {
            rank,
            alpha,
            ..Default::default()
        }
    }

    /// Create a QLoRA config.
    pub fn new_qlora(rank: usize, alpha: f32, double_quant: bool) -> Self {
        LoRAConfig {
            rank,
            alpha,
            use_qlora: true,
            double_quant,
            ..Default::default()
        }
    }

    /// Create a DoRA config.
    pub fn new_dora(rank: usize, alpha: f32) -> Self {
        LoRAConfig {
            rank,
            alpha,
            use_dora: true,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_lora_adapter_creation() {
        let adapter = LoRAAdapter::new(768, 768, 8, 16.0, 0.0);
        assert_eq!(adapter.rank, 8);
        assert_eq!(adapter.alpha, 16.0);
        assert_eq!(adapter.dropout, 0.0);
    }

    #[test]
    fn test_lora_update_shape() {
        let adapter = LoRAAdapter::new(768, 768, 8, 16.0, 0.0);
        let update = adapter.get_update();
        let shape = update.lock().storage.shape().to_vec();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0], 768);
        assert_eq!(shape[1], 768);
    }

    #[test]
    fn test_lora_update_zero_initial() {
        // B is initialized to zero, so update should be zero
        let adapter = LoRAAdapter::new(768, 768, 8, 16.0, 0.0);
        let update = adapter.get_update();
        let arr = update.lock().storage.to_f32_array();
        for &v in arr.iter() {
            assert!((v - 0.0).abs() < 1e-6, "Expected zero update, got {}", v);
        }
    }

    #[test]
    fn test_lora_parameters() {
        let adapter = LoRAAdapter::new(768, 768, 8, 16.0, 0.0);
        let params = adapter.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_lora_named_parameters() {
        let adapter = LoRAAdapter::new(768, 768, 8, 16.0, 0.0);
        let params = adapter.named_parameters("test");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "test.lora_down.weight");
        assert_eq!(params[1].0, "test.lora_up.weight");
    }

    #[test]
    #[should_panic(expected = "LoRA rank must be positive")]
    fn test_lora_zero_rank_panics() {
        LoRAAdapter::new(768, 768, 0, 16.0, 0.0);
    }

    #[test]
    fn test_q_lora_adapter() {
        let qlora = QLoRAAdapter::new(768, 768, 8, 16.0, 0.0, 0.1, 0.0, true);
        let params = qlora.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_dora_adapter() {
        let dora = DoRAAdapter::new(768, 768, 8, 16.0, 0.0, 1.0);
        let params = dora.parameters();
        assert_eq!(params.len(), 3); // lora down, lora up, magnitude
    }

    #[test]
    fn test_lora_config_defaults() {
        let config = LoRAConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert!(!config.use_qlora);
        assert!(!config.use_dora);
    }

    #[test]
    fn test_lora_config_qlora() {
        let config = LoRAConfig::new_qlora(4, 8.0, true);
        assert!(config.use_qlora);
        assert!(config.double_quant);
        assert_eq!(config.rank, 4);
    }

    #[test]
    fn test_lora_config_dora() {
        let config = LoRAConfig::new_dora(16, 32.0);
        assert!(config.use_dora);
        assert_eq!(config.rank, 16);
    }
}
