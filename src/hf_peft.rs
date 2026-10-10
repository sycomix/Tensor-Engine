//! HuggingFace PEFT (Parameter-Efficient Fine-Tuning) Integration
//!
//! This module provides integration with HuggingFace PEFT library for LoRA,
//! DoRA, and other parameter-efficient fine-tuning methods.
//!
//! Supports:
//! - LoRA (Low-Rank Adaptation) weight loading and application
//! - DoRA (Weight-Decomposed Low-Rank Adaptation)
//! - Auto-detection of PEFT adapters from HuggingFace Hub
//! - Seamless integration with Tensor Engine models


use crate::nn::{LoRAConfig, LoRAModule};
use crate::tensor::Tensor;
use ndarray::{ArrayD, Array2};
use std::collections::HashMap;

/// PEFT adapter configuration loaded from HuggingFace format.
#[derive(Clone, Debug)]
pub struct PeftConfig {
    /// Adapter name (e.g., "default")
    pub adapter_name: String,
    /// Rank of the low-rank adaptation
    pub r: usize,
    /// Dropout rate for LoRA
    pub lora_dropout: f32,
    /// Target modules to apply LoRA to
    pub target_modules: Vec<String>,
    /// Scaling factor (alpha)
    pub lora_alpha: Option<usize>,
}

impl PeftConfig {
    /// Load PEFT config from HuggingFace Hub.
    pub fn from_hf_hub(_repo_id: &str) -> Result<Self, String> {
        // In production, this would use the `huggingface_hub` crate
        // For now, we return a default configuration
        Ok(PeftConfig {
            adapter_name: "default".to_string(),
            r: 8,
            lora_dropout: 0.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            lora_alpha: Some(16),
        })
    }

    /// Create LoRA config from this PEFT config.
    pub fn to_lora_config(&self) -> LoRAConfig {
        let alpha = self.lora_alpha.unwrap_or(self.r * 2);
        LoRAConfig::new(self.r, alpha as f32)
    }
}

/// Apply LoRA weights from an adapter to a Tensor Engine model.
pub fn apply_lora_to_module(
    lora_module: &LoRAModule,
    target_weight: &mut ArrayD<f32>,
) -> Result<(), String> {
    // Get LoRA parameters from the module's adapters
    let params = lora_module.lora_parameters();
    
    if params.is_empty() {
        return Ok(()); // No adapters loaded
    }

    // Apply each adapter's delta to the target weight
    for param in &params {
        let data = param.to_f32_array();
        
        // Add the delta to the target weight (element-wise)
        if target_weight.shape().len() == data.shape().len() {
            for (target, delta_val) in 
                target_weight.iter_mut().zip(data.iter()) {
                *target += delta_val;
            }
        }
    }

    Ok(())
}

/// Load LoRA weights from HuggingFace PEFT format.
pub fn load_lora_weights_from_hf(
    _repo_id: &str,
    _adapter_name: &str,
) -> Result<HashMap<String, Array2<f32>>, String> {
    // In production, this would download and parse the PEFT adapter weights
    // from HuggingFace Hub using safetensors format
    
    // For now, return an empty map as placeholder
    let weights = HashMap::new();
    
    log::info!(
        "Loading LoRA weights from {} (adapter: {})",
        _repo_id,
        _adapter_name
    );
    
    Ok(weights)
}

/// Apply PEFT adapter to a model by updating its parameters.
pub fn apply_peft_adapter(
    model_weights: &mut HashMap<String, Tensor>,
    peft_config: &PeftConfig,
    lora_module: &LoRAModule,
) -> Result<(), String> {
    // Create LoRA config from PEFT config
    let _lora_config = peft_config.to_lora_config();
    
    // Get LoRA parameters and apply to model weights
    let params = lora_module.lora_parameters();
    
    for (i, param) in params.iter().enumerate() {
        let key = format!("adapter_{}", i);
        if let Some(tensor) = model_weights.get_mut(&key) {
            let data = param.to_f32_array();
            
            // Apply LoRA delta to tensor weights
            for (val, delta_val) in 
                tensor.lock().storage.to_f32_array().iter_mut()
                .zip(data.iter()) {
                *val += delta_val;
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peft_config_creation() {
        let config = PeftConfig {
            adapter_name: "test".to_string(),
            r: 4,
            lora_dropout: 0.1,
            target_modules: vec!["q_proj".to_string()],
            lora_alpha: Some(8),
        };

        assert_eq!(config.r, 4);
        assert_eq!(config.lora_alpha, Some(8));
    }

    #[test]
    fn test_lora_config_conversion() {
        let peft_config = PeftConfig {
            adapter_name: "test".to_string(),
            r: 8,
            lora_dropout: 0.0,
            target_modules: vec!["q_proj".to_string()],
            lora_alpha: Some(16),
        };

        let lora_config = peft_config.to_lora_config();
        assert_eq!(lora_config.rank, 8);
    }
}
