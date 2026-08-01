//! Autoregressive text generation and sampling utilities.
//!
//! Provides token sampling strategies (greedy, top-k, top-p), KV-cache-backed
//! incremental decoding, speculative decoding for accelerated inference, and a
//! high-level `generate()` function that orchestrates the full loop.

use crate::tensor::Tensor;

pub mod sampling;
pub mod speculative;

// Re-export public types so consumers can use `crate::generation::*`
pub use sampling::{Sampler, SamplingResult};
pub use speculative::{SpeculativeModel, SpeculativeSampler};

/// Error types for text generation failures.
#[derive(Debug, Clone)]
pub enum GenerationError {
    /// Prompt contains no tokens.
    EmptyPrompt,
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Model inference failed during token step.
    InferenceFailed(String),
}

impl From<String> for GenerationError {
    fn from(s: String) -> Self {
        GenerationError::InferenceFailed(s)
    }
}

impl std::fmt::Display for GenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerationError::EmptyPrompt => write!(f, "prompt must contain at least one token"),
            GenerationError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            GenerationError::InferenceFailed(msg) => write!(f, "inference failed: {}", msg),
        }
    }
}

impl std::error::Error for GenerationError {}

/// Configuration for autoregressive text generation.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate (excluding prompt).
    pub max_new_tokens: usize,
    /// Sampling temperature; higher = more random, lower = more deterministic.
    pub temperature: f32,
    /// Top-k sampling parameter (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) sampling parameter (1.0 = disabled).
    pub top_p: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Whether to use KV cache for incremental decoding.
    pub use_kv_cache: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
            use_kv_cache: true,
        }
    }
}

impl GenerationConfig {
    /// Create a greedy generation config (temperature=1.0, no sampling).
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
            use_kv_cache: true,
        }
    }

    /// Create a nucleus-sampling config.
    pub fn nucleus(max_new_tokens: usize, temperature: f32, top_p: f32, seed: u64) -> Self {
        Self {
            max_new_tokens,
            temperature,
            top_k: 0,
            top_p,
            seed,
            use_kv_cache: true,
        }
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return Err(format!(
                "temperature must be finite and > 0, got {}",
                self.temperature
            ));
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err(format!("top_p must be in [0, 1], got {}", self.top_p));
        }
        if self.max_new_tokens == 0 && self.use_kv_cache {
            log::warn!("KV cache enabled but max_new_tokens=0; cache will not be used");
        }
        Ok(())
    }

    /// Create a config for deterministic greedy decoding.
    pub fn greedy_deterministic(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            temperature: 0.0, // Will use argmax directly
            top_k: 1,
            top_p: 1.0,
            seed: 42,
            use_kv_cache: true,
        }
    }
}

/// Generate text autoregressively using a Llama-style model with KV cache.
///
/// This is the canonical generation loop: it feeds the prompt through the model
/// to initialize KV caches (or processes incrementally), then generates tokens
/// one at a time using `forward_single_token` until max_new_tokens or EOS is reached.
///
/// # Arguments
/// * `model` — Mutable reference to a model with `forward_single_token` API (Llama, Mistral, etc.)
/// * `prompt_ids` — Token IDs for the prompt
/// * `config` — Generation configuration
/// * `eos_token_id` — Optional EOS token ID to stop generation early
/// * `causal_offset` — Optional offset for multimodal contexts (image-token count)
///
/// # Returns
/// Vector of generated token IDs including the prompt.
pub fn generate_with_kv_cache(
    model: &mut dyn std::any::Any, // We'll downcast to specific model types
    prompt_ids: &[u32],
    config: &GenerationConfig,
    eos_token_id: Option<u32>,
    causal_offset: Option<usize>,
) -> Result<Vec<u32>, GenerationError> {
    if prompt_ids.is_empty() {
        return Err(GenerationError::EmptyPrompt);
    }

    let mut sampler = Sampler::new(config.temperature, config.top_k, config.top_p, config.seed);
    let mut generated = prompt_ids.to_vec();

    // Process prompt: feed through model to initialize or update KV caches
    // We process the prompt in chunks if needed, but for simplicity we use forward_with_mask first
    // Then switch to single-token steps

    // For now, generate tokens one at a time using the model's forward_single_token API
    // The caller is responsible for having initialized KV caches via init_kv_caches()

    for _ in 0..config.max_new_tokens {
        let last_token = {
            let idx = generated.len() - 1;
            let data = vec![prompt_ids[idx] as f32];
            let shape = ndarray::IxDyn(&[1][..]);
            let arr = ndarray::Array::from_shape_vec(shape, data).map_err(|e| {
                GenerationError::InferenceFailed(format!("tensor creation failed: {e}"))
            })?;
            Tensor::new(arr, false)
        };

        // Get logits from model (downcast to Llama-style models)
        let logits = if let Some(llama) = model.downcast_mut::<crate::nn::Llama>() {
            llama.forward_single_token(&last_token, causal_offset)?
        } else if let Some(mistral) = model.downcast_mut::<crate::nn::Mistral>() {
            mistral.forward_single_token(&last_token, causal_offset)?
        } else if let Some(phi) = model.downcast_mut::<crate::nn::Phi>() {
            phi.forward_single_token(&last_token, causal_offset)?
        } else if let Some(qwen) = model.downcast_mut::<crate::nn::Qwen>() {
            qwen.forward_single_token(&last_token, causal_offset)?
        } else if let Some(gemma) = model.downcast_mut::<crate::nn::Gemma>() {
            gemma.forward_single_token(&last_token, causal_offset)?
        } else {
            return Err(GenerationError::InferenceFailed(
                "unsupported model type for generation".to_string(),
            ));
        };

        // Sample next token
        let result = sampler.sample(&logits);
        let next_token = result.token as u32;
        generated.push(next_token);

        log::debug!(
            "Generated token {} (prob={:.4}), total tokens: {}",
            next_token,
            result.prob,
            generated.len()
        );

        // Stop at EOS token
        if let Some(eos) = eos_token_id {
            if next_token == eos {
                log::info!(
                    "EOS token reached at step {}, stopping generation",
                    generated.len()
                );
                break;
            }
        }
    }

    Ok(generated)
}

/// Generate text using a Llama-style model directly (convenience wrapper).
///
/// This is the canonical generation loop: it feeds the prompt through the model
/// to initialize or update KV caches, then generates tokens one at a time using
/// `forward_single_token` until max_new_tokens or EOS is reached.
///
/// # Arguments
/// * `model` — Mutable reference to a Llama-style model (Llama, Mistral, Phi, Qwen, Gemma)
/// * `prompt_ids` — Token IDs for the prompt
/// * `config` — Generation configuration
/// * `eos_token_id` — Optional EOS token ID to stop generation early
/// * `causal_offset` — Optional offset for multimodal contexts (image-token count)
///
/// # Returns
/// Vector of generated token IDs including the prompt.
pub fn generate_llama_style<M>(
    model: &mut M,
    prompt_ids: &[u32],
    config: &GenerationConfig,
    eos_token_id: Option<u32>,
    causal_offset: Option<usize>,
) -> Result<Vec<u32>, GenerationError>
where
    M: crate::nn::Module + crate::nn::LlamaStyleModel,
{
    if prompt_ids.is_empty() {
        return Err(GenerationError::EmptyPrompt);
    }

    let mut sampler = Sampler::new(config.temperature, config.top_k, config.top_p, config.seed);
    let mut generated = prompt_ids.to_vec();

    // Initialize KV caches for the full sequence length we might need
    let max_seq_len = prompt_ids.len() + config.max_new_tokens;
    model.init_kv_caches(max_seq_len).map_err(|e| {
        GenerationError::InferenceFailed(format!("failed to init KV caches: {}", e))
    })?;

    // Process prompt tokens one at a time to populate KV cache
    for &token_id in &prompt_ids[..prompt_ids.len().saturating_sub(1)] {
        let last_token = {
            let arr =
                ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1][..]), vec![token_id as f32])
                    .map_err(|e| {
                        GenerationError::InferenceFailed(format!("tensor creation failed: {e}"))
                    })?;
            Tensor::new(arr, false)
        };
        model
            .forward_single_token(&last_token, causal_offset)
            .map_err(|e| {
                GenerationError::InferenceFailed(format!("prompt processing failed: {}", e))
            })?;
    }

    // Generate new tokens
    for _ in 0..config.max_new_tokens {
        let last_token = {
            let idx = generated.len() - 1;
            let arr = ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[1][..]),
                vec![generated[idx] as f32],
            )
            .map_err(|e| {
                GenerationError::InferenceFailed(format!("tensor creation failed: {e}"))
            })?;
            Tensor::new(arr, false)
        };

        let logits = model
            .forward_single_token(&last_token, causal_offset)
            .map_err(|e| GenerationError::InferenceFailed(format!("inference failed: {}", e)))?;

        let result = sampler.sample(&logits);
        let next_token = result.token as u32;
        generated.push(next_token);

        log::debug!(
            "Generated token {} (prob={:.4}), total tokens: {}",
            next_token,
            result.prob,
            generated.len()
        );

        if let Some(eos) = eos_token_id {
            if next_token == eos {
                log::info!(
                    "EOS token reached at step {}, stopping generation",
                    generated.len()
                );
                break;
            }
        }
    }

    Ok(generated)
}
