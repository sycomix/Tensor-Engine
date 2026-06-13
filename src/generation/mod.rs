//! Autoregressive text generation and sampling utilities.
//!
//! Provides token sampling strategies (greedy, top-k, top-p), KV-cache-backed
//! incremental decoding, and speculative decoding for accelerated inference.

pub mod sampling;
pub mod speculative;

// Re-export public types so consumers can use `crate::generation::*`
pub use sampling::{Sampler, SamplingResult};
pub use speculative::{SpeculativeModel, SpeculativeSampler};

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
            log::warn!(
                "KV cache enabled but max_new_tokens=0; cache will not be used"
            );
        }
        Ok(())
    }
}
