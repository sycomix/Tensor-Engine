use super::model::{GPTModel, GPTModelError};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    Greedy,
    TopK { k: usize },
    TopP { p: f32 },
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub strategy: SamplingStrategy,
    pub eos_token_id: Option<u32>,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            temperature: 1.0,
            strategy: SamplingStrategy::Greedy,
            eos_token_id: None,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceError {
    EmptyPrompt,
    InvalidConfig(&'static str),
    Model(GPTModelError),
}

impl Display for InferenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::EmptyPrompt => write!(f, "prompt must contain at least one token"),
            InferenceError::InvalidConfig(msg) => write!(f, "invalid generation config: {}", msg),
            InferenceError::Model(err) => write!(f, "model inference failed: {}", err),
        }
    }
}

impl Error for InferenceError {}

pub fn generate(
    model: &GPTModel,
    prompt: &[u32],
    cfg: GenerationConfig,
) -> Result<Vec<u32>, InferenceError> {
    if prompt.is_empty() {
        return Err(InferenceError::EmptyPrompt);
    }
    if cfg.max_new_tokens == 0 {
        return Ok(prompt.to_vec());
    }
    if cfg.temperature <= 0.0 || !cfg.temperature.is_finite() {
        return Err(InferenceError::InvalidConfig(
            "temperature must be finite and > 0",
        ));
    }
    if let SamplingStrategy::TopP { p } = cfg.strategy {
        if !(0.0..=1.0).contains(&p) {
            return Err(InferenceError::InvalidConfig(
                "top-p value must be in [0, 1]",
            ));
        }
    }

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut generated = prompt.to_vec();
    let mut cache = model.new_decode_cache();
    let mut logits = model
        .try_prefill_decode_cache(prompt, &mut cache)
        .map_err(InferenceError::Model)?;

    for step in 0..cfg.max_new_tokens {
        let next = sample_next_token(&logits, cfg.temperature, cfg.strategy, &mut rng)?;
        generated.push(next as u32);

        if cfg.eos_token_id == Some(next as u32) {
            break;
        }
        if step + 1 < cfg.max_new_tokens {
            logits = model
                .try_decode_next_logits(next as u32, &mut cache)
                .map_err(InferenceError::Model)?;
        }
    }

    Ok(generated)
}

fn sample_next_token(
    logits: &[f32],
    temperature: f32,
    strategy: SamplingStrategy,
    rng: &mut StdRng,
) -> Result<usize, InferenceError> {
    if logits.is_empty() {
        return Err(InferenceError::InvalidConfig("logit row is empty"));
    }

    match strategy {
        SamplingStrategy::Greedy => {
            let mut best_idx = 0usize;
            let mut best_logit = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_logit {
                    best_logit = v;
                    best_idx = i;
                }
            }
            Ok(best_idx)
        }
        SamplingStrategy::TopK { k } => {
            if k == 0 {
                return Err(InferenceError::InvalidConfig("top-k must be > 0"));
            }
            let selected = top_k_indices(logits, k);
            sample_from_indices(logits, selected, temperature, rng)
        }
        SamplingStrategy::TopP { p } => {
            if p <= 0.0 {
                return Err(InferenceError::InvalidConfig("top-p must be > 0"));
            }

            let probs = softmax_temperature(logits, temperature);
            let mut idxs: Vec<usize> = (0..probs.len()).collect();
            idxs.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));

            let mut cumulative = 0.0_f32;
            let mut selected = Vec::new();
            for idx in idxs {
                selected.push(idx);
                cumulative += probs[idx];
                if cumulative >= p {
                    break;
                }
            }

            if selected.is_empty() {
                return Err(InferenceError::InvalidConfig(
                    "top-p selected no candidates",
                ));
            }

            sample_from_probs(&probs, &selected, rng)
        }
    }
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let keep = k.min(logits.len());
    let mut idxs: Vec<usize> = (0..logits.len()).collect();
    if keep == idxs.len() {
        idxs.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
        return idxs;
    }

    let (selected, _, _) = idxs.select_nth_unstable_by(keep, |&a, &b| {
        logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b))
    });
    selected.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    selected.to_vec()
}

fn sample_from_indices(
    logits: &[f32],
    indices: Vec<usize>,
    temperature: f32,
    rng: &mut StdRng,
) -> Result<usize, InferenceError> {
    let mut filtered = Vec::with_capacity(indices.len());
    for &idx in &indices {
        filtered.push(logits[idx]);
    }
    let probs = softmax_temperature(&filtered, temperature);

    let mut cumulative = 0.0_f32;
    let draw = rng.random::<f32>();
    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if draw <= cumulative {
            return Ok(indices[i]);
        }
    }

    indices
        .last()
        .copied()
        .ok_or(InferenceError::InvalidConfig("empty candidate set"))
}

fn sample_from_probs(
    probs: &[f32],
    indices: &[usize],
    rng: &mut StdRng,
) -> Result<usize, InferenceError> {
    let mut cumulative = 0.0_f32;
    let draw = rng.random::<f32>();

    let mut total = 0.0_f32;
    for &idx in indices {
        total += probs[idx];
    }
    if total <= 0.0 || !total.is_finite() {
        return Err(InferenceError::InvalidConfig(
            "invalid probability mass for sampling",
        ));
    }

    for &idx in indices {
        cumulative += probs[idx] / total;
        if draw <= cumulative {
            return Ok(idx);
        }
    }

    indices.last().copied().ok_or(InferenceError::InvalidConfig(
        "empty probability candidate set",
    ))
}

fn softmax_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let mut max_v = f32::NEG_INFINITY;
    for &v in logits {
        let scaled = v / temperature;
        if scaled > max_v {
            max_v = scaled;
        }
    }

    let mut exps = vec![0.0_f32; logits.len()];
    let mut sum = 0.0_f32;
    for (i, &v) in logits.iter().enumerate() {
        let e = ((v / temperature) - max_v).exp();
        exps[i] = e;
        sum += e;
    }

    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }

    exps.iter().map(|v| v / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::{generate, GenerationConfig, SamplingStrategy};
    use crate::nn::gpt::model::{GPTConfig, GPTModel};

    #[test]
    fn greedy_generation_extends_sequence() {
        let model = GPTModel::from_config(GPTConfig {
            vocab_size: 32,
            max_seq_len: 16,
            embedding_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 2,
            seed: 7,
            tie_weights: true,
        })
        .unwrap();

        let out = generate(
            &model,
            &[1, 2, 3],
            GenerationConfig {
                max_new_tokens: 4,
                temperature: 1.0,
                strategy: SamplingStrategy::Greedy,
                eos_token_id: None,
                seed: 123,
            },
        )
        .unwrap();

        assert_eq!(out.len(), 7);
    }

    #[test]
    fn top_k_is_deterministic_for_fixed_seed() {
        let model = GPTModel::from_config(GPTConfig {
            vocab_size: 24,
            max_seq_len: 12,
            embedding_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 2,
            seed: 11,
            tie_weights: true,
        })
        .unwrap();

        let cfg = GenerationConfig {
            max_new_tokens: 3,
            temperature: 0.9,
            strategy: SamplingStrategy::TopK { k: 5 },
            eos_token_id: None,
            seed: 999,
        };

        let a = generate(&model, &[1, 5], cfg).unwrap();
        let b = generate(&model, &[1, 5], cfg).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn greedy_generation_matches_full_recompute_loop() {
        let model = GPTModel::from_config(GPTConfig {
            vocab_size: 28,
            max_seq_len: 16,
            embedding_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 3,
            seed: 202,
            tie_weights: true,
        })
        .unwrap();

        let prompt = [2, 4, 6];
        let cfg = GenerationConfig {
            max_new_tokens: 5,
            temperature: 1.0,
            strategy: SamplingStrategy::Greedy,
            eos_token_id: None,
            seed: 1,
        };

        let cached = generate(&model, &prompt, cfg).unwrap();
        let mut recompute = prompt.to_vec();
        for _ in 0..cfg.max_new_tokens {
            let logits = model.try_forward(&recompute, false).unwrap();
            let last = logits.last().unwrap();
            let mut best_idx = 0usize;
            let mut best_logit = f32::NEG_INFINITY;
            for (idx, &value) in last.iter().enumerate() {
                if value > best_logit {
                    best_logit = value;
                    best_idx = idx;
                }
            }
            recompute.push(best_idx as u32);
        }

        assert_eq!(cached, recompute);
    }
}
