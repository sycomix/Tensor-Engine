use super::tokenizer::Tokenizer;
use std::collections::BTreeMap;

/// A minimal token sampler that operates on logit vectors of shape (vocab_size, 1).
pub struct TokenSampler {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
}

impl Default for TokenSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenSampler {
    pub fn new() -> Self {
        Self {
            temperature: 0.2,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 0.8,
        }
    }

    pub fn temperature(self, temperature: f32) -> Self {
        Self { temperature, ..self }
    }

    pub fn top_p(self, top_p: f32) -> Self {
        Self { top_p, ..self }
    }

    pub fn top_k(self, top_k: usize) -> Self {
        Self { top_k, ..self }
    }

    pub fn repetition_penalty(self, repetition_penalty: f32) -> Self {
        Self { repetition_penalty, ..self }
    }

    pub fn sample_ids(
        &self,
        logits: &[f32],
        vocab_size: usize,
        _tokenizer: &Tokenizer,
        existing_tokens: &[usize],
    ) -> (usize, f32) {
        assert!(logits.len() == vocab_size);
        let mut times_used: BTreeMap<usize, usize> = BTreeMap::new();
        for token in existing_tokens {
            times_used.entry(*token).and_modify(|e| *e += 1).or_insert(1);
        }

        // Make a mutable copy in Vec<f32> for processing
        let mut l: Vec<f32> = logits.to_vec();

        // Apply temperature
        if self.temperature > 0.0 {
            for v in l.iter_mut() {
                *v /= self.temperature;
            }
        }

        // Apply repetition penalty
        if (self.repetition_penalty - 1.0).abs() > f32::EPSILON {
            for (i, val) in l.iter_mut().enumerate() {
                if let Some(count) = times_used.get(&i) {
                    let penalty = self.repetition_penalty.powf(*count as f32);
                    *val *= penalty;
                }
            }
        }

        // Numerically stable softmax
        let maxv = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for v in l.iter_mut() {
            *v -= maxv;
        }
        // Convert to probabilities
        let expv: Vec<f32> = l.iter().map(|x| x.exp()).collect();

        // Pair (id, score)
        let mut logitsf: Vec<(usize, f32)> = (0..vocab_size).map(|i| (i, expv[i])).collect();
        // Sort descending
        logitsf.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to top_k
        if logitsf.len() > self.top_k {
            logitsf.truncate(self.top_k);
        }

        // Apply top_p
        let mut p_accum: f32 = 0.0;
        for (idx, v) in logitsf.iter().enumerate() {
            p_accum += v.1;
            if p_accum >= self.top_p {
                logitsf.truncate(idx + 1);
                break;
            }
        }

        let total_p: f32 = logitsf.iter().map(|x| x.1).sum();
        // Use `rand::random::<f32>()` to avoid deprecated gen_range warnings
        let p: f32 = if total_p > 0.0 { rand::random::<f32>() * total_p } else { 0.0 };
        p_accum = 0.0;
        for v in logitsf.into_iter() {
            p_accum += v.1;
            if p_accum >= p {
                return (v.0, v.1 / total_p);
            }
        }
        (0, 0.0)
    }

    pub fn logits_to_btreemap(&self, logits: &[f32], vocab_size: usize, tokenizer: &Tokenizer) -> BTreeMap<String, f32> {
        let mut result = BTreeMap::new();
        for (token_idx, &score) in logits.iter().enumerate().take(vocab_size) {
            let tok = tokenizer.vocab.iter().find(|(_, &id)| id == token_idx).map(|(s, _)| s.clone()).unwrap_or_else(|| format!("<{}>", token_idx));
            result.insert(tok, score);
        }
        result
    }
}
