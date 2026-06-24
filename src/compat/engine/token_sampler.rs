use super::tensor::Tensor;
use super::tokenizer::{TokenId, Tokenizer};
use rand::Rng;
use std::collections::{BTreeMap, HashMap};

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
            top_k: 1,                // same as argmax
            repetition_penalty: 1.0, // 1.0 = no penalty. above 1.0 penalizes repeats, below 1.0 encourages
        }
    }

    pub fn get_temperature(&self) -> f32 {
        self.temperature
    }

    pub fn get_top_p(&self) -> f32 {
        self.top_p
    }

    pub fn get_top_k(&self) -> usize {
        self.top_k
    }

    pub fn get_repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }

    pub fn temperature(self, temperature: f32) -> Self {
        Self {
            temperature,
            ..self
        }
    }

    pub fn top_p(self, top_p: f32) -> Self {
        Self { top_p, ..self }
    }

    pub fn top_k(self, top_k: usize) -> Self {
        Self { top_k, ..self }
    }

    pub fn repetition_penalty(self, repetition_penalty: f32) -> Self {
        Self {
            repetition_penalty,
            ..self
        }
    }

    pub fn logits_to_btreemap(
        &self,
        logits: &Tensor,
        tokenizer: &Tokenizer,
    ) -> BTreeMap<String, f32> {
        let mut result = BTreeMap::new();
        for token_idx in 0..logits.rows() {
            result.insert(
                tokenizer.id_to_str(token_idx as TokenId).to_string(),
                logits.get_f32(token_idx, 0),
            );
        }
        result
    }

    pub fn sample(
        &self,
        logits: &Tensor,
        _tokenizer: &Tokenizer,
        existing_tokens: &[TokenId],
    ) -> (TokenId, f32) {
        self.sample_logits(logits, existing_tokens)
    }

    fn sample_logits(&self, logits: &Tensor, existing_tokens: &[TokenId]) -> (TokenId, f32) {
        assert!(logits.cols() == 1);
        if logits.rows() == 0 {
            return (0, 0.0);
        }

        if (self.top_k == 1 || self.temperature <= 0.0) && self.repetition_penalty == 1.0 {
            let nrows = logits.rows();
            let mut best_idx: TokenId = 0;
            let mut best_val = logits.get_f32(0, 0);
            for token_idx in 1..nrows {
                let v = logits.get_f32(token_idx, 0);
                if v > best_val {
                    best_val = v;
                    best_idx = token_idx as TokenId;
                }
            }
            return (best_idx, 1.0);
        }

        let mut times_used: HashMap<TokenId, usize> = HashMap::new();
        for &token in existing_tokens {
            times_used.entry(token).and_modify(|e| *e += 1).or_insert(1);
        }

        let nrows = logits.rows();
        let mut logitsf: Vec<(TokenId, f32)> = Vec::with_capacity(nrows as usize);
        let inv_temperature = if self.temperature > 0.0 {
            1.0 / self.temperature
        } else {
            1.0
        };
        for token_idx in 0..nrows {
            let token_id = token_idx as TokenId;
            let mut score = logits.get_f32(token_idx, 0) * inv_temperature;
            if self.repetition_penalty != 1.0 {
                if let Some(count) = times_used.get(&token_id) {
                    let penalty = self.repetition_penalty.powf(*count as f32);
                    score /= penalty;
                }
            }
            logitsf.push((token_id, score));
        }

        sample_from_scored(&mut logitsf, self.top_k, self.top_p)
    }

    pub fn sample_projected(
        &self,
        hidden: &Tensor,
        output_weight: &Tensor,
        existing_tokens: &[TokenId],
    ) -> (TokenId, f32) {
        assert!(hidden.rows() == 1);
        assert!(hidden.cols() == output_weight.cols());

        let vocab_size = output_weight.rows();
        if vocab_size == 0 {
            return (0, 0.0);
        }
        if (self.top_k == 1 || self.temperature <= 0.0) && self.repetition_penalty == 1.0 {
            let logits = output_weight.matrix_vector_mul_transposed(hidden);
            let nrows = logits.rows();
            let mut best_idx: TokenId = 0;
            let mut best_val = f32::NEG_INFINITY;
            for row in 0..nrows {
                let val = logits.get_f32(row, 0);
                if val > best_val {
                    best_val = val;
                    best_idx = row as TokenId;
                }
            }
            return (best_idx, 1.0);
        }

        let logits = output_weight.matrix_mul_transposed(hidden);
        self.sample_logits(&logits, existing_tokens)
    }
}

fn compare_scores_desc(a: &(TokenId, f32), b: &(TokenId, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}

fn sample_from_scored(
    scored: &mut Vec<(TokenId, f32)>,
    top_k: usize,
    top_p: f32,
) -> (TokenId, f32) {
    if scored.is_empty() {
        return (0, 0.0);
    }

    let keep = if top_k == 0 {
        scored.len()
    } else {
        top_k.min(scored.len())
    };
    if keep < scored.len() {
        let (_, _, _) = scored.select_nth_unstable_by(keep, |a, b| compare_scores_desc(a, b));
        scored.truncate(keep);
    }
    scored.sort_unstable_by(compare_scores_desc);

    let maxv = scored
        .first()
        .map(|(_, score)| *score)
        .unwrap_or(f32::NEG_INFINITY);
    let mut total_exp = 0.0_f32;
    for (_, score) in scored.iter_mut() {
        let prob = (*score - maxv).exp();
        *score = prob;
        total_exp += prob;
    }
    if total_exp <= 0.0 || !total_exp.is_finite() {
        let uniform = 1.0 / scored.len().max(1) as f32;
        for (_, prob) in scored.iter_mut() {
            *prob = uniform;
        }
    } else {
        for (_, prob) in scored.iter_mut() {
            *prob /= total_exp;
        }
    }

    let mut p_accum: f32 = 0.0;
    for (idx, value) in scored.iter().enumerate() {
        p_accum += value.1;
        if p_accum >= top_p {
            scored.truncate(idx + 1);
            break;
        }
    }

    let mut total_p: f32 = 0.0;
    for value in scored.iter() {
        total_p += value.1;
    }
    let mut rng = rand::rng();
    let p: f32 = if total_p > 0.0 {
        rng.random_range(0.0..=total_p)
    } else {
        0.0
    };
    p_accum = 0.0;
    for value in scored.iter() {
        p_accum += value.1;
        if p_accum >= p {
            return (value.0, value.1 / total_p);
        }
    }
    (0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::TokenSampler;
    use crate::compat::engine::tensor::{Tensor, TensorDType};
    use crate::compat::engine::tokenizer::Tokenizer;

    #[test]
    fn greedy_sampling_picks_largest_logit() {
        let tokenizer = Tokenizer::empty_for_tests();
        let mut logits = Tensor::zeros(4, 1, TensorDType::Float32);
        logits.set_f32(0, 0, -1.0);
        logits.set_f32(1, 0, 0.5);
        logits.set_f32(2, 0, 2.0);
        logits.set_f32(3, 0, 1.5);

        let (token, probability) = TokenSampler::new().sample(&logits, &tokenizer, &[]);
        assert_eq!(token, 2);
        assert_eq!(probability, 1.0);
    }

    #[test]
    fn repetition_penalty_applies_to_all_vocabulary_rows() {
        let tokenizer = Tokenizer::empty_for_tests();
        let mut logits = Tensor::zeros(3, 1, TensorDType::Float32);
        logits.set_f32(0, 0, 0.0);
        logits.set_f32(1, 0, 3.0);
        logits.set_f32(2, 0, 2.9);

        let sampler = TokenSampler::new()
            .temperature(0.8)
            .top_k(1)
            .repetition_penalty(2.0);
        let (token, _) = sampler.sample(&logits, &tokenizer, &[1]);
        assert_eq!(token, 2);
    }

    #[test]
    fn projected_sampling_matches_explicit_logits_for_greedy() {
        let tokenizer = Tokenizer::empty_for_tests();
        let hidden = Tensor::random(1, 6, TensorDType::Float32);
        let output = Tensor::random(9, 6, TensorDType::Float32);
        let logits = output.matrix_mul_transposed(&hidden);
        let sampler = TokenSampler::new();

        let projected = sampler.sample_projected(&hidden, &output, &[]);
        let explicit = sampler.sample(&logits, &tokenizer, &[]);
        assert_eq!(projected, explicit);
    }

    #[test]
    fn projected_sampling_matches_explicit_logits_with_repetition_penalty() {
        let tokenizer = Tokenizer::empty_for_tests();
        let mut hidden = Tensor::zeros(1, 3, TensorDType::Float32);
        hidden.set_f32(0, 0, 1.0);
        hidden.set_f32(0, 1, 0.5);
        hidden.set_f32(0, 2, -0.25);

        let mut output = Tensor::zeros(4, 3, TensorDType::Float32);
        output.set_f32(0, 0, 0.2);
        output.set_f32(0, 1, 0.1);
        output.set_f32(0, 2, 0.0);
        output.set_f32(1, 0, 3.0);
        output.set_f32(1, 1, 0.0);
        output.set_f32(1, 2, 0.0);
        output.set_f32(2, 0, 2.8);
        output.set_f32(2, 1, 0.0);
        output.set_f32(2, 2, 0.0);
        output.set_f32(3, 0, 0.0);
        output.set_f32(3, 1, 0.0);
        output.set_f32(3, 2, 1.0);

        let logits = output.matrix_mul_transposed(&hidden);
        let sampler = TokenSampler::new()
            .temperature(0.8)
            .top_k(1)
            .repetition_penalty(2.0);

        let projected = sampler.sample_projected(&hidden, &output, &[1]);
        let explicit = sampler.sample(&logits, &tokenizer, &[1]);
        assert_eq!(projected, explicit);
        assert_eq!(projected.0, 2);
    }
}
