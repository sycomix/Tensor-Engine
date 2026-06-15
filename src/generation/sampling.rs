use crate::tensor::Tensor;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct SamplingResult {
    pub token: usize,
    pub prob: f32,                 // Probability of the selected token
    pub distribution: Array1<f32>, // Full distribution for debugging/rejection
}

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub rng: StdRng,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Sampler {
            temperature,
            top_k,
            top_p,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample from logits.
    /// Input logits: [vocab_size] (1D tensor or 2D [1, vocab])
    pub fn sample(&mut self, logits: &Tensor) -> SamplingResult {
        let logits_arr = logits.to_f32_array();

        // Flatten
        let flattened: Vec<f32> = logits_arr.iter().cloned().collect();
        let mut probs = Array1::from_vec(flattened);

        // 1. Temperature
        if self.temperature > 0.0 {
            probs.mapv_inplace(|x| x / self.temperature);
        }

        // 2. Softmax
        self.softmax_inplace(&mut probs);

        // 3. Top-K / Top-P
        self.apply_top_k_p(&mut probs);

        // 4. Sample
        let (token, prob) = self.sample_from_probs(&probs);

        SamplingResult {
            token,
            prob,
            distribution: probs,
        }
    }

    pub fn sample_from_probs(&mut self, probs: &Array1<f32>) -> (usize, f32) {
        let r: f32 = self.rng.random();
        let mut cdf = 0.0;
        let mut chosen_token = probs.len() - 1;
        let mut chosen_prob = probs[chosen_token];

        for (i, &p) in probs.iter().enumerate() {
            cdf += p;
            if r < cdf {
                chosen_token = i;
                chosen_prob = p;
                break;
            }
        }
        (chosen_token, chosen_prob)
    }

    fn softmax_inplace(&self, x: &mut Array1<f32>) {
        let max = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        x.mapv_inplace(|v| (v - max).exp());
        let sum = x.sum();
        if sum > 0.0 {
            x.mapv_inplace(|v| v / sum);
        }
    }

    fn apply_top_k_p(&self, probs: &mut Array1<f32>) {
        let mut pairs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        let top_k_limit = if self.top_k > 0 && self.top_k < pairs.len() {
            self.top_k
        } else {
            pairs.len()
        };
        if top_k_limit < pairs.len() {
            let (selected, _, _) = pairs.select_nth_unstable_by(top_k_limit, |a, b| {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });
            selected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            pairs.truncate(top_k_limit);
        } else {
            pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        }

        let mut cutoff_index = pairs.len();

        // Top-P
        if self.top_p < 1.0 {
            let mut cum_prob = 0.0;
            for (i, &(_, p)) in pairs.iter().enumerate() {
                if i >= cutoff_index {
                    break;
                }
                cum_prob += p;
                if cum_prob > self.top_p {
                    cutoff_index = i + 1; // Include this one to cross threshold
                    break;
                }
            }
        }

        let truncated = cutoff_index < pairs.len();

        if truncated {
            // Filter pairs
            pairs.truncate(cutoff_index);

            // Zero out original probs
            probs.fill(0.0);
            let mut sum = 0.0;
            for (idx, p) in pairs.iter() {
                probs[*idx] = *p;
                sum += *p;
            }

            // Renormalize
            if sum > 0.0 {
                probs.mapv_inplace(|x| x / sum);
            } else {
                // Fallback: uniform over selected
                let n = pairs.len() as f32;
                for (idx, _) in pairs.iter() {
                    probs[*idx] = 1.0 / n;
                }
            }
        }
    }
}
