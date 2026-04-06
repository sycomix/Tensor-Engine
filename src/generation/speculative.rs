use crate::generation::sampling::Sampler;
use crate::nn::transformer::Llama;
use crate::nn::KVCache;
use crate::ops::Slice;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

/// Trait for models capable of speculative decoding (draft or target).
/// Requires access to KV cache state for rollback.
pub trait SpeculativeModel {
    /// Forward pass for a sequence of tokens.
    /// Returns logits: [batch, seq, vocab_size]
    /// Mutates internal KV cache.
    fn forward_t(&mut self, input: &Tensor) -> Tensor;

    /// Capture current KV cache state (shallow clone of storage handles).
    fn get_cache_snapshot(&self) -> Vec<Option<KVCache>>;

    /// Restore KV cache state from snapshot.
    fn restore_cache_snapshot(&mut self, snapshot: Vec<Option<KVCache>>);

    /// Truncate the last n tokens from the KV cache.
    fn truncate_cache(&mut self, n: usize);
}

impl SpeculativeModel for Llama {
    fn forward_t(&mut self, input: &Tensor) -> Tensor {
        // Llama::forward_with_mask updates cache if initialized
        self.forward_with_mask(input, None)
    }

    fn get_cache_snapshot(&self) -> Vec<Option<KVCache>> {
        self.layers.iter().map(|l| l.kv_cache_clone()).collect()
    }

    fn restore_cache_snapshot(&mut self, snapshot: Vec<Option<KVCache>>) {
        if snapshot.len() != self.layers.len() {
            log::error!(
                "Snapshot length mismatch: {} vs {}",
                snapshot.len(),
                self.layers.len()
            );
            return;
        }
        for (layer, snap) in self.layers.iter_mut().zip(snapshot.into_iter()) {
            layer.set_kv_cache(snap.unwrap_or_else(|| KVCache::new()));
        }
    }

    fn truncate_cache(&mut self, n: usize) {
        self.truncate_kv_cache(n);
    }
}

pub struct SpeculativeSampler {
    pub draft_model: Box<dyn SpeculativeModel>,
    pub target_model: Box<dyn SpeculativeModel>,
    pub gamma: usize, // lookahead steps
    pub sampler: Sampler,
}

impl SpeculativeSampler {
    pub fn new(
        draft_model: Box<dyn SpeculativeModel>,
        target_model: Box<dyn SpeculativeModel>,
        gamma: usize,
        sampler: Sampler,
    ) -> Self {
        SpeculativeSampler {
            draft_model,
            target_model,
            gamma,
            sampler,
        }
    }

    /// Helper: Slice tensor along specific axis
    fn slice_axis(t: &Tensor, axis: usize, start: usize, len: usize) -> Tensor {
        Tensor::apply(Arc::new(Slice::new(axis, start, len)), &[t.clone()][..])
    }

    /// Helper: Slice last token from sequence. [batch, seq] -> [batch, 1]
    fn slice_last(t: &Tensor) -> Tensor {
        let shape = t.lock().storage.shape().to_vec();
        let axis = shape.len() - 1;
        let len = shape[axis];
        if len == 0 {
            return t.clone();
        }
        Tensor::apply(Arc::new(Slice::new(axis, len - 1, 1)), &[t.clone()][..])
    }

    /// Speculative decoding with rejection sampling.
    /// input: [batch, seq] - The prompt.
    /// max_new_tokens: number of tokens to generate
    pub fn generate(&mut self, input: &Tensor, max_new_tokens: usize) -> Tensor {
        let mut all_tokens = input.clone();

        // 1. Prime draft model with input
        let _ = self.draft_model.forward_t(&input);

        // 2. Prime target model and get initial logits
        let target_logits_full = self.target_model.forward_t(&input);
        // Extract logits for the last token position: [1, seq, vocab] -> [1, 1, vocab]
        let seq_len = target_logits_full.lock().storage.shape()[1];
        let mut last_target_logits = Self::slice_axis(&target_logits_full, 1, seq_len - 1, 1);

        let mut n_generated = 0;

        while n_generated < max_new_tokens {
            // Current loop context
            let mut loop_input = Self::slice_last(&all_tokens);

            // 3. Draft Rollout
            let mut draft_tokens = Vec::with_capacity(self.gamma);
            let mut draft_probs = Vec::with_capacity(self.gamma);

            for _ in 0..self.gamma {
                let logits = self.draft_model.forward_t(&loop_input); // [batch, 1, vocab]
                let sample = self.sampler.sample(&logits);

                let token_tensor = Tensor::new(
                    ArrayD::from_elem(IxDyn(&[1usize, 1][..]), sample.token as f32),
                    false,
                );

                draft_tokens.push(token_tensor.clone());
                draft_probs.push(sample);

                loop_input = token_tensor;
            }

            // 4. Target Verification
            // Input: drafted token sequence [d1 through dK]
            let verification_input = Tensor::concat(&draft_tokens[..], 1);

            // target_logits_full: logits for the drafted token sequence [d1 through dK]
            let target_logits_full = self.target_model.forward_t(&verification_input);

            let k_steps = draft_tokens.len();
            let mut accepted_count = 0;
            let mut correct_token = None;
            let mut next_target_logits = None;

            for k in 0..k_steps {
                let p_logits = if k == 0 {
                    last_target_logits.clone()
                } else {
                    Self::slice_axis(&target_logits_full, 1, k - 1, 1)
                };

                let p_res = self.sampler.sample(&p_logits);
                let p_dist = p_res.distribution;

                let q_res = &draft_probs[k];
                let q_dist = q_res.distribution.clone();

                let draft_token_id = q_res.token;
                let q_x = q_dist[draft_token_id];
                let p_x = p_dist[draft_token_id];

                let acceptance_prob = (p_x / q_x).min(1.0);

                let r: f32 = rand::random();

                if r < acceptance_prob {
                    accepted_count += 1;
                    // Logits for next token (prediction from THIS accepted token)
                    next_target_logits = Some(Self::slice_axis(&target_logits_full, 1, k, 1));
                } else {
                    // Rejected
                    let mut diff = &p_dist - &q_dist;
                    diff.mapv_inplace(|v| v.max(0.0));
                    let sum = diff.sum();
                    let (resampled_id, _) = if sum > 0.0 {
                        diff.mapv_inplace(|v| v / sum);
                        self.sampler.sample_from_probs(&diff)
                    } else {
                        self.sampler.sample_from_probs(&p_dist)
                    };

                    let token_tensor = Tensor::new(
                        ArrayD::from_elem(IxDyn(&[1usize, 1][..]), resampled_id as f32),
                        false,
                    );
                    correct_token = Some(token_tensor);
                    break;
                }
            }

            // 5. Commit/Rollback
            if accepted_count == k_steps {
                // All accepted, sample one more
                let next_logits = next_target_logits.unwrap();
                let sample = self.sampler.sample(&next_logits);
                let token_tensor = Tensor::new(
                    ArrayD::from_elem(IxDyn(&[1usize, 1][..]), sample.token as f32),
                    false,
                );
                correct_token = Some(token_tensor);
            }

            let final_token = correct_token.unwrap();

            // Append accepted tokens to all_tokens
            if accepted_count > 0 {
                let accepted_slice = Tensor::concat(&draft_tokens[0..accepted_count], 1);
                all_tokens = Tensor::concat(&[all_tokens, accepted_slice][..], 1);
            }
            all_tokens = Tensor::concat(&[all_tokens, final_token.clone()][..], 1);

            n_generated += accepted_count + 1;

            // Rollback Caches
            // Target added `gamma` tokens. Retain `accepted_count`. Append `final_token`.
            self.target_model
                .truncate_cache(self.gamma - accepted_count);
            last_target_logits = self.target_model.forward_t(&final_token);

            // Draft added `gamma`. Retain `accepted_count`. Append `final_token`.
            self.draft_model.truncate_cache(self.gamma - accepted_count);
            let _ = self.draft_model.forward_t(&final_token);
        }

        all_tokens
    }
}
