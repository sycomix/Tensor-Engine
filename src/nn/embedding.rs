use crate::nn::{linear_dispatch::LinearLayer, Module};
use crate::ops::EmbeddingLookup;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

/// Sparse Embedding layer.
/// Currently functionally equivalent to a standard Embedding layer but marked for
/// potential sparse gradient optimization in the future.
#[derive(Clone)]
pub struct SparseEmbedding {
    pub weight: Tensor,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub sparse: bool, // Indicator for optimizers
}

impl SparseEmbedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let w = ndarray::Array::zeros(ndarray::IxDyn(&[num_embeddings, embedding_dim]));
        // In a real scenario, we'd initialize better (e.g. normal distribution)
        // But for consistency with other parts, we start with zeros or let user init.
        // Usually embeddings are random.
        // Let's use standard random initialization if possible, or just new_with_data logic.
        // For now, let's create it and let initializer handle it, similar to Linear.
        // But Linear uses Xavier/He.
        // We'll stick to zeros here for safety, or small random.
        // Let's use simple logic:
        SparseEmbedding {
            weight: Tensor::new(w, true),
            num_embeddings,
            embedding_dim,
            sparse: true,
        }
    }

    pub fn new_with_init(num_embeddings: usize, embedding_dim: usize) -> Self {
        let s = Self::new(num_embeddings, embedding_dim);
        // Simple normal init 0.0, 1.0
        // This would require a random op.
        // For now, we rely on the user to initialize weights if they want something specific,
        // or we could add a `reset_parameters` method later.
        s
    }
}

impl Module for SparseEmbedding {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x: indices [batch, seq] or whatever
        Tensor::apply(
            Arc::new(EmbeddingLookup::new()),
            &[self.weight.clone(), x.clone()],
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![(format!("{}.weight", prefix), self.weight.clone())]
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        let key = format!("{}.weight", prefix);
        if let Some(t) = state.get(&key) {
            self.weight = t.clone();
            Ok(())
        } else {
            Err(format!("Missing key: {}", key))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Adaptive Input Embedding
/// Based on "Adaptive Input Representations for Neural Language Modeling" (Baevski & Auli, 2018).
/// Splits vocabulary into clusters. The first cluster (head) projects directly to d_model.
/// Subsequent clusters have smaller dimension and are projected up to d_model.
#[derive(Clone)]
pub struct AdaptiveEmbedding {
    pub head: SparseEmbedding,
    pub tail: Vec<(SparseEmbedding, LinearLayer)>,
    pub cutoffs: Vec<usize>,
    pub div_value: f32,
    pub d_model: usize,
}

impl AdaptiveEmbedding {
    /// cutoffs: e.g. [2000, 10000] for vocab size 50000.
    /// ranges:
    ///   - Head: 0..2000 (dim = d_model)
    ///   - Tail 0: 2000..10000 (dim = d_model / div)
    ///   - Tail 1: 10000..vocab (dim = d_model / div^2)
    pub fn new(vocab_size: usize, d_model: usize, cutoffs: Vec<usize>, div_value: f32) -> Self {
        let _head_dim = d_model;
        let mut tail = Vec::new();

        let head_size = if cutoffs.is_empty() {
            vocab_size
        } else {
            cutoffs[0]
        };

        let head = SparseEmbedding::new(head_size, d_model);

        let mut last_cutoff = head_size;
        let mut current_dim = d_model;

        for (_i, &cutoff) in cutoffs
            .iter()
            .skip(1)
            .chain(std::iter::once(&vocab_size))
            .enumerate()
        {
            if cutoff <= last_cutoff {
                continue; // Should return error?
            }
            current_dim = (current_dim as f32 / div_value) as usize;
            // ensure at least 1
            if current_dim < 1 {
                current_dim = 1;
            }

            let cluster_size = cutoff - last_cutoff;
            let emb = SparseEmbedding::new(cluster_size, current_dim);
            let proj = LinearLayer::new_f32(current_dim, d_model, false);

            tail.push((emb, proj));
            last_cutoff = cutoff;
        }

        AdaptiveEmbedding {
            head,
            tail,
            cutoffs,
            div_value,
            d_model,
        }
    }
}

impl Module for AdaptiveEmbedding {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x: indices [batch, sequence]
        // Strategy:
        // 1. Identify masks for each region.
        // 2. Gather indices.
        // 3. Forward sub-embeddings.
        // 4. Scatter add.

        // This requires boolean masks and scatter selection which we might not have efficiently exposed yet.
        // However, we can do this functionally via loops if necessary, or use `topk`-like strategies? No.
        // We can use `Tensor` comparison operations to make masks.
        // x < cutoffs[0] -> mask_head
        // Continue with per-cluster dispatch logic.

        // BUT: Tensor operations for "get values at mask" -> Sparse Gather?
        // We don't have `masked_select` explicitly exposed as differentiable Op easily maybe.
        // Let's assume we run the full batch computation but zero-out invalid results?
        // No, that's wasteful for Adaptive Embedding (point is speed).
        // The point is to only compute for the indices in the cluster.

        // Given existing Ops constraint, maybe we iterate over the flattened input?
        // Similar to MoE dispatch.

        // Flatten input conceptualization for iteration
        let shape = x.lock().storage.shape();
        let flat_len: usize = shape.iter().product();
        let x_data = x.lock().storage.to_f32_array();

        // Buckets of (original_index, vocab_index)
        let mut head_indices = Vec::new(); // (orig_idx, vocab_idx)
        let mut tail_indices = vec![Vec::new(); self.tail.len()];

        // Iterate over all elements in standard layout order
        for (i, &val) in x_data.iter().enumerate() {
            let idx = val as usize;
            if self.cutoffs.is_empty() || idx < self.cutoffs[0] {
                head_indices.push((i, val));
            } else {
                // Find which tail cluster
                let mut found = false;
                let mut start = self.cutoffs[0];
                for (cluster_i, &end) in self.cutoffs.iter().skip(1).enumerate() {
                    if idx < end {
                        tail_indices[cluster_i].push((i, val - start as f32));
                        found = true;
                        break;
                    }
                    start = end;
                }
                if !found {
                    // last cluster
                    // check bounds?
                    let last_cluster_idx = self.tail.len() - 1;
                    tail_indices[last_cluster_idx].push((i, val - start as f32));
                }
            }
        }

        // We need to construct result tensor.
        // Since we are doing scatter-add logic manually again (like in MoE),
        // we need to be careful about differentiability.
        // If we gather subsets, process, and scatter back.

        // Helper to process a bucket
        // (indices_in_batch, indices_in_vocab)
        // We need to form a tensor for indices_in_vocab to feed to Embedding.

        let mut output_parts = Vec::new();

        // 1. Head
        if !head_indices.is_empty() {
            let (batch_idxs, vocab_idxs): (Vec<usize>, Vec<f32>) = head_indices.into_iter().unzip();
            let subset_len = batch_idxs.len();
            let vocab_t = Tensor::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[subset_len]), vocab_idxs)
                    .expect("emb"),
                false, // no grad for indices
            );
            let out = self.head.forward(&vocab_t); // [subset, d_model]
            output_parts.push((batch_idxs, out));
        }

        // 2. Tail
        for (i, indices) in tail_indices.into_iter().enumerate() {
            if indices.is_empty() {
                continue;
            }
            let (batch_idxs, vocab_idxs): (Vec<usize>, Vec<f32>) = indices.into_iter().unzip();
            let subset_len = batch_idxs.len();
            let vocab_t = Tensor::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[subset_len]), vocab_idxs)
                    .expect("emb"),
                false,
            );

            let (emb, proj) = &self.tail[i];
            let sub_emb = emb.forward(&vocab_t); // [subset, d_curr]
            let sub_proj = proj.forward(&sub_emb); // [subset, d_model]
            output_parts.push((batch_idxs, sub_proj));
        }

        // 3. Scatter outputs back
        // We accumulate into a zero tensor.
        // Similar to MoE: OneHot scatter.
        // P: [flat_len, subset_len]

        let mut final_out = Tensor::zeros(&[flat_len, self.d_model]);

        for (batch_idxs, out_tensor) in output_parts {
            let subset_len = batch_idxs.len();

            // Construct sparse P matrix (dense for now)
            let mut p_data = vec![0.0f32; flat_len * subset_len];
            for (j, &global_i) in batch_idxs.iter().enumerate() {
                p_data[global_i * subset_len + j] = 1.0;
            }

            let p_mat = Tensor::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[flat_len, subset_len]), p_data)
                    .expect("emb"),
                false,
            );

            let scattered = p_mat.matmul(&out_tensor); // [flat_len, d_model]
            final_out = final_out.add(&scattered);
        }

        final_out
            .reshape(
                x.lock()
                    .storage
                    .shape()
                    .iter()
                    .chain(std::iter::once(&self.d_model))
                    .cloned()
                    .collect(),
            )
            .expect("emb2")
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.head.parameters();
        for (emb, proj) in &self.tail {
            p.extend(emb.parameters());
            p.extend(proj.parameters());
        }
        p
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut p = self.head.named_parameters(&format!("{}.head", prefix));
        for (i, (emb, proj)) in self.tail.iter().enumerate() {
            p.extend(emb.named_parameters(&format!("{}.tail.{}.emb", prefix, i)));
            p.extend(proj.named_parameters(&format!("{}.tail.{}.proj", prefix, i)));
        }
        p
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.head
            .load_state_dict(state, &format!("{}.head", prefix))?;
        for (i, (emb, proj)) in self.tail.iter_mut().enumerate() {
            emb.load_state_dict(state, &format!("{}.tail.{}.emb", prefix, i))?;
            proj.load_state_dict(state, &format!("{}.tail.{}.proj", prefix, i))?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
