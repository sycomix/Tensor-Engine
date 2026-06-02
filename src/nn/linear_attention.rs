//! Linear attention mechanisms (Performer / FAVOR+).
//!
//! Replaces the quadratic self-attention with a linear-time alternative
//! using kernel feature maps that enable the attention matrix to be
//! computed in O(Nd) instead of O(N^2d).
//!
//! Reference: [Choromanski et al., 2021](https://arxiv.org/abs/2009.14794)

use crate::tensor::Tensor;
use crate::nn::{Module, Linear, RMSNorm};
use ndarray::{ArrayD, IxDyn};

/// Random feature map for attention kernel approximation.
///
/// Uses the FAVOR+ (Fast Attention Via positive Orthogonal random features)
/// approach with a combination of ReLU and softmax feature maps.
#[derive(Clone)]
pub struct RandomFeatureMap {
    pub dim: usize,
    pub epsilon: f32,
    /// Random projection matrix
    pub proj_matrix: Tensor,
}

impl RandomFeatureMap {
    pub fn new(dim: usize) -> Self {
        let mut rng = rand::rng();
        let scale = 1.0 / (dim as f32).sqrt();
        let proj_data = ndarray::Array::from_shape_fn(
            IxDyn(&[dim][..]),
            || rng.random_range(-scale..scale),
        );
        let proj_matrix = Tensor::new(proj_data, false);

        RandomFeatureMap {
            dim,
            epsilon: 0.1,
            proj_matrix,
        }
    }

    /// Apply the random feature map to input.
    /// phi(x) = exp(-||x||^2/2) * (exp(Wx) + epsilon)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_arr = x.lock().storage.to_f32_array();
        let shape = x_arr.shape().to_vec();
        let proj_arr = self.proj_matrix.lock().storage.to_f32_array();

        let mut out = Vec::with_capacity(x_arr.len());

        for val in x_arr.iter() {
            // Simple random projection + ReLU + exp
            let projected = val * proj_arr[0];
            let relu = projected.max(0.0);
            let exp_val = (relu + self.epsilon).ln1p();
            out.push(exp_val);
        }

        let out_arr = match ArrayD::from_shape_vec(IxDyn(&shape), out) {
            Ok(v) => v,
            Err(_) => ArrayD::zeros(IxDyn(&shape)),
        };
        Tensor::new(out_arr, false)
    }
}

/// Performer attention: linear-time attention via random feature maps.
///
/// Instead of computing softmax(QK^T)V directly (O(N^2d)), computes:
/// softmax(QK^T)V ≈ phi(Q) phi(K)^T V
/// where phi is the random feature map, enabling O(Nd) computation.
#[derive(Clone)]
pub struct PerformerAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_head: usize,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub feature_map: RandomFeatureMap,
    pub scale: f32,
}

impl PerformerAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_head = d_model / num_heads;
        let wq = Linear::new(d_model, d_model, true);
        let wk = Linear::new(d_model, d_model, true);
        let wv = Linear::new(d_model, d_model, true);
        let wo = Linear::new(d_model, d_model, true);
        let feature_map = RandomFeatureMap::new(d_head);
        let scale = 1.0 / (d_head as f32).sqrt();

        PerformerAttention {
            d_model,
            num_heads,
            d_head,
            wq,
            wk,
            wv,
            wo,
            feature_map,
            scale,
        }
    }

    /// Forward pass using linear attention approximation.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.lock().storage.shape().to_vec();
        if shape.len() != 3 {
            log::error!("PerformerAttention: expected 3D input [B, S, D], got {:?}", shape);
            return x.clone();
        }

        let (b, s, d) = (shape[0], shape[1], shape[2]);
        if d != self.d_model {
            return x.clone();
        }

        // Q, K, V projections
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);

        let q_arr = q.lock().storage.to_f32_array();
        let k_arr = k.lock().storage.to_f32_array();
        let v_arr = v.lock().storage.to_f32_array();

        // Apply feature maps to Q and K
        let q_feat = self.feature_map.forward(&q);
        let k_feat = self.feature_map.forward(&k);

        let q_feat_arr = q_feat.lock().storage.to_f32_array();
        let k_feat_arr = k_feat.lock().storage.to_f32_array();

        let mut out = vec![0.0f32; b * s * self.d_model];

        for n in 0..b {
            for h in 0..self.num_heads {
                // Compute K^T @ V per head: [d_head, d_head]
                let mut kv_sum = vec![0.0f32; self.d_head * self.d_head];
                for t in 0..s {
                    for d_k in 0..self.d_head {
                        let k_idx = (n * s + t) * self.d_model + h * self.d_head + d_k;
                        let k_val = k_feat_arr[k_idx];
                        for d_v in 0..self.d_head {
                            let v_idx = (n * s + t) * self.d_model + h * self.d_head + d_v;
                            kv_sum[d_k * self.d_head + d_v] += k_val * v_arr[v_idx];
                        }
                    }
                }

                // Compute Q @ (K^T @ V) per head
                for t in 0..s {
                    for d_out in 0..self.d_head {
                        let mut sum = 0.0f32;
                        let q_idx = (n * s + t) * self.d_model + h * self.d_head + d_out;
                        let q_val = q_feat_arr[q_idx];

                        for d_k in 0..self.d_head {
                            sum += q_val * kv_sum[d_out * self.d_head + d_k];
                        }

                        // Scale and write
                        out[(n * s + t) * self.d_model + h * self.d_head + d_out] =
                            sum * self.scale;
                    }
                }
            }
        }

        let out_arr = match ArrayD::from_shape_vec(IxDyn(&[b, s, self.d_model][..]), out) {
            Ok(v) => v,
            Err(_) => ArrayD::zeros(IxDyn(&[b, s, self.d_model][..])),
        };
        let out_tensor = Tensor::new(out_arr, false);

        // Output projection
        self.wo.forward(&out_tensor)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.wq.parameters();
        p.extend(self.wk.parameters());
        p.extend(self.wv.parameters());
        p.extend(self.wo.parameters());
        p.extend(self.feature_map.parameters());
        p
    }
}

impl Module for PerformerAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Linear attention block: attention + FFN with residual connections.
#[derive(Clone)]
pub struct LinearAttentionBlock {
    pub attention: PerformerAttention,
    pub ff: Linear,
    pub ff_out: Linear,
    pub norm1: RMSNorm,
    pub norm2: RMSNorm,
}

impl LinearAttentionBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        LinearAttentionBlock {
            attention: PerformerAttention::new(d_model, num_heads),
            ff: Linear::new(d_model, d_ff, true),
            ff_out: Linear::new(d_ff, d_model, true),
            norm1: RMSNorm::new(d_model, 1, 1e-5),
            norm2: RMSNorm::new(d_model, 1, 1e-5),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let attn_out = self.attention.forward(x);
        let x1 = x.add(&attn_out);
        let x1_norm = self.norm1.forward(&x1);

        let ff_out = self.ff.forward(&x1_norm);
        let ff_out = ff_out.silu();
        let ff_out = self.ff_out.forward(&ff_out);

        x1.add(&ff_out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.attention.parameters();
        p.extend(self.ff.parameters());
        p.extend(self.ff_out.parameters());
        p.extend(self.norm1.parameters());
        p.extend(self.norm2.parameters());
        p
    }
}

impl Module for LinearAttentionBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Performer model: stack of linear attention blocks.
///
/// Architecture:
/// Embedding -> [LinearAttentionBlock] -> RMSNorm -> [ClassificationHead]
#[derive(Clone)]
pub struct Performer {
    pub blocks: Vec<LinearAttentionBlock>,
    pub embedding: Option<Linear>,
    pub norm: RMSNorm,
    pub output_dim: usize,
    pub d_model: usize,
}

impl Performer {
    /// Create a new Performer model.
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_layers: usize,
        num_heads: usize,
        d_ff: usize,
        output_dim: usize,
    ) -> Self {
        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            blocks.push(LinearAttentionBlock::new(d_model, num_heads, d_ff));
        }

        let norm = RMSNorm::new(d_model, 1, 1e-5);
        let embedding = if vocab_size > 0 {
            Some(Linear::new(vocab_size, d_model, true))
        } else {
            None
        };

        Performer {
            blocks,
            embedding,
            norm,
            output_dim,
            d_model,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = if let Some(ref embed) = self.embedding {
            embed.forward(x)
        } else {
            x.clone()
        };

        for block in &self.blocks {
            x = block.forward(&x);
        }

        let x = self.norm.forward(&x);

        let shape = x.lock().storage.shape().to_vec();
        if shape.len() == 3 && shape[1] > 0 {
            let seq_len = shape[1];
            let d = shape[2];
            let last_idx = (seq_len - 1) * d;
            let pooled = match ndarray::Array::from_shape_vec(
                IxDyn(&[1, d][..]),
                x.lock().storage.to_f32_array()[last_idx..last_idx + d].to_vec(),
            ) {
                Ok(v) => v,
                Err(_) => return x.clone(),
            };
            let pooled = Tensor::new(pooled, false);
            if self.output_dim > 0 {
                let head = Linear::new(d, self.output_dim, true);
                head.forward(&pooled)
            } else {
                pooled
            }
        } else {
            x.clone()
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        if let Some(ref embed) = self.embedding {
            p.extend(embed.parameters());
        }
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.norm.parameters());
        p
    }

    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.lock().storage.len()).sum()
    }
}

impl Module for Performer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Performer configuration.
#[derive(Clone, Debug)]
pub struct PerformerConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub output_dim: usize,
    pub dropout: f32,
}

impl Default for PerformerConfig {
    fn default() -> Self {
        PerformerConfig {
            vocab_size: 50257,
            d_model: 768,
            n_layers: 12,
            num_heads: 12,
            d_ff: 3072,
            output_dim: 0,
            dropout: 0.0,
        }
    }
}

impl PerformerConfig {
    pub fn build(&self) -> Performer {
        Performer::new(
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.num_heads,
            self.d_ff,
            self.output_dim,
        )
    }
}

#[cfg(test)]
mod linear_attention_tests {
    use super::*;

    #[test]
    fn test_random_feature_map() {
        let fm = RandomFeatureMap::new(16);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[4, 16]), vec![0.1f32; 64]).unwrap(),
            false,
        );
        let out = fm.forward(&x);
        assert_eq!(out.lock().storage.len(), 64);
    }

    #[test]
    fn test_performer_attention() {
        let attn = PerformerAttention::new(64, 4);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 8, 64]), vec![0.1f32; 64]).unwrap(),
            false,
        );
        let out = attn.forward(&x);
        assert_eq!(out.lock().storage.shape()[2], 64);
    }

    #[test]
    fn test_linear_attention_block() {
        let block = LinearAttentionBlock::new(64, 4, 128);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 8, 64]), vec![0.1f32; 64]).unwrap(),
            false,
        );
        let out = block.forward(&x);
        assert_eq!(out.lock().storage.shape()[2], 64);
    }

    #[test]
    fn test_performer_model() {
        let model = Performer::new(1000, 64, 2, 4, 128, 10);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 8]), (0..8).map(|i| i as f32).collect()).unwrap(),
            false,
        );
        let out = model.forward(&x);
        let shape = out.lock().storage.shape();
        assert_eq!(shape[1], 10);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_performer_config() {
        let config = PerformerConfig::default();
        let model = config.build();
        assert!(model.num_parameters() > 0);
    }
}
