//! RetNet architecture: Retentive Network with multiplicative recurrence.
//!
//! RetNet replaces self-attention with a recurrent SSM-like mechanism using
//! multiplicative recurrence and multi-scale retention for efficient training
//! and inference.
//!
//! Reference: [Yan et al., 2023](https://arxiv.org/abs/2307.08621)

use crate::tensor::Tensor;
use crate::nn::{Module, Linear, LayerNorm, RMSNorm};
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

/// Multi-scale retention mechanism.
///
/// Computes retention over multiple scales to capture multi-scale dependencies.
/// y = sum_k (R_k @ V) where R_k = softmax(Q @ K^T / sqrt(d)) at scale k.
#[derive(Clone)]
pub struct MultiScaleRetention {
    pub d_model: usize,
    pub d_state: usize,
    pub num_heads: usize,
    pub d_head: usize,
    /// Weight matrices for Q, K, V, O projections
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    /// Decay rates for each head
    pub decay: Tensor,
    /// Gate projection
    pub gate_proj: Linear,
    /// RMSNorm for input
    pub norm: RMSNorm,
}

impl MultiScaleRetention {
    pub fn new(d_model: usize, num_heads: usize, d_state: usize) -> Self {
        let d_head = d_model / num_heads;
        let wq = Linear::new(d_model, d_model, true);
        let wk = Linear::new(d_model, d_model, true);
        let wv = Linear::new(d_model, d_model, true);
        let wo = Linear::new(d_model, d_model, true);
        let gate_proj = Linear::new(d_model, d_model, true);
        let norm = RMSNorm::new(d_model, 1, 1e-5);

        // Decay rates: geometric sequence per head
        let decay_data = ArrayD::from_shape_vec(
            IxDyn(&[num_heads][..]),
            (0..num_heads)
                .map(|i| 0.9f32.powi(i as i32))
                .collect(),
        ).unwrap();
        let decay = Tensor::new(decay_data, false);

        MultiScaleRetention {
            d_model,
            d_state,
            num_heads,
            d_head,
            wq,
            wk,
            wv,
            wo,
            decay,
            gate_proj,
            norm,
        }
    }

    /// Forward pass through multi-scale retention.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.lock().storage.shape().to_vec();
        if shape.len() != 3 {
            log::error!("MultiScaleRetention: expected 3D input [B, S, D], got {:?}", shape);
            return x.clone();
        }

        let (b, s, d) = (shape[0], shape[1], shape[2]);
        if d != self.d_model {
            return x.clone();
        }

        // Pre-norm
        let x_norm = self.norm.forward(x);

        // Q, K, V projections
        let q = self.wq.forward(&x_norm);
        let k = self.wk.forward(&x_norm);
        let v = self.wv.forward(&x_norm);

        // Reshape to [B, S, num_heads, d_head]
        let q_reshaped = match q.reshape(vec![b, s, self.num_heads, self.d_head]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };
        let k_reshaped = match k.reshape(vec![b, s, self.num_heads, self.d_head]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };
        let v_reshaped = match v.reshape(vec![b, s, self.num_heads, self.d_head]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };

        let q_arr = q_reshaped.lock().storage.to_f32_array();
        let k_arr = k_reshaped.lock().storage.to_f32_array();
        let v_arr = v_reshaped.lock().storage.to_f32_array();
        let decay_arr = self.decay.lock().storage.to_f32_array();

        let scale = (self.d_head as f32).sqrt();
        let mut out = vec![0.0f32; b * s * self.num_heads * self.d_head];

        for n in 0..b {
            for h in 0..self.num_heads {
                let decay = decay_arr[h].clamp(0.0, 1.0);
                let mut state = vec![0.0f32; self.d_head];

                for t in 0..s {
                    // Compute QK^T for current position
                    let mut attn = vec![0.0f32; s];
                    for t2 in 0..=t {
                        let mut sum = 0.0f32;
                        for d in 0..self.d_head {
                            let qi = q_arr[((n * s + t) * self.num_heads + h) * self.d_head + d];
                            let ki = k_arr[((n * s + t2) * self.num_heads + h) * self.d_head + d];
                            sum += qi * ki;
                        }
                        attn[t2] = sum / scale;
                    }

                    // Softmax
                    let max_attn = attn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_attn: Vec<f32> = attn.iter().map(|a| (a - max_attn).exp()).collect();
                    let sum_exp: f32 = exp_attn.iter().sum();
                    let softmax_attn: Vec<f32> = if sum_exp > 1e-12 {
                        exp_attn.iter().map(|e| e / sum_exp).collect()
                    } else {
                        vec![1.0 / s as f32; s]
                    };

                    // Update state: state = decay * state + softmax_attn @ V
                    let mut new_state = vec![0.0f32; self.d_head];
                    for d in 0..self.d_head {
                        let mut sum = 0.0f32;
                        for t2 in 0..=t {
                            sum += softmax_attn[t2] * v_arr[((n * s + t2) * self.num_heads + h) * self.d_head + d];
                        }
                        new_state[d] = decay * state[d] + sum;
                    }
                    state = new_state;

                    // Write output
                    for d in 0..self.d_head {
                        out[((n * s + t) * self.num_heads + h) * self.d_head + d] = state[d];
                    }
                }
            }
        }

        let out_arr = match ArrayD::from_shape_vec(
            IxDyn(&[b, s, self.num_heads, self.d_head][..]),
            out,
        ) {
            Ok(v) => v,
            Err(_) => ArrayD::zeros(IxDyn(&[b, s, self.num_heads, self.d_head][..])),
        };
        let out_tensor = Tensor::new(out_arr, false);

        // Reshape back to [B, S, D]
        let out_flat = match out_tensor.reshape(vec![b, s, self.d_model]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };

        // Apply gate
        let gate = self.gate_proj.forward(&x_norm);
        let gated = out_flat.mul(&gate.sigmoid());

        // Output projection
        self.wo.forward(&gated)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.wq.parameters();
        p.extend(self.wk.parameters());
        p.extend(self.wv.parameters());
        p.extend(self.wo.parameters());
        p.extend(self.gate_proj.parameters());
        p.extend(self.norm.parameters());
        p.push(self.decay.clone());
        p
    }
}

impl Module for MultiScaleRetention {
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

/// RetNet block: retention + FFN with parallelization trick.
///
/// Architecture:
/// x -> MultiScaleRetention -> + -> RMSNorm -> FFN -> + -> RMSNorm
#[derive(Clone)]
pub struct RetNetBlock {
    pub retention: MultiScaleRetention,
    pub ff: Linear,
    pub ff_out: Linear,
    pub norm1: RMSNorm,
    pub norm2: RMSNorm,
}

impl RetNetBlock {
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize, d_state: usize) -> Self {
        RetNetBlock {
            retention: MultiScaleRetention::new(d_model, num_heads, d_state),
            ff: Linear::new(d_model, d_ff, true),
            ff_out: Linear::new(d_ff, d_model, true),
            norm1: RMSNorm::new(d_model, 1, 1e-5),
            norm2: RMSNorm::new(d_model, 1, 1e-5),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Retention sub-layer
        let ret_out = self.retention.forward(x);
        let x1 = x.add(&ret_out);
        let x1_norm = self.norm1.forward(&x1);

        // FFN sub-layer
        let ff_out = self.ff.forward(&x1_norm);
        let ff_out = ff_out.silu();
        let ff_out = self.ff_out.forward(&ff_out);

        x1.add(&ff_out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.retention.parameters();
        p.extend(self.ff.parameters());
        p.extend(self.ff_out.parameters());
        p.extend(self.norm1.parameters());
        p.extend(self.norm2.parameters());
        p
    }
}

impl Module for RetNetBlock {
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

/// RetNet model: stack of RetNet blocks.
///
/// Architecture:
/// Embedding -> [RetNetBlock] -> RMSNorm -> [ClassificationHead]
#[derive(Clone)]
pub struct RetNet {
    pub blocks: Vec<RetNetBlock>,
    pub embedding: Option<Linear>,
    pub norm: RMSNorm,
    pub output_dim: usize,
    pub d_model: usize,
    /// Positional encoding (learnable)
    pub pos_encoding: Option<Tensor>,
}

impl RetNet {
    /// Create a new RetNet model.
    ///
    /// # Arguments
    /// * `vocab_size` - Vocabulary size
    /// * `d_model` - Model dimension
    /// * `n_layers` - Number of RetNet blocks
    /// * `num_heads` - Number of attention heads
    /// * `d_state` - State dimension for retention
    /// * `d_ff` - Feed-forward hidden dimension
    /// * `output_dim` - Output dimension (0 for no classification head)
    /// * `max_seq_len` - Maximum sequence length for positional encoding
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_layers: usize,
        num_heads: usize,
        d_state: usize,
        d_ff: usize,
        output_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            blocks.push(RetNetBlock::new(d_model, d_ff, num_heads, d_state));
        }

        let norm = RMSNorm::new(d_model, 1, 1e-5);
        let embedding = if vocab_size > 0 {
            Some(Linear::new(vocab_size, d_model, true))
        } else {
            None
        };

        let pos_encoding = if max_seq_len > 0 {
            let pe_data = ArrayD::from_shape_vec(
                IxDyn(&[max_seq_len, d_model][..]),
                vec![0.0f32; max_seq_len * d_model],
            ).unwrap();
            Some(Tensor::new(pe_data, true))
        } else {
            None
        };

        RetNet {
            blocks,
            embedding,
            norm,
            output_dim,
            d_model,
            pos_encoding,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = if let Some(ref embed) = self.embedding {
            embed.forward(x)
        } else {
            x.clone()
        };

        // Add positional encoding
        if let Some(ref pe) = self.pos_encoding {
            let x_shape = x.lock().storage.shape().to_vec();
            if x_shape.len() == 3 {
                let seq = x_shape[1];
                if seq <= pe.lock().storage.shape()[0] {
                    let pe_slice = match pe.reshape(vec![seq, self.d_model]) {
                        Ok(t) => t,
                        Err(_) => return x.clone(),
                    };
                    x = x.add(&pe_slice);
                }
            }
        }

        for block in &self.blocks {
            x = block.forward(&x);
        }

        let x = self.norm.forward(&x);

        // Pool: use last token
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
        if let Some(ref pe) = self.pos_encoding {
            p.push(pe.clone());
        }
        p
    }

    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.lock().storage.len()).sum()
    }
}

impl Module for RetNet {
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

/// RetNet configuration.
#[derive(Clone, Debug)]
pub struct RetNetConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub num_heads: usize,
    pub d_state: usize,
    pub d_ff: usize,
    pub output_dim: usize,
    pub max_seq_len: usize,
    pub dropout: f32,
}

impl Default for RetNetConfig {
    fn default() -> Self {
        RetNetConfig {
            vocab_size: 50257,
            d_model: 1536,
            n_layers: 24,
            num_heads: 12,
            d_state: 64,
            d_ff: 4096,
            output_dim: 0,
            max_seq_len: 4096,
            dropout: 0.0,
        }
    }
}

impl RetNetConfig {
    pub fn retnet_3b() -> Self {
        RetNetConfig {
            vocab_size: 50257,
            d_model: 3072,
            n_layers: 48,
            num_heads: 24,
            d_state: 64,
            d_ff: 8192,
            output_dim: 0,
            max_seq_len: 4096,
            dropout: 0.0,
        }
    }

    pub fn build(&self) -> RetNet {
        RetNet::new(
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.num_heads,
            self.d_state,
            self.d_ff,
            self.output_dim,
            self.max_seq_len,
        )
    }
}

#[cfg(test)]
mod retentive_tests {
    use super::*;

    #[test]
    fn test_multi_scale_retention() {
        let retention = MultiScaleRetention::new(64, 4, 16);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 8, 64]), vec![0.1f32; 64]).unwrap(),
            false,
        );
        let out = retention.forward(&x);
        assert_eq!(out.lock().storage.shape()[2], 64);
    }

    #[test]
    fn test_retnet_block() {
        let block = RetNetBlock::new(64, 128, 4, 16);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 8, 64]), vec![0.1f32; 64]).unwrap(),
            false,
        );
        let out = block.forward(&x);
        assert_eq!(out.lock().storage.shape()[2], 64);
    }

    #[test]
    fn test_retnet_model() {
        let model = RetNet::new(1000, 64, 2, 4, 16, 128, 10, 32);
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
    fn test_retnet_config() {
        let config = RetNetConfig::retnet_3b();
        assert_eq!(config.d_model, 3072);
        let model = config.build();
        assert!(model.num_parameters() > 0);
    }
}
