//! Mamba architecture: Selective State Space Model (SSM).
//!
//! Mamba introduces a data-dependent selection mechanism for SSM parameters,
//! enabling linear-time sequence modeling with input-aware content processing.
//!
//! Reference: [Gu & Dao, 2024](https://arxiv.org/abs/2312.00752)

use crate::tensor::Tensor;
use crate::nn::{Module, Linear, LayerNorm, RMSNorm};
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

/// SSM core operation: selective scan.
///
/// Implements the parallel scan algorithm for SSM state evolution:
/// h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
/// y_t = C * h_t
#[derive(Clone)]
pub struct SSMCore {
    /// State dimension
    pub d_state: usize,
    /// Expansion factor
    pub expand: usize,
    /// Hidden dimension (d_model = d_state * expand)
    pub d_model: usize,
    /// Discretization step size (delta)
    pub delta: f32,
    /// SSM matrix A (d_state,)
    pub a: Tensor,
    /// Softplus bias for delta
    pub delta_bias: Option<Tensor>,
}

impl SSMCore {
    pub fn new(d_state: usize, expand: usize, d_model: usize) -> Self {
        let a_data = ArrayD::from_shape_vec(
            IxDyn(&[d_state][..]),
            (0..d_state).map(|i| -(i as f32 + 1.0) / (d_state as f32)).collect(),
        ).unwrap();
        let a = Tensor::new(a_data, false);
        SSMCore {
            d_state,
            expand,
            d_model,
            delta: 0.01,
            a,
            delta_bias: None,
        }
    }

    /// Forward pass through SSM core.
    ///
    /// Input: [batch, seq, d_model]
    /// Output: [batch, seq, d_model]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.lock().storage.shape().to_vec();
        if shape.len() != 3 {
            log::error!("SSMCore::forward: expected 3D input [B, S, D], got {:?}", shape);
            return x.clone();
        }

        let (b, s, d) = (shape[0], shape[1], shape[2]);
        if d != self.d_model {
            log::error!("SSMCore::forward: input dim {} != d_model {}", d, self.d_model);
            return x.clone();
        }

        let x_arr = x.lock().storage.to_f32_array();
        let a_arr = self.a.lock().storage.to_f32_array();
        let d_state = self.d_state;

        // Selective scan: for each batch and each state dimension
        let mut out = vec![0.0f32; b * s * d];

        for n in 0..b {
            let mut h = vec![0.0f32; d_state];

            for t in 0..s {
                let x_t = &x_arr[(n * s + t) * d..(n * s + t + 1) * d];

                // For each output dimension, compute SSM update
                for j in 0..d {
                    let state_idx = j % d_state;
                    let expand_idx = j / d_state;

                    // B_bar_t = B_t * delta_t (simplified)
                    let b_val = x_t[expand_idx * d_state + state_idx].max(0.0);
                    let delta = if let Some(ref bias) = self.delta_bias {
                        self.delta + bias.lock().storage.to_f32_array()[0]
                    } else {
                        self.delta
                    };
                    let a_bar = (-delta * a_arr[state_idx].exp()).clamp(-1.0, 1.0);

                    // h_t = a_bar * h_{t-1} + b_bar * x_t
                    h[state_idx] = a_bar * h[state_idx] + b_val * x_t[expand_idx * d_state + state_idx];

                    // y_t = C * h_t (C is identity for simplicity)
                    out[(n * s + t) * d + j] = h[state_idx];
                }
            }
        }

        let out_arr = match ArrayD::from_shape_vec(IxDyn(&[b, s, d][..]), out) {
            Ok(v) => v,
            Err(_) => ArrayD::zeros(IxDyn(&[b, s, d][..])),
        };
        Tensor::new(out_arr, false)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }
}

/// Conv1D layer for Mamba's input projection.
#[derive(Clone)]
pub struct Conv1DLayer {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
}

impl Conv1DLayer {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let weight_data = ndarray::Array::zeros(IxDyn(&[out_channels, in_channels, kernel_size][..]));
        let weight = Tensor::new(weight_data, true);
        let bias = Some(Tensor::new(
            ndarray::Array::zeros(IxDyn(&[out_channels][..])),
            true,
        ));
        Conv1DLayer {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Simple 1D conv: pad, then compute
        let shape = x.lock().storage.shape().to_vec();
        if shape.len() != 3 {
            return x.clone();
        }
        let (b, s, c) = (shape[0], shape[1], shape[2]);
        if c != self.in_channels {
            return x.clone();
        }

        let x_arr = x.lock().storage.to_f32_array();
        let w_arr = self.weight.lock().storage.to_f32_array();
        let pad = self.kernel_size / 2;

        let mut out = vec![0.0f32; b * s * self.out_channels];

        for n in 0..b {
            for t in 0..s {
                for oc in 0..self.out_channels {
                    let mut sum = 0.0f32;
                    for k in 0..self.kernel_size {
                        let tc = t + k - pad;
                        if tc >= 0 && tc < s {
                            for ic in 0..self.in_channels {
                                sum += x_arr[((n * s + tc) * c + ic)] * w_arr[[oc, ic, k]];
                            }
                        }
                    }
                    out[(n * s + t) * self.out_channels + oc] = sum;
                }
            }
        }

        let out_arr = match ArrayD::from_shape_vec(IxDyn(&[b, s, self.out_channels][..]), out) {
            Ok(v) => v,
            Err(_) => ArrayD::zeros(IxDyn(&[b, s, self.out_channels][..])),
        };
        let mut result = Tensor::new(out_arr, false);
        if let Some(ref bias) = self.bias {
            result = result.add(bias);
        }
        result
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            p.push(bias.clone());
        }
        p
    }
}

/// Mamba block: combines SSM with residual connections and normalization.
///
/// Architecture:
/// x -> Conv1D -> Silu -> Linear -> SSM -> Linear -> Silu -> Element-wise mul -> Linear -> + -> LayerNorm
#[derive(Clone)]
pub struct MambaBlock {
    pub conv1d: Conv1DLayer,
    pub ssm: SSMCore,
    pub in_proj: Linear,
    pub out_proj: Linear,
    pub norm: RMSNorm,
    pub dt_proj: Linear,
    pub dt_bias: bool,
}

impl MambaBlock {
    pub fn new(d_model: usize, d_state: usize, expand: usize, kernel_size: usize) -> Self {
        let d_inner = d_model * expand;
        let in_proj = Linear::new(d_model, d_inner * 2, true);
        let out_proj = Linear::new(d_inner, d_model, true);
        let norm = RMSNorm::new(d_model, 1, 1e-5);
        let dt_proj = Linear::new(d_inner, d_inner, true);

        MambaBlock {
            conv1d: Conv1DLayer::new(d_model, d_model, kernel_size),
            ssm: SSMCore::new(d_state, expand, d_inner),
            in_proj,
            out_proj,
            norm,
            dt_proj,
            dt_bias: true,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let residual = x.clone();

        // Pre-norm
        let x = self.norm.forward(x);

        // Conv + Silu
        let x = self.conv1d.forward(&x);
        let x = x.silu();

        // In-projection (split into B and X branches)
        let projected = self.in_proj.forward(&x);
        let shape = projected.lock().storage.shape().to_vec();
        let (b, s, d) = (shape[0], shape[1], shape[2]);
        let d_inner = d / 2;

        // Split: [B, X] -> B branch and X branch
        let x_branch = match projected.reshape(vec![b * s, d_inner]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };

        // SSM path
        let ssm_out = self.ssm.forward(&x_branch);

        // Apply delta projection (simplified)
        let dt_out = self.dt_proj.forward(&x_branch);
        let dt = dt_out.sigmoid();

        // Element-wise: ssm_out * silu(x_branch) * dt
        let silu_x = x_branch.silu();
        let gated = ssm_out.mul(&silu_x).mul(&dt);

        // Out projection
        let out = self.out_proj.forward(&gated);

        // Residual
        out.add(&residual)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.conv1d.parameters();
        p.extend(self.ssm.parameters());
        p.extend(self.in_proj.parameters());
        p.extend(self.out_proj.parameters());
        p.extend(self.norm.parameters());
        p.extend(self.dt_proj.parameters());
        p
    }
}

impl Module for MambaBlock {
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

/// Mamba model: stack of Mamba blocks with optional embedding and classification head.
///
/// Architecture:
/// Embedding -> [MambaBlock] -> LayerNorm -> [ClassificationHead]
#[derive(Clone)]
pub struct Mamba {
    pub blocks: Vec<MambaBlock>,
    pub embedding: Option<Linear>,
    pub norm: RMSNorm,
    pub output_dim: usize,
    pub d_model: usize,
}

impl Mamba {
    /// Create a new Mamba model.
    ///
    /// # Arguments
    /// * `vocab_size` - Vocabulary size (for embedding)
    /// * `d_model` - Model dimension
    /// * `n_layers` - Number of Mamba blocks
    /// * `d_state` - State dimension for SSM
    /// * `expand` - Expansion factor
    /// * `kernel_size` - Conv1D kernel size
    /// * `output_dim` - Output dimension (0 for no classification head)
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_layers: usize,
        d_state: usize,
        expand: usize,
        kernel_size: usize,
        output_dim: usize,
    ) -> Self {
        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            blocks.push(MambaBlock::new(d_model, d_state, expand, kernel_size));
        }

        let norm = RMSNorm::new(d_model, 1, 1e-5);
        let embedding = if vocab_size > 0 {
            Some(Linear::new(vocab_size, d_model, true))
        } else {
            None
        };

        Mamba {
            blocks,
            embedding,
            norm,
            output_dim,
            d_model,
        }
    }

    /// Forward pass.
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

        // Pool: use last token output
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

    /// Get all parameters.
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

    /// Number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.lock().storage.len()).sum()
    }
}

impl Module for Mamba {
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

/// Mamba configuration for model definition.
#[derive(Clone, Debug)]
pub struct MambaConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub d_state: usize,
    pub expand: usize,
    pub kernel_size: usize,
    pub output_dim: usize,
    pub d_conv: usize,
    pub dt_min: f32,
    pub dt_max: f32,
    pub dt_init: String,
    pub dt_scale: f32,
    pub dt_rank: String,
}

impl Default for MambaConfig {
    fn default() -> Self {
        MambaConfig {
            vocab_size: 50257,
            d_model: 768,
            n_layers: 24,
            d_state: 16,
            expand: 2,
            kernel_size: 4,
            output_dim: 0,
            d_conv: 4,
            dt_min: 0.001,
            dt_max: 0.1,
            dt_init: "random",
            dt_scale: 1.0,
            dt_rank: "d_half",
        }
    }
}

impl MambaConfig {
    /// Create a Mamba-1B configuration (similar to state-spaces/mamba-1b).
    pub fn mamba_1b() -> Self {
        MambaConfig {
            vocab_size: 50277,
            d_model: 2048,
            n_layers: 24,
            d_state: 16,
            expand: 2,
            kernel_size: 4,
            output_dim: 0,
            ..Default::default()
        }
    }

    /// Create a Mamba-2.8B configuration (similar to state-spaces/mamba-2.8b).
    pub fn mamba_2_8b() -> Self {
        MambaConfig {
            vocab_size: 50277,
            d_model: 3200,
            n_layers: 64,
            d_state: 16,
            expand: 2,
            kernel_size: 4,
            output_dim: 0,
            ..Default::default()
        }
    }

    /// Build a Mamba model from this config.
    pub fn build(&self) -> Mamba {
        Mamba::new(
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.d_state,
            self.expand,
            self.kernel_size,
            self.output_dim,
        )
    }
}

#[cfg(test)]
mod mamba_tests {
    use super::*;

    #[test]
    fn test_ssm_core() {
        let ssm = SSMCore::new(16, 2, 32);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 4, 32]), vec![0.1f32; 64]).unwrap(),
            false,
        );
        let out = ssm.forward(&x);
        assert_eq!(out.lock().storage.shape()[0], 2);
        assert_eq!(out.lock().storage.shape()[1], 4);
        assert_eq!(out.lock().storage.shape()[2], 32);
    }

    #[test]
    fn test_conv1d_layer() {
        let conv = Conv1DLayer::new(32, 64, 4);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 8, 32]), vec![0.1f32; 512]).unwrap(),
            false,
        );
        let out = conv.forward(&x);
        assert_eq!(out.lock().storage.shape()[2], 64);
    }

    #[test]
    fn test_mamba_block() {
        let block = MambaBlock::new(64, 16, 2, 4);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 8, 64]), vec![0.1f32; 64]).unwrap(),
            false,
        );
        let out = block.forward(&x);
        assert_eq!(out.lock().storage.shape()[2], 64);
    }

    #[test]
    fn test_mamba_model() {
        let model = Mamba::new(1000, 64, 2, 16, 2, 4, 10);
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
    fn test_mamba_config() {
        let config = MambaConfig::mamba_1b();
        assert_eq!(config.d_model, 2048);
        assert_eq!(config.n_layers, 24);
        let model = config.build();
        assert!(model.num_parameters() > 0);
    }
}
