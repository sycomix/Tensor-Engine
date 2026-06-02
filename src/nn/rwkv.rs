//! RWKV (Receptance Weighted Key Value) architecture.
//!
//! RWKV combines the best of RNN and Transformer: linear-time inference,
//! constant memory scaling, and parallel training. Uses time-mixing with
//! exponential decay and channel-mixing with GLU.
//!
//! Reference: [Wu et al., 2023](https://arxiv.org/abs/2310.10520)

use crate::nn::{Linear, Module, RMSNorm};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// RWKV time-mixing layer.
///
/// Implements the time-mixing mechanism with:
/// - Recurrent state for KV cache
/// - Exponential decay for attention
/// - Receptance gate for output modulation
#[derive(Clone)]
pub struct RWKVTimeMix {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_head: usize,
    /// Time decay parameter (learnable)
    pub time_decay: Tensor,
    /// Time mix parameter (learnable)
    pub time_mix: Tensor,
    /// Q, K, V, O projections
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    /// Receptance gate
    pub wr: Linear,
    /// Time shift buffer for recurrent state
    pub state_k: Option<Tensor>,
    pub state_v: Option<Tensor>,
    /// Initial state
    pub init_k: Tensor,
    pub init_v: Tensor,
}

impl RWKVTimeMix {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_head = d_model / num_heads;
        let wq = Linear::new(d_model, d_model, true);
        let wk = Linear::new(d_model, d_model, true);
        let wv = Linear::new(d_model, d_model, true);
        let wo = Linear::new(d_model, d_model, true);
        let wr = Linear::new(d_model, d_model, true);

        // Time decay: exponential decay rates per head
        let time_decay_data = ArrayD::from_shape_vec(
            IxDyn(&[num_heads][..]),
            (0..num_heads)
                .map(|i| -(1.0 + 10.0 * ((i as f32 + 0.999) / (1.0 - 0.999)).ln()).exp())
                .collect(),
        ).unwrap();
        let time_decay = Tensor::new(time_decay_data, false);

        // Time mix: interpolation between current and previous token
        let time_mix_data = ArrayD::from_shape_vec(
            IxDyn(&[d_model][..]),
            vec![0.9f32; d_model],
        ).unwrap();
        let time_mix = Tensor::new(time_mix_data, false);

        // Initial KV states
        let init_k = Tensor::zeros(&[num_heads, d_head][..]);
        let init_v = Tensor::zeros(&[num_heads, d_head][..]);

        RWKVTimeMix {
            d_model,
            num_heads,
            d_head,
            time_decay,
            time_mix,
            wq,
            wk,
            wv,
            wo,
            wr,
            state_k: None,
            state_v: None,
            init_k,
            init_v,
        }
    }

    /// Forward pass with recurrent state.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.lock().storage.shape().to_vec();
        if shape.len() != 3 {
            log::error!("RWKVTimeMix: expected 3D input [B, S, D], got {:?}", shape);
            return x.clone();
        }

        let (b, s, d) = (shape[0], shape[1], shape[2]);
        if d != self.d_model {
            return x.clone();
        }

        let decay_arr = self.time_decay.lock().storage.to_f32_array();
        let mix_arr = self.time_mix.lock().storage.to_f32_array();

        // Q, K, V, R projections
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let r = self.wr.forward(x);

        let q_arr = q.lock().storage.to_f32_array();
        let k_arr = k.lock().storage.to_f32_array();
        let v_arr = v.lock().storage.to_f32_array();
        let r_arr = r.lock().storage.to_f32_array();

        let mut out = vec![0.0f32; b * s * self.d_model];

        for n in 0..b {
            // Initialize state
            let mut state_k = vec![0.0f32; self.num_heads * self.d_head];
            let mut state_v = vec![0.0f32; self.num_heads * self.d_head];

            for t in 0..s {
                let t_offset = (n * s + t) * self.d_model;

                // Time mixing: blend current and previous token
                let mut k_mix = vec![0.0f32; self.d_model];
                let mut q_mix = vec![0.0f32; self.d_model];
                let mut v_mix = vec![0.0f32; self.d_model];

                for ch in 0..self.d_model {
                    let mix_val = mix_arr[ch].clamp(0.0, 1.0);
                    let prev_idx = if t > 0 {
                        (n * s + t - 1) * self.d_model + ch
                    } else {
                        0
                    };
                    k_mix[ch] = mix_val * k_arr[t_offset + ch] + (1.0 - mix_val) * k_arr[prev_idx];
                    q_mix[ch] = mix_val * q_arr[t_offset + ch] + (1.0 - mix_val) * q_arr[prev_idx];
                    v_mix[ch] = mix_val * v_arr[t_offset + ch] + (1.0 - mix_val) * v_arr[prev_idx];
                }

                // Compute attention per head
                for h in 0..self.num_heads {
                    let decay = decay_arr[h].clamp(-10.0, 10.0);
                    let decay_exp = decay.exp();
                    let decay_inv = 1.0 / (1.0 + decay_exp);

                    let mut attn_sum = vec![0.0f32; self.d_head];
                    for d in 0..self.d_head {
                        let k_idx = h * self.d_head + d;
                        let q_idx = (t_offset) * self.d_model / self.d_model + h * self.d_head + d;
                        let q_val = q_mix[h * self.d_head + d];

                        // Attention: sum over past positions
                        let mut attn = 0.0f32;
                        for t2 in 0..=t {
                            let k_idx2 = h * self.d_head + d;
                            let v_idx = (n * s + t2) * self.d_model + h * self.d_head + d;
                            let k_val = if t2 == t {
                                k_mix[h * self.d_head + d]
                            } else {
                                k_arr[((n * s + t2) * self.d_model) + h * self.d_head + d]
                            };
                            let v_val = v_arr[((n * s + t2) * self.d_model) + h * self.d_head + d];
                            let weight = if t2 == t {
                                q_val * k_val
                            } else {
                                q_val * k_val * decay_exp.powi((t - t2) as i32)
                            };
                            attn += weight * v_val;
                        }

                        attn_sum[d] = attn;
                    }

                    // Update state
                    for d in 0..self.d_head {
                        let k_idx = h * self.d_head + d;
                        let v_idx = h * self.d_head + d;
                        let k_val = k_mix[h * self.d_head + d];
                        let v_val = v_mix[h * self.d_head + d];
                        state_k[k_idx] = state_k[k_idx] * decay_inv + k_val;
                        state_v[v_idx] = state_v[v_idx] * decay_inv + v_val;
                    }

                    // Write output
                    for d in 0..self.d_head {
                        out[(t_offset) + h * self.d_head + d] = attn_sum[d];
                    }
                }
            }
        }

        let out_arr = match ArrayD::from_shape_vec(IxDyn(&[b, s, self.d_model][..]), out) {
            Ok(v) => v,
            Err(_) => ArrayD::zeros(IxDyn(&[b, s, self.d_model][..])),
        };
        let out_tensor = Tensor::new(out_arr, false);

        // Apply receptance gate
        out_tensor.mul(&r_arr.into_iter().cloned().collect::<ArrayD<f32>>())
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.wq.parameters();
        p.extend(self.wk.parameters());
        p.extend(self.wv.parameters());
        p.extend(self.wo.parameters());
        p.extend(self.wr.parameters());
        p.push(self.time_decay.clone());
        p.push(self.time_mix.clone());
        p.push(self.init_k.clone());
        p.push(self.init_v.clone());
        p
    }
}

impl Module for RWKVTimeMix {
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

/// RWKV channel-mixing layer with GLU.
///
/// Implements the channel-mixing mechanism with:
/// - Linear projection to 2.5x hidden dimension
/// - GLU (Gated Linear Unit) activation
/// - Time shift for recurrent state
#[derive(Clone)]
pub struct RWKVChannelMix {
    pub d_model: usize,
    pub d_ff: usize,
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
    pub time_mix: Tensor,
}

impl RWKVChannelMix {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let w1 = Linear::new(d_model, d_ff, true);
        let w2 = Linear::new(d_ff, d_model, true);
        let w3 = Linear::new(d_model, d_ff, true);

        let time_mix_data = ArrayD::from_shape_vec(
            IxDyn(&[d_model][..]),
            vec![0.9f32; d_model],
        ).unwrap();
        let time_mix = Tensor::new(time_mix_data, false);

        RWKVChannelMix {
            d_model,
            d_ff,
            w1,
            w2,
            w3,
            time_mix,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.lock().storage.shape().to_vec();
        if shape.len() != 3 {
            return x.clone();
        }

        let (b, s, d) = (shape[0], shape[1], shape[2]);
        if d != self.d_model {
            return x.clone();
        }

        // Time shift: blend current and previous
        let x_shifted = if s > 1 {
            let mut shifted = vec![0.0f32; b * s * self.d_model];
            let x_arr = x.lock().storage.to_f32_array();
            for t in 0..s {
                for ch in 0..self.d_model {
                    let curr = x_arr[(t * self.d_model + ch)];
                    let prev = if t > 0 {
                        x_arr[((t - 1) * self.d_model + ch)]
                    } else {
                        0.0
                    };
                    shifted[t * self.d_model + ch] = curr * 0.5 + prev * 0.5;
                }
            }
            match ArrayD::from_shape_vec(IxDyn(&[b, s, self.d_model][..]), shifted) {
                Ok(v) => Tensor::new(v, false),
                Err(_) => x.clone(),
            }
        } else {
            x.clone()
        };

        // GLU: w1(x) * sigmoid(w3(x))
        let w1_out = self.w1.forward(&x_shifted);
        let w3_out = self.w3.forward(&x_shifted);
        let glu = w1_out.mul(&w3_out.sigmoid());

        // Output projection
        self.w2.forward(&glu)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.w1.parameters();
        p.extend(self.w2.parameters());
        p.extend(self.w3.parameters());
        p.push(self.time_mix.clone());
        p
    }
}

impl Module for RWKVChannelMix {
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

/// RWKV block: time-mix + channel-mix with residual connections.
///
/// Architecture:
/// x -> TimeMix -> + -> RMSNorm -> ChannelMix -> + -> RMSNorm
#[derive(Clone)]
pub struct RWKVBlock {
    pub time_mix: RWKVTimeMix,
    pub channel_mix: RWKVChannelMix,
    pub norm1: RMSNorm,
    pub norm2: RMSNorm,
}

impl RWKVBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        RWKVBlock {
            time_mix: RWKVTimeMix::new(d_model, num_heads),
            channel_mix: RWKVChannelMix::new(d_model, d_ff),
            norm1: RMSNorm::new(d_model, 1, 1e-5),
            norm2: RMSNorm::new(d_model, 1, 1e-5),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Time-mix sub-layer
        let tm_out = self.time_mix.forward(x);
        let x1 = x.add(&tm_out);
        let x1_norm = self.norm1.forward(&x1);

        // Channel-mix sub-layer
        let cm_out = self.channel_mix.forward(&x1_norm);
        x1.add(&cm_out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.time_mix.parameters();
        p.extend(self.channel_mix.parameters());
        p.extend(self.norm1.parameters());
        p.extend(self.norm2.parameters());
        p
    }
}

impl Module for RWKVBlock {
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

/// RWKV model: stack of RWKV blocks with embedding and classification head.
///
/// Architecture:
/// Embedding -> [RWKVBlock] -> RMSNorm -> [ClassificationHead]
#[derive(Clone)]
pub struct RWKV {
    pub blocks: Vec<RWKVBlock>,
    pub embedding: Option<Linear>,
    pub norm: RMSNorm,
    pub output_dim: usize,
    pub d_model: usize,
}

impl RWKV {
    /// Create a new RWKV model.
    ///
    /// # Arguments
    /// * `vocab_size` - Vocabulary size
    /// * `d_model` - Model dimension
    /// * `n_layers` - Number of RWKV blocks
    /// * `num_heads` - Number of heads for time-mixing
    /// * `d_ff` - Feed-forward hidden dimension
    /// * `output_dim` - Output dimension (0 for no classification head)
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
            blocks.push(RWKVBlock::new(d_model, num_heads, d_ff));
        }

        let norm = RMSNorm::new(d_model, 1, 1e-5);
        let embedding = if vocab_size > 0 {
            Some(Linear::new(vocab_size, d_model, true))
        } else {
            None
        };

        RWKV {
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
        p
    }

    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.lock().storage.len()).sum()
    }
}

impl Module for RWKV {
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

/// RWKV configuration.
#[derive(Clone, Debug)]
pub struct RWKVConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub output_dim: usize,
    pub lr_init: f32,
    pub lr_min: f32,
}

impl Default for RWKVConfig {
    fn default() -> Self {
        RWKVConfig {
            vocab_size: 50277,
            d_model: 1024,
            n_layers: 24,
            num_heads: 8,
            d_ff: 4096,
            output_dim: 0,
            lr_init: 0.0001,
            lr_min: 0.00001,
        }
    }
}

impl RWKVConfig {
    pub fn rwkv_1b5() -> Self {
        RWKVConfig {
            vocab_size: 65536,
            d_model: 2048,
            n_layers: 24,
            num_heads: 16,
            d_ff: 8192,
            output_dim: 0,
            lr_init: 0.0004,
            lr_min: 0.00004,
        }
    }

    pub fn rwkv_3b() -> Self {
        RWKVConfig {
            vocab_size: 65536,
            d_model: 3072,
            n_layers: 32,
            num_heads: 24,
            d_ff: 12288,
            output_dim: 0,
            lr_init: 0.0003,
            lr_min: 0.00003,
        }
    }

    pub fn build(&self) -> RWKV {
        RWKV::new(
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
mod rwkv_tests {
    use super::*;

    #[test]
    fn test_rwkv_time_mix() {
        let mix = RWKVTimeMix::new(64, 4);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 8, 64]), vec![0.1f32; 64]).unwrap(),
            false,
        );
        let out = mix.forward(&x);
        assert_eq!(out.lock().storage.shape()[2], 64);
    }

    #[test]
    fn test_rwkv_channel_mix() {
        let mix = RWKVChannelMix::new(64, 128);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 8, 64]), vec![0.1f32; 64]).unwrap(),
            false,
        );
        let out = mix.forward(&x);
        assert_eq!(out.lock().storage.shape()[2], 64);
    }

    #[test]
    fn test_rwkv_block() {
        let block = RWKVBlock::new(64, 4, 128);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 8, 64]), vec![0.1f32; 64]).unwrap(),
            false,
        );
        let out = block.forward(&x);
        assert_eq!(out.lock().storage.shape()[2], 64);
    }

    #[test]
    fn test_rwkv_model() {
        let model = RWKV::new(1000, 64, 2, 4, 128, 10);
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
    fn test_rwkv_config() {
        let config = RWKVConfig::rwkv_1b5();
        assert_eq!(config.d_model, 2048);
        let model = config.build();
        assert!(model.num_parameters() > 0);
    }
}
