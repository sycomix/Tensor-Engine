use crate::nn::{Module, Linear, Conv2D};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

/// Timestep embeddings: sinusoidal embedding followed by a linear projection.
pub struct TimestepEmbedding {
    pub linear1: Linear,
    pub linear2: Linear,
}

impl TimestepEmbedding {
    pub fn new(d_model: usize, hidden: usize) -> Self {
        TimestepEmbedding {
            linear1: Linear::new(d_model, hidden, true),
            linear2: Linear::new(hidden, d_model, true),
        }
    }

    pub fn forward(&self, t: &Tensor) -> Tensor {
        // t is expected as scalar batch or [B, 1]
        // Use simple linear projection: out = linear2(relu(linear1(t))) for now
        let h = self.linear1.forward(t);
        let h = h.relu();
        self.linear2.forward(&h)
    }
}

/// Group Normalization module expects NCHW tensors and normalizes over groups.
pub struct GroupNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub num_groups: usize,
    pub eps: f32,
}

impl GroupNorm {
    pub fn new(num_channels: usize, num_groups: usize, eps: f32) -> Self {
        let gamma = Tensor::new(ndarray::Array::from_shape_vec(IxDyn(&[num_channels]), vec![1.0; num_channels]).unwrap(), true);
        let beta = Tensor::new(ndarray::Array::from_shape_vec(IxDyn(&[num_channels]), vec![0.0; num_channels]).unwrap(), true);
        GroupNorm { gamma, beta, num_groups, eps }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let arr = x.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();
        if shape.len() != 4 { return x.clone(); }
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let g = self.num_groups;
        assert!(c % g == 0, "num_channels must be divisible by num_groups");
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, c, h, w]));
        let channels_per_group = c / g;
        for ni in 0..n {
            for gi in 0..g {
                let cstart = gi * channels_per_group;
                let cend = cstart + channels_per_group;
                // compute mean/var over channel block and spatial dims
                let mut sum = 0.0f32;
                let mut sumsq = 0.0f32;
                let mut count = 0usize;
                for ci in cstart..cend {
                    for hi in 0..h {
                        for wi in 0..w {
                            let v = arr[[ni, ci, hi, wi]];
                            sum += v;
                            sumsq += v * v;
                            count += 1;
                        }
                    }
                }
                let mean = sum / count as f32;
                let var = (sumsq / count as f32) - (mean * mean);
                let denom = (var + self.eps).sqrt();
                for ci in cstart..cend {
                    for hi in 0..h {
                        for wi in 0..w {
                            let v = arr[[ni, ci, hi, wi]];
                            let nval = (v - mean) / denom;
                            let gamma = self.gamma.lock().storage.to_f32_array();
                            let beta = self.beta.lock().storage.to_f32_array();
                            let gval = gamma[[ci]];
                            let bval = beta[[ci]];
                            out[[ni, ci, hi, wi]] = nval * gval + bval;
                        }
                    }
                }
            }
        }
        Tensor::new(out, false)
    }
}

/// ResNetBlock: GroupNorm -> SiLU -> Conv2D + time embedding injection
pub struct ResNetBlock {
    pub gn1: GroupNorm,
    pub conv1: crate::nn::Conv2D,
    pub conv2: crate::nn::Conv2D,
    pub proj: Option<crate::nn::Linear>,
}

impl ResNetBlock {
    pub fn new(in_channels: usize, out_channels: usize, num_groups: usize) -> Self {
        ResNetBlock {
            gn1: GroupNorm::new(in_channels, num_groups, 1e-5),
            conv1: Conv2D::new(in_channels, out_channels, 3, 1, 1, true),
            conv2: Conv2D::new(out_channels, out_channels, 3, 1, 1, true),
            proj: if in_channels != out_channels { Some(Linear::new(in_channels, out_channels, true)) } else { None },
        }
    }

    pub fn forward(&self, x: &Tensor, t_emb: Option<&Tensor>) -> Tensor {
        let mut h = self.gn1.forward(x);
        h = h.silu();
        h = self.conv1.forward(&h);
        if let Some(te) = t_emb {
            // project t_emb to spatial dims and add (broadcast)
            let te_proj = crate::nn::Linear::new(te.lock().storage.shape()[te.lock().storage.shape().len()-1], h.lock().storage.shape()[1], true);
            let tp = te_proj.forward(te);
            h = h.add(&tp);
        }
        h = self.gn1.forward(&h);
        h = h.silu();
        h = self.conv2.forward(&h);
        // residual
        let res = if let Some(proj) = &self.proj { proj.forward(x) } else { x.clone() };
        res.add(&h)
    }
}

/// UNet skeleton - minimal forward using blocks above
pub struct UNetModel {
    pub in_channels: usize,
    pub base_channels: usize,
    pub blocks: Vec<ResNetBlock>,
}

impl UNetModel {
    pub fn new(in_channels: usize, base_channels: usize, depth: usize) -> Self {
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            blocks.push(ResNetBlock::new(base_channels << i, base_channels << i, 8));
        }
        UNetModel { in_channels, base_channels, blocks }
    }
    pub fn forward(&self, x: &Tensor, t_emb: &Tensor) -> Tensor {
        let mut h = x.clone();
        for b in &self.blocks {
            h = b.forward(&h, Some(t_emb));
        }
        h
    }
}

impl Module for UNetModel {
    fn forward(&self, input: &Tensor) -> Tensor { input.clone() }
    fn parameters(&self) -> Vec<Tensor> { Vec::new() }
    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Tensor)> { Vec::new() }
    fn load_state_dict(&mut self, _state: &HashMap<String, Tensor>, _prefix: &str) -> Result<(), String> { Ok(()) }
}
