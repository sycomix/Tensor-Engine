//! WaveNet and HiFi-GAN audio generation models.
//!
//! WaveNet: Deep residual network for raw audio waveform generation.
//! HiFi-GAN: Generative adversarial network for high-fidelity audio synthesis.

use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// WaveNet residual block with dilated convolutions and gated activations.
#[derive(Clone)]
pub struct WaveNetBlock {
    /// Dilated convolution
    pub dilated_conv: Conv1D,
    /// 1x1 convolution for skip connection
    pub skip_conv: Conv1D,
    /// Input gate filter
    pub filter_conv: Conv1D,
    /// Input gate gate
    pub gate_conv: Conv1D,
    /// Dilation rate
    pub dilation: usize,
}

impl WaveNetBlock {
    /// Create a new WaveNet block.
    pub fn new(
        in_channels: usize,
        residual_channels: usize,
        dilation_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Self {
        WaveNetBlock {
            dilated_conv: Conv1D::new(
                residual_channels,
                residual_channels,
                kernel_size,
                1,
                dilation,
                true,
            ),
            skip_conv: Conv1D::new(residual_channels, residual_channels, 1, 1, 0, true),
            filter_conv: Conv1D::new(in_channels, dilation_channels, 1, 1, 0, true),
            gate_conv: Conv1D::new(in_channels, dilation_channels, 1, 1, 0, true),
            dilation,
        }
    }

    /// Forward pass through the WaveNet block.
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        // Gated activation: tanh(filter) * sigmoid(gate)
        let filter = self.filter_conv.forward(x).tanh();
        let gate = self.gate_conv.forward(x).sigmoid();
        let gated = filter.mul(&gate);

        // Apply dilated convolution
        let dilated = self.dilated_conv.forward(&gated);

        // Skip connection
        let skip = self.skip_conv.forward(&dilated);

        // Residual connection
        let residual = x.add(&dilated);

        (residual, skip)
    }
}

/// WaveNet model for audio generation.
///
/// Stacks multiple WaveNet blocks with increasing dilation rates.
#[derive(Clone)]
pub struct WaveNet {
    pub blocks: Vec<WaveNetBlock>,
    pub input_projection: Conv1D,
    pub output_proj1: Conv1D,
    pub output_proj2: Conv1D,
    pub num_layers: usize,
    pub num_stacks: usize,
}

impl WaveNet {
    /// Create a new WaveNet model.
    pub fn new(
        in_channels: usize,
        residual_channels: usize,
        dilation_channels: usize,
        kernel_size: usize,
        num_layers: usize,
        num_stacks: usize,
    ) -> Self {
        let mut blocks = Vec::new();
        for i in 0..num_layers {
            let dilation = 2usize.pow((i % num_layers) as u32);
            blocks.push(WaveNetBlock::new(
                in_channels,
                residual_channels,
                dilation_channels,
                kernel_size,
                dilation,
            ));
        }

        WaveNet {
            blocks,
            input_projection: Conv1D::new(in_channels, residual_channels, 1, 1, 0, true),
            output_proj1: Conv1D::new(residual_channels, dilation_channels, 1, 1, 0, true),
            output_proj2: Conv1D::new(dilation_channels, in_channels, 1, 1, 0, true),
            num_layers,
            num_stacks,
        }
    }

    /// Forward pass through the WaveNet.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.input_projection.forward(x);
        let mut skip_connections = Vec::new();

        for block in &self.blocks {
            let (_, skip) = block.forward(&x);
            skip_connections.push(skip);
        }

        // Sum all skip connections
        let mut out = skip_connections
            .pop()
            .unwrap_or_else(|| Tensor::zeros(&[0]));
        for skip in skip_connections {
            out = out.add(&skip);
        }

        // Output projection
        out = self.output_proj1.forward(&out).relu();
        self.output_proj2.forward(&out)
    }
}

impl Module for WaveNet {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.input_projection.weight.clone(),
            self.output_proj1.weight.clone(),
            self.output_proj2.weight.clone(),
        ];
        for block in &self.blocks {
            params.extend(block.dilated_conv.parameters());
            params.extend(block.skip_conv.parameters());
            params.extend(block.filter_conv.parameters());
            params.extend(block.gate_conv.parameters());
        }
        params
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// HiFi-GAN generator block with transposed convolutions.
#[derive(Clone)]
pub struct HifiGanBlock {
    /// Main convolution path
    pub convs: Vec<Conv1D>,
    /// Residual path
    pub res_blocks: Vec<ResBlock>,
    /// Number of residual blocks
    pub num_res_blocks: usize,
}

impl HifiGanBlock {
    /// Create a new HiFi-GAN block.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        num_res_blocks: usize,
    ) -> Self {
        let mut convs = Vec::new();
        let mut res_blocks = Vec::new();

        // Main path: Conv -> LeakyReLU -> BatchNorm
        convs.push(Conv1D::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            1,
            true,
        ));

        // Residual blocks
        for _ in 0..num_res_blocks {
            res_blocks.push(ResBlock::new(out_channels));
        }

        HifiGanBlock {
            convs,
            res_blocks,
            num_res_blocks,
        }
    }

    /// Forward pass through the HiFi-GAN block.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = self.convs[0].forward(x).leaky_relu();

        // Residual path
        let mut residual = Tensor::zeros(&[0]);
        for res_block in &self.res_blocks {
            residual = residual.add(&res_block.forward(&out));
        }

        // Add residual
        out.add(&residual)
    }
}

/// Residual block for HiFi-GAN.
#[derive(Clone)]
pub struct ResBlock {
    pub conv1: Conv1D,
    pub conv2: Conv1D,
    pub kernel_sizes: [usize; 2],
    pub dilations: [usize; 2],
}

impl ResBlock {
    /// Create a new residual block.
    pub fn new(channels: usize) -> Self {
        ResBlock {
            conv1: Conv1D::new(channels, channels, 3, 1, 1, true),
            conv2: Conv1D::new(channels, channels, 3, 1, 1, true),
            kernel_sizes: [3, 3],
            dilations: [1, 3],
        }
    }

    /// Forward pass through the residual block.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = self.conv1.forward(x).leaky_relu();
        self.conv2.forward(&out).leaky_relu()
    }
}

/// HiFi-GAN generator model.
#[derive(Clone)]
pub struct HifiGanGenerator {
    pub blocks: Vec<HifiGanBlock>,
    pub final_conv: Conv1D,
}

impl HifiGanGenerator {
    /// Create a new HiFi-GAN generator.
    pub fn new(channel_sizes: Vec<usize>, kernel_sizes: Vec<usize>, stride: usize) -> Self {
        let mut blocks = Vec::new();
        let mut in_ch = 128; // Input is typically 128-dim latent

        for (i, &out_ch) in channel_sizes.iter().enumerate() {
            let kernel = *kernel_sizes.get(i).unwrap_or(&7);
            blocks.push(HifiGanBlock::new(in_ch, out_ch, kernel, stride, 8));
            in_ch = out_ch;
        }

        HifiGanGenerator {
            blocks,
            final_conv: Conv1D::new(in_ch, 1, 7, 1, 1, true),
        }
    }

    /// Forward pass through the generator.
    pub fn forward(&self, z: &Tensor) -> Tensor {
        let mut out = z.clone();
        for block in &self.blocks {
            out = block.forward(&out);
        }
        self.final_conv.forward(&out).tanh()
    }
}

impl Module for HifiGanGenerator {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for block in &self.blocks {
            params.extend(block.convs.iter().flat_map(|c| c.parameters()));
            params.extend(block.res_blocks.iter().flat_map(|r| r.conv1.parameters()));
            params.extend(block.res_blocks.iter().flat_map(|r| r.conv2.parameters()));
        }
        params.extend(self.final_conv.parameters());
        params
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// HiFi-GAN discriminator block.
#[derive(Clone)]
pub struct HifiGanDiscriminatorBlock {
    pub conv: Conv1D,
    pub batch_norm: BatchNorm1d,
}

impl HifiGanDiscriminatorBlock {
    /// Create a new discriminator block.
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        HifiGanDiscriminatorBlock {
            conv: Conv1D::new(in_channels, out_channels, 7, 2, 1, true),
            batch_norm: BatchNorm1d::new(out_channels),
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = self.conv.forward(x);
        self.batch_norm.forward(&out).leaky_relu()
    }
}

/// HiFi-GAN discriminator model.
#[derive(Clone)]
pub struct HifiGanDiscriminator {
    pub blocks: Vec<HifiGanDiscriminatorBlock>,
    pub final_conv: Conv1D,
}

impl HifiGanDiscriminator {
    /// Create a new discriminator.
    pub fn new() -> Self {
        HifiGanDiscriminator {
            blocks: vec![
                HifiGanDiscriminatorBlock::new(1, 64),
                HifiGanDiscriminatorBlock::new(64, 128),
                HifiGanDiscriminatorBlock::new(128, 256),
                HifiGanDiscriminatorBlock::new(256, 512),
                HifiGanDiscriminatorBlock::new(512, 1024),
                HifiGanDiscriminatorBlock::new(1024, 1024),
            ],
            final_conv: Conv1D::new(1024, 1, 1, 1, 0, true),
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        for block in &self.blocks {
            out = block.forward(&out);
        }
        self.final_conv.forward(&out)
    }
}

/// Multi-scale discriminator for HiFi-GAN training.
#[derive(Clone)]
pub struct MultiScaleDiscriminator {
    pub discriminators: Vec<HifiGanDiscriminator>,
}

impl MultiScaleDiscriminator {
    /// Create a new multi-scale discriminator.
    pub fn new(num_scales: usize) -> Self {
        MultiScaleDiscriminator {
            discriminators: (0..num_scales)
                .map(|_| HifiGanDiscriminator::new())
                .collect(),
        }
    }

    /// Forward pass through all scales.
    pub fn forward(&self, x: &Tensor) -> Vec<Tensor> {
        let mut outputs = Vec::new();
        let mut input = x.clone();

        for (i, disc) in self.discriminators.iter().enumerate() {
            outputs.push(disc.forward(&input));

            // Downsample for next scale
            if i < self.discriminators.len() - 1 {
                input = self.downsample(&input);
            }
        }

        outputs
    }

    /// Downsample by factor of 2.
    fn downsample(&self, x: &Tensor) -> Tensor {
        let arr = x.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();
        if shape.len() != 3 {
            return x.clone();
        }

        let c = shape[0];
        let h = shape[1];
        let w = shape[2];

        let new_w = w / 2;
        let mut downsampled = ArrayD::<f32>::zeros(IxDyn(&[c, h, new_w][..]));

        for ch in 0..c {
            for y in 0..h {
                for x in 0..new_w {
                    downsampled[[ch, y, x]] = arr[[ch, y, x * 2]];
                }
            }
        }

        Tensor::new(downsampled.into_dyn(), false)
    }
}

/// 1D convolution layer (NCL format).
#[derive(Clone)]
pub struct Conv1D {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
}

impl Conv1D {
    /// Create a new 1D convolution.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let weight_data = Tensor::new(
            ndarray::Array::zeros(IxDyn(&[out_channels, in_channels, kernel_size][..])),
            true,
        );
        let bias = if bias {
            Some(Tensor::new(
                ndarray::Array::zeros(IxDyn(&[out_channels][..])),
                true,
            ))
        } else {
            None
        };
        Conv1D {
            weight: weight_data,
            bias,
            stride,
            padding,
        }
    }
}

impl Module for Conv1D {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut inputs = vec![input.clone(), self.weight.clone()];
        if let Some(b) = &self.bias {
            inputs.push(b.clone());
        }
        Tensor::apply(
            std::sync::Arc::new(crate::ops::Conv1D::new(self.stride, self.padding)),
            &inputs,
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Batch normalization for 1D inputs.
#[derive(Clone)]
pub struct BatchNorm1d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub gamma: Tensor,
    pub beta: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub training: bool,
}

impl BatchNorm1d {
    /// Create a new batch normalization layer.
    pub fn new(num_features: usize) -> Self {
        let rm = Tensor::zeros(&[num_features][..]);
        rm.set_requires_grad(false);
        let rv = Tensor::ones(&[num_features][..]);
        rv.set_requires_grad(false);

        BatchNorm1d {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            gamma: Tensor::ones(&[num_features][..]),
            beta: Tensor::zeros(&[num_features][..]),
            running_mean: rm,
            running_var: rv,
            training: true,
        }
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let config = crate::tensor::BatchNormConfig {
            momentum: self.momentum,
            eps: self.eps,
            training: self.training,
        };
        input.batch_norm(
            &self.gamma,
            &self.beta,
            &self.running_mean,
            &self.running_var,
            config,
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// LeakyReLU activation extension.
pub trait LeakyReLUExt {
    /// Apply LeakyReLU with negative slope 0.2.
    fn leaky_relu(&self) -> Tensor;
}

impl LeakyReLUExt for Tensor {
    fn leaky_relu(&self) -> Tensor {
        let arr = self.lock().storage.to_f32_array();
        let mut out = arr.clone();
        for v in out.iter_mut() {
            if *v < 0.0 {
                *v *= 0.2;
            }
        }
        Tensor::new(out.into_dyn(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavenet_block() {
        let block = WaveNetBlock::new(80, 128, 128, 3, 1);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 80, 16]), vec![0.0f32; 1280]).unwrap(),
            false,
        );
        let (residual, skip) = block.forward(&x);
        let res_shape = residual.lock().storage.shape().to_vec();
        let skip_shape = skip.lock().storage.shape().to_vec();
        assert_eq!(res_shape.len(), 3);
        assert_eq!(skip_shape.len(), 3);
    }

    #[test]
    fn test_hifi_gan_generator() {
        let channel_sizes = vec![128, 64, 32];
        let kernel_sizes = vec![7, 5, 3];
        let generator = HifiGanGenerator::new(channel_sizes, kernel_sizes, 2);
        let z = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 128, 8]), vec![0.0f32; 1024]).unwrap(),
            false,
        );
        let out = generator.forward(&z);
        let out_shape = out.lock().storage.shape().to_vec();
        assert_eq!(out_shape[0], 1);
        assert_eq!(out_shape[1], 1);
    }

    #[test]
    fn test_hifi_gan_discriminator() {
        let disc = HifiGanDiscriminator::new();
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 16000]), vec![0.0f32; 16000]).unwrap(),
            false,
        );
        let out = disc.forward(&x);
        let out_shape = out.lock().storage.shape().to_vec();
        assert_eq!(out_shape[0], 1);
        assert_eq!(out_shape[1], 1);
    }

    #[test]
    fn test_multi_scale_discriminator() {
        let ms_disc = MultiScaleDiscriminator::new(3);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 16000]), vec![0.0f32; 16000]).unwrap(),
            false,
        );
        let outputs = ms_disc.forward(&x);
        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn test_leaky_relu() {
        let pos = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
            false,
        );
        let neg = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -2.0]).unwrap(),
            false,
        );

        let pos_out = pos.leaky_relu();
        let neg_out = neg.leaky_relu();

        let pos_arr = pos_out.lock().storage.to_f32_array();
        let neg_arr = neg_out.lock().storage.to_f32_array();

        assert!((pos_arr[0] - 1.0).abs() < 1e-6);
        assert!((pos_arr[1] - 2.0).abs() < 1e-6);
        assert!((neg_arr[0] - (-0.2)).abs() < 1e-6);
        assert!((neg_arr[1] - (-0.4)).abs() < 1e-6);
    }
}
