use super::{Conv1D, ConvTranspose1D, Module};
use crate::tensor::Tensor;

/// Audio Encoder: stack of Conv1D downsampling layers
pub struct AudioEncoder {
    pub layers: Vec<Conv1D>,
}

impl AudioEncoder {
    pub fn new(in_channels: usize, hidden: usize, layers: usize) -> Self {
        let mut convs = Vec::new();
        let mut in_ch = in_channels;
        for i in 0..layers {
            let out_ch = hidden * (1 << i); // exponential growth
            convs.push(Conv1D::new(in_ch, out_ch, 4, 2, 1, true));
            in_ch = out_ch;
        }
        AudioEncoder { layers: convs }
    }
}

impl Module for AudioEncoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.clone();
        for l in &self.layers {
            out = l.forward(&out);
            out = out.relu();
        }
        out
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l: &Conv1D| l.parameters()).collect::<Vec<Tensor>>()
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Audio Decoder: stack of ConvTranspose1D upsampling layers (mirror of encoder)
pub struct AudioDecoder {
    pub layers: Vec<ConvTranspose1D>,
}

impl AudioDecoder {
    pub fn new(in_channels: usize, hidden: usize, layers: usize) -> Self {
        let mut convs = Vec::new();
        let mut in_ch = in_channels;
        for i in 0..layers {
            let out_ch = if i == layers - 1 { 1 } else { hidden * (1 << (layers - i - 1)) };
            convs.push(ConvTranspose1D::new(in_ch, out_ch, 4, 2, 1, true));
            in_ch = out_ch;
        }
        AudioDecoder { layers: convs }
    }
}

impl Module for AudioDecoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.clone();
        for l in &self.layers {
            out = l.forward(&out);
            out = out.relu();
        }
        out
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l: &ConvTranspose1D| l.parameters()).collect::<Vec<Tensor>>()
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
