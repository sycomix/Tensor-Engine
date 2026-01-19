use crate::nn::quantized::QuantizedLinear;
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;
use std::any::Any;
use std::collections::HashMap;

/// An enum wrapper that can hold either a standard F32 Linear layer or a QuantizedLinear layer.
/// This allows models to mix and match precision or load quantized weights into a standard architecture.
#[derive(Clone)]
pub enum LinearLayer {
    F32(Linear),
    Quantized(QuantizedLinear),
}

impl LinearLayer {
    /// Create a new F32 linear layer (wrapper)
    pub fn new_f32(in_features: usize, out_features: usize, bias: bool) -> Self {
        LinearLayer::F32(Linear::new(in_features, out_features, bias))
    }

    pub fn as_f32(&self) -> Option<&Linear> {
        match self {
            LinearLayer::F32(l) => Some(l),
            _ => None,
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut Linear> {
        match self {
            LinearLayer::F32(l) => Some(l),
            _ => None,
        }
    }
}

impl Module for LinearLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            LinearLayer::F32(l) => l.forward(input),
            LinearLayer::Quantized(l) => l.forward(input),
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        match self {
            LinearLayer::F32(l) => l.parameters(),
            LinearLayer::Quantized(l) => l.parameters(),
        }
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        match self {
            LinearLayer::F32(l) => l.named_parameters(prefix),
            LinearLayer::Quantized(l) => l.named_parameters(prefix),
        }
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        match self {
            LinearLayer::F32(l) => l.load_state_dict(state, prefix),
            LinearLayer::Quantized(l) => l.load_state_dict(state, prefix),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
