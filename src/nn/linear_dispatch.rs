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
        // Check for quantization keys first
        let qweight_key = format!("{}.qweight", prefix);
        if state.contains_key(&qweight_key) {
            // Found quantized weights, likely need to switch to QuantizedLinear
            // If we are already Quantized, just load.
            // If we are F32, we need to replace ourselves with a QuantizedLinear.

            // We need metadata to construct QuantizedLinear if we are switching.
            // Specifically: in_features, out_features, group_size.
            // These might be inferable from shapes or user config.
            // For now, we attempt to infer from shapes or use existing F32 dimensions.

            match self {
                LinearLayer::F32(l) => {
                    log::info!(
                        "LinearLayer: switching from F32 to QuantizedLinear for {}",
                        prefix
                    );
                    // Infer dimensions from current F32 layer
                    let in_features = l.in_features;
                    let out_features = l.out_features;

                    // Default group_size if not known?
                    // Usually 128 is a safe bet for AWQ unless specified otherwise.
                    // Ideally we'd pass this in or it would be in the state dict metadata.
                    // For this implementation, we'll assume 128 or try to infer.
                    let group_size = 128;

                    let qweight = state.get(&qweight_key).unwrap().clone();
                    // qzeros/scales must exist
                    let qzeros = state
                        .get(&format!("{}.qzeros", prefix))
                        .ok_or_else(|| format!("Missing qzeros for {}", prefix))?
                        .clone();
                    let scales = state
                        .get(&format!("{}.scales", prefix))
                        .ok_or_else(|| format!("Missing scales for {}", prefix))?
                        .clone();
                    let bias = state.get(&format!("{}.bias", prefix)).cloned();

                    let ql = crate::nn::quantized::QuantizedLinear::new(
                        qweight,
                        qzeros,
                        scales,
                        bias,
                        in_features,
                        out_features,
                        group_size,
                    );
                    // No need to call load_state_dict again since we just constructed it with the data
                    *self = LinearLayer::Quantized(ql);
                }
                LinearLayer::Quantized(l) => {
                    l.load_state_dict(state, prefix)?;
                }
            }
            Ok(())
        } else {
            // Standard F32 load
            match self {
                LinearLayer::F32(l) => l.load_state_dict(state, prefix),
                LinearLayer::Quantized(_) => {
                    // If we are quantized but keys are missing, that's an error or we revert?
                    // Usually we shouldn't revert automatically without explicit instructions.
                    // It means the state dict doesn't match the current architecture.
                    Err(format!(
                        "LinearLayer: expected quantized keys for {} but found none",
                        prefix
                    ))
                }
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
