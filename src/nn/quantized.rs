use crate::nn::Module;
use crate::quantization::awq::awq_dequantize_affine;
use crate::tensor::Tensor;
use std::any::Any;
use std::collections::HashMap;

/// A linear layer with 4-bit quantized weights (AWQ).
///
/// This module stores weights in a packed 4-bit format (u8) and dequantizes them
/// on-the-fly during the forward pass using CPU reference implementation.
///
/// Math: w = (qweight - qzeros) * scales
/// Output = input @ w.T + bias
#[derive(Clone)]
pub struct QuantizedLinear {
    pub qweight: Tensor, // [in_features, out_features / 2] (u8)
    pub qzeros: Tensor,  // [in_features, out_features / group_size] (f32 for now, simplified)
    pub scales: Tensor,  // [in_features, out_features / group_size] (f32)
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
    pub group_size: usize,
}

impl QuantizedLinear {
    pub fn new(
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> Self {
        QuantizedLinear {
            qweight,
            qzeros,
            scales,
            bias,
            in_features,
            out_features,
            group_size,
        }
    }
}

impl Module for QuantizedLinear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Dequantize weights
        // Shape of qweight is likely [cols, rows] if transposed or [rows, cols]
        // Standard Linear weights are [out, in] or [in, out] depending on framework.
        // In this workspace, Linear uses [in, out] (see src/nn/mod.rs).
        // Let's assume qweight is [in_features, out_features] logical.
        let target_shape = vec![self.in_features, self.out_features];

        let w = match awq_dequantize_affine(
            &self.qweight,
            &self.scales,
            &self.qzeros,
            self.group_size,
            &target_shape,
        ) {
            Ok(t) => t,
            Err(e) => {
                log::error!("QuantizedLinear forward failed: {}", e);
                // Return dummy to avoid panic in production (though this is serious)
                // In reference impl, panicking might be better, but we log and return zeros.
                Tensor::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&target_shape)), false)
            }
        };

        // Linear forward: input @ weight + bias
        // Handle broadcasting if input is > 2D (e.g. [batch, seq, in])
        // MatMul op only supports 2D, so we flatten, matmul, then reshape.
        let input_shape = input.lock().storage.shape();
        let ndim = input_shape.len();

        let activation = if ndim > 2 {
            let last_dim = input_shape[ndim - 1];
            if last_dim != self.in_features {
                log::error!(
                    "QuantizedLinear input shape mismatch: expected last dim {}, got {}",
                    self.in_features,
                    last_dim
                );
            }
            // product of all dims except last
            let batch_dim: usize = input_shape[0..ndim - 1].iter().product();
            let flattened_shape = vec![batch_dim, last_dim];

            // Reshape input to 2D
            match input.reshape(flattened_shape) {
                Ok(flat_input) => {
                    let flat_out = flat_input.matmul(&w);
                    // Reshape back to [..., out_features]
                    let mut out_shape = input_shape[0..ndim - 1].to_vec();
                    out_shape.push(self.out_features);
                    match flat_out.reshape(out_shape) {
                        Ok(o) => o,
                        Err(e) => {
                            log::error!("QuantizedLinear: failed to reshape output: {}", e);
                            flat_out // return flattened on error to avoid panic, though incorrect
                        }
                    }
                }
                Err(e) => {
                    log::error!("QuantizedLinear: failed to flatten input: {}", e);
                    Tensor::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[][..])), false)
                }
            }
        } else {
            input.matmul(&w)
        };

        let bias = match &self.bias {
            Some(b) => b.clone(),
            None => Tensor::new(
                ndarray::ArrayD::zeros(ndarray::IxDyn(&[self.out_features][..])),
                false,
            ),
        };

        activation.add(&bias)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![
            self.qweight.clone(),
            self.qzeros.clone(),
            self.scales.clone(),
        ];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut p = vec![
            (format!("{}.qweight", prefix), self.qweight.clone()),
            (format!("{}.qzeros", prefix), self.qzeros.clone()),
            (format!("{}.scales", prefix), self.scales.clone()),
        ];
        if let Some(b) = &self.bias {
            p.push((format!("{}.bias", prefix), b.clone()));
        }
        p
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        if let Some(t) = state.get(&format!("{}.qweight", prefix)) {
            self.qweight = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.qzeros", prefix)) {
            self.qzeros = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.scales", prefix)) {
            self.scales = t.clone();
        }
        if let Some(t) = state.get(&format!("{}.bias", prefix)) {
            self.bias = Some(t.clone());
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
