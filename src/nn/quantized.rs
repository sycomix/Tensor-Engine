use crate::nn::Module;
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

    /// Dequantize weights to F32 using AWQ affine logic.
    pub fn dequantize_to_float(&self) -> Result<Tensor, String> {
        // Assuming qweight is [Out, In_Packed] and we want [Out, In]
        crate::quantization::awq::awq_dequantize_affine(
            &self.qweight,
            &self.scales,
            &self.qzeros,
            self.group_size,
            &[self.out_features, self.in_features],
        )
    }
}

impl Module for QuantizedLinear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Backend::matmul_quantized not yet implemented
        // Fallback: dequantize to float since backend doesn't support packed matmul yet
        if let Ok(weights) = self.dequantize_to_float() {
            let w_t = weights.transpose();
            let out = input.matmul(&w_t);
            if let Some(b) = &self.bias {
                out.add(b)
            } else {
                out
            }
        } else {
            panic!("QuantizedLinear: no backend implementation available for matmul_quantized and dequantization failed")
        }
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
