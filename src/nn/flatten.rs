use crate::nn::Module;
use crate::tensor::Tensor;

/// Flatten layer: converts 4D tensors [N, C, H, W] into 2D tensors [N, C*H*W].
/// Keeps requires_grad flag, so the new tensor is backpropagable.
#[derive(Debug, Clone, Default)]
pub struct Flatten {}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        let guard = input.lock();
        let data = guard.storage.to_f32_array();
        let requires_grad = guard.requires_grad;
        drop(guard);

        match data.ndim() {
            2 => Tensor::new(data.into_dyn(), requires_grad),
            4 => {
                let shape = data.shape().to_vec();
                let batch = shape[0];
                let features = shape[1] * shape[2] * shape[3];
                let arr = data.into_shape_with_order((batch, features)).unwrap();
                Tensor::new(arr.into_dyn(), requires_grad)
            }
            _ => {
                let total = data.len();
                let arr = data.into_shape_with_order((1, total)).unwrap();
                Tensor::new(arr.into_dyn(), requires_grad)
            }
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
