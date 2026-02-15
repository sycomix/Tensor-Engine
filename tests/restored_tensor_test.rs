use ndarray::{ArrayD, IxDyn};
use std::sync::{Arc, Mutex};
use tensor_engine::dtype::{DType, TensorStorage};
use tensor_engine::ops::Operation;

/// `TensorData` contains the actual data of a tensor, along with metadata for automatic differentiation.
pub struct TensorData {
    /// The tensor's data, stored as a dynamically-dimensioned array.
    pub storage: TensorStorage,
    /// The gradient of tensor, if it has one.
    pub grad: Option<ArrayD<f32>>,
    /// The operation that created this tensor, if any.
    pub creator: Option<Arc<dyn Operation + Send + Sync>>,
    /// The input tensors that were used to create this tensor.
    pub inputs: Vec<Tensor>,
    /// Whether this tensor requires a gradient.
    pub requires_grad: bool,
    /// Data type indicator for storage/representation purposes (MVP: data stays f32 but dtype captures intended storage semantics)
    pub dtype: DType,
}

/// A multi-dimensional array (tensor) that supports automatic differentiation.
///
/// Tensors are fundamental data structure in `tensor_engine`. They can be created from scratch
/// or as a result of operations on other tensors. If a tensor is created from operations on other
/// tensors that have `requires_grad = true`, then it will also have `requires_grad = true` and will
/// be part of a computation graph.
#[derive(Clone)]
pub struct Tensor(Arc<Mutex<TensorData>>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0);
        let t_data = TensorData {
            storage: TensorStorage::from_f32_array(&data, DType::F32),
            grad: None,
            creator: None,
            inputs: vec![],
            requires_grad: false,
            dtype: DType::F32,
        };
        let t = Tensor(Arc::new(Mutex::new(t_data)));

        let lock = t.0.lock().unwrap();
        assert_eq!(lock.dtype, DType::F32);
        assert!(!lock.requires_grad);
    }
}
