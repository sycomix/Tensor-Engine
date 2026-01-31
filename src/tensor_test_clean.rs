use crate::dtype::{DType, TensorStorage};
use crate::ops::{
    Add, BinaryCrossEntropy, BinaryCrossEntropyWithLogits, Concat, CrossEntropyLogits, Div,
    EmbeddingLookup, KVCacheAppend, LayerNorm, Log, LogSoftmax, MatMul, Mean, Mul, NLLLoss,
    Operation, PermuteAxes, Pow, RMSNorm, ReLU, RoPE, Sigmoid, Softmax, SoftmaxCrossEntropyLogits,
    Stack, Sub, Sum, SwiGLU, Tanh,
};
use ndarray::{ArrayD, IxDyn};
use std::sync::{Arc, Mutex, MutexGuard};

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
