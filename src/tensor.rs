use crate::ops::{
    Add, Concat, CrossEntropyLogits, Div, LayerNorm, Log, LogSoftmax, MatMul, Mean, Mul, NLLLoss,
    Operation, Pow, ReLU, Sigmoid, Softmax, SoftmaxCrossEntropyLogits, Stack, Sub, Sum, Tanh,
};
use ndarray::{ArrayD, IxDyn};
use std::sync::{Arc, Mutex, MutexGuard};

/// `TensorData` contains the actual data of a tensor, along with metadata for automatic differentiation.
pub struct TensorData {
    /// The tensor's data, stored as a dynamically-dimensioned array.
    pub data: ArrayD<f32>,
    /// The gradient of the tensor, if it has one.
    pub grad: Option<ArrayD<f32>>,
    /// The operation that created this tensor, if any.
    pub creator: Option<Arc<dyn Operation + Send + Sync>>,
    /// The input tensors that were used to create this tensor.
    pub inputs: Vec<Tensor>,
    /// Whether this tensor requires a gradient.
    pub requires_grad: bool,
}

/// A multi-dimensional array (tensor) that supports automatic differentiation.
///
/// Tensors are the fundamental data structure in `tensor_engine`. They can be created from scratch
/// or as the result of operations on other tensors. If a tensor is created from operations on other
/// tensors that have `requires_grad = true`, then it will also have `requires_grad = true` and will
/// be part of a computation graph.
#[derive(Clone)]
pub struct Tensor(Arc<Mutex<TensorData>>);

impl Tensor {
    /// Creates a new tensor.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor's data.
    /// * `requires_grad` - Whether this tensor should have a gradient.
    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> Self {
        Tensor(Arc::new(Mutex::new(TensorData {
            data,
            grad: None,
            creator: None,
            inputs: vec![],
            requires_grad,
        })))
    }

    /// Applies an operation to a set of input tensors.
    ///
    /// This is the primary way that computation graphs are constructed.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to apply.
    /// * `inputs` - The input tensors.
    pub fn apply(op: Arc<dyn Operation + Send + Sync>, inputs: &[Tensor]) -> Tensor {
        let requires_grad = inputs.iter().any(|t| t.lock().requires_grad);
        // Determine output shape, supporting broadcasting for element-wise ops.
        let out_shape: Vec<usize> = if op.as_any().is::<Sum>() || op.as_any().is::<Mean>() {
            vec![] // scalar
        } else if op.as_any().is::<crate::ops::Concat>() || op.as_any().is::<crate::ops::Stack>() {
            // Concat/Stack manage their own shapes in ops implementations; default to first input
            inputs[0].lock().data.shape().to_vec()
        } else {
            // Generic element-wise broadcast across inputs
            fn broadcast_shape_from(shapes: &[Vec<usize>]) -> Result<Vec<usize>, String> {
                let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
                let mut result = vec![1usize; max_ndim];
                for s in shapes {
                    for (i, &dim) in s.iter().rev().enumerate() {
                        let ridx = max_ndim - 1 - i;
                        let cur = result[ridx];
                        if cur == 1 {
                            result[ridx] = dim;
                        } else if dim == 1 { /* keep cur */
                        } else if cur == dim { /* ok */
                        } else {
                            return Err(format!("Cannot broadcast shapes: {:?}", shapes));
                        }
                    }
                }
                Ok(result)
            }

            let shapes: Vec<Vec<usize>> = inputs
                .iter()
                .map(|t| t.lock().data.shape().to_vec())
                .collect();
            match broadcast_shape_from(&shapes) {
                Ok(s) => s,
                Err(_e) => inputs[0].lock().data.shape().to_vec(),
            }
        };

        let mut data = ArrayD::zeros(IxDyn(&out_shape));
        op.forward(inputs, &mut data);

        Tensor(Arc::new(Mutex::new(TensorData {
            data,
            grad: None,
            creator: Some(op),
            inputs: inputs.to_vec(),
            requires_grad,
        })))
    }

    /// Adds two tensors.
    pub fn add(&self, other: &Tensor) -> Tensor {
        Tensor::apply(Arc::new(Add), &[self.clone(), other.clone()])
    }

    /// Multiplies two tensors.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        Tensor::apply(Arc::new(Mul), &[self.clone(), other.clone()])
    }

    /// Subtracts two tensors.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        Tensor::apply(Arc::new(Sub), &[self.clone(), other.clone()])
    }

    /// Divides two tensors.
    pub fn div(&self, other: &Tensor) -> Tensor {
        Tensor::apply(Arc::new(Div), &[self.clone(), other.clone()])
    }

    /// Performs matrix multiplication.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        Tensor::apply(Arc::new(MatMul), &[self.clone(), other.clone()])
    }

    /// Raises a tensor to a power.
    pub fn pow(&self, power: f32) -> Tensor {
        Tensor::apply(Arc::new(Pow(power)), &[self.clone()])
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Tensor {
        // Use imported `Log` symbol (avoids unused-import warnings for `Log` in module imports)
        Tensor::apply(Arc::new(Log), &[self.clone()])
    }

    /// Negates the tensor (multiply by -1 scalar).
    pub fn neg(&self) -> Tensor {
        let scalar = Tensor::new(ArrayD::from_elem(IxDyn(&[]), -1.0), false);
        Tensor::apply(Arc::new(Mul), &[self.clone(), scalar])
    }

    /// Applies the ReLU activation function.
    pub fn relu(&self) -> Tensor {
        Tensor::apply(Arc::new(ReLU), &[self.clone()])
    }

    /// Applies the sigmoid activation function.
    pub fn sigmoid(&self) -> Tensor {
        Tensor::apply(Arc::new(Sigmoid), &[self.clone()])
    }

    /// Applies the tanh activation function.
    pub fn tanh(&self) -> Tensor {
        Tensor::apply(Arc::new(Tanh), &[self.clone()])
    }

    /// Computes the sum of the tensor's elements.
    pub fn sum(&self) -> Tensor {
        Tensor::apply(Arc::new(Sum), &[self.clone()])
    }

    /// Computes the mean of the tensor's elements.
    pub fn mean(&self) -> Tensor {
        Tensor::apply(Arc::new(Mean), &[self.clone()])
    }

    /// Element-wise softmax along the specified axis (default last axis)
    pub fn softmax(&self, axis: usize) -> Tensor {
        Tensor::apply(Arc::new(Softmax::new(axis)), &[self.clone()])
    }

    /// Stable log-softmax along the specified axis
    pub fn log_softmax(&self, axis: usize) -> Tensor {
        Tensor::apply(Arc::new(LogSoftmax::new(axis)), &[self.clone()])
    }

    /// Cross-entropy with logits (logits + targets), targets may be a vector of indices (float ints) or one-hot vectors.
    /// `axis` may be negative to index from the right (e.g., -1). Pass axis as signed integer.
    pub fn cross_entropy_with_logits(&self, target: &Tensor, axis: isize) -> Tensor {
        let ndim = self.lock().data.ndim() as isize;
        let axis_norm = if axis < 0 {
            (ndim + axis) as usize
        } else {
            axis as usize
        };
        Tensor::apply(
            Arc::new(CrossEntropyLogits::new(axis_norm)),
            &[self.clone(), target.clone()],
        )
    }

    /// Combined softmax and cross-entropy for logits to avoid extra allocations.
    /// `axis` may be negative to index from the right (e.g., -1).
    pub fn softmax_cross_entropy_with_logits(&self, target: &Tensor, axis: isize) -> Tensor {
        let ndim = self.lock().data.ndim() as isize;
        let axis_norm = if axis < 0 {
            (ndim + axis) as usize
        } else {
            axis as usize
        };
        Tensor::apply(
            Arc::new(SoftmaxCrossEntropyLogits::new(axis_norm)),
            &[self.clone(), target.clone()],
        )
    }

    /// NLLLoss expects log-probabilities (log_softmax output) and integer label indices (as floats) or one-hot.
    pub fn nll_loss(&self, target: &Tensor) -> Tensor {
        Tensor::apply(Arc::new(NLLLoss::new()), &[self.clone(), target.clone()])
    }

    /// Layer normalization along axis with learnable gamma and beta tensors.
    pub fn layer_norm(&self, axis: usize, eps: f32, gamma: &Tensor, beta: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(LayerNorm::new(axis, eps)),
            &[self.clone(), gamma.clone(), beta.clone()],
        )
    }

    /// Reshapes the tensor.
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor, String> {
        // Validate target shape first to produce same error semantics
        let lock = self.lock();
        let data_clone = lock.data.clone();
        let _requires_grad = lock.requires_grad;
        drop(lock);
        // Validate shape
        match data_clone.to_shape(shape.clone()) {
            Ok(_) => Ok(Tensor::apply(
                Arc::new(crate::ops::Reshape::new(shape)),
                &[self.clone()],
            )),
            Err(e) => Err(format!(
                "Cannot reshape tensor from {:?} to {:?}: {}",
                data_clone.shape(),
                shape,
                e
            )),
        }
    }

    /// Transposes the tensor.
    pub fn transpose(&self) -> Tensor {
        let lock = self.lock();
        let data = lock.data.clone().reversed_axes();
        let requires_grad = lock.requires_grad;
        drop(lock);
        Tensor::new(data, requires_grad)
    }

    /// Concatenates a list of tensors along a given axis.
    pub fn concat(tensors: &[Tensor], axis: usize) -> Tensor {
        Tensor::apply(Arc::new(Concat(axis)), tensors)
    }

    /// Stacks a list of tensors along a new axis.
    pub fn stack(tensors: &[Tensor], axis: usize) -> Tensor {
        Tensor::apply(Arc::new(Stack(axis)), tensors)
    }

    /// Locks the tensor's data for reading or writing.
    pub fn lock(&self) -> MutexGuard<'_, TensorData> {
        self.0.lock().unwrap()
    }

    /// Sets the gradient of this tensor to zero.
    pub fn zero_grad(&self) {
        let mut lock = self.lock();
        lock.grad = None;
    }

    /// Detaches the tensor from the computation graph.
    pub fn detach(&self) -> Tensor {
        let lock = self.lock();
        Tensor::new(lock.data.clone(), false)
    }

    /// Returns whether this tensor requires gradients.
    pub fn requires_grad(&self) -> bool {
        self.lock().requires_grad
    }

    /// Sets whether this tensor requires gradients.
    pub fn set_requires_grad(&self, requires_grad: bool) {
        let mut lock = self.lock();
        lock.requires_grad = requires_grad;
    }

    /// Performs backpropagation starting from this tensor.
    ///
    /// This will compute the gradients of all tensors in the computation graph that have
    /// `requires_grad = true`.
    pub fn backward(&self) {
        // Set gradient for the output tensor if not already set (root call)
        {
            let mut self_lock = self.lock();
            if self_lock.grad.is_none() {
                self_lock.grad = Some(ArrayD::ones(self_lock.data.dim()));
            }
        } // Lock is released here

        let self_lock = self.lock();
        if let Some(creator) = &self_lock.creator {
            let output_grad = self_lock
                .grad
                .as_ref()
                .expect("Backward called but output gradient is None");
            let inputs = self_lock.inputs.clone();
            let input_grads = creator.backward(&inputs, output_grad);

            // Collect inputs that need recursive backward to avoid holding locks while recursing.
            let mut to_backward: Vec<Tensor> = Vec::new();
            for (i, input) in inputs.iter().enumerate() {
                if input.lock().requires_grad {
                    let mut input_lock = input.lock();
                    let grad_to_add = &input_grads[i];
                    if let Some(grad) = &mut input_lock.grad {
                        *grad += grad_to_add;
                    } else {
                        input_lock.grad = Some(grad_to_add.clone());
                    }
                    to_backward.push(input.clone());
                }
            }
            for input in to_backward.iter() {
                input.backward();
            }
        }
    }

    /// Builds a topological sort of the computation graph.
    #[allow(dead_code)]
    fn build_topo(
        &self,
        visited: &mut std::collections::HashSet<*const Mutex<TensorData>>,
        topo_order: &mut Vec<Tensor>,
    ) {
        let ptr = Arc::as_ptr(&self.0);
        if !visited.contains(&ptr) {
            visited.insert(ptr);
            for input in &self.lock().inputs {
                input.build_topo(visited, topo_order);
            }
            topo_order.push(self.clone());
        }
    }
}

// Implement Deref to allow treating Tensor like Arc<Mutex<TensorData>>
use std::ops::Deref;

impl Deref for Tensor {
    type Target = Arc<Mutex<TensorData>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Implement PartialEq for Tensors
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

// Implement Eq for Tensors
impl Eq for Tensor {}

// Implement Hash for Tensors
use std::hash::{Hash, Hasher};

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}
