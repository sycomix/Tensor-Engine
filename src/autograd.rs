use crate::tensor::Tensor;
use ndarray::ArrayD;
use std::collections::HashSet;

/// The `AutogradEngine` is responsible for orchestrating the backward pass.
///
/// This is the main entry point for performing backpropagation.
pub struct AutogradEngine;

impl AutogradEngine {
    /// Creates a new `AutogradEngine`.
    pub fn new() -> Self {
        AutogradEngine
    }

    /// Starts the backpropagation process from a given tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to start the backward pass from.
    pub fn backward(&self, tensor: &Tensor) {
        // 1. Initialize gradient of the starting tensor to 1.0 if not already set.
        {
            let mut lock = tensor.lock();
            if lock.grad.is_none() {
                let shape = lock.storage.shape();
                lock.grad = Some(ArrayD::ones(ndarray::IxDyn(&shape)));
            }
        }

        // 2. Build topological order
        let mut visited = HashSet::new();
        let mut topo_order = Vec::new();
        tensor.build_topo(&mut visited, &mut topo_order);

        // 3. Process nodes in reverse topological order (children before parents)
        for t in topo_order.into_iter().rev() {
            let (creator, grad, inputs) = {
                let lock = t.lock();
                (lock.creator.clone(), lock.grad.clone(), lock.inputs.clone())
            };

            // If this tensor has a creator operation and gradient to propagate.
            if let (Some(op), Some(out_grad)) = (creator, grad) {
                let input_grads = op.backward(&inputs, &out_grad);
                for (i, input) in inputs.iter().enumerate() {
                    if input.requires_grad() {
                        let mut input_lock = input.lock();
                        let grad_to_add = &input_grads[i];
                        if let Some(existing_grad) = &mut input_lock.grad {
                            *existing_grad += grad_to_add;
                        } else {
                            input_lock.grad = Some(grad_to_add.clone());
                        }
                    }
                }
            }
        }
    }
}

impl Default for AutogradEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Applies gradient clipping to a list of parameters.
///
/// Clips gradients to a maximum L2 norm. If the total L2 norm of all gradients
/// exceeds `max_norm`, all gradients are scaled down proportionally.
///
/// # Arguments
///
/// * `params` - The parameters whose gradients should be clipped.
/// * `max_norm` - The maximum allowed L2 norm of the gradients.
///
/// # Returns
///
/// The actual total L2 norm of the gradients before clipping.
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32) -> f32 {
    // Compute total L2 norm of all gradients
    let mut total_norm = 0.0f32;
    for param in params {
        let lock = param.lock();
        if let Some(grad) = &lock.grad {
            // Compute L2 norm of this gradient tensor
            let mut local_norm = 0.0f32;
            for &val in grad.iter() {
                local_norm += val * val;
            }
            total_norm += local_norm;
        }
    }
    total_norm = total_norm.sqrt();

    // If total norm exceeds max_norm, scale all gradients
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for param in params {
            let mut lock = param.lock();
            if let Some(ref mut grad) = lock.grad {
                grad.mapv_inplace(|g| g * clip_coef);
            }
        }
    }

    total_norm
}

/// Applies gradient clipping to a list of parameters by value.
///
/// Clips each gradient tensor element-wise to the range `[-clip_value, clip_value]`.
///
/// # Arguments
///
/// * `params` - The parameters whose gradients should be clipped.
/// * `clip_value` - The maximum absolute value for each gradient element.
pub fn clip_grad_value(params: &[Tensor], clip_value: f32) {
    for param in params {
        let mut lock = param.lock();
        if let Some(ref mut grad) = lock.grad {
            grad.mapv_inplace(|g| g.clamp(-clip_value, clip_value));
        }
    }
}

/// Applies gradient checkpointing to a function `f` with given `inputs`.
///
/// This wraps the function execution in a `Checkpoint` operation, which
/// forgets intermediate results during the forward pass and recomputes them
/// during the backward pass to save memory.
///
/// # Arguments
///
/// * `f` - The closure defining the subgraph to checkpoint.
/// * `inputs` - The input tensors to the closure.
pub fn checkpoint<F>(f: F, inputs: &[Tensor]) -> Tensor
where
    F: Fn(&[Tensor]) -> Tensor + Send + Sync + 'static,
{
    use std::sync::Arc;
    Tensor::apply(Arc::new(crate::ops::Checkpoint::new(f)), inputs)
}
