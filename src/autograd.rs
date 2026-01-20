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

            // If there's an operation that created this tensor and we have a gradient to propagate...
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
