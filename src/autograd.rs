use crate::tensor::Tensor;

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
        tensor.backward();
    }
}

impl Default for AutogradEngine {
    fn default() -> Self {
        Self::new()
    }
}
