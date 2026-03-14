use crate::dtype::DType;
use ndarray::{ArrayD, IxDyn};
use std::any::Any;

/// Represents a storage backend (CPU, WGPU, etc.)
pub trait Backend: Send + Sync + 'static {
    fn name(&self) -> &str;

    // Factory methods
    fn create_from_data(&self, data: ArrayD<f32>, dtype: DType) -> Box<dyn Storage>;
    fn create_zeros(&self, shape: &[usize]) -> Box<dyn Storage>;
    fn create_ones(&self, shape: &[usize]) -> Box<dyn Storage>;

    // Core tensor operations - backends can implement optimized versions
    fn matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        // Default to CPU implementation if not overridden
        None
    }

    fn conv2d(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        bias: Option<&ArrayD<f32>>,
        stride: usize,
        padding: usize,
    ) -> Option<ArrayD<f32>> {
        None
    }

    fn softmax(&self, input: &ArrayD<f32>, axis: isize) -> Option<ArrayD<f32>> {
        None
    }

    fn layer_norm(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        bias: &ArrayD<f32>,
        eps: f32,
        axis: isize,
    ) -> Option<ArrayD<f32>> {
        None
    }

    // Llama/Mistral specific operations
    fn rms_norm(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        eps: f32,
        axis: isize,
    ) -> Option<ArrayD<f32>> {
        None
    }

    fn rope(
        &self,
        x: &ArrayD<f32>,
        freqs: &ArrayD<f32>,
        seq_len: usize,
        head_dim: usize,
    ) -> Option<ArrayD<f32>> {
        None
    }

    // Device management
    fn synchronize(&self) {
        // Default no-op for CPU
    }

    fn memory_info(&self) -> (usize, usize); // (used_bytes, total_bytes)
}

/// Abstract storage for tensor data
pub trait Storage: Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DType;

    // For now, we might need a way to get data back to CPU for debugging/interop
    fn to_cpu(&self) -> ArrayD<f32>;

    fn box_clone(&self) -> Box<dyn Storage>;

    // Optional Helpers for efficiency
    fn to_f32_array(&self) -> ArrayD<f32> {
        self.to_cpu()
    }

    fn as_u8_array(&self) -> Option<&ArrayD<u8>> {
        None
    }

    fn element_count(&self) -> usize {
        self.shape().iter().product()
    }
}
