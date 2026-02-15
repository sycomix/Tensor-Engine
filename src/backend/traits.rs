use crate::dtype::DType;
use ndarray::ArrayD;
use std::any::Any;

/// Represents a storage backend (CPU, WGPU, etc.)
pub trait Backend: Send + Sync + 'static {
    fn name(&self) -> &str;

    // Factory methods
    fn create_from_data(&self, data: ArrayD<f32>, dtype: DType) -> Box<dyn Storage>;
    fn create_zeros(&self, shape: &[usize]) -> Box<dyn Storage>;
    fn create_ones(&self, shape: &[usize]) -> Box<dyn Storage>;

    // Device-specific operations would go here or in a separate Ops trait
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
}
