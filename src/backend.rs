//! Backend abstraction placeholder.
//!
//! The goal is to abstract Rust ops behind a Backend trait so it's easier to add GPU/accelerated backends
//! (wgpu/cudarc) later. This file provides a minimal trait definition and a CPU backend stub to start
//! refactoring ops.

use crate::tensor::Tensor;
use ndarray::ArrayD;
use std::sync::Arc;
use std::sync::OnceLock;

/// Minimal backend trait to prepare for future backends. Implementations will provide
/// optimized kernels for operations like matmul, conv, etc.
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;

    // Return an optional result for matmul. If a backend returns `Some(ArrayD)` the
    // op can short-circuit and use that result; if `None` the op should fall back
    // to its built-in implementation.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Option<ArrayD<f32>>;
}

/// CPU backend stub — delegates to existing op implementations for now.
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor) -> Option<ArrayD<f32>> {
        // Default: return None to allow MatMul op to use its optimized CPU code path.
        None
    }
}

// TODO: migrate existing ops in `src/ops.rs` to backend trait implementations, and use
// `Backend` in `Tensor::apply` to call backend-specific optimized kernels.

static GLOBAL_BACKEND: OnceLock<Arc<dyn Backend>> = OnceLock::new();

/// Set the global backend for operations. Only callable once per process.
pub fn set_global_backend(b: Arc<dyn Backend>) -> Result<(), String> {
    GLOBAL_BACKEND
        .set(b)
        .map_err(|_| "Global backend already set".to_string())
}

/// Returns the global backend. Defaults to CPU backend if not set.
pub fn get_global_backend() -> &'static Arc<dyn Backend> {
    GLOBAL_BACKEND.get_or_init(|| Arc::new(CpuBackend {}))
}

/// Convenience function to set the backend to CPU explicitly.
pub fn set_cpu_backend() -> Result<(), String> {
    set_global_backend(Arc::new(CpuBackend {}))
}
