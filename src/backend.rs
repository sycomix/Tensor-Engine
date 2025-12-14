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

// NOTE: A backend trait migration is planned to allow backend-specific optimizations
// (e.g. CUDA, wgpu). See `docs/backend_migration_plan.md` for design notes and steps
// to migrate `src/ops.rs` to use the `Backend` trait and to call backend-specific
// kernels from `Tensor::apply`.

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

/// Minimal Cuda backend implementation to start Phase 2 integration.
/// For now this `CudaBackend` dispatches to a CPU-hosted matmul implementation
/// but provides a concise integration point for adding CUDA accelerated matmul
/// and other kernels in the future. It implements `Backend` and is usable
/// directly or as a global backend via `set_cuda_backend()`.
pub struct CudaBackend;

impl Backend for CudaBackend {
    fn name(&self) -> &'static str { "cuda" }
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Option<ArrayD<f32>> {
        // Start of integration: perform a CPU matmul for now, returning the result.
        // Future changes should replace this with an actual cuBLAS or CUDA kernel call
        // guarded by a runtime feature-flag and appropriate dependencies.
        let a_lock = a.lock();
        let b_lock = b.lock();
        let a_arr = a_lock.storage.to_f32_array();
        let b_arr = b_lock.storage.to_f32_array();
        // Only allow 2D matrices for this simplified path
        if a_arr.ndim() == 2 && b_arr.ndim() == 2 {
            if let (Ok(a2), Ok(b2)) = (
                a_arr.into_dimensionality::<ndarray::Ix2>(),
                b_arr.into_dimensionality::<ndarray::Ix2>(),
            ) {
                let c = a2.dot(&b2);
                return Some(c.into_dyn());
            }
        }
        None
    }
}

/// Convenience function to set the backend to a minimal CudaBackend.
pub fn set_cuda_backend() -> Result<(), String> {
    set_global_backend(Arc::new(CudaBackend {}))
}
