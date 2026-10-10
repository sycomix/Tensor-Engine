//! Backend abstraction layer for Tensor Engine.
//!
//! This module provides a unified interface for different compute backends:
//! - CPU (default, always available)
//! - CUDA (GPU acceleration, Linux-only)
//! - WGPU (WebGPU, cross-platform)

pub mod cpu;
#[cfg(feature = "backend_cuda")]
pub mod cuda;
#[cfg(feature = "backend_cuda")]
pub(crate) mod cuda_kernels;
#[cfg(feature = "backend_wgpu")]
pub mod gpu_memory;
pub mod traits;
#[cfg(feature = "backend_wgpu")]
pub mod wgpu;

pub use self::cpu::CpuBackend;
pub use self::traits::{ActivationKind, Backend};
use std::sync::OnceLock;

static GLOBAL_BACKEND: OnceLock<Box<dyn Backend>> = OnceLock::new();

/// Initialize the global backend. Priority: CUDA > WGPU > CPU.
pub fn get_global_backend() -> &'static dyn Backend {
    GLOBAL_BACKEND
        .get_or_init(|| {
            // 1. Try CUDA first (highest priority on Linux)
            #[cfg(feature = "backend_cuda")]
            {
                if let Ok(backend) = cuda::CudaBackend::new_default() {
                    log::info!("Auto-detected CUDA backend");
                    return Box::new(backend);
                }
                log::info!("CUDA unavailable, trying next backend");
            }

            // 2. Try WGPU if enabled
            #[cfg(feature = "backend_wgpu")]
            {
                match wgpu::WgpuBackend::new() {
                    Ok(backend) => {
                        log::info!("Auto-detected WGPU backend");
                        return Box::new(backend);
                    }
                    Err(e) => {
                        log::info!("WGPU unavailable ({}), falling back to CPU", e);
                    }
                }
            }

            // 3. Fall back to CPU
            log::info!("Using CPU backend");
            Box::new(CpuBackend::default())
        })
        .as_ref()
}

pub fn set_cpu_backend() -> Result<(), crate::error::TensorError> {
    let backend = CpuBackend::default();
    GLOBAL_BACKEND
        .set(Box::new(backend))
        .map_err(|_| crate::error::TensorError::Generic {
            message: "Backend already initialized".to_string(),
        })
}

/// Set CUDA as the global backend on a specific device.
pub fn set_cuda_backend(device: usize) -> Result<(), crate::error::TensorError> {
    #[cfg(feature = "backend_cuda")]
    {
        let backend = cuda::CudaBackend::new(device).map_err(|e| crate::error::TensorError::BackendError {
            backend_name: "cuda".to_string(),
            message: e.to_string(),
        })?;
        GLOBAL_BACKEND
            .set(Box::new(backend))
            .map_err(|_| crate::error::TensorError::Generic {
                message: "Backend already initialized".to_string(),
            })
    }
    #[cfg(not(feature = "backend_cuda"))]
    Err(crate::error::TensorError::BackendError {
        backend_name: "cuda".to_string(),
        message: "CUDA feature not enabled".to_string(),
    })
}

#[cfg(feature = "backend_wgpu")]
pub fn set_wgpu_backend() -> Result<(), crate::error::TensorError> {
    let backend =
        wgpu::WgpuBackend::new().map_err(|e| crate::error::TensorError::BackendError {
            backend_name: "wgpu".to_string(),
            message: e.to_string(),
        })?;
    GLOBAL_BACKEND
        .set(Box::new(backend))
        .map_err(|_| crate::error::TensorError::Generic {
            message: "Backend already initialized".to_string(),
        })
}

/// Check if CUDA is available on this system.
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "backend_cuda")]
    {
        cuda_kernels::is_cuda_available()
    }
    #[cfg(not(feature = "backend_cuda"))]
    false
}

/// Check if WGPU is available on this system.
#[cfg(feature = "backend_wgpu")]
pub fn is_wgpu_available() -> bool {
    wgpu::WgpuBackend::new().is_ok()
}

/// Get the number of CUDA devices available.
#[cfg(feature = "backend_cuda")]
pub fn cuda_device_count() -> usize {
    use cudarc::driver::CudaContext;
    
    let mut count = 0;
    for i in 0..16 { // Check up to 16 possible GPUs
        if CudaContext::new(i).is_ok() {
            count += 1;
        } else {
            break;
        }
    }
    count
}

#[cfg(not(feature = "backend_cuda"))]
pub fn cuda_device_count() -> usize {
    0
}
