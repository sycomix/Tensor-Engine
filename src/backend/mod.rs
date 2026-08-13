pub mod cpu;
#[cfg(feature = "backend_wgpu")]
pub mod gpu_memory;
pub mod traits;
#[cfg(feature = "backend_wgpu")]
pub mod wgpu;

pub use self::cpu::CpuBackend;
pub use self::traits::{ActivationKind, Backend};
use std::sync::OnceLock;

static GLOBAL_BACKEND: OnceLock<Box<dyn Backend>> = OnceLock::new();

pub fn get_global_backend() -> &'static dyn Backend {
    GLOBAL_BACKEND
        .get_or_init(|| {
            // Auto-detect: try WGPU if feature enabled, otherwise fall back to CPU
            #[cfg(feature = "backend_wgpu")]
            {
                match wgpu::WgpuBackend::new() {
                    Ok(backend) => {
                        log::info!("Auto-detected WGPU backend");
                        Box::new(backend)
                    }
                    Err(e) => {
                        log::info!("WGPU unavailable ({}), using CPU backend", e);
                        Box::new(CpuBackend::default())
                    }
                }
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                log::info!("Using CPU backend (WGPU feature not enabled)");
                Box::new(CpuBackend::default())
            }
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

#[cfg(feature = "backend_wgpu")]
pub fn is_wgpu_available() -> bool {
    wgpu::WgpuBackend::new().is_ok()
}
