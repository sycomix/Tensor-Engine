pub mod cpu;
#[cfg(feature = "backend_wgpu")]
pub mod gpu_memory;
pub mod traits;
#[cfg(feature = "backend_wgpu")]
pub mod wgpu;

use self::cpu::CpuBackend;
use self::traits::Backend;
use std::sync::OnceLock;

static GLOBAL_BACKEND: OnceLock<Box<dyn Backend>> = OnceLock::new();

pub fn get_global_backend() -> &'static dyn Backend {
    GLOBAL_BACKEND.get_or_init(|| Box::new(CpuBackend::default())).as_ref()
}

pub fn set_cpu_backend() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let backend = CpuBackend::default();
    GLOBAL_BACKEND
        .set(Box::new(backend))
        .map_err(|_| "Backend already initialized".into())
}

#[cfg(feature = "backend_wgpu")]
pub fn set_wgpu_backend() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let backend = wgpu::WgpuBackend::new().map_err(|e| format!("Failed to initialize WGPU: {}", e))?;
    GLOBAL_BACKEND
        .set(Box::new(backend))
        .map_err(|_| "Backend already initialized".into())
}

#[cfg(feature = "backend_wgpu")]
pub fn is_wgpu_available() -> bool {
    wgpu::WgpuBackend::new().is_ok()
}
