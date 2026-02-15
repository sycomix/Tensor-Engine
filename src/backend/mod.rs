pub mod cpu;
pub mod traits;
// pub mod wgpu; // later

use self::cpu::CpuBackend;
use self::traits::Backend;
use std::sync::OnceLock;

static GLOBAL_BACKEND: OnceLock<Box<dyn Backend>> = OnceLock::new();

pub fn get_global_backend() -> &'static dyn Backend {
    GLOBAL_BACKEND.get_or_init(|| Box::new(CpuBackend)).as_ref()
}
