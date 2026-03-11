use crate::backend::traits::{Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use ndarray::{ArrayD, IxDyn};

pub struct WgpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuBackend {
    pub fn new() -> Result<Self, String> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .ok_or("Failed to find an appropriate adapter".to_string())?;

            log::info!("WGPU Adapter: {:?}", adapter.get_info());

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("TensorEngine WGPU Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await
                .map_err(|e| format!("Failed to create device: {}", e))?;

            Ok(Self { device, queue })
        })
    }
}

impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn create_from_data(&self, data: ArrayD<f32>, dtype: DType) -> Box<dyn Storage> {
        if dtype == DType::F32 {
            Box::new(TensorStorage::F32(data))
        } else {
            Box::new(TensorStorage::from_f32_array(&data, dtype))
        }
    }

    fn create_zeros(&self, shape: &[usize]) -> Box<dyn Storage> {
        let shape_ix = IxDyn(shape);
        let data = ArrayD::zeros(shape_ix);
        Box::new(TensorStorage::F32(data))
    }

    fn create_ones(&self, shape: &[usize]) -> Box<dyn Storage> {
        let shape_ix = IxDyn(shape);
        let data = ArrayD::from_elem(shape_ix, 1.0);
        Box::new(TensorStorage::F32(data))
    }

    fn matmul(&self, _a: &ArrayD<f32>, _b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        // TODO: Implement GPU-accelerated matrix multiplication using wgpu shaders
        log::warn!("WGPU matmul not yet implemented - falling back to CPU");
        None
    }

    fn softmax(&self, _input: &ArrayD<f32>, _axis: isize) -> Option<ArrayD<f32>> {
        // TODO: Implement GPU-accelerated softmax
        None
    }

    fn memory_info(&self) -> (usize, usize) {
        // Get GPU memory info - approximate for now
        let total = 8 * 1024 * 1024 * 1024; // Assume 8GB GPU
        let used = 512 * 1024 * 1024; // Estimate 512MB used
        (used, total)
    }

    fn synchronize(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}
