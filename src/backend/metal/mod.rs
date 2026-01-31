use crate::tensor::Tensor;
use metal::*;
use ndarray::{ArrayD, IxDyn};

use std::mem;

mod shaders;

pub struct MetalBackend {
    device: Device,
    command_queue: CommandQueue,
    matmul_pipeline: ComputePipelineState,
}

impl MetalBackend {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("Failed to get system default Metal device")?;
        let command_queue = device.new_command_queue();

        let library = device
            .new_library_with_source(shaders::MATMUL_SHADER, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile MSL library: {}", e))?;

        let kernel = library
            .get_function("matmul_naive", None)
            .map_err(|e| format!("Failed to get matmul_naive function: {}", e))?;

        let matmul_pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create compute pipeline state: {}", e))?;

        log::info!("Initialized Metal Backend on device: {}", device.name());

        Ok(Self {
            device,
            command_queue,
            matmul_pipeline,
        })
    }
}

unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

impl crate::backend::Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Option<ArrayD<f32>> {
        let a_lock = a.lock();
        let b_lock = b.lock();
        let a_arr = a_lock.storage.to_f32_array();
        let b_arr = b_lock.storage.to_f32_array();

        if a_arr.ndim() != 2 || b_arr.ndim() != 2 {
            log::warn!("MetalBackend::matmul: Only 2D matrices supported");
            return None;
        }

        let m = a_arr.shape()[0] as u32;
        let k = a_arr.shape()[1] as u32;
        let k2 = b_arr.shape()[0] as u32;
        let n = b_arr.shape()[1] as u32;

        if k != k2 {
            log::warn!("MetalBackend::matmul: Dimension mismatch: {} != {}", k, k2);
            return None;
        }

        let a_data = a_arr
            .as_slice()
            .expect("Matrix A must be contiguous in memory");
        let b_data = b_arr
            .as_slice()
            .expect("Matrix B must be contiguous in memory");
        let c_len = (m * n) as usize;
        let c_size = (c_len * mem::size_of::<f32>()) as u64;

        let options = MTLResourceOptions::StorageModeShared;

        let buffer_a = self.device.new_buffer_with_data(
            a_data.as_ptr() as *const _,
            (a_data.len() * mem::size_of::<f32>()) as u64,
            options,
        );

        let buffer_b = self.device.new_buffer_with_data(
            b_data.as_ptr() as *const _,
            (b_data.len() * mem::size_of::<f32>()) as u64,
            options,
        );

        let buffer_c = self.device.new_buffer(c_size, options);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.matmul_pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_c), 0);

        encoder.set_bytes(3, mem::size_of::<u32>() as u64, &m as *const _ as *const _);
        encoder.set_bytes(4, mem::size_of::<u32>() as u64, &k as *const _ as *const _);
        encoder.set_bytes(5, mem::size_of::<u32>() as u64, &n as *const _ as *const _);

        let w = self.matmul_pipeline.thread_execution_width();
        let h = self.matmul_pipeline.max_total_threads_per_threadgroup() / w;

        let thread_group_size = MTLSize {
            width: w,
            height: h,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: n as u64,
            height: m as u64,
            depth: 1,
        };

        encoder.dispatch_threads(grid_size, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = buffer_c.contents() as *mut f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, c_len) };
        let vec_out = slice.to_vec();

        match ArrayD::from_shape_vec(IxDyn(&[m as usize, n as usize]), vec_out) {
            Ok(arr) => Some(arr),
            Err(e) => {
                log::error!("MetalBackend: Failed to reshape output: {}", e);
                None
            }
        }
    }

    fn matmul_quantized(
        &self,
        _input: &Tensor,
        _qweight: &Tensor,
        _scales: &Tensor,
        _qzeros: &Tensor,
        _bias: Option<&Tensor>,
        _group_size: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Option<ArrayD<f32>> {
        log::warn!("MetalBackend: Quantized matmul not implemented yet");
        None
    }
}
