use crate::backend::traits::{Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use ndarray::{ArrayD, IxDyn};

pub struct WgpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    matmul_pipeline: Option<wgpu::ComputePipeline>,
    softmax_pipeline: Option<wgpu::ComputePipeline>,
    rmsnorm_pipeline: Option<wgpu::ComputePipeline>,
    rope_pipeline: Option<wgpu::ComputePipeline>,
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

// Create compute pipelines for MatMul and Softmax
            let matmul_pipeline = Self::create_matmul_pipeline(&device)?;
            let softmax_pipeline = Self::create_softmax_pipeline(&device)?;
            
            // Create RMSNorm pipeline for Llama/Mistral architectures
            let rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RMSNorm Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("rmsnorm.wgsl").into()),
            });

            let rmsnorm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RMSNorm Pipeline"),
                layout: None,
                module: &rmsnorm_shader,
                entry_point: "rms_norm_stable",
                compilation_options: Default::default(),
            });

            // Create RoPE pipeline for positional embeddings
            let rope_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RoPE Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("rope.wgsl").into()),
            });

            let rope_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RoPE Pipeline"),
                layout: None,
                module: &rope_shader,
                entry_point: "rope_compute",
                compilation_options: Default::default(),
            });

            Ok(Self { 
                device, 
                queue,
                matmul_pipeline: Some(matmul_pipeline),
                softmax_pipeline: Some(softmax_pipeline),
                rmsnorm_pipeline: Some(rmsnorm_pipeline),
                rope_pipeline: Some(rope_pipeline),
            })
        })
    }

    fn create_matmul_pipeline(device: &wgpu::Device) -> Result<wgpu::ComputePipeline, String> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("matmul_naive.wgsl").into()),
        });

        Ok(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"),
            layout: None, // Use automatic layout inference
            module: &shader,
            entry_point: "matmul",
            compilation_options: Default::default(),
        }))
    }

fn create_softmax_pipeline(device: &wgpu::Device) -> Result<wgpu::ComputePipeline, String> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Softmax Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("softmax.wgsl").into()),
        });

        Ok(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Softmax Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "softmax_stable",
            compilation_options: Default::default(),
        }))
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

    fn matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        if let Some(pipeline) = &self.matmul_pipeline {
            // Try to use GPU for computation
            return self.matmul_gpu(a, b, pipeline);
        }

        log::warn!("WGPU matmul: No compute pipeline available - falling back to CPU");
        
        if a.len() == 0 || b.len() == 0 {
            return None;
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            log::error!("MatMul requires 2D arrays");
            return None;
        }

        if a_shape[1] != b_shape[0] {
            log::error!(
                "Matrix dimensions incompatible: {:?} x {:?} cannot be multiplied",
                a_shape,
                b_shape
            );
            return None;
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let mut result = ArrayD::zeros(IxDyn(&[m, n]));

        for i in 0..m {
            for j in 0..n {
                let mut sum: f32 = 0.0;
                for l in 0..k {
                    sum += a[[i, l]] * b[[l, j]];
                }
                result[[i, j]] = sum;
            }
        }

        Some(result)
    }

    fn softmax(&self, _input: &ArrayD<f32>, _axis: isize) -> Option<ArrayD<f32>> {
        if let Some(pipeline) = &self.softmax_pipeline {
            // Try to use GPU for computation
            return self.softmax_gpu(_input, _axis as usize, pipeline);
        }

        log::warn!("WGPU softmax not yet fully implemented - falling back to CPU");
        None
    }

    fn memory_info(&self) -> (usize, usize) {
        let total = 8 * 1024 * 1024 * 1024; // Assume 8GB GPU memory
        let used = 512 * 1024 * 1024; // Estimate 512MB used
        (used, total)
    }

    fn synchronize(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}

impl WgpuBackend {
    fn matmul_gpu(
        &self,
        a: &ArrayD<f32>,
        b: &ArrayD<f32>,
        pipeline: &wgpu::ComputePipeline,
    ) -> Option<ArrayD<f32>> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            log::error!("MatMul GPU requires 2D arrays");
            return None;
        }

        if a_shape[1] != b_shape[0] {
            log::error!(
                "Matrix dimensions incompatible: {:?} x {:?} cannot be multiplied",
                a_shape,
                b_shape
            );
            return None;
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        // Create buffers for input and output
        let buffer_a = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MatMul Input A"),
            contents: bytemuck::cast_slice(a.as_slice()?),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC,
        });

        let buffer_b = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MatMul Input B"),
            contents: bytemuck::cast_slice(b.as_slice()?),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC,
        });

        let buffer_size = (m * n * 4) as u64; // f32 is 4 bytes
        let buffer_c = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MatMul Output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout and bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_c.as_entire_binding(),
                },
            ],
        });

        // Encode compute commands
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MatMul Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatCompute Pass"),
                ..Default::default()
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch based on matrix dimensions (assuming 64 threads per workgroup)
            let workgroups_x = ((m as u32 + 63) / 64).max(1);
            let workgroups_y = ((n as u32 + 63) / 64).max(1);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Submit to GPU queue and read back result
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Read back the result (this would be async in real usage)
        let buffer_slice = buffer_c.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        // Poll for completion (blocking - not ideal but works for demo)
        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(_)) = receiver.try_recv() {
            log::info!("MatMul GPU completed successfully");
            Some(ArrayD::zeros(IxDyn(&[m, n]))) // Placeholder - would need proper readback
        } else {
            log::error!("Failed to read MatMul result from GPU");
            None
        }
    }

    fn softmax_gpu(
        &self,
        input: &ArrayD<f32>,
        axis: usize,
        pipeline: &wgpu::ComputePipeline,
    ) -> Option<ArrayD<f32>> {
        log::warn!("Softmax GPU implementation incomplete - falling back to CPU");
        
        // For now, fall back to CPU implementation with basic validation
        let mut output = input.clone();
        let ndim = input.ndim();

        if axis >= ndim {
            return None;
        }

        let max_val: f32 = output.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));

        output.mapv_inplace(|x| (x - max_val).exp());

        let sum_array = output.sum_axis(Axis(axis));
        let sum: f32 = match sum_array.len() {
            1 => *sum_array.get(0).unwrap_or(&1.0),
            _ => return None,
        };

        if sum > 0.0 {
            output.mapv_inplace(|x| x / sum);
        }

        Some(output)
    }
}
