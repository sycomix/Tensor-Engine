use crate::backend::traits::{Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use ndarray::{ArrayD, IxDyn};
use wgpu::util::DeviceExt;

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

    fn matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        // Only support 2D matrices for now
        if a.ndim() != 2 || b.ndim() != 2 || a.shape()[1] != b.shape()[0] {
            log::warn!("WgpuBackend::matmul: Only aligned 2D matrices supported");
            return None;
        }

        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];

        let a_flat: Vec<f32> = a.iter().cloned().collect();
        let b_flat: Vec<f32> = b.iter().cloned().collect();

        // Create buffers using wgpu::util::DeviceExt trait
        use wgpu::util::DeviceExt;

        let buffer_a = self.device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: Some("Buffer A"),
            contents: bytemuck::cast_slice(&a_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let buffer_b = self.device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: Some("Buffer B"),
            contents: bytemuck::cast_slice(&b_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

let output_size = (m * n) * std::mem::size_of::<f32>();
        let buffer_c = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer C"),
            size: output_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shader & Pipeline
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "matmul_naive.wgsl"
            ))),
        });

        // Uniform for dimensions
        let params = [m as u32, k as u32, n as u32];
        let param_buffer = self.device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MatMul Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: param_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MatMul Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_size = 16u32;
            cpass.dispatch_workgroups((n as u32 + wg_size - 1) / wg_size, (m as u32 + wg_size - 1) / wg_size, 1);
        }

        encoder.copy_buffer_to_buffer(
            &buffer_c,
            0,
            &staging_buffer,
            0,
            output_size as wgpu::BufferAddress,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Readback
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            return Some(ArrayD::from_shape_vec(IxDyn(&[m, n]), result).unwrap());
        }

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
