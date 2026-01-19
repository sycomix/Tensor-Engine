use crate::tensor::Tensor;
use ndarray::ArrayD;
use ndarray::IxDyn;

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

impl crate::backend::Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Option<ArrayD<f32>> {
        // Naive Synchronous Implementation for Preamble Verification
        let a_lock = a.lock();
        let b_lock = b.lock();
        let a_arr = a_lock.storage.to_f32_array();
        let b_arr = b_lock.storage.to_f32_array();

        let a_shape = a_arr.shape();
        let b_shape = b_arr.shape();

        // Only support 2D for rudimentary test
        if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[1] != b_shape[0] {
            log::warn!("WgpuBackend::matmul: Only aligned 2D matrices supported for now.");
            return None;
        }

        let m = a_shape[0] as u32;
        let k = a_shape[1] as u32;
        let n = b_shape[1] as u32;

        let a_flat: Vec<f32> = a_arr.iter().cloned().collect();
        let b_flat: Vec<f32> = b_arr.iter().cloned().collect();

        // 1. Create Buffers
        let buffer_a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Buffer A"),
                contents: bytemuck::cast_slice(&a_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Transpose B is usually needed for efficient coalescing, but naive shader reads directly
        // Currently naive shader expects B in row-major as well (standard format)
        let buffer_b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Buffer B"),
                contents: bytemuck::cast_slice(&b_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (m * n) as usize * std::mem::size_of::<f32>();
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

        // 2. Shader & Pipeline
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MatMul Shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "matmul_naive.wgsl"
                ))),
            });

        // Uniform for dimensions
        let params = [m, k, n];
        let param_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MatMul Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            // A
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
                            // B
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
                            // C
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
                            // Params
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

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MatMul Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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

        // 3. Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MatMul Encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroup_size = 16;
            let dispatch_x = (n + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (m + workgroup_size - 1) / workgroup_size;
            cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        encoder.copy_buffer_to_buffer(
            &buffer_c,
            0,
            &staging_buffer,
            0,
            output_size as wgpu::BufferAddress,
        );
        self.queue.submit(Some(encoder.finish()));

        // 4. Readback
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait); // Block until done

        if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();

            // Reshape back to ArrayD
            return Some(ArrayD::from_shape_vec(IxDyn(&[m as usize, n as usize]), result).unwrap());
        }

        None
    }

    fn matmul_quantized(
        &self,
        input: &Tensor,
        qweight: &Tensor,
        scales: &Tensor,
        qzeros: &Tensor,
        bias: Option<&Tensor>,
        group_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> Option<ArrayD<f32>> {
        let input_lock = input.lock();
        let q_lock = qweight.lock(); // U8 packed
        let s_lock = scales.lock();
        let z_lock = qzeros.lock();

        let input_arr = input_lock.storage.to_f32_array();
        let input_shape = input_arr.shape();

        // Validation (simplified)
        if input_shape.len() != 2 {
            log::warn!("WgpuBackend::matmul_quantized: Only 2D input supported for now.");
            return None;
        }

        let m = input_shape[0] as u32; // Batch
        let k = input_shape[1] as u32; // In features
        let n = out_features as u32; // Out features
        let g = group_size as u32;

        if k as usize != in_features {
            log::warn!(
                "WgpuBackend::matmul_quantized: K mismatch {} != {}",
                k,
                in_features
            );
            return None;
        }

        // 1. Data Prep
        let input_flat: Vec<f32> = input_arr.iter().cloned().collect();
        // Pack qweight to U32 for alignment if needed, or just cast slice.
        // WGSL expects array<u32>.
        // Our storage is U8. We need to pad/cast to U32.
        // Simplest: copy U8 bytes to U8 vec, then padding to multiple of 4, then cast.
        let q_bytes = match &q_lock.storage {
            crate::dtype::TensorStorage::U8(arr) => arr.as_slice().unwrap(),
            _ => return None,
        };
        // Pad to 4 bytes alignment
        let mut q_padded = q_bytes.to_vec();
        while q_padded.len() % 4 != 0 {
            q_padded.push(0);
        }
        let q_u32: &[u32] = bytemuck::cast_slice(&q_padded);

        let s_arr = s_lock.storage.to_f32_array();
        let z_arr = z_lock.storage.to_f32_array();
        let s_flat: Vec<f32> = s_arr.iter().cloned().collect();
        let z_flat: Vec<f32> = z_arr.iter().cloned().collect();

        let bias_flat: Vec<f32> = match bias {
            Some(b) => {
                let b_lock = b.lock();
                b_lock.storage.to_f32_array().iter().cloned().collect()
            }
            None => vec![0.0; n as usize],
        };

        // 2. Create Buffers
        let buffer_a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("In A"),
                contents: bytemuck::cast_slice(&input_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buffer_q = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("In Q"),
                contents: bytemuck::cast_slice(q_u32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buffer_s = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("In S"),
                contents: bytemuck::cast_slice(&s_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buffer_z = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("In Z"),
                contents: bytemuck::cast_slice(&z_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buffer_b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("In B"),
                contents: bytemuck::cast_slice(&bias_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (m * n) as usize * std::mem::size_of::<f32>();
        let buffer_c = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Out C"),
            size: output_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: output_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Pipeline
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("QuantMatMul Shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "matmul_quantized.wgsl"
                ))),
            });

        let params = [m, k, n, g];
        let param_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("QuantMatMul Layout"),
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
                        }, // A
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }, // Q
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }, // S
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }, // Z
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }, // B
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }, // C
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }, // Params
                    ],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("QuantMatMul Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: "main",
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_s.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffer_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: param_buffer.as_entire_binding(),
                },
            ],
        });

        // 4. Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_size = 16u32;
            cpass.dispatch_workgroups((n + wg_size - 1) / wg_size, (m + wg_size - 1) / wg_size, 1);
        }

        encoder.copy_buffer_to_buffer(
            &buffer_c,
            0,
            &staging_buffer,
            0,
            output_size as wgpu::BufferAddress,
        );
        self.queue.submit(Some(encoder.finish()));

        // 5. Readback
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            return Some(ArrayD::from_shape_vec(IxDyn(&[m as usize, n as usize]), result).unwrap());
        }

        None
    }
}
