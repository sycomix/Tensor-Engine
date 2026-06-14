use crate::backend::traits::{Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use ndarray::{ArrayD, Axis, Dimension, IxDyn};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

const MATMUL_SHADER: &str = r#"
struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: MatMulParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    if (row >= params.m || col >= params.n) {
        return;
    }

    var sum = 0.0;
    for (var idx = 0u; idx < params.k; idx = idx + 1u) {
        sum = sum + a[row * params.k + idx] * b[idx * params.n + col];
    }
    c[row * params.n + col] = sum;
}
"#;

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

            log::info!("WGPU Backend initialized successfully");

            Ok(Self { device, queue })
        })
    }

    /// Get a reference to the WGPU device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a reference to the WGPU queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn matmul_2d_gpu(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>, String> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(format!(
                "WGPU matmul expects 2D tensors, got {:?} and {:?}",
                a_shape, b_shape
            ));
        }
        if a_shape[1] != b_shape[0] {
            return Err(format!(
                "WGPU matmul shape mismatch: {:?} cannot multiply {:?}",
                a_shape, b_shape
            ));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        if m == 0 || k == 0 || n == 0 {
            return Err("WGPU matmul does not accept zero-sized dimensions".to_string());
        }

        let a_standard = a.as_standard_layout().into_owned();
        let b_standard = b.as_standard_layout().into_owned();
        let a_slice = a_standard
            .as_slice()
            .ok_or_else(|| "WGPU matmul could not create contiguous lhs buffer".to_string())?;
        let b_slice = b_standard
            .as_slice()
            .ok_or_else(|| "WGPU matmul could not create contiguous rhs buffer".to_string())?;

        let output_len = m
            .checked_mul(n)
            .ok_or_else(|| "WGPU matmul output element count overflowed".to_string())?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "WGPU matmul output byte count overflowed".to_string())?;

        let a_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU MatMul A"),
                contents: bytemuck::cast_slice(a_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU MatMul B"),
                contents: bytemuck::cast_slice(b_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU MatMul C"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU MatMul Readback"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = MatMulParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            _pad: 0,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU MatMul Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("TensorEngine WGPU MatMul Shader"),
                source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
            });
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("TensorEngine WGPU MatMul BindGroupLayout"),
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
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("TensorEngine WGPU MatMul PipelineLayout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TensorEngine WGPU MatMul Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TensorEngine WGPU MatMul BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TensorEngine WGPU MatMul Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TensorEngine WGPU MatMul Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((m as u32) + 15) / 16, ((n as u32) + 15) / 16, 1);
        }
        encoder.copy_buffer_to_buffer(&c_buffer, 0, &readback_buffer, 0, output_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|_| "WGPU matmul readback channel closed".to_string())?
            .map_err(|e| format!("WGPU matmul readback failed: {}", e))?;

        let mapped = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        readback_buffer.unmap();

        ArrayD::from_shape_vec(IxDyn(&[m, n]), result)
            .map_err(|e| format!("WGPU matmul result shape failed: {}", e))
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
        match self.matmul_2d_gpu(a, b) {
            Ok(result) => Some(result),
            Err(err) => {
                log::warn!("WGPU matmul unavailable: {}", err);
                None
            }
        }
    }

    fn softmax(&self, input: &ArrayD<f32>, axis: isize) -> Option<ArrayD<f32>> {
        // For now, fall back to CPU implementation
        log::debug!("WGPU softmax: falling back to CPU");

        let mut output = input.clone();
        let ndim = input.ndim();

        if axis < 0 || (axis as usize) >= ndim {
            log::error!(
                "Softmax: Invalid axis {} for tensor with {} dimensions",
                axis,
                ndim
            );
            return None;
        }

        let norm_axis = axis as usize;

        // Compute max for numerical stability along the normalization axis
        let max_val: f32 = output
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

        // Subtract max and compute exp (numerically stable)
        output.mapv_inplace(|x| (x - max_val).exp());

        // Sum along axis for normalization
        let sum_array = output.sum_axis(Axis(norm_axis));

        // Handle edge case where sum might be zero or NaN
        if sum_array.iter().any(|&x| x.is_nan() || x <= 0.0) {
            log::warn!("Softmax: Invalid sum detected, returning zeros");
            return Some(ArrayD::zeros(input.shape().to_vec()));
        }

        // Convert to scalar f32 and normalize
        let sum: f32 = match sum_array.len() {
            1 => *sum_array.get(0).unwrap_or(&1.0),
            _ => return None, // Should not happen for valid input
        };

        // Divide by sum for each element along the axis
        if sum > 1e-8 {
            output.mapv_inplace(|x| x / sum);
        } else {
            log::warn!("Softmax: Near-zero denominator detected");
        }

        Some(output)
    }

    fn rms_norm(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        eps: f32,
        axis: isize,
    ) -> Option<ArrayD<f32>> {
        // For now, fall back to CPU implementation
        log::debug!("WGPU rms_norm: falling back to CPU");

        let mut output = input.clone();
        let ndim = input.ndim();

        if axis < 0 || (axis as usize) >= ndim {
            log::error!(
                "RMSNorm: Invalid axis {} for tensor with {} dimensions",
                axis,
                ndim
            );
            return None;
        }

        let norm_axis = axis as usize;

        // Compute RMS along the normalization axis with numerical stability
        let sum_sq = output.sum_axis(Axis(norm_axis));

        // Convert to f32 and compute RMS with epsilon for stability
        let rms_values: Vec<f32> = sum_sq
            .iter()
            .map(|&x| ((x / weight.len() as f32) + eps).sqrt())
            .collect();

        // Normalize by dividing input by RMS - sequential iteration to avoid borrow issues
        for (idx, val) in output.indexed_iter_mut() {
            let pos = idx.slice()[norm_axis];
            if pos < rms_values.len() {
                let norm_val = rms_values[pos as usize];
                if norm_val > 1e-8 {
                    *val /= norm_val;
                } else {
                    log::warn!("RMSNorm: Near-zero RMS value detected");
                }
            }
        }

        // Apply weight scaling - sequential iteration to avoid borrow issues
        for (idx, val) in output.indexed_iter_mut() {
            let pos = idx.slice()[norm_axis];
            if pos < weight.len() {
                *val *= weight[pos as usize];
            }
        }

        Some(output)
    }

    fn rope(
        &self,
        x: &ArrayD<f32>,
        freqs: &ArrayD<f32>,
        seq_len: usize,
        head_dim: usize,
    ) -> Option<ArrayD<f32>> {
        // For now, fall back to CPU implementation
        log::debug!("WGPU rope: falling back to CPU");

        if x.ndim() < 3 || x.shape()[1] != seq_len || x.shape()[2] != head_dim {
            log::error!(
                "RoPE: Invalid input shape {:?} for seq_len={} and head_dim={}",
                x.shape(),
                seq_len,
                head_dim
            );
            return None;
        }

        let mut output = x.clone();

        for batch in 0..output.shape()[0] {
            for pos in 0..output.shape()[1] {
                for i in (0..output.shape()[2]).step_by(2) {
                    if i + 1 >= output.shape()[2] {
                        break;
                    }

                    let freq_idx = i / 2;
                    if freq_idx >= freqs.len() {
                        continue;
                    }

                    let theta = freqs[freq_idx];

                    let x_i = output[[batch, pos, i]];
                    let x_i1 = output[[batch, pos, i + 1]];

                    let cos_theta = (pos as f32 * theta).cos();
                    let sin_theta = (pos as f32 * theta).sin();

                    output[[batch, pos, i]] = x_i * cos_theta - x_i1 * sin_theta;
                    output[[batch, pos, i + 1]] = x_i * sin_theta + x_i1 * cos_theta;
                }
            }
        }

        Some(output)
    }

    fn memory_info(&self) -> (usize, usize) {
        // WGPU backend - estimate based on GPU memory
        let total = 4 * 1024 * 1024 * 1024; // Assume 4GB GPU memory
        let used = 512 * 1024 * 1024; // Estimate 512MB used
        (used, total)
    }

    fn synchronize(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}
