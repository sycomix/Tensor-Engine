use crate::backend::traits::{ActivationKind, Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use ndarray::{ArrayD, IxDyn};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchedMatMulParams {
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsNormParams {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerNormParams {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UnaryActivationParams {
    len: u32,
    kind: u32,
    _pad0: u32,
    _pad1: u32,
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

const BATCHED_MATMUL_SHADER: &str = r#"
struct BatchedMatMulParams {
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: BatchedMatMulParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    let batch_idx = id.z;
    if (batch_idx >= params.batch || row >= params.m || col >= params.n) {
        return;
    }

    let a_batch_offset = batch_idx * params.m * params.k;
    let b_batch_offset = batch_idx * params.k * params.n;
    let c_batch_offset = batch_idx * params.m * params.n;

    var sum = 0.0;
    for (var idx = 0u; idx < params.k; idx = idx + 1u) {
        let a_index = a_batch_offset + row * params.k + idx;
        let b_index = b_batch_offset + idx * params.n + col;
        sum = sum + a[a_index] * b[b_index];
    }
    c[c_batch_offset + row * params.n + col] = sum;
}
"#;

const SOFTMAX_SHADER: &str = r#"
struct SoftmaxParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SoftmaxParams;

var<workgroup> partial_max: array<f32, 256>;
var<workgroup> partial_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = workgroup_id.x;
    let lane = local_id.x;
    if (row >= params.rows) {
        return;
    }

    let row_offset = row * params.cols;
    var row_max = -3.4028234663852886e38;
    var col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        row_max = max(row_max, input[row_offset + col]);
        col = col + 256u;
    }
    partial_max[lane] = row_max;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            partial_max[lane] = max(partial_max[lane], partial_max[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let max_value = partial_max[0];

    var row_sum = 0.0;
    col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let value = exp(input[row_offset + col] - max_value);
        row_sum = row_sum + value;
        col = col + 256u;
    }
    partial_sum[lane] = row_sum;
    workgroupBarrier();

    stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            partial_sum[lane] = partial_sum[lane] + partial_sum[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let sum_value = partial_sum[0];
    let use_uniform = !(sum_value > 0.0);

    col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let index = row_offset + col;
        if (use_uniform) {
            output[index] = 1.0 / f32(params.cols);
        } else {
            output[index] = exp(input[index] - max_value) / sum_value;
        }
        col = col + 256u;
    }
}
"#;

const RMS_NORM_SHADER: &str = r#"
struct RmsNormParams {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: RmsNormParams;

var<workgroup> partial_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = workgroup_id.x;
    let lane = local_id.x;
    if (row >= params.rows) {
        return;
    }

    let row_offset = row * params.cols;
    var local_sum = 0.0;
    var col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let value = input[row_offset + col];
        local_sum = local_sum + value * value;
        col = col + 256u;
    }
    partial_sum[lane] = local_sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            partial_sum[lane] = partial_sum[lane] + partial_sum[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let mean_square = partial_sum[0] / f32(params.cols);
    let denom = sqrt(mean_square + params.eps);
    col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let index = row_offset + col;
        output[index] = (input[index] / denom) * weight[col];
        col = col + 256u;
    }
}
"#;

const LAYER_NORM_SHADER: &str = r#"
struct LayerNormParams {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: LayerNormParams;

var<workgroup> partial_sum: array<f32, 256>;
var<workgroup> partial_var: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = workgroup_id.x;
    let lane = local_id.x;
    if (row >= params.rows) {
        return;
    }

    let row_offset = row * params.cols;
    var local_sum = 0.0;
    var col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        local_sum = local_sum + input[row_offset + col];
        col = col + 256u;
    }
    partial_sum[lane] = local_sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            partial_sum[lane] = partial_sum[lane] + partial_sum[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean = partial_sum[0] / f32(params.cols);

    var local_var = 0.0;
    col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let centered = input[row_offset + col] - mean;
        local_var = local_var + centered * centered;
        col = col + 256u;
    }
    partial_var[lane] = local_var;
    workgroupBarrier();

    stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            partial_var[lane] = partial_var[lane] + partial_var[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let variance = partial_var[0] / f32(params.cols);
    let inv_std = inverseSqrt(variance + params.eps);
    col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let index = row_offset + col;
        output[index] = ((input[index] - mean) * inv_std) * weight[col] + bias[col];
        col = col + 256u;
    }
}
"#;

const UNARY_ACTIVATION_SHADER: &str = r#"
struct UnaryActivationParams {
    len: u32,
    kind: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: UnaryActivationParams;

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

fn gelu(value: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845834732056;
    let cubic = value * value * value;
    let inner = sqrt_2_over_pi * (value + 0.044715 * cubic);
    return 0.5 * value * (1.0 + tanh(inner));
}

fn activate(value: f32, kind: u32) -> f32 {
    if (kind == 0u) {
        return max(value, 0.0);
    }
    if (kind == 1u) {
        return sigmoid(value);
    }
    if (kind == 2u) {
        return tanh(value);
    }
    if (kind == 3u) {
        return gelu(value);
    }
    if (kind == 4u) {
        return value * sigmoid(value);
    }
    return value;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.len) {
        return;
    }
    output[index] = activate(input[index], params.kind);
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

    fn checked_u32_dim(name: &str, value: usize) -> Result<u32, String> {
        u32::try_from(value)
            .map_err(|_| format!("WGPU matmul dimension {}={} exceeds u32::MAX", name, value))
    }

    fn activation_kind_id(kind: ActivationKind) -> u32 {
        match kind {
            ActivationKind::Relu => 0,
            ActivationKind::Sigmoid => 1,
            ActivationKind::Tanh => 2,
            ActivationKind::Gelu => 3,
            ActivationKind::Silu => 4,
        }
    }

    fn unary_activation_gpu(
        &self,
        input: &ArrayD<f32>,
        kind: ActivationKind,
    ) -> Result<ArrayD<f32>, String> {
        let len = input.len();
        if len == 0 {
            return Err("WGPU unary activation does not accept empty tensors".to_string());
        }
        let input_standard = input.as_standard_layout().into_owned();
        let input_slice = input_standard.as_slice().ok_or_else(|| {
            "WGPU unary activation could not create contiguous input buffer".to_string()
        })?;
        let output_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "WGPU unary activation output byte count overflowed".to_string())?;

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU UnaryActivation Input"),
                contents: bytemuck::cast_slice(input_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU UnaryActivation Output"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU UnaryActivation Readback"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = UnaryActivationParams {
            len: Self::checked_u32_dim("len", len)?,
            kind: Self::activation_kind_id(kind),
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU UnaryActivation Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("TensorEngine WGPU UnaryActivation Shader"),
                source: wgpu::ShaderSource::Wgsl(UNARY_ACTIVATION_SHADER.into()),
            });
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("TensorEngine WGPU UnaryActivation BindGroupLayout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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
                label: Some("TensorEngine WGPU UnaryActivation PipelineLayout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TensorEngine WGPU UnaryActivation Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TensorEngine WGPU UnaryActivation BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TensorEngine WGPU UnaryActivation Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TensorEngine WGPU UnaryActivation Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((params.len + 255) / 256, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if sender.send(result).is_err() {
                log::warn!("WGPU unary activation readback receiver dropped before map completion");
            }
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|_| "WGPU unary activation readback channel closed".to_string())?
            .map_err(|e| format!("WGPU unary activation readback failed: {}", e))?;

        let mapped = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        readback_buffer.unmap();

        ArrayD::from_shape_vec(IxDyn(input.shape()), result)
            .map_err(|e| format!("WGPU unary activation result shape failed: {}", e))
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
            m: Self::checked_u32_dim("m", m)?,
            n: Self::checked_u32_dim("n", n)?,
            k: Self::checked_u32_dim("k", k)?,
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
            pass.dispatch_workgroups((params.m + 15) / 16, (params.n + 15) / 16, 1);
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

    fn matmul_3d_gpu(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>, String> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 3 || b_shape.len() != 3 {
            return Err(format!(
                "WGPU batched matmul expects 3D tensors, got {:?} and {:?}",
                a_shape, b_shape
            ));
        }
        if a_shape[0] != b_shape[0] {
            return Err(format!(
                "WGPU batched matmul batch mismatch: {:?} cannot multiply {:?}",
                a_shape, b_shape
            ));
        }
        if a_shape[2] != b_shape[1] {
            return Err(format!(
                "WGPU batched matmul inner dimension mismatch: {:?} cannot multiply {:?}",
                a_shape, b_shape
            ));
        }

        let batch = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];
        if batch == 0 || m == 0 || k == 0 || n == 0 {
            return Err("WGPU batched matmul does not accept zero-sized dimensions".to_string());
        }

        let a_standard = a.as_standard_layout().into_owned();
        let b_standard = b.as_standard_layout().into_owned();
        let a_slice = a_standard.as_slice().ok_or_else(|| {
            "WGPU batched matmul could not create contiguous lhs buffer".to_string()
        })?;
        let b_slice = b_standard.as_slice().ok_or_else(|| {
            "WGPU batched matmul could not create contiguous rhs buffer".to_string()
        })?;

        let output_len = batch
            .checked_mul(m)
            .and_then(|value| value.checked_mul(n))
            .ok_or_else(|| "WGPU batched matmul output element count overflowed".to_string())?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "WGPU batched matmul output byte count overflowed".to_string())?;

        let a_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU BatchedMatMul A"),
                contents: bytemuck::cast_slice(a_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU BatchedMatMul B"),
                contents: bytemuck::cast_slice(b_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU BatchedMatMul C"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU BatchedMatMul Readback"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = BatchedMatMulParams {
            batch: Self::checked_u32_dim("batch", batch)?,
            m: Self::checked_u32_dim("m", m)?,
            n: Self::checked_u32_dim("n", n)?,
            k: Self::checked_u32_dim("k", k)?,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU BatchedMatMul Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("TensorEngine WGPU BatchedMatMul Shader"),
                source: wgpu::ShaderSource::Wgsl(BATCHED_MATMUL_SHADER.into()),
            });
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("TensorEngine WGPU BatchedMatMul BindGroupLayout"),
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
                label: Some("TensorEngine WGPU BatchedMatMul PipelineLayout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TensorEngine WGPU BatchedMatMul Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TensorEngine WGPU BatchedMatMul BindGroup"),
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
                label: Some("TensorEngine WGPU BatchedMatMul Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TensorEngine WGPU BatchedMatMul Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((params.m + 15) / 16, (params.n + 15) / 16, params.batch);
        }
        encoder.copy_buffer_to_buffer(&c_buffer, 0, &readback_buffer, 0, output_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if sender.send(result).is_err() {
                log::warn!("WGPU batched matmul readback receiver dropped before map completion");
            }
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|_| "WGPU batched matmul readback channel closed".to_string())?
            .map_err(|e| format!("WGPU batched matmul readback failed: {}", e))?;

        let mapped = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        readback_buffer.unmap();

        ArrayD::from_shape_vec(IxDyn(&[batch, m, n]), result)
            .map_err(|e| format!("WGPU batched matmul result shape failed: {}", e))
    }

    fn softmax_last_axis_gpu(
        &self,
        input: &ArrayD<f32>,
        axis: isize,
    ) -> Result<ArrayD<f32>, String> {
        let ndim = input.ndim();
        if ndim == 0 {
            return Err("WGPU softmax requires at least one dimension".to_string());
        }
        if axis < 0 || axis as usize >= ndim {
            return Err(format!(
                "WGPU softmax invalid axis {} for tensor with {} dimensions",
                axis, ndim
            ));
        }
        let norm_axis = axis as usize;
        let last_axis = ndim - 1;
        if norm_axis != last_axis {
            return Err(format!(
                "WGPU softmax supports last-axis execution only, got axis {} for shape {:?}",
                axis,
                input.shape()
            ));
        }

        let shape = input.shape();
        let cols = shape[last_axis];
        if cols == 0 {
            return Err("WGPU softmax does not accept zero-sized normalization axis".to_string());
        }
        let total_len = input.len();
        if total_len == 0 {
            return Err("WGPU softmax does not accept empty tensors".to_string());
        }
        let rows = total_len
            .checked_div(cols)
            .ok_or_else(|| "WGPU softmax row count calculation failed".to_string())?;
        if rows == 0 {
            return Err("WGPU softmax computed zero rows".to_string());
        }

        let input_standard = input.as_standard_layout().into_owned();
        let input_slice = input_standard
            .as_slice()
            .ok_or_else(|| "WGPU softmax could not create contiguous input buffer".to_string())?;
        let output_bytes = total_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "WGPU softmax output byte count overflowed".to_string())?;

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU Softmax Input"),
                contents: bytemuck::cast_slice(input_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU Softmax Output"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU Softmax Readback"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = SoftmaxParams {
            rows: Self::checked_u32_dim("rows", rows)?,
            cols: Self::checked_u32_dim("cols", cols)?,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU Softmax Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("TensorEngine WGPU Softmax Shader"),
                source: wgpu::ShaderSource::Wgsl(SOFTMAX_SHADER.into()),
            });
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("TensorEngine WGPU Softmax BindGroupLayout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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
                label: Some("TensorEngine WGPU Softmax PipelineLayout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TensorEngine WGPU Softmax Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TensorEngine WGPU Softmax BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TensorEngine WGPU Softmax Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TensorEngine WGPU Softmax Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(params.rows, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if sender.send(result).is_err() {
                log::warn!("WGPU softmax readback receiver dropped before map completion");
            }
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|_| "WGPU softmax readback channel closed".to_string())?
            .map_err(|e| format!("WGPU softmax readback failed: {}", e))?;

        let mapped = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        readback_buffer.unmap();

        ArrayD::from_shape_vec(IxDyn(shape), result)
            .map_err(|e| format!("WGPU softmax result shape failed: {}", e))
    }

    fn rms_norm_last_axis_gpu(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        eps: f32,
        axis: isize,
    ) -> Result<ArrayD<f32>, String> {
        if !eps.is_finite() || eps < 0.0 {
            return Err(format!(
                "WGPU RMSNorm requires finite non-negative eps, got {}",
                eps
            ));
        }
        let ndim = input.ndim();
        if ndim == 0 {
            return Err("WGPU RMSNorm requires at least one dimension".to_string());
        }
        if axis < 0 || axis as usize >= ndim {
            return Err(format!(
                "WGPU RMSNorm invalid axis {} for tensor with {} dimensions",
                axis, ndim
            ));
        }
        let norm_axis = axis as usize;
        let last_axis = ndim - 1;
        if norm_axis != last_axis {
            return Err(format!(
                "WGPU RMSNorm supports last-axis execution only, got axis {} for shape {:?}",
                axis,
                input.shape()
            ));
        }

        let shape = input.shape();
        let cols = shape[last_axis];
        if cols == 0 {
            return Err("WGPU RMSNorm does not accept zero-sized normalization axis".to_string());
        }
        if weight.ndim() != 1 || weight.len() != cols {
            return Err(format!(
                "WGPU RMSNorm requires 1D weight of length {}, got shape {:?}",
                cols,
                weight.shape()
            ));
        }
        let total_len = input.len();
        if total_len == 0 {
            return Err("WGPU RMSNorm does not accept empty tensors".to_string());
        }
        let rows = total_len
            .checked_div(cols)
            .ok_or_else(|| "WGPU RMSNorm row count calculation failed".to_string())?;
        if rows == 0 {
            return Err("WGPU RMSNorm computed zero rows".to_string());
        }

        let input_standard = input.as_standard_layout().into_owned();
        let input_slice = input_standard
            .as_slice()
            .ok_or_else(|| "WGPU RMSNorm could not create contiguous input buffer".to_string())?;
        let weight_standard = weight.as_standard_layout().into_owned();
        let weight_slice = weight_standard
            .as_slice()
            .ok_or_else(|| "WGPU RMSNorm could not create contiguous weight buffer".to_string())?;
        let output_bytes = total_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "WGPU RMSNorm output byte count overflowed".to_string())?;

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU RMSNorm Input"),
                contents: bytemuck::cast_slice(input_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let weight_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU RMSNorm Weight"),
                contents: bytemuck::cast_slice(weight_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU RMSNorm Output"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU RMSNorm Readback"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = RmsNormParams {
            rows: Self::checked_u32_dim("rows", rows)?,
            cols: Self::checked_u32_dim("cols", cols)?,
            eps,
            _pad: 0,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU RMSNorm Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("TensorEngine WGPU RMSNorm Shader"),
                source: wgpu::ShaderSource::Wgsl(RMS_NORM_SHADER.into()),
            });
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("TensorEngine WGPU RMSNorm BindGroupLayout"),
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
                label: Some("TensorEngine WGPU RMSNorm PipelineLayout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TensorEngine WGPU RMSNorm Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TensorEngine WGPU RMSNorm BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
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
                label: Some("TensorEngine WGPU RMSNorm Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TensorEngine WGPU RMSNorm Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(params.rows, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if sender.send(result).is_err() {
                log::warn!("WGPU RMSNorm readback receiver dropped before map completion");
            }
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|_| "WGPU RMSNorm readback channel closed".to_string())?
            .map_err(|e| format!("WGPU RMSNorm readback failed: {}", e))?;

        let mapped = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        readback_buffer.unmap();

        ArrayD::from_shape_vec(IxDyn(shape), result)
            .map_err(|e| format!("WGPU RMSNorm result shape failed: {}", e))
    }

    fn layer_norm_last_axis_gpu(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        bias: &ArrayD<f32>,
        eps: f32,
        axis: isize,
    ) -> Result<ArrayD<f32>, String> {
        if !eps.is_finite() || eps < 0.0 {
            return Err(format!(
                "WGPU LayerNorm requires finite non-negative eps, got {}",
                eps
            ));
        }
        let ndim = input.ndim();
        if ndim == 0 {
            return Err("WGPU LayerNorm requires at least one dimension".to_string());
        }
        if axis < 0 || axis as usize >= ndim {
            return Err(format!(
                "WGPU LayerNorm invalid axis {} for tensor with {} dimensions",
                axis, ndim
            ));
        }
        let norm_axis = axis as usize;
        let last_axis = ndim - 1;
        if norm_axis != last_axis {
            return Err(format!(
                "WGPU LayerNorm supports last-axis execution only, got axis {} for shape {:?}",
                axis,
                input.shape()
            ));
        }

        let shape = input.shape();
        let cols = shape[last_axis];
        if cols == 0 {
            return Err("WGPU LayerNorm does not accept zero-sized normalization axis".to_string());
        }
        if weight.ndim() != 1 || weight.len() != cols {
            return Err(format!(
                "WGPU LayerNorm requires 1D weight of length {}, got shape {:?}",
                cols,
                weight.shape()
            ));
        }
        if bias.ndim() != 1 || bias.len() != cols {
            return Err(format!(
                "WGPU LayerNorm requires 1D bias of length {}, got shape {:?}",
                cols,
                bias.shape()
            ));
        }
        let total_len = input.len();
        if total_len == 0 {
            return Err("WGPU LayerNorm does not accept empty tensors".to_string());
        }
        let rows = total_len
            .checked_div(cols)
            .ok_or_else(|| "WGPU LayerNorm row count calculation failed".to_string())?;
        if rows == 0 {
            return Err("WGPU LayerNorm computed zero rows".to_string());
        }

        let input_standard = input.as_standard_layout().into_owned();
        let input_slice = input_standard
            .as_slice()
            .ok_or_else(|| "WGPU LayerNorm could not create contiguous input buffer".to_string())?;
        let weight_standard = weight.as_standard_layout().into_owned();
        let weight_slice = weight_standard.as_slice().ok_or_else(|| {
            "WGPU LayerNorm could not create contiguous weight buffer".to_string()
        })?;
        let bias_standard = bias.as_standard_layout().into_owned();
        let bias_slice = bias_standard
            .as_slice()
            .ok_or_else(|| "WGPU LayerNorm could not create contiguous bias buffer".to_string())?;
        let output_bytes = total_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "WGPU LayerNorm output byte count overflowed".to_string())?;

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU LayerNorm Input"),
                contents: bytemuck::cast_slice(input_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let weight_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU LayerNorm Weight"),
                contents: bytemuck::cast_slice(weight_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bias_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU LayerNorm Bias"),
                contents: bytemuck::cast_slice(bias_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU LayerNorm Output"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU LayerNorm Readback"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = LayerNormParams {
            rows: Self::checked_u32_dim("rows", rows)?,
            cols: Self::checked_u32_dim("cols", cols)?,
            eps,
            _pad: 0,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TensorEngine WGPU LayerNorm Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("TensorEngine WGPU LayerNorm Shader"),
                source: wgpu::ShaderSource::Wgsl(LAYER_NORM_SHADER.into()),
            });
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("TensorEngine WGPU LayerNorm BindGroupLayout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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
                label: Some("TensorEngine WGPU LayerNorm PipelineLayout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TensorEngine WGPU LayerNorm Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TensorEngine WGPU LayerNorm BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TensorEngine WGPU LayerNorm Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TensorEngine WGPU LayerNorm Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(params.rows, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if sender.send(result).is_err() {
                log::warn!("WGPU LayerNorm readback receiver dropped before map completion");
            }
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|_| "WGPU LayerNorm readback channel closed".to_string())?
            .map_err(|e| format!("WGPU LayerNorm readback failed: {}", e))?;

        let mapped = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        readback_buffer.unmap();

        ArrayD::from_shape_vec(IxDyn(shape), result)
            .map_err(|e| format!("WGPU LayerNorm result shape failed: {}", e))
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
        let result = match (a.ndim(), b.ndim()) {
            (2, 2) => self.matmul_2d_gpu(a, b),
            (3, 3) => self.matmul_3d_gpu(a, b),
            _ => Err(format!(
                "WGPU matmul supports 2D or 3D tensors, got {:?} and {:?}",
                a.shape(),
                b.shape()
            )),
        };
        match result {
            Ok(result) => Some(result),
            Err(err) => {
                log::warn!("WGPU matmul unavailable: {}", err);
                None
            }
        }
    }

    fn unary_activation(&self, input: &ArrayD<f32>, kind: ActivationKind) -> Option<ArrayD<f32>> {
        match self.unary_activation_gpu(input, kind) {
            Ok(result) => Some(result),
            Err(err) => {
                log::warn!("WGPU unary activation unavailable: {}", err);
                None
            }
        }
    }

    fn softmax(&self, input: &ArrayD<f32>, axis: isize) -> Option<ArrayD<f32>> {
        match self.softmax_last_axis_gpu(input, axis) {
            Ok(result) => Some(result),
            Err(err) => {
                log::warn!("WGPU softmax unavailable: {}", err);
                None
            }
        }
    }

    fn layer_norm(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        bias: &ArrayD<f32>,
        eps: f32,
        axis: isize,
    ) -> Option<ArrayD<f32>> {
        match self.layer_norm_last_axis_gpu(input, weight, bias, eps, axis) {
            Ok(result) => Some(result),
            Err(err) => {
                log::warn!("WGPU LayerNorm unavailable: {}", err);
                None
            }
        }
    }

    fn rms_norm(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        eps: f32,
        axis: isize,
    ) -> Option<ArrayD<f32>> {
        match self.rms_norm_last_axis_gpu(input, weight, eps, axis) {
            Ok(result) => Some(result),
            Err(err) => {
                log::warn!("WGPU RMSNorm unavailable: {}", err);
                None
            }
        }
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
