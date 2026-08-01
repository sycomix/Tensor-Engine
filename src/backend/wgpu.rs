use crate::backend::traits::{ActivationKind, Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RoPEParams {
    total_elements: u32,
    seq_len: u32,
    head_dim: u32,
    freq_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv2DParams {
    n: u32,
    cin: u32,
    cout: u32,
    kh: u32,
    kw: u32,
    hin: u32,
    win: u32,
    hout: u32,
    wout: u32,
    stride: u32,
    padding: u32,
    has_bias: u32,
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const CONV2D_SHADER: &str = r#"
struct Conv2DParams {
    n: u32,
    cin: u32,
    cout: u32,
    kh: u32,
    kw: u32,
    hin: u32,
    win: u32,
    hout: u32,
    wout: u32,
    stride: u32,
    padding: u32,
    has_bias: u32,
    total_elements: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv2DParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.total_elements) {
        return;
    }

    let wout = params.wout;
    let hout = params.hout;
    let cout = params.cout;

    let ow = idx % wout;
    let oh = (idx / wout) % hout;
    let oc = (idx / (wout * hout)) % cout;
    let batch = idx / (wout * hout * cout);

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.cin; ic = ic + 1u) {
        for (var kh_i: u32 = 0u; kh_i < params.kh; kh_i = kh_i + 1u) {
            for (var kw_i: u32 = 0u; kw_i < params.kw; kw_i = kw_i + 1u) {
                let ih = oh * params.stride + kh_i - params.padding;
                let iw = ow * params.stride + kw_i - params.padding;
                if (ih < params.hin && iw < params.win) {
                    let in_idx = batch * params.cin * params.hin * params.win
                               + ic * params.hin * params.win
                               + ih * params.win + iw;
                    let w_idx = oc * params.cin * params.kh * params.kw
                              + ic * params.kh * params.kw
                              + kh_i * params.kw + kw_i;
                    sum = sum + input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    if (params.has_bias != 0u) {
        sum = sum + bias[oc];
    }

    output[idx] = sum;
}
"#;

const ROPE_SHADER: &str = r#"
struct RoPEParams {
    total_elements: u32,
    seq_len: u32,
    head_dim: u32,
    freq_len: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> freqs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: RoPEParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.total_elements) {
        return;
    }

    let head_dim = params.head_dim;
    let seq_len = params.seq_len;

    let dim_idx = idx % head_dim;
    let pos = (idx / head_dim) % seq_len;

    let pair_idx = dim_idx / 2u;
    if (pair_idx >= params.freq_len) {
        output[idx] = input[idx];
        return;
    }

    let theta = freqs[pair_idx];
    let angle = f32(pos) * theta;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let base_idx = idx - dim_idx;
    let pair_offset = (dim_idx & 1u) ^ 1u;
    let partner_idx = base_idx + pair_offset;

    let x0 = input[base_idx];
    let x1 = input[base_idx + 1u];

    if (dim_idx & 1u) == 0u {
        output[idx] = x0 * cos_val - x1 * sin_val;
    } else {
        output[idx] = x0 * sin_val + x1 * cos_val;
    }
}
"#;

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
    if (kind == 5u) {
        return gelu(value);
    }
    if (kind == 6u) {
        let r = max(value, 0.0);
        return r * r;
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

type PipelineCache =
    Mutex<HashMap<&'static str, (Arc<wgpu::BindGroupLayout>, Arc<wgpu::ComputePipeline>)>>;

pub struct WgpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipeline_cache: PipelineCache,
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

            Ok(Self {
                device,
                queue,
                pipeline_cache: Mutex::new(HashMap::new()),
            })
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
            ActivationKind::GeluTanh => 5,
            ActivationKind::Relu2 => 6,
        }
    }

    fn get_or_create_pipeline(
        &self,
        name: &'static str,
        shader_src: &'static str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> (Arc<wgpu::BindGroupLayout>, Arc<wgpu::ComputePipeline>) {
        {
            let cache = self.pipeline_cache.lock().unwrap();
            if let Some((bgl, pipeline)) = cache.get(name) {
                return (Arc::clone(bgl), Arc::clone(pipeline));
            }
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(name),
                    entries,
                });
        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(name),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });
        let bgl = Arc::new(bind_group_layout);
        let pl = Arc::new(pipeline);
        self.pipeline_cache
            .lock()
            .unwrap()
            .insert(name, (Arc::clone(&bgl), Arc::clone(&pl)));
        (bgl, pl)
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

        let (bind_group_layout, pipeline) = self.get_or_create_pipeline(
            "unary_activation",
            UNARY_ACTIVATION_SHADER,
            &[
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
        );
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
            pass.dispatch_workgroups(params.len.div_ceil(256), 1, 1);
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

        let max_binding = self.device.limits().max_storage_buffer_binding_size as u64;
        let max_buffer = self.device.limits().max_buffer_size;
        let a_bytes = a_slice.len() * std::mem::size_of::<f32>();
        let b_bytes = b_slice.len() * std::mem::size_of::<f32>();
        if a_bytes as u64 > max_binding || b_bytes as u64 > max_binding || output_bytes as u64 > max_buffer {
            log::info!(
                "WGPU matmul operand too large for binding (max_binding={} bytes, max_buffer={} bytes): a={} bytes, b={} bytes, out={} bytes; falling back to chunked matmul",
                max_binding, max_buffer, a_bytes, b_bytes, output_bytes
            );
            return self.matmul_2d_gpu_chunked(a, b);
        }

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

        let (bind_group_layout, pipeline) = self.get_or_create_pipeline(
            "matmul_2d",
            MATMUL_SHADER,
            &[
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
        );
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
            pass.dispatch_workgroups(params.m.div_ceil(16), params.n.div_ceil(16), 1);
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

    /// Computes a 2D matmul in blocks so that each buffer fits within the
    /// device's max buffer size. Falls back to the single-shot path for each
    /// block (via recursion) and concatenates the results.
    fn matmul_2d_gpu_chunked(
        &self,
        a: &ArrayD<f32>,
        b: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>, String> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        let max_binding = self.device.limits().max_storage_buffer_binding_size as usize;

        // Each chunk needs: a_block [m_chunk,k], b_block [k,n_chunk], out [m_chunk,n_chunk]
        // All three must fit within max_binding (since they're storage bindings).
        // a_block_bytes = m_chunk * k * 4, b_block_bytes = k * n_chunk * 4
        // out_block_bytes = m_chunk * n_chunk * 4
        // Constraint: max(a_block_bytes, b_block_bytes, out_block_bytes) <= max_binding
        // => m_chunk * k <= max_binding / 4 and k * n_chunk <= max_binding / 4 and m_chunk * n_chunk <= max_binding / 4
        let elem_budget = (max_binding / 4).max(1);

        // Choose chunk sizes respecting all constraints
        let n_chunk = ((elem_budget / k.max(1)).max(16)).min(n);
        let m_chunk = ((elem_budget / n.max(1)).max(16))
            .min(m)
            .min(elem_budget / k.max(1));

        let a2 = a
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("WGPU chunked matmul a dim: {}", e))?;
        let b2 = b
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("WGPU chunked matmul b dim: {}", e))?;

        let mut out = ndarray::Array2::<f32>::zeros((m, n));
        for mi in (0..m).step_by(m_chunk) {
            let m_end = (mi + m_chunk).min(m);
            for ni in (0..n).step_by(n_chunk) {
                let n_end = (ni + n_chunk).min(n);
                let a_block = a2.slice(ndarray::s![mi..m_end, ..]).to_owned();
                let b_block = b2.slice(ndarray::s![.., ni..n_end]).to_owned();
                let res = self.matmul_2d_gpu(&a_block.into_dyn(), &b_block.into_dyn())?;
                let res2 = res
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| format!("WGPU chunked matmul result dim: {}", e))?;
                out.slice_mut(ndarray::s![mi..m_end, ni..n_end])
                    .assign(&res2);
            }
        }
        Ok(out.into_dyn())
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

        let max_binding = self.device.limits().max_storage_buffer_binding_size as u64;
        let max_buffer = self.device.limits().max_buffer_size;
        let a_bytes = a_slice.len() * std::mem::size_of::<f32>();
        let b_bytes = b_slice.len() * std::mem::size_of::<f32>();
        if a_bytes as u64 > max_binding || b_bytes as u64 > max_binding || output_bytes as u64 > max_buffer {
            return Err(format!(
                "WGPU batched matmul operand exceeds max binding (max_binding={} bytes, max_buffer={} bytes): a={} bytes, b={} bytes, out={} bytes",
                max_binding, max_buffer, a_bytes, b_bytes, output_bytes
            ));
        }

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

        let (bind_group_layout, pipeline) = self.get_or_create_pipeline(
            "matmul_3d",
            BATCHED_MATMUL_SHADER,
            &[
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
        );
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
            pass.dispatch_workgroups(params.m.div_ceil(16), params.n.div_ceil(16), params.batch);
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

        let (bind_group_layout, pipeline) = self.get_or_create_pipeline(
            "softmax",
            SOFTMAX_SHADER,
            &[
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
        );
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

        let (bind_group_layout, pipeline) = self.get_or_create_pipeline(
            "rms_norm",
            RMS_NORM_SHADER,
            &[
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
        );
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

        let (bind_group_layout, pipeline) = self.get_or_create_pipeline(
            "layer_norm",
            LAYER_NORM_SHADER,
            &[
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
        );
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

    fn conv2d_gpu(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        bias: Option<&ArrayD<f32>>,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<f32>, String> {
        let in_std = input.as_standard_layout().into_owned();
        let in_slice = in_std.as_slice().ok_or("WGPU conv2d: input not contiguous")?;
        let w_std = weight.as_standard_layout().into_owned();
        let w_slice = w_std.as_slice().ok_or("WGPU conv2d: weight not contiguous")?;

        let in_shape = input.shape();
        let w_shape = weight.shape();
        if in_shape.len() != 4 || w_shape.len() != 4 {
            return Err(format!(
                "WGPU conv2d expects 4D input and weight, got {:?} and {:?}",
                in_shape, w_shape
            ));
        }
        let n = in_shape[0];
        let cin = in_shape[1];
        let hin = in_shape[2];
        let win = in_shape[3];
        let cout = w_shape[0];
        let cin2 = w_shape[1];
        let kh = w_shape[2];
        let kw = w_shape[3];
        if cin != cin2 {
            return Err(format!(
                "WGPU conv2d channel mismatch: input cin={} weight cin={}",
                cin, cin2
            ));
        }

        let s = stride as isize;
        let p = padding as isize;
        let hout = ((hin as isize - kh as isize + 2 * p) / s + 1) as usize;
        let wout = ((win as isize - kw as isize + 2 * p) / s + 1) as usize;
        let total_elements = n * cout * hout * wout;
        if total_elements == 0 {
            return Ok(ArrayD::zeros(IxDyn(&[n, cout, hout, wout])));
        }

        let bias_slice = match bias {
            Some(b) => {
                let b_std = b.as_standard_layout().into_owned();
                b_std.as_slice()
                    .ok_or("WGPU conv2d: bias not contiguous")?
                    .to_vec()
            }
            None => vec![0.0f32],
        };

        let output_bytes = total_elements * std::mem::size_of::<f32>();

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TensorEngine WGPU Conv2D Input"),
            contents: bytemuck::cast_slice(in_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let weight_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TensorEngine WGPU Conv2D Weight"),
            contents: bytemuck::cast_slice(w_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bias_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TensorEngine WGPU Conv2D Bias"),
            contents: bytemuck::cast_slice(&bias_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU Conv2D Output"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU Conv2D Readback"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = Conv2DParams {
            n: n as u32,
            cin: cin as u32,
            cout: cout as u32,
            kh: kh as u32,
            kw: kw as u32,
            hin: hin as u32,
            win: win as u32,
            hout: hout as u32,
            wout: wout as u32,
            stride: stride as u32,
            padding: padding as u32,
            has_bias: if bias.is_some() { 1 } else { 0 },
            total_elements: total_elements as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TensorEngine WGPU Conv2D Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let (bind_group_layout, pipeline) = self.get_or_create_pipeline(
            "conv2d",
            CONV2D_SHADER,
            &[
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
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TensorEngine WGPU Conv2D BindGroup"),
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

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("TensorEngine WGPU Conv2D Encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TensorEngine WGPU Conv2D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (total_elements as u32).div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_bytes as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .ok()
            .ok_or_else(|| "WGPU conv2d: channel closed".to_string())?
            .map_err(|e| format!("WGPU conv2d: map_async failed: {:?}", e))?;

        let data = slice.get_mapped_range();
        let output_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buffer.unmap();

        ArrayD::from_shape_vec(IxDyn(&[n, cout, hout, wout]), output_vec)
            .map_err(|e| format!("WGPU conv2d result shape failed: {}", e))
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

    fn conv2d(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        bias: Option<&ArrayD<f32>>,
        stride: usize,
        padding: usize,
    ) -> Option<ArrayD<f32>> {
        match self.conv2d_gpu(input, weight, bias, stride, padding) {
            Ok(result) => Some(result),
            Err(err) => {
                log::warn!("WGPU Conv2D unavailable: {}", err);
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
        if x.ndim() < 3 || x.shape()[1] != seq_len || x.shape()[2] != head_dim {
            log::error!(
                "RoPE: Invalid input shape {:?} for seq_len={} and head_dim={}",
                x.shape(),
                seq_len,
                head_dim
            );
            return None;
        }

        let total_elements = x.len();
        let freq_len = freqs.len();

        let x_std = x.as_standard_layout().into_owned();
        let x_slice = x_std.as_slice()?;
        let freq_std = freqs.as_standard_layout().into_owned();
        let freq_slice = freq_std.as_slice()?;

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TensorEngine WGPU RoPE Input"),
            contents: bytemuck::cast_slice(x_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let freq_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TensorEngine WGPU RoPE Freqs"),
            contents: bytemuck::cast_slice(freq_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_bytes = total_elements * std::mem::size_of::<f32>();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU RoPE Output"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TensorEngine WGPU RoPE Readback"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = RoPEParams {
            total_elements: total_elements as u32,
            seq_len: seq_len as u32,
            head_dim: head_dim as u32,
            freq_len: freq_len as u32,
        };
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TensorEngine WGPU RoPE Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let (bind_group_layout, pipeline) = self.get_or_create_pipeline(
            "rope",
            ROPE_SHADER,
            &[
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
        );
        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TensorEngine WGPU RoPE BindGroup"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: freq_buffer.as_entire_binding(),
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

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("TensorEngine WGPU RoPE Encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TensorEngine WGPU RoPE ComputePass"),
                ..Default::default()
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (total_elements as u32).div_ceil(256);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_bytes as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().ok()?.ok()?;

        let data = slice.get_mapped_range();
        let output_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buffer.unmap();

        let shape = x.shape().to_vec();
        ArrayD::from_shape_vec(IxDyn(&shape), output_vec).ok()
    }

    fn memory_info(&self) -> (usize, usize) {
        let limits = self.device.limits();
        let total = limits.max_storage_buffer_binding_size as usize * 4;
        let used = 0;
        (used, total)
    }

    fn synchronize(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    fn matmul_backward(
        &self,
        d_out: &ArrayD<f32>,
        a: &ArrayD<f32>,
        b: &ArrayD<f32>,
    ) -> Option<(ArrayD<f32>, ArrayD<f32>)> {
        // dA = d_out @ B^T — transpose B then multiply
        let b_t = b.clone().reversed_axes();
        let grad_a = match self.matmul_2d_gpu(d_out, &b_t) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("matmul_backward dA failed: {}", e);
                return None;
            }
        };
        // dB = A^T @ d_out — transpose A then multiply
        let a_t = a.clone().reversed_axes();
        let grad_b = match self.matmul_2d_gpu(&a_t, d_out) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("matmul_backward dB failed: {}", e);
                return None;
            }
        };
        Some((grad_a, grad_b))
    }

    fn unary_activation_backward(
        &self,
        d_out: &ArrayD<f32>,
        x: &ArrayD<f32>,
        kind: ActivationKind,
    ) -> Option<ArrayD<f32>> {
        if x.shape() != d_out.shape() {
            log::error!("unary_activation_backward: shape mismatch");
            return None;
        }
        let mut d_input = d_out.clone();
        d_input.iter_mut().zip(x.iter()).for_each(|(di, &xi)| {
            *di *= match kind {
                ActivationKind::Relu => {
                    if xi > 0.0 { 1.0 } else { 0.0 }
                }
                ActivationKind::Sigmoid => {
                    let s = 1.0 / (1.0 + (-xi).exp());
                    s * (1.0 - s)
                }
                ActivationKind::Tanh => {
                    let t = xi.tanh();
                    1.0 - t * t
                }
                ActivationKind::Gelu => {
                    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
                    let u = sqrt_2_over_pi * (xi + 0.044715 * xi * xi * xi);
                    let tanh_u = u.tanh();
                    let sech2 = 1.0 - tanh_u * tanh_u;
                    0.5 * (1.0 + tanh_u) + 0.5 * xi * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * xi * xi)
                }
                ActivationKind::Silu => {
                    let s = 1.0 / (1.0 + (-xi).exp());
                    s + xi * s * (1.0 - s)
                }
                ActivationKind::GeluTanh => {
                    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
                    let u = sqrt_2_over_pi * (xi + 0.044715 * xi * xi * xi);
                    let tanh_u = u.tanh();
                    let sech2 = 1.0 - tanh_u * tanh_u;
                    0.5 * (1.0 + tanh_u) + 0.5 * xi * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * xi * xi)
                }
                ActivationKind::Relu2 => {
                    if xi > 0.0 { 2.0 * xi } else { 0.0 }
                }
            };
        });
        Some(d_input)
    }

    fn softmax_backward(
        &self,
        d_out: &ArrayD<f32>,
        softmax_out: &ArrayD<f32>,
        axis: isize,
    ) -> Option<ArrayD<f32>> {
        let ndim = softmax_out.ndim();
        let norm_axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };
        if norm_axis >= ndim || d_out.shape() != softmax_out.shape() {
            log::error!("softmax_backward: shape mismatch or invalid axis");
            return None;
        }
        // d_input = softmax * (d_out - sum(d_out * softmax, axis))
        let prod = d_out.clone() * softmax_out.clone();
        let sum_array = prod.sum_axis(ndarray::Axis(norm_axis));
        let s_b = sum_array.insert_axis(ndarray::Axis(norm_axis));
        let d_input = softmax_out * (d_out - s_b);
        Some(d_input)
    }

    fn layer_norm_backward(
        &self,
        d_out: &ArrayD<f32>,
        x: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        normalized: &ArrayD<f32>,
        rstd: &ArrayD<f32>,
        axis: isize,
    ) -> Option<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)> {
        let ndim = x.ndim();
        let norm_axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };
        if norm_axis >= ndim || d_out.shape() != x.shape() {
            log::error!("WGPU layer_norm_backward: shape mismatch or invalid axis");
            return None;
        }
        let features = x.shape()[norm_axis];
        let total: usize = x.shape().iter().product();
        let nrows = total / features;

        let d_out_2d = match d_out.to_shape((nrows, features)) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                log::error!("WGPU layer_norm_backward: reshape d_out failed: {}", e);
                return None;
            }
        };
        let norm_2d = match normalized.to_shape((nrows, features)) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                log::error!("WGPU layer_norm_backward: reshape norm failed: {}", e);
                return None;
            }
        };
        let rstd_1d = match rstd.to_shape((nrows,)) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                log::error!("WGPU layer_norm_backward: reshape rstd failed: {}", e);
                return None;
            }
        };

        let mut grad_gamma = ArrayD::zeros(IxDyn(&[features]));
        let mut grad_beta = ArrayD::zeros(IxDyn(&[features]));
        for j in 0..features {
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            for irow in 0..nrows {
                let dop = d_out_2d[[irow, j]];
                let norm = norm_2d[[irow, j]];
                sum_g += dop * norm;
                sum_b += dop;
            }
            grad_gamma[[j]] = sum_g;
            grad_beta[[j]] = sum_b;
        }

        let mut grad_x2 = ArrayD::zeros(IxDyn(&[nrows, features]));
        for irow in 0..nrows {
            let inv = rstd_1d[irow];
            let mut mean1 = 0.0f32;
            let mut mean2 = 0.0f32;
            for j in 0..features {
                let g = d_out_2d[[irow, j]];
                let gam = weight[[j]];
                let dnormalized = g * gam;
                mean1 += dnormalized;
                mean2 += dnormalized * norm_2d[[irow, j]];
            }
            mean1 /= features as f32;
            mean2 /= features as f32;
            for j in 0..features {
                let dnormalized = d_out_2d[[irow, j]] * weight[[j]];
                let norm = norm_2d[[irow, j]];
                grad_x2[[irow, j]] = inv * (dnormalized - mean1 - norm * mean2);
            }
        }

        let grad_x = match grad_x2.into_dyn().to_shape(IxDyn(x.shape())) {
            Ok(g) => g.to_owned(),
            Err(e) => {
                log::error!("WGPU layer_norm_backward: reshape grad_x failed: {}", e);
                return None;
            }
        };
        Some((grad_x, grad_gamma, grad_beta))
    }

    fn rms_norm_backward(
        &self,
        d_out: &ArrayD<f32>,
        x: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        rstd: &ArrayD<f32>,
        axis: isize,
    ) -> Option<(ArrayD<f32>, ArrayD<f32>)> {
        let ndim = x.ndim();
        let norm_axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };
        if norm_axis >= ndim || d_out.shape() != x.shape() {
            log::error!("WGPU rms_norm_backward: shape mismatch or invalid axis");
            return None;
        }
        let features = x.shape()[norm_axis];
        let total: usize = x.shape().iter().product();
        let nrows = total / features;

        let d_out_2d = d_out.to_shape((nrows, features)).ok()?.to_owned();
        let x_2d = x.to_shape((nrows, features)).ok()?.to_owned();

        let mut gamma_shape = vec![1usize; ndim];
        gamma_shape[norm_axis] = features;
        let gamma_broadcast = match weight.to_shape(IxDyn(&gamma_shape)) {
            Ok(v) => v.to_owned(),
            Err(_) => weight.clone(),
        };

        let g = if let Ok(reshaped) = gamma_broadcast.to_shape(d_out_2d.shape()) {
            &d_out_2d * &reshaped
        } else {
            &d_out_2d * &gamma_broadcast
        };
        let gx = (&g * &x_2d).sum_axis(ndarray::Axis(1));

        let mut grad_gamma = ArrayD::zeros(IxDyn(&[features]));
        for j in 0..features {
            let mut sum = 0.0f32;
            for irow in 0..nrows {
                sum += g[[irow, j]] * x_2d[[irow, j]];
            }
            grad_gamma[[j]] = sum;
        }

        let mut grad_x2 = ArrayD::zeros(IxDyn(&[nrows, features]));
        for irow in 0..nrows {
            let inv = if rstd.ndim() == 1 {
                rstd[irow]
            } else if rstd.ndim() == 2 {
                rstd[[irow, 0]]
            } else {
                1.0f32
            };
            for j in 0..features {
                let gi = g[[irow, j]];
                let xi = x_2d[[irow, j]];
                grad_x2[[irow, j]] = inv * (gi - xi * gx[irow] * inv * inv / features as f32);
            }
        }

        let grad_x = match grad_x2.into_dyn().to_shape(IxDyn(x.shape())) {
            Ok(g) => g.to_owned(),
            Err(e) => {
                log::error!("WGPU rms_norm_backward: reshape grad_x failed: {}", e);
                return None;
            }
        };
        Some((grad_x, grad_gamma))
    }
}
