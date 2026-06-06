use crate::backend::traits::{Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use ndarray::{ArrayD, Axis, Dimension, IxDyn};

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
        // For now, fall back to CPU implementation since full GPU pipeline requires more setup
        log::debug!("WGPU matmul: falling back to CPU");

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
