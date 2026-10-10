//! CUDA backend for Tensor Engine operations
//!
//! This module provides optimized CUDA kernels for high-performance
//! tensor operations used in machine learning workloads.
//! It uses the cudarc crate to interface with the NVIDIA CUDA driver API.
//!
//! Supported on Linux only. Requires CUDA toolkit installed.

use super::cuda_kernels;
use super::traits::{Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use cudarc::driver::CudaContext;
use ndarray::{ArrayD, Axis, IxDyn};
use std::sync::Arc;

/// CUDA backend for tensor operations.
pub struct CudaBackend {
    /// CUDA context managing device resources
    ctx: Arc<CudaContext>,
    /// Device ordinal (which GPU this backend uses)
    device: usize,
}

impl CudaBackend {
    /// Create a new CUDA backend on the specified device ordinal.
    pub fn new(device: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(device)
            .map_err(|e| format!("CUDA init failed for device {}: {:?}", device, e))?;

        log::info!(
            "CUDA backend initialized on device {} ({})",
            device,
            ctx.name().unwrap_or_else(|_| "Unknown".to_string())
        );

        Ok(Self { ctx, device })
    }

    /// Create a new CUDA backend on the first available device (ordinal 0).
    pub fn new_default() -> Result<Self, String> {
        Self::new(0)
    }

    /// Check if CUDA is available with at least one GPU.
    pub fn is_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// Get the device ordinal for this backend instance.
    pub fn device(&self) -> usize {
        self.device
    }

    /// Get CUDA memory info (free, total in bytes).
    pub fn memory_info(&self) -> Result<(usize, usize), String> {
        self.ctx.mem_get_info().map_err(|e| format!("Memory query failed: {:?}", e))
    }

    /// Synchronize the CUDA stream.
    pub fn synchronize(&self) -> Result<(), String> {
        self.ctx.synchronize().map_err(|e| format!("CUDA sync failed: {:?}", e))
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn create_from_data(
        &self,
        data: ArrayD<f32>,
        dtype: DType,
    ) -> Box<dyn Storage> {
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
        match (a.ndim(), b.ndim()) {
            (2, 2) => Some(self.matmul_2d(a, b)?),
            (3, 3) => Some(self.matmul_3d(a, b)?),
            _ => None,
        }
    }

    fn softmax(&self, input: &ArrayD<f32>, axis: isize) -> Option<ArrayD<f32>> {
        let ndim = input.ndim();
        if axis < 0 || (axis as usize) >= ndim {
            log::error!("Softmax: Invalid axis {} for tensor with {} dimensions", axis, ndim);
            return None;
        }
        let norm_axis = axis as usize;
        let mut output = input.clone();

        for mut lane in output.lanes_mut(Axis(norm_axis)) {
            let max_val = lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in lane.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            if !(sum > 0.0f32 && sum.is_finite()) {
                let len = lane.len() as f32;
                for v in lane.iter_mut() {
                    *v = 1.0f32 / len;
                }
                continue;
            }
            for v in lane.iter_mut() {
                *v /= sum;
            }
        }

        Some(output)
    }

    fn rms_norm(&self, input: &ArrayD<f32>, weight: &ArrayD<f32>, eps: f32, axis: isize) -> Option<ArrayD<f32>> {
        let ndim = input.ndim();
        let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
        if norm_axis >= ndim {
            log::error!("RMSNorm: Invalid axis {} for tensor with {} dimensions", axis, ndim);
            return None;
        }
        let lane_len = input.shape()[norm_axis];
        if weight.len() != lane_len {
            log::error!("RMSNorm: Weight length {} does not match axis {} dimension {}", weight.len(), norm_axis, lane_len);
            return None;
        }
        let n = lane_len as f32;
        let mut output = input.clone();

        for mut lane in output.lanes_mut(Axis(norm_axis)) {
            let sum_sq: f32 = lane.iter().map(|&x| x * x).sum();
            let rms = (sum_sq / n + eps).sqrt();
            if rms < 1e-8 {
                log::warn!("RMSNorm: Near-zero RMS value detected; skipping normalization for lane");
                continue;
            }
            for (val, &w) in lane.iter_mut().zip(weight.iter()) {
                *val = (*val / rms) * w;
            }
        }

        Some(output)
    }

    fn rope(&self, x: &ArrayD<f32>, freqs: &ArrayD<f32>, seq_len: usize, head_dim: usize) -> Option<ArrayD<f32>> {
        if x.ndim() < 3 || x.shape()[1] != seq_len || x.shape()[2] != head_dim {
            log::error!("RoPE: Invalid input shape {:?} for seq_len={} and head_dim={}", x.shape(), seq_len, head_dim);
            return None;
        }
        let mut output = x.clone();

        for pos in 0..seq_len {
            for batch in 0..x.shape()[0] {
                for i in (0..head_dim).step_by(2) {
                    if i + 1 >= head_dim { break; }
                    let freq_idx = i / 2;
                    if freq_idx >= freqs.len() { continue; }
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

    fn synchronize(&self) {
        if let Err(e) = self.ctx.synchronize() {
            log::warn!("CUDA sync failed: {:?}", e);
        }
    }

    fn memory_info(&self) -> (usize, usize) {
        match self.ctx.mem_get_info() {
            Ok((free, total)) => (total - free, total),
            Err(_) => (0, 0),
        }
    }
}

impl CudaBackend {
    fn matmul_2d(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];

        let a_flat = a.as_standard_layout().as_slice()?.to_vec();
        let b_flat = b.as_standard_layout().as_slice()?.to_vec();
        let mut output = vec![0.0f32; m * n];

        if let Err(e) = cuda_kernels::matmul_2d(&self.ctx, &a_flat, &b_flat, m, n, k, &mut output) {
            log::warn!("CUDA matmul_2d failed: {}, falling back to CPU", e);
            return None;
        }

        Some(ArrayD::<f32>::from_shape_vec(IxDyn(&[m, n]), output).ok()?)
    }

    fn matmul_3d(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        let batch = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        let n = b.shape()[2];

        let a_flat = a.as_standard_layout().as_slice()?.to_vec();
        let b_flat = b.as_standard_layout().as_slice()?.to_vec();
        let mut output = vec![0.0f32; batch * m * n];

        if let Err(e) = cuda_kernels::matmul_3d(&self.ctx, &a_flat, &b_flat, batch, m, k, n, &mut output) {
            log::warn!("CUDA matmul_3d failed: {}, falling back to CPU", e);
            return None;
        }

        Some(ArrayD::<f32>::from_shape_vec(IxDyn(&[batch, m, n]), output).ok()?)
    }
}
