use crate::dtype::DType;
use ndarray::ArrayD;
use std::any::Any;

/// Represents a storage backend (CPU, WGPU, etc.)
pub trait Backend: Send + Sync + 'static {
    fn name(&self) -> &str;

    // Factory methods
    fn create_from_data(&self, data: ArrayD<f32>, dtype: DType) -> Box<dyn Storage>;
    fn create_zeros(&self, shape: &[usize]) -> Box<dyn Storage>;
    fn create_ones(&self, shape: &[usize]) -> Box<dyn Storage>;

    // Core tensor operations - backends can implement optimized versions
    fn matmul(&self, _a: &ArrayD<f32>, _b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        // Default to CPU implementation if not overridden
        None
    }

    fn conv2d(
        &self,
        _input: &ArrayD<f32>,
        _weight: &ArrayD<f32>,
        _bias: Option<&ArrayD<f32>>,
        _stride: usize,
        _padding: usize,
    ) -> Option<ArrayD<f32>> {
        None
    }

    fn softmax(&self, input: &ArrayD<f32>, axis: isize) -> Option<ArrayD<f32>> {
        let mut output = input.clone();
        let ndim = input.ndim();
        if axis < 0 || (axis as usize) >= ndim {
            log::error!("Softmax: Invalid axis {} for tensor with {} dimensions", axis, ndim);
            return None;
        }
        let norm_axis = axis as usize;
        let max_val: f32 = output.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
        output.mapv_inplace(|x| (x - max_val).exp());
        let sum_array = output.sum_axis(ndarray::Axis(norm_axis));
        if sum_array.iter().any(|&x| x.is_nan() || x <= 0.0) {
            log::warn!("Softmax: Invalid sum detected, returning zeros");
            return Some(ArrayD::zeros(input.shape().to_vec()));
        }
        let sum: f32 = match sum_array.len() {
            1 => *sum_array.get(0).unwrap_or(&1.0),
            _ => return None,
        };
        if sum > 1e-8 {
            output.mapv_inplace(|x| x / sum);
        }
        Some(output)
    }

    fn layer_norm(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        bias: &ArrayD<f32>,
        eps: f32,
        axis: isize,
    ) -> Option<ArrayD<f32>> {
        let ndim = input.ndim();
        let norm_axis = if axis < 0 {
            let positive = ndim as isize + axis;
            if positive < 0 {
                log::error!("LayerNorm: Invalid axis {} for tensor with {} dimensions", axis, ndim);
                return None;
            }
            positive as usize
        } else {
            axis as usize
        };
        if norm_axis >= ndim {
            log::error!("LayerNorm: Invalid axis {} for tensor with {} dimensions", axis, ndim);
            return None;
        }
        let shape = input.shape();
        let lane_len = shape[norm_axis];
        if weight.len() != lane_len || bias.len() != lane_len {
            log::error!("LayerNorm: weight/bias length mismatch");
            return None;
        }
        let mut output = input.clone();
        for mut lane in output.lanes_mut(ndarray::Axis(norm_axis)) {
            let mean: f32 = lane.iter().sum::<f32>() / lane.len() as f32;
            let var: f32 = lane.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / lane.len() as f32;
            let denom = (var + eps).sqrt();
            if denom < 1e-8 {
                log::warn!("LayerNorm: Near-zero denominator detected; skipping normalization for lane");
                continue;
            }
            for ((val, &w), &b) in lane.iter_mut().zip(weight.iter()).zip(bias.iter()) {
                *val = ((*val - mean) / denom) * w + b;
            }
        }
        Some(output)
    }

    // Llama/Mistral specific operations
    fn rms_norm(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        eps: f32,
        axis: isize,
    ) -> Option<ArrayD<f32>> {
        let ndim = input.ndim();
        let norm_axis = if axis < 0 {
            let positive = ndim as isize + axis;
            if positive < 0 || (positive as usize) >= ndim {
                log::error!("RMSNorm: Invalid axis {} for tensor with {} dimensions", axis, ndim);
                return None;
            }
            positive as usize
        } else {
            axis as usize
        };
        if norm_axis >= ndim {
            log::error!("RMSNorm: Invalid axis {} for tensor with {} dimensions", axis, ndim);
            return None;
        }
        let lane_len = input.shape()[norm_axis];
        if weight.len() != lane_len {
            log::error!("RMSNorm: weight length {} != lane length {}", weight.len(), lane_len);
            return None;
        }
        let mut output = input.clone();
        for mut lane in output.lanes_mut(ndarray::Axis(norm_axis)) {
            let rms = lane.iter().map(|&x| x * x).sum::<f32>().sqrt() / lane.len() as f32;
            let denom = if rms > eps { rms } else { eps };
            for (val, &w) in lane.iter_mut().zip(weight.iter()) {
                *val = (*val / denom) * w;
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
        if x.ndim() != 3 {
            log::error!("RoPE: input must be 3D [batch, seq, dim], got {:?}", x.shape());
            return None;
        }
        let batch = x.shape()[0];
        let seq = x.shape()[1];
        let dim = x.shape()[2];
        if dim != head_dim {
            log::error!("RoPE: dim {} != head_dim {}", dim, head_dim);
            return None;
        }
        let mut output = x.clone();
        for b in 0..batch {
            for s in 0..seq.min(seq_len) {
                for h in (0..dim).step_by(2) {
                    if h + 1 >= dim {
                        break;
                    }
                    let x0 = output[[b, s, h]];
                    let x1 = output[[b, s, h + 1]];
                    let freq_idx = if s < freqs.shape().first().copied().unwrap_or(1) {
                        s
                    } else {
                        freqs.shape().first().copied().unwrap_or(1) - 1
                    };
                    let freq = if freqs.ndim() >= 1 {
                        if freqs.shape().len() == 1 {
                            *freqs.get(freq_idx).unwrap_or(&1.0)
                        } else {
                            *freqs.get([freq_idx, 0]).unwrap_or(&1.0)
                        }
                    } else {
                        1.0
                    };
                    let inv_freq = if freq > 0.0 { 1.0 / freq } else { 1.0 };
                    let angle = (s as f32) * inv_freq;
                    let cos = angle.cos();
                    let sin = angle.sin();
                    output[[b, s, h]] = x0 * cos - x1 * sin;
                    output[[b, s, h + 1]] = x0 * sin + x1 * cos;
                }
            }
        }
        Some(output)
    }

    // Device management
    fn synchronize(&self) {
        // Default no-op for CPU
    }

    fn memory_info(&self) -> (usize, usize); // (used_bytes, total_bytes)
}

/// Abstract storage for tensor data
pub trait Storage: Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DType;

    // For now, we might need a way to get data back to CPU for debugging/interop
    fn to_cpu(&self) -> ArrayD<f32>;

    fn box_clone(&self) -> Box<dyn Storage>;

    // Optional Helpers for efficiency
    fn to_f32_array(&self) -> ArrayD<f32> {
        self.to_cpu()
    }

    fn as_u8_array(&self) -> Option<&ArrayD<u8>> {
        None
    }

    fn element_count(&self) -> usize {
        self.shape().iter().product()
    }
}
