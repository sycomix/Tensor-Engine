//! Dynamic Quantization Utilities for efficient model inference.
//!
//! Provides tools for dynamic (runtime) quantization of models without
//! requiring calibration data. This is useful for on-the-fly optimization
//! of models for inference.

use crate::dtype::TensorStorage;
use crate::tensor::Tensor;

/// Configuration for dynamic quantization.
#[derive(Clone, Debug)]
pub struct DynamicQuantConfig {
    /// Target bit-width (e.g., 8 for INT8, 4 for INT4)
    pub bits: u32,
    /// Whether to use symmetric quantization (vs asymmetric)
    pub symmetric: bool,
    /// Reduce range for signed integers (e.g., -127 to 127 for INT8)
    pub reduce_range: bool,
    /// Per-channel quantization for weight matrices
    pub per_channel: bool,
}

impl Default for DynamicQuantConfig {
    fn default() -> Self {
        DynamicQuantConfig {
            bits: 8,
            symmetric: true,
            reduce_range: false,
            per_channel: true,
        }
    }
}

impl DynamicQuantConfig {
    /// Create configuration for INT8 quantization.
    pub fn int8() -> Self {
        DynamicQuantConfig {
            bits: 8,
            symmetric: true,
            reduce_range: false,
            per_channel: true,
        }
    }

    /// Create configuration for INT4 quantization.
    pub fn int4() -> Self {
        DynamicQuantConfig {
            bits: 4,
            symmetric: true,
            reduce_range: true,
            per_channel: true,
        }
    }

    /// Create configuration for asymmetric quantization.
    pub fn asymmetric(bits: u32) -> Self {
        DynamicQuantConfig {
            bits,
            symmetric: false,
            reduce_range: false,
            per_channel: true,
        }
    }
}

/// Statistics for dynamic quantization.
#[derive(Clone, Debug)]
pub struct QuantStats {
    /// Minimum value in tensor
    pub min: f32,
    /// Maximum value in tensor
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Absolute maximum (for symmetric quantization)
    pub abs_max: f32,
}

impl QuantStats {
    /// Compute statistics from a tensor.
    pub fn from_tensor(tensor: &Tensor) -> Self {
        let lock = tensor.lock();
        let arr = match &lock.storage {
            TensorStorage::F32(a) => a.clone(),
            _ => lock.storage.to_f32_array(),
        };

        let data: Vec<f32> = arr.iter().copied().collect();
        let n = data.len() as f32;

        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let abs_max = data.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let mean = data.iter().copied().sum::<f32>() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        QuantStats {
            min,
            max,
            mean,
            std,
            abs_max,
        }
    }
}

/// Quantization parameters (scale and zero-point).
#[derive(Clone, Debug)]
pub struct QuantParams {
    /// Quantization scale factor
    pub scale: f32,
    /// Zero-point for asymmetric quantization
    pub zero_point: i32,
    /// Whether quantization is symmetric
    pub symmetric: bool,
}

impl QuantParams {
    /// Compute quantization parameters from tensor.
    pub fn compute(tensor: &Tensor, config: &DynamicQuantConfig) -> Self {
        let stats = QuantStats::from_tensor(tensor);

        let qmax = if config.reduce_range {
            (1i64 << (config.bits - 1)) - 1
        } else {
            (1i64 << (config.bits - 1)) - 1
        };

        let (scale, zero_point) = if config.symmetric {
            let scale = stats.abs_max / qmax as f32;
            (scale, 0)
        } else {
            // Asymmetric: map [min, max] to [0, qmax]
            let range = stats.max - stats.min;
            let scale = range / qmax as f32;
            let zero_point = -((stats.min / scale).round() as i32);
            (scale, zero_point)
        };

        QuantParams {
            scale,
            zero_point,
            symmetric: config.symmetric,
        }
    }

    /// Clamp value to quantization range.
    fn clamp_value(&self, val: f32, max_val: f32) -> i32 {
        let qval = if self.symmetric {
            (val / self.scale).round() as i32
        } else {
            ((val / self.scale) + self.zero_point as f32).round() as i32
        };
        qval.clamp(-(max_val as i32), max_val as i32)
    }

    /// Dequantize value back to float.
    fn dequantize(&self, qval: i32) -> f32 {
        if self.symmetric {
            qval as f32 * self.scale
        } else {
            (qval as f32 - self.zero_point as f32) * self.scale
        }
    }
}

/// Dynamically quantize a tensor.
pub fn quantize_dynamic(tensor: &Tensor, config: &DynamicQuantConfig) -> (Vec<i8>, QuantParams) {
    // Compute params first (which locks the tensor internally), then extract data separately
    // to avoid holding the lock while calling compute (which would deadlock on re-entrant lock).
    let params = QuantParams::compute(tensor, config);

    let lock = tensor.lock();
    let arr = match &lock.storage {
        TensorStorage::F32(a) => a.clone(),
        _ => lock.storage.to_f32_array(),
    };

    let qmax = (1i64 << (config.bits - 1)) - 1;

    let quantized: Vec<i8> = arr
        .iter()
        .map(|&val| params.clamp_value(val, qmax as f32) as i8)
        .collect();

    (quantized, params)
}

/// Dequantize a tensor.
pub fn dequantize_dynamic(data: &[i8], params: &QuantParams) -> Tensor {
    let dequantized: Vec<f32> = data.iter().map(|&q| params.dequantize(q as i32)).collect();
    let arr = ndarray::Array::from_vec(dequantized).into_dyn();
    Tensor::new(arr, false)
}

/// Compute quantization error metrics.
#[derive(Clone, Debug)]
pub struct QuantErrorMetrics {
    /// Mean squared error between original and quantized
    pub mse: f32,
    /// Relative error (MSE / variance)
    pub rel_error: f32,
    /// Maximum absolute error
    pub max_abs_error: f32,
}

impl QuantErrorMetrics {
    /// Compute error between original and quantized tensor.
    pub fn compute(
        _original: &Tensor,
        quantized_data: &[i8],
        params: &QuantParams,
        original_data: &[f32],
    ) -> Self {
        let mut mse = 0.0f32;
        let mut max_abs_error = 0.0f32;

        for (orig, &q) in original_data.iter().zip(quantized_data.iter()) {
            let dequant = params.dequantize(q as i32);
            let error = orig - dequant;
            mse += error * error;
            max_abs_error = max_abs_error.max(error.abs());
        }

        mse /= original_data.len() as f32;

        // Compute variance for relative error
        let mean: f32 = original_data.iter().sum::<f32>() / original_data.len() as f32;
        let variance: f32 = original_data
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / original_data.len() as f32;

        let rel_error = if variance > 1e-10 {
            mse / variance
        } else {
            0.0
        };

        QuantErrorMetrics {
            mse,
            rel_error,
            max_abs_error,
        }
    }
}

/// Per-channel quantization for weight matrices.
pub fn quantize_per_channel(
    tensor: &Tensor,
    config: &DynamicQuantConfig,
    dim: usize,
) -> (Vec<Vec<i8>>, Vec<QuantParams>) {
    let lock = tensor.lock();
    let shape = lock.storage.shape();

    if dim >= shape.len() {
        panic!(
            "Dimension {} out of bounds for tensor with {} dimensions",
            dim,
            shape.len()
        );
    }

    // For simplicity, quantize along the specified dimension
    let channel_size: usize = shape.iter().skip(dim + 1).product();
    let num_channels: usize = shape[..=dim].iter().product::<usize>() / (channel_size + 1).max(1);

    let arr = match &lock.storage {
        TensorStorage::F32(a) => a.clone(),
        _ => lock.storage.to_f32_array(),
    };

    let data: Vec<f32> = arr.iter().copied().collect();

    let mut quantized_channels = Vec::new();
    let mut channel_params = Vec::new();

    for ch in 0..num_channels {
        let start = ch * channel_size;
        let end = (start + channel_size).min(data.len());
        let channel_data: Vec<f32> = data[start..end].to_vec();

        // Create temporary tensor for this channel
        let channel_tensor = Tensor::new(ndarray::Array::from_vec(channel_data).into_dyn(), false);
        let params = QuantParams::compute(&channel_tensor, config);

        let qmax = (1i64 << (config.bits - 1)) - 1;
        let channel_quantized: Vec<i8> = data[start..end]
            .iter()
            .map(|&val| params.clamp_value(val, qmax as f32) as i8)
            .collect();

        quantized_channels.push(channel_quantized);
        channel_params.push(params);
    }

    (quantized_channels, channel_params)
}

#[cfg(test)]
mod dynamic_quant_tests {
    use super::*;

    #[test]
    fn test_quant_config_int8() {
        let config = DynamicQuantConfig::int8();
        assert_eq!(config.bits, 8);
        assert!(config.symmetric);
    }

    #[test]
    fn test_quant_config_int4() {
        let config = DynamicQuantConfig::int4();
        assert_eq!(config.bits, 4);
        assert!(config.reduce_range);
    }

    #[test]
    fn test_quant_stats() {
        use ndarray::Array;
        let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]).into_dyn();
        let tensor = Tensor::new(data, false);
        let stats = QuantStats::from_tensor(&tensor);

        assert!((stats.min - 1.0).abs() < 1e-5);
        assert!((stats.max - 5.0).abs() < 1e-5);
        assert!(stats.mean > 0.0);
        assert!(stats.std > 0.0);
    }

    #[test]
    fn test_quant_params_symmetric() {
        use ndarray::Array;
        let data = Array::from_vec(vec![-4.0, -2.0, 0.0, 2.0, 4.0]).into_dyn();
        let tensor = Tensor::new(data, false);
        let config = DynamicQuantConfig::int8();
        let params = QuantParams::compute(&tensor, &config);

        assert!(params.symmetric);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        use ndarray::Array;
        let data = Array::from_vec(vec![1.0, 2.5, 3.5, 4.0]).into_dyn();
        let data_clone = data.clone();
        let tensor = Tensor::new(data, false);
        let config = DynamicQuantConfig::int8();

        let (quantized, params) = quantize_dynamic(&tensor, &config);
        let dequantized = dequantize_dynamic(&quantized, &params);

        let original: Vec<f32> = data_clone.iter().copied().collect();

        let dequant_lock = dequantized.lock();
        let dequant_arr = match &dequant_lock.storage {
            TensorStorage::F32(a) => a.clone(),
            _ => dequant_lock.storage.to_f32_array(),
        };
        let dequantized_vec: Vec<f32> = dequant_arr.iter().copied().collect();

        // Verify roundtrip error is small
        for (orig, dequant) in original.iter().zip(dequantized_vec.iter()) {
            assert!((orig - dequant).abs() < 0.1); // INT8 has limited precision
        }
    }

    #[test]
    fn test_quant_error_metrics() {
        let original_data = vec![1.0, 2.0, 3.0, 4.0];
        let quantized = vec![1i8, 2, 3, 4];
        let params = QuantParams {
            scale: 1.0,
            zero_point: 0,
            symmetric: true,
        };

        let metrics =
            QuantErrorMetrics::compute(&Tensor::ones(&[1]), &quantized, &params, &original_data);

        assert!(metrics.mse >= 0.0);
        assert!(metrics.rel_error >= 0.0);
        assert!(metrics.max_abs_error >= 0.0);
    }
}
