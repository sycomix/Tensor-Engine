#[cfg(feature = "dtype_f16")]
use half::{bf16, f16};
use ndarray::ArrayD;
use ndarray::ArrayViewD;
#[allow(unused_imports)]
use ndarray::IxDyn;
use std::fmt;
use std::fmt::Debug;

/// Supported DType for Tensors. MVP keeps data in f32 storage but tracks dtype so
/// we can convert/serialize and later implement optimized storage paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    F8,
    I8,
    I8Rowwise,
    I8Blockwise,
}

impl DType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F8 => "f8",
            DType::I8 => "int8",
            DType::I8Rowwise => "i8_rowwise",
            DType::I8Blockwise => "i8_blockwise",
        }
    }

    pub fn parse(s: &str) -> Option<DType> {
        match s.to_lowercase().as_str() {
            "f32" => Some(DType::F32),
            "float32" => Some(DType::F32),
            "f16" => Some(DType::F16),
            "float16" => Some(DType::F16),
            "bf16" => Some(DType::BF16),
            "bfloat16" => Some(DType::BF16),
            "f8" => Some(DType::F8),
            "int8" | "i8" => Some(DType::I8),
            "i8_rowwise" | "i8rowwise" => Some(DType::I8Rowwise),
            "i8_blockwise" | "i8blockwise" => Some(DType::I8Blockwise),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Tensor storage variants that hold the concrete representation.
/// For MVP: F16/BF16 require features `dtype_f16`/`dtype_bf16`.
#[derive(Clone, Debug)]
pub enum TensorStorage {
    F32(ArrayD<f32>),
    #[cfg(feature = "dtype_f16")]
    F16(ArrayD<f16>),
    #[cfg(feature = "dtype_bf16")]
    BF16(ArrayD<bf16>),
    /// Emulated f8 encoding (bytes) plus per-tensor scale and shape.
    F8(Vec<u8>, f32, Vec<usize>),
    I8(Vec<i8>, f32, Vec<usize>),
    /// Per-row scales for I8 quantized storage. Data stored as i8 flattened row-major with one scale per row.
    I8Rowwise(Vec<i8>, Vec<f32>, Vec<usize>),
    /// Blockwise I8 quantization with per-block scales. block_size is the right-dimension block size used.
    I8Blockwise(Vec<i8>, Vec<f32>, Vec<usize>, usize),
}

impl TensorStorage {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            TensorStorage::F32(arr) => arr.shape().to_vec(),
            #[cfg(feature = "dtype_f16")]
            TensorStorage::F16(arr) => arr.shape().to_vec(),
            #[cfg(feature = "dtype_bf16")]
            TensorStorage::BF16(arr) => arr.shape().to_vec(),
            TensorStorage::F8(_, _, shape) => shape.clone(),
            TensorStorage::I8(_, _, shape) => shape.clone(),
            TensorStorage::I8Rowwise(_, _, shape) => shape.clone(),
            TensorStorage::I8Blockwise(_, _, shape, _block) => shape.clone(),
        }
    }

    pub fn to_f32_array(&self) -> ArrayD<f32> {
        match self {
            TensorStorage::F32(arr) => arr.clone(),
            #[cfg(feature = "dtype_f16")]
            TensorStorage::F16(arr) => crate::dtype::f16_helpers::from_f16(arr),
            #[cfg(feature = "dtype_bf16")]
            TensorStorage::BF16(arr) => crate::dtype::f16_helpers::from_bf16(arr),
            TensorStorage::F8(bytes, scale, shape) => {
                crate::dtype::f8::dequantize_from_f8(bytes, *scale, shape)
            }
            TensorStorage::I8(bytes, scale, shape) => {
                crate::dtype::int8::dequantize_from_i8(bytes, *scale, shape)
            }
            TensorStorage::I8Rowwise(bytes, scales, shape) => {
                crate::dtype::int8::dequantize_from_i8_rowwise(bytes, scales, shape)
            }
            TensorStorage::I8Blockwise(bytes, scales, shape, block_size) => {
                crate::dtype::int8::dequantize_from_i8_blockwise(bytes, scales, shape, *block_size)
            }
        }
    }

    /// If storage is F32, return an ArrayViewD to avoid cloning; otherwise None.
    pub fn as_f32_view(&self) -> Option<ArrayViewD<'_, f32>> {
        match self {
            TensorStorage::F32(arr) => Some(arr.view()),
            _ => None,
        }
    }

    pub fn from_f32_array(arr: &ArrayD<f32>, dtype: DType) -> Self {
        match dtype {
            DType::F32 => TensorStorage::F32(arr.clone()),
            DType::F16 => {
                #[cfg(feature = "dtype_f16")]
                {
                    TensorStorage::F16(crate::dtype::f16_helpers::to_f16(arr))
                }
                #[cfg(not(feature = "dtype_f16"))]
                {
                    TensorStorage::F32(arr.clone())
                }
            }
            DType::BF16 => {
                #[cfg(feature = "dtype_bf16")]
                {
                    TensorStorage::BF16(crate::dtype::f16_helpers::to_bf16(arr))
                }
                #[cfg(not(feature = "dtype_bf16"))]
                {
                    TensorStorage::F32(arr.clone())
                }
            }
            DType::F8 => {
                let (bytes, scale) = crate::dtype::f8::quantize_to_f8(arr);
                TensorStorage::F8(bytes, scale, arr.shape().to_vec())
            }
            DType::I8 => {
                let (bytes, scale) = crate::dtype::int8::quantize_to_i8(arr);
                TensorStorage::I8(bytes, scale, arr.shape().to_vec())
            }
            DType::I8Rowwise => {
                let (bytes, scales) = match crate::dtype::int8::quantize_rowwise_to_i8(arr) {
                    Ok((b, s)) => (b, s),
                    Err(e) => {
                        log::error!("I8Rowwise quantization failed: {}", e);
                        return TensorStorage::F32(arr.clone());
                    }
                };
                TensorStorage::I8Rowwise(bytes, scales, arr.shape().to_vec())
            }
            DType::I8Blockwise => {
                // Default block size heuristics: use 32
                let block_size = 32usize;
                let (bytes, scales) = match crate::dtype::int8::quantize_blockwise_to_i8(arr, block_size) {
                    Ok((b, s)) => (b, s),
                    Err(e) => {
                        log::error!("I8Blockwise quantization failed: {}", e);
                        return TensorStorage::F32(arr.clone());
                    }
                };
                TensorStorage::I8Blockwise(bytes, scales, arr.shape().to_vec(), block_size)
            }
        }
    }
}

#[cfg(feature = "dtype_f16")]
pub mod f16_helpers {
    use half::{bf16, f16};
    use ndarray::{ArrayD, IxDyn};

    /// Convert from ArrayD<f32> to ArrayD<f16>
    pub fn to_f16(src: &ArrayD<f32>) -> ArrayD<f16> {
        let v: Vec<f16> = src.iter().map(|x| f16::from_f32(*x)).collect();
        match ArrayD::from_shape_vec(IxDyn(src.shape()), v) {
            Ok(a) => a,
            Err(e) => { log::error!("to_f16: shape mismatch when building ArrayD: {}", e); ArrayD::zeros(IxDyn(src.shape())) }
        }
    }

    /// Convert from ArrayD<f16> to ArrayD<f32>
    pub fn from_f16(src: &ArrayD<f16>) -> ArrayD<f32> {
        let v: Vec<f32> = src.iter().map(|x| f32::from(*x)).collect();
        match ArrayD::from_shape_vec(IxDyn(src.shape()), v) {
            Ok(a) => a,
            Err(e) => { log::error!("from_f16: shape mismatch when building ArrayD: {}", e); ArrayD::zeros(IxDyn(src.shape())) }
        }
    }

    /// Convert from ArrayD<f32> to ArrayD<bf16>
    pub fn to_bf16(src: &ArrayD<f32>) -> ArrayD<bf16> {
        let v: Vec<bf16> = src.iter().map(|x| bf16::from_f32(*x)).collect();
        match ArrayD::from_shape_vec(IxDyn(src.shape()), v) {
            Ok(a) => a,
            Err(e) => { log::error!("to_bf16: shape mismatch when building ArrayD: {}", e); ArrayD::zeros(IxDyn(src.shape())) }
        }
    }

    /// Convert from ArrayD<bf16> to ArrayD<f32>
    pub fn from_bf16(src: &ArrayD<bf16>) -> ArrayD<f32> {
        let v: Vec<f32> = src.iter().map(|x| f32::from(*x)).collect();
        match ArrayD::from_shape_vec(IxDyn(src.shape()), v) {
            Ok(a) => a,
            Err(e) => { log::error!("from_bf16: shape mismatch when building ArrayD: {}", e); ArrayD::zeros(IxDyn(src.shape())) }
        }
    }
}

/// Simple symmetric quantization to 8-bit float emulation (MVP).
/// This method is intentionally lightweight; it is meant for testing and I/O emulation.
pub mod f8 {
    use ndarray::{ArrayD, IxDyn};

    /// Emulate f8 quantize: scale by max absolute and map to u8 with symmetric quantization.
    pub fn quantize_to_f8(src: &ArrayD<f32>) -> (Vec<u8>, f32) {
        let max = src.iter().cloned().fold(0f32, |a, b| a.max(b.abs()));
        let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
        let data: Vec<u8> = src
            .iter()
            .map(|v| {
                // clamp to [-127,127]
                let q = (v / scale).round();
                let q = q.max(-127.0).min(127.0) as i32;
                (q as i8 as i32 - i8::MIN as i32) as u8
            })
            .collect();
        (data, scale)
    }

    /// Dequantize from f8 quantized bytes + scale back to f32 ArrayD
    pub fn dequantize_from_f8(data: &[u8], scale: f32, shape: &[usize]) -> ArrayD<f32> {
        let v: Vec<f32> = data
            .iter()
            .map(|b| {
                let signed = *b as i32 + i8::MIN as i32;
                signed as f32 * scale
            })
            .collect();
        match ArrayD::from_shape_vec(IxDyn(shape), v) {
            Ok(a) => a,
            Err(e) => { log::error!("dequantize_from_f8: shape mismatch when building ArrayD: {}", e); ArrayD::zeros(IxDyn(shape)) }
        }
    }
}

/// Simple INT8 quantization helpers
pub mod int8 {
    use ndarray::{ArrayD, IxDyn};

    pub fn quantize_to_i8(src: &ArrayD<f32>) -> (Vec<i8>, f32) {
        let max = src.iter().cloned().fold(0f32, |a, b| a.max(b.abs()));
        let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
        let data: Vec<i8> = src
            .iter()
            .map(|v| {
                let q = (v / scale).round();
                let q = q.max(-127.0).min(127.0) as i32;
                q as i8
            })
            .collect();
        (data, scale)
    }

    pub fn dequantize_from_i8(data: &[i8], scale: f32, shape: &[usize]) -> ArrayD<f32> {
        let v: Vec<f32> = data.iter().map(|b| *b as f32 * scale).collect();
        match ArrayD::from_shape_vec(IxDyn(shape), v) {
            Ok(a) => a,
            Err(e) => { log::error!("dequantize_from_i8: shape mismatch when building ArrayD: {}", e); ArrayD::zeros(IxDyn(shape)) }
        }
    }

    pub fn quantize_rowwise_to_i8(src: &ArrayD<f32>) -> Result<(Vec<i8>, Vec<f32>), String> {
        let shape = src.shape();
        if shape.len() != 2 {
            return Err("quantize_rowwise_to_i8 expects 2D matrix".to_string());
        }
        let rows = shape[0];
        let cols = shape[1];
        let mut bytes: Vec<i8> = Vec::with_capacity(rows * cols);
        let mut scales: Vec<f32> = Vec::with_capacity(rows);
        for r in 0..rows {
            let row = src.slice(ndarray::s![r, ..]);
            let max = row.iter().cloned().fold(0f32, |a, b| a.max(b.abs()));
            let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
            scales.push(scale);
            for c in 0..cols {
                let v = src[[r, c]];
                let q = (v / scale).round();
                let q = q.max(-127.0).min(127.0) as i32;
                bytes.push(q as i8);
            }
        }
        Ok((bytes, scales))
    }

    pub fn dequantize_from_i8_rowwise(data: &[i8], scales: &[f32], shape: &[usize]) -> ArrayD<f32> {
        let rows = shape[0];
        let cols = shape[1];
        let mut v: Vec<f32> = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let scale = scales[r];
            for c in 0..cols {
                let idx = r * cols + c;
                let q = data[idx] as f32;
                v.push(q * scale);
            }
        }
        match ArrayD::from_shape_vec(IxDyn(shape), v) {
            Ok(a) => a,
            Err(e) => { log::error!("dequantize_from_i8_rowwise: shape mismatch when building ArrayD: {}", e); ArrayD::zeros(IxDyn(shape)) }
        }
    }

    pub fn quantize_blockwise_to_i8(src: &ArrayD<f32>, block_size: usize) -> Result<(Vec<i8>, Vec<f32>), String> {
        let shape = src.shape();
        if shape.len() != 2 {
            return Err("quantize_blockwise_to_i8 expects 2D matrix".to_string());
        }
        let rows = shape[0];
        let cols = shape[1];
        let blocks_per_row = (cols + block_size - 1) / block_size;
        let mut bytes: Vec<i8> = Vec::with_capacity(rows * cols);
        let mut scales: Vec<f32> = Vec::with_capacity(rows * blocks_per_row);
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let start = b * block_size;
                let end = ((b + 1) * block_size).min(cols);
                let mut max = 0f32;
                for c in start..end {
                    let v = src[[r, c]];
                    max = max.max(v.abs());
                }
                let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
                scales.push(scale);
                for c in start..end {
                    let v = src[[r, c]];
                    let q = (v / scale).round();
                    let q = q.max(-127.0).min(127.0) as i32;
                    bytes.push(q as i8);
                }
            }
        }
        Ok((bytes, scales))
    }

    pub fn dequantize_from_i8_blockwise(data: &[i8], scales: &[f32], shape: &[usize], block_size: usize) -> ArrayD<f32> {
        let rows = shape[0];
        let cols = shape[1];
        let blocks_per_row = (cols + block_size - 1) / block_size;
        let mut v: Vec<f32> = Vec::with_capacity(rows * cols);
        let mut idx = 0usize;
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let scale = scales[r * blocks_per_row + b];
                let start = b * block_size;
                let end = ((b + 1) * block_size).min(cols);
                for _c in start..end {
                    let q = data[idx] as f32;
                    v.push(q * scale);
                    idx += 1;
                }
            }
        }
        match ArrayD::from_shape_vec(IxDyn(shape), v) {
            Ok(a) => a,
            Err(e) => { log::error!("dequantize_from_i8_blockwise: shape mismatch when building ArrayD: {}", e); ArrayD::zeros(IxDyn(shape)) }
        }
    }
}
