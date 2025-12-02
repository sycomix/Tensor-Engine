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
}

impl DType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F8 => "f8",
            DType::I8 => "int8",
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
        ArrayD::from_shape_vec(IxDyn(src.shape()), v).expect("to_f16: shape mismatch")
    }

    /// Convert from ArrayD<f16> to ArrayD<f32>
    pub fn from_f16(src: &ArrayD<f16>) -> ArrayD<f32> {
        let v: Vec<f32> = src.iter().map(|x| f32::from(*x)).collect();
        ArrayD::from_shape_vec(IxDyn(src.shape()), v).expect("from_f16: shape mismatch")
    }

    /// Convert from ArrayD<f32> to ArrayD<bf16>
    pub fn to_bf16(src: &ArrayD<f32>) -> ArrayD<bf16> {
        let v: Vec<bf16> = src.iter().map(|x| bf16::from_f32(*x)).collect();
        ArrayD::from_shape_vec(IxDyn(src.shape()), v).expect("to_bf16: shape mismatch")
    }

    /// Convert from ArrayD<bf16> to ArrayD<f32>
    pub fn from_bf16(src: &ArrayD<bf16>) -> ArrayD<f32> {
        let v: Vec<f32> = src.iter().map(|x| f32::from(*x)).collect();
        ArrayD::from_shape_vec(IxDyn(src.shape()), v).expect("from_bf16: shape mismatch")
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
        ArrayD::from_shape_vec(IxDyn(shape), v).expect("dequantize_from_f8: shape mismatch")
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
        ArrayD::from_shape_vec(IxDyn(shape), v).expect("dequantize_from_i8: shape mismatch")
    }
}
