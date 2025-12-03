#[cfg(feature = "safe_tensors")]
use crate::dtype::DType;
#[cfg(feature = "safe_tensors")]
use crate::tensor::Tensor;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use half::{bf16, f16};
#[cfg(feature = "safe_tensors")]
use ndarray::{Array, IxDyn};
#[cfg(feature = "safe_tensors")]
use safetensors::tensor::Dtype;
#[cfg(feature = "safe_tensors")]
use safetensors::SafeTensors;
#[cfg(feature = "safe_tensors")]
use std::collections::HashMap;

#[cfg(feature = "safe_tensors")]
pub fn load_safetensors_from_bytes(
    bytes: &[u8],
    transpose_two_dim_weights: bool,
) -> Result<HashMap<String, Tensor>, String> {
    // Parse SafeTensors file from bytes and map to Tensor instances
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| format!("safetensors deserialize error: {}", e))?;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for (key, tensor) in st.tensors() {
        // gather dtype -- currently only support f32
        match tensor.dtype() {
            Dtype::F32 => {
                let shape: Vec<usize> = tensor.shape().iter().map(|d| *d as usize).collect();
                // convert bytes to f32 vec by reading the raw bytes
                let bytes = tensor.data();
                let mut data = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(chunk);
                    data.push(f32::from_le_bytes(b));
                }
                let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                    .map_err(|e| format!("ndarray shape creation error: {}", e))?;
                let out =
                    if transpose_two_dim_weights && shape.len() == 2 && key.ends_with(".weight") {
                        // transpose [out, in] -> [in, out]
                        let mut mat = arr
                            .into_dimensionality::<ndarray::Ix2>()
                            .map_err(|e| format!("transpose dim error: {}", e))?;
                        mat = mat.reversed_axes();
                        Tensor::new(mat.into_dyn(), false)
                    } else {
                        Tensor::new(arr.into_dyn(), false)
                    };
                map.insert(key.clone(), out);
            }
            Dtype::F16 => {
                #[cfg(not(feature = "multi_precision"))]
                return Err(
                    "F16 dtype requested but crate not built with 'multi_precision' feature"
                        .to_string(),
                );
                #[cfg(feature = "multi_precision")]
                {
                    let shape: Vec<usize> = tensor.shape().iter().map(|d| *d as usize).collect();
                    if tensor.data().len() % 2 != 0 {
                        return Err("Invalid byte length for f16 tensor".to_string());
                    }
                    let bytes_slice = tensor.data();
                    let mut data: Vec<f32> = Vec::with_capacity(bytes_slice.len() / 2);
                    for chunk in bytes_slice.chunks_exact(2) {
                        let mut b = [0u8; 2];
                        b.copy_from_slice(chunk);
                        let bits = u16::from_le_bytes(b);
                        let val = f16::from_bits(bits);
                        data.push(f32::from(val));
                    }
                    let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                        .map_err(|e| format!("ndarray shape creation error: {}", e))?;
                    let out = if transpose_two_dim_weights
                        && shape.len() == 2
                        && key.ends_with(".weight")
                    {
                        let mut mat = arr
                            .into_dimensionality::<ndarray::Ix2>()
                            .map_err(|e| format!("transpose dim error: {}", e))?;
                        mat = mat.reversed_axes();
                        Tensor::new_with_dtype(mat.into_dyn(), false, DType::F16)
                    } else {
                        Tensor::new_with_dtype(arr.into_dyn(), false, DType::F16)
                    };
                    map.insert(key.clone(), out);
                }
            }
            Dtype::BF16 => {
                #[cfg(not(feature = "multi_precision"))]
                return Err(
                    "BF16 dtype requested but crate not built with 'multi_precision' feature"
                        .to_string(),
                );
                #[cfg(feature = "multi_precision")]
                {
                    let shape: Vec<usize> = tensor.shape().iter().map(|d| *d as usize).collect();
                    if tensor.data().len() % 2 != 0 {
                        return Err("Invalid byte length for bf16 tensor".to_string());
                    }
                    let bytes_slice = tensor.data();
                    let mut data: Vec<f32> = Vec::with_capacity(bytes_slice.len() / 2);
                    for chunk in bytes_slice.chunks_exact(2) {
                        let mut b = [0u8; 2];
                        b.copy_from_slice(chunk);
                        let bits = u16::from_le_bytes(b);
                        let val = bf16::from_bits(bits);
                        data.push(f32::from(val));
                    }
                    let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                        .map_err(|e| format!("ndarray shape creation error: {}", e))?;
                    let out = if transpose_two_dim_weights
                        && shape.len() == 2
                        && key.ends_with(".weight")
                    {
                        let mut mat = arr
                            .into_dimensionality::<ndarray::Ix2>()
                            .map_err(|e| format!("transpose dim error: {}", e))?;
                        mat = mat.reversed_axes();
                        Tensor::new_with_dtype(mat.into_dyn(), false, DType::BF16)
                    } else {
                        Tensor::new_with_dtype(arr.into_dyn(), false, DType::BF16)
                    };
                    map.insert(key.clone(), out);
                }
            }
            _ => return Err(format!("Unsupported dtype: {:?}", tensor.dtype())),
        }
    }
    Ok(map)
}

#[cfg(feature = "safe_tensors")]
/// Parse a single tensor from raw bytes and return a `Tensor` instance.
/// This helper is useful for unit tests that only want to validate dtype conversions
/// without constructing a full SafeTensors archive.
pub fn parse_safetensors_tensor(
    dtype: Dtype,
    shape: Vec<usize>,
    data_bytes: &[u8],
    transpose_two_dim_weights: bool,
    key: Option<&str>,
) -> Result<Tensor, String> {
    match dtype {
        Dtype::F32 => {
            if data_bytes.len() % 4 != 0 {
                return Err("Invalid byte length for f32 tensor".to_string());
            }
            let mut data = Vec::with_capacity(data_bytes.len() / 4);
            for chunk in data_bytes.chunks_exact(4) {
                let mut b = [0u8; 4];
                b.copy_from_slice(chunk);
                data.push(f32::from_le_bytes(b));
            }
            let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                .map_err(|e| format!("ndarray shape creation error: {}", e))?;
            let out = if transpose_two_dim_weights
                && shape.len() == 2
                && key.unwrap_or("").ends_with(".weight")
            {
                let mut mat = arr
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| format!("transpose dim error: {}", e))?;
                mat = mat.reversed_axes();
                Tensor::new_with_dtype(mat.into_dyn(), false, DType::F32)
            } else {
                Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32)
            };
            Ok(out)
        }
        Dtype::F16 => {
            #[cfg(not(feature = "multi_precision"))]
            return Err(
                "F16 dtype requested but crate not built with 'multi_precision' feature"
                    .to_string(),
            );
            #[cfg(feature = "multi_precision")]
            {
                if data_bytes.len() % 2 != 0 {
                    return Err("Invalid byte length for f16 tensor".to_string());
                }
                let mut data: Vec<f32> = Vec::with_capacity(data_bytes.len() / 2);
                for chunk in data_bytes.chunks_exact(2) {
                    let mut b = [0u8; 2];
                    b.copy_from_slice(chunk);
                    let bits = u16::from_le_bytes(b);
                    let val = f16::from_bits(bits);
                    data.push(f32::from(val));
                }
                let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                    .map_err(|e| format!("ndarray shape creation error: {}", e))?;
                let out = if transpose_two_dim_weights
                    && shape.len() == 2
                    && key.unwrap_or("").ends_with(".weight")
                {
                    let mut mat = arr
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|e| format!("transpose dim error: {}", e))?;
                    mat = mat.reversed_axes();
                    Tensor::new_with_dtype(mat.into_dyn(), false, DType::F16)
                } else {
                    Tensor::new_with_dtype(arr.into_dyn(), false, DType::F16)
                };
                Ok(out)
            }
        }
        Dtype::BF16 => {
            #[cfg(not(feature = "multi_precision"))]
            return Err(
                "BF16 dtype requested but crate not built with 'multi_precision' feature"
                    .to_string(),
            );
            #[cfg(feature = "multi_precision")]
            {
                if data_bytes.len() % 2 != 0 {
                    return Err("Invalid byte length for bf16 tensor".to_string());
                }
                let mut data: Vec<f32> = Vec::with_capacity(data_bytes.len() / 2);
                for chunk in data_bytes.chunks_exact(2) {
                    let mut b = [0u8; 2];
                    b.copy_from_slice(chunk);
                    let bits = u16::from_le_bytes(b);
                    let val = bf16::from_bits(bits);
                    data.push(f32::from(val));
                }
                let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                    .map_err(|e| format!("ndarray shape creation error: {}", e))?;
                let out = if transpose_two_dim_weights
                    && shape.len() == 2
                    && key.unwrap_or("").ends_with(".weight")
                {
                    let mut mat = arr
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|e| format!("transpose dim error: {}", e))?;
                    mat = mat.reversed_axes();
                    Tensor::new_with_dtype(mat.into_dyn(), false, DType::BF16)
                } else {
                    Tensor::new_with_dtype(arr.into_dyn(), false, DType::BF16)
                };
                Ok(out)
            }
        }
        _ => Err(format!("Unsupported dtype: {:?}", dtype)),
    }
}

/// Apply a state dict mapping (names -> Tensors) to a module by delegating
/// to the module's `load_state_dict` implementation. This will return an error
/// if the module fails to load a parameter.
pub fn apply_state_dict_to_module(
    module: &mut dyn crate::nn::Module,
    state: &HashMap<String, Tensor>,
    root: &str,
) -> Result<(), String> {
    module.load_state_dict(state, root)
}
