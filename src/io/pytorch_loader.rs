//! Pure Rust PyTorch / TorchScript state-dict loader.
//! Reads `.pt` files by parsing protobuf-wire format — no `tch`, no libtorch, no Python.

use crate::dtype::DType;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};

#[derive(Debug, Clone, Copy, PartialEq)]
enum WireType {
    VarInt = 0,
    SixtyFourBit = 1,
    LengthDelimited = 2,
}

const WIRE_VARINT: u8 = WireType::VarInt as u8;
const WIRE_64BIT: u8 = WireType::SixtyFourBit as u8;
const WIRE_LENGTH_DELIMITED: u8 = WireType::LengthDelimited as u8;

fn read_varint(buf: &mut &[u8]) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if buf.is_empty() {
            return None;
        }
        let byte = (*buf)[0];
        *buf = &(*buf)[1..];
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift >= 63 {
            return None;
        }
    }
}

#[allow(dead_code)]
fn read_le_u32(buf: &mut &[u8]) -> Option<u32> {
    if buf.len() < 4 {
        return None;
    }
    let val = u32::from_le_bytes([(*buf)[0], (*buf)[1], (*buf)[2], (*buf)[3]]);
    *buf = &(*buf)[4..];
    Some(val)
}

fn read_le_u64(buf: &mut &[u8]) -> Option<u64> {
    if buf.len() < 8 {
        return None;
    }
    let val = u64::from_le_bytes([(*buf)[0], (*buf)[1], (*buf)[2], (*buf)[3], (*buf)[4], (*buf)[5], (*buf)[6], (*buf)[7]]);
    *buf = &(*buf)[8..];
    Some(val)
}

fn read_bytes<'a>(buf: &mut &'a [u8]) -> Option<&'a [u8]> {
    let len = read_varint(buf)? as usize;
    if buf.len() < len {
        return None;
    }
    let slice = &(*buf)[..len];
    *buf = &(*buf)[len..];
    Some(slice)
}

struct RawTensor {
    data: Vec<u8>,
    dtype: u64,
    shape: Vec<usize>,
}

fn half_f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let sign = (bits >> 15) & 1;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = bits & 0x3FF;
    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits((sign as u32) << 31);
        }
        let mut m = mantissa;
        let mut e: i32 = 1;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        f32::from_bits(((sign as u32) << 31) | (((e + 112i32) as u32) << 23) | ((m as u32) << 13))
    } else if exp == 0x1F {
        f32::from_bits(((sign as u32) << 31) | (0xFFu32 << 23) | (if mantissa != 0 { 1u32 << 22 } else { 0 }))
    } else {
        f32::from_bits(((sign as u32) << 31) | (((exp + 112i32) as u32) << 23) | ((mantissa as u32) << 13))
    }
}

fn half_bf16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    f32::from_bits((bits as u32) << 16)
}

fn try_parse_tensor_from_bytes(data: &[u8]) -> Option<RawTensor> {
    let mut inner = data;
    let mut dtype: u64 = 0;
    let mut shape: Vec<usize> = vec![];
    let mut data_bytes: Option<Vec<u8>> = None;
    while !inner.is_empty() {
        let tag2 = read_varint(&mut inner)?;
        let field_num = tag2 >> 3;
        let wire2 = (tag2 & 0x07) as u8;
        if field_num == 1 && wire2 == WIRE_VARINT {
            dtype = read_varint(&mut inner)?;
        } else if field_num == 2 && wire2 == WIRE_LENGTH_DELIMITED {
            let dim_bytes = read_bytes(&mut inner)?;
            if dim_bytes.len() % 4 == 0 && !dim_bytes.is_empty() {
                for chunk in dim_bytes.chunks_exact(4) {
                    shape.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize);
                }
            }
        } else if field_num == 3 && wire2 == WIRE_LENGTH_DELIMITED {
            data_bytes = Some(read_bytes(&mut inner)?.to_vec());
        } else {
            match wire2 {
                WIRE_VARINT => { let _ = read_varint(&mut inner); }
                WIRE_64BIT => { let _ = read_le_u64(&mut inner); }
                WIRE_LENGTH_DELIMITED => { let _ = read_bytes(&mut inner); }
                _ => break,
            }
        }
    }
    if shape.is_empty() || data_bytes.is_none() {
        return None;
    }
    Some(RawTensor { data: data_bytes.unwrap(), dtype, shape })
}

fn raw_tensor_to_f32(rt: &RawTensor) -> Result<ArrayD<f32>, String> {
    let numel: usize = rt.shape.iter().product();
    let values: Vec<f32> = match rt.dtype {
        1 => {
            if rt.data.len() != numel * 4 {
                return Err(format!("float32 data length {} mismatch (expected {})", rt.data.len(), numel * 4));
            }
            let mut out = Vec::with_capacity(numel);
            for chunk in rt.data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            out
        }
        2 => {
            if rt.data.len() != numel * 4 {
                return Err(format!("int32 data length {} mismatch (expected {})", rt.data.len(), numel * 4));
            }
            rt.data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i32 as f32).collect()
        }
        3 => {
            if rt.data.len() != numel * 8 {
                return Err(format!("int64 data length {} mismatch (expected {})", rt.data.len(), numel * 8));
            }
            rt.data.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32).collect()
        }
        4 => {
            if rt.data.len() != numel * 2 {
                return Err(format!("float16 data length {} mismatch (expected {})", rt.data.len(), numel * 2));
            }
            rt.data.chunks_exact(2).map(|c| half_f16_to_f32(c[0], c[1])).collect()
        }
        5 => {
            if rt.data.len() != numel * 2 {
                return Err(format!("bfloat16 data length {} mismatch (expected {})", rt.data.len(), numel * 2));
            }
            rt.data.chunks_exact(2).map(|c| half_bf16_to_f32(c[0], c[1])).collect()
        }
        0 => {
            if !rt.data.len().is_multiple_of(4) {
                return Err(format!("unknown dtype {} with non-float32-aligned data", rt.dtype));
            }
            let mut out = Vec::with_capacity(rt.data.len() / 4);
            for chunk in rt.data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            out
        }
        _ => return Err(format!("unsupported torch dtype {}", rt.dtype)),
    };
    if values.len() != numel {
        return Err(format!("tensor data length {} does not match computed size {}", values.len(), numel));
    }
    ArrayD::from_shape_vec(IxDyn(rt.shape.as_slice()), values).map_err(|e| e.to_string())
}

fn parse_torchscript_file(path: &str) -> Result<HashMap<String, Tensor>, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    parse_torchscript_protobuf(&mut &buf[..], "", &mut map)?;
    if map.is_empty() {
        Err(format!("No parameters found in {}. Use examples/convert_torch_to_safetensors.py", path))
    } else {
        Ok(map)
    }
}

enum TensorReadResult<'a> {
    Named(String, RawTensor),
    Nested(&'a [u8], String),
}

fn try_read_tensor<'a>(data: &mut &'a [u8]) -> Result<TensorReadResult<'a>, ()> {
    let tag = read_varint(data).ok_or(())?;
    let outer_wire = (tag & 0x07) as u8;
    if outer_wire != WIRE_LENGTH_DELIMITED {
        return Err(());
    }
    let payload_len = read_varint(data).ok_or(())? as usize;
    if (*data).len() < payload_len {
        return Err(());
    }
    let payload = &(*data)[..payload_len];
    *data = &(*data)[payload_len..];
    let mut inner = payload;
    let mut name: Option<String> = None;
    while !inner.is_empty() {
        let tag2 = read_varint(&mut inner).ok_or(())?;
        let _field_num = tag2 >> 3;
        let wire2 = (tag2 & 0x07) as u8;
        if wire2 == WIRE_LENGTH_DELIMITED {
            if let Some(bytes) = read_bytes(&mut inner) {
                if let Ok(s) = String::from_utf8(bytes.to_vec()) {
                    if s.contains('.') || s.ends_with(".weight") || s.ends_with(".bias") {
                        name = Some(s);
                    } else if !s.is_empty() && name.is_none() {
                        name = Some(s.clone());
                    }
                }
            }
        } else if wire2 == WIRE_VARINT {
            let _ = read_varint(&mut inner);
        } else if wire2 == WIRE_64BIT {
            let _ = read_le_u64(&mut inner);
        } else {
            break;
        }
    }
    if let Some(ref n) = name {
        if let Some(rt_parsed) = try_parse_tensor_from_bytes(payload) {
            return Ok(TensorReadResult::Named(n.clone(), rt_parsed));
        }
    }
    if !payload.is_empty() && (name.is_some() || payload.len() > 10) {
        return Ok(TensorReadResult::Nested(payload, name.unwrap_or_default()));
    }
    Err(())
}

fn parse_torchscript_protobuf(data: &mut &[u8], prefix: &str, map: &mut HashMap<String, Tensor>) -> Result<(), String> {
    let original_len = data.len();
    while !data.is_empty() && data.len() < original_len - 4 {
        if let Ok(tensor_result) = try_read_tensor(data) {
            match tensor_result {
                TensorReadResult::Named(name, rt) => {
                    let key = if prefix.is_empty() { name.clone() } else { format!("{}.{}", prefix, name) };
                    if let Ok(arr) = raw_tensor_to_f32(&rt) {
                        map.insert(key, Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32));
                    }
                }
                TensorReadResult::Nested(inner_data, inner_prefix) => {
                    let _ = parse_torchscript_protobuf(&mut &inner_data[..], &inner_prefix, map);
                }
            }
        } else {
            break;
        }
    }
    Ok(())
}

fn try_safetensors_fallback(path: &str) -> Result<HashMap<String, Tensor>, String> {
    #[cfg(feature = "safe_tensors")]
    {
        let sf_path = format!("{}.safetensors", path.trim_end_matches(".pt"));
        if std::path::Path::new(&sf_path).exists() {
            log::info!("No torch tensors in {}; falling back to safetensors: {}", path, sf_path);
            let bytes = std::fs::read(&sf_path).map_err(|e| format!("Cannot read {}: {}", sf_path, e))?;
            return crate::io::safetensors_loader::load_safetensors_from_bytes(&bytes, false);
        }
    }
    Err(format!("No parameters found in {}. No safetensors fallback available. Use examples/convert_torch_to_safetensors.py", path))
}

/// Load a PyTorch/TorchScript state dict from a `.pt` file into a HashMap of named tensors.
/// Uses pure Rust protobuf parsing — no `tch`, no libtorch, no Python dependency.
pub fn load_torch_state_dict_to_map(path: &str, _transpose_two_dim_weights: bool) -> Result<HashMap<String, Tensor>, String> {
    match parse_torchscript_file(path) {
        Ok(map) => {
            if !map.is_empty() {
                log::info!("Loaded {} parameters from {}", map.len(), path);
                return Ok(map);
            }
        }
        Err(e) => {
            log::debug!("TorchScript parse failed for {}: {}", path, e);
        }
    }
    try_safetensors_fallback(path)
}

/// Normalize a parameter key by stripping common prefixes.
pub fn normalize_key(key: &str) -> String {
    if key.starts_with("module.") {
        return key[7..].to_string();
    }
    if key.starts_with("model.") {
        return key[6..].to_string();
    }
    key.to_string()
}

/// Transpose 2D weight tensors if needed.
pub fn maybe_transpose_weight(tensor: Tensor, _key: &str, transpose_two_dim_weights: bool) -> Tensor {
    if !transpose_two_dim_weights || !_key.ends_with(".weight") {
        return tensor;
    }
    let shape = tensor.shape();
    if shape.len() == 2 {
        log::debug!("Transposing weight tensor {} with shape {:?}", _key, shape);
        let vec = tensor.to_vec();
        let rows = shape[0];
        let cols = shape[1];
        let mut transposed = Vec::with_capacity(vec.len());
        for c in 0..cols {
            for r in 0..rows {
                transposed.push(vec[r * cols + c]);
            }
        }
        Tensor::new_with_dtype(
            ndarray::ArrayD::<f32>::from_shape_vec(IxDyn(&[cols, rows]), transposed).unwrap().into_dyn(),
            true, DType::F32,
        )
    } else {
        tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_key_strips_module_prefix() {
        assert_eq!(normalize_key("module.linear.weight"), "linear.weight");
        assert_eq!(normalize_key("model.linear.weight"), "linear.weight");
        assert_eq!(normalize_key("linear.weight"), "linear.weight");
    }

    #[test]
    fn test_f16_to_f32_basic() {
        let result = half_f16_to_f32(0x00, 0x3C);
        assert!((result - 1.0).abs() < 1e-5, "f16 +1.0 failed: got {}", result);
        let result = half_f16_to_f32(0x00, 0xBC);
        assert!((result - (-1.0)).abs() < 1e-5, "f16 -1.0 failed: got {}", result);
        let result = half_f16_to_f32(0x00, 0x38);
        assert!((result - 0.5).abs() < 1e-5, "f16 +0.5 failed: got {}", result);
        let result = half_f16_to_f32(0x00, 0x00);
        assert!(result == 0.0 || result == -0.0, "f16 zero failed: got {}", result);
        let result = half_f16_to_f32(0x00, 0x7C);
        assert!(result.is_infinite() && result > 0.0, "f16 inf failed: got {}", result);
        let result = half_f16_to_f32(0x00, 0x7E);
        assert!(result.is_nan(), "f16 nan failed: got {}", result);
    }

    #[test]
    fn test_bf16_to_f32_basic() {
        let result = half_bf16_to_f32(0x80, 0x3F);
        assert!((result - 1.0).abs() < 1e-4, "bf16 +1.0 failed: got {}", result);
        let result = half_bf16_to_f32(0x00, 0x3F); // bf16 le: +0.5 (bits=0x3F00)
        assert!((result - 0.5).abs() < 1e-4, "bf16 +0.5 failed: got {}", result);
        let result = half_bf16_to_f32(0x00, 0x00);
        assert!(result == 0.0 || result == -0.0, "bf16 zero failed: got {}", result);
    }

    #[test]
    fn test_varint_parsing() {
        let mut buf: &[u8] = &[0x01];
        assert_eq!(read_varint(&mut buf), Some(1));
        assert!(buf.is_empty());
        let mut buf: &[u8] = &[0xAC, 0x02];
        assert_eq!(read_varint(&mut buf), Some(300));
    }

    #[test]
    fn test_missing_file_returns_error() {
        let res = load_torch_state_dict_to_map("nonexistent_file.pt", false);
        assert!(res.is_err());
        if let Err(msg) = res {
            assert!(!msg.is_empty(), "Error message should not be empty");
        } else {
            panic!("Expected error for missing path");
        }
    }

    #[test]
    fn test_raw_tensor_to_f32_float32() {
        let data: Vec<u8> = vec![0, 0, 128, 63]; // f32 le: +1.0 (0x3F800000)
        let rt = RawTensor { data, dtype: 1, shape: vec![1] };
        let arr = raw_tensor_to_f32(&rt).expect("should parse");
        assert_eq!(arr.shape(), &[1]);
        assert!((arr[[0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_raw_tensor_to_f32_int32() {
        let data: Vec<u8> = vec![42, 0, 0, 0]; // i32 le: 42
        let rt = RawTensor { data, dtype: 2, shape: vec![1] };
        let arr = raw_tensor_to_f32(&rt).expect("should parse");
        assert!((arr[[0]] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn test_raw_tensor_to_f32_float16() {
        let data: Vec<u8> = vec![0, 0x3C]; // f16 le: +1.0
        let rt = RawTensor { data, dtype: 4, shape: vec![1] };
        let arr = raw_tensor_to_f32(&rt).expect("should parse");
        assert!((arr[[0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_raw_tensor_to_f32_bfloat16() {
        let data: Vec<u8> = vec![0x80, 0x3F]; // bf16 le: +1.0 (bits=0x3F80)
        let rt = RawTensor { data, dtype: 5, shape: vec![1] };
        let arr = raw_tensor_to_f32(&rt).expect("should parse");
        assert!((arr[[0]] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_raw_tensor_to_f32_multidim() {
        let mut data = Vec::new();
        for v in &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let rt = RawTensor { data, dtype: 1, shape: vec![2, 3] };
        let arr = raw_tensor_to_f32(&rt).expect("should parse");
        assert_eq!(arr.shape(), &[2, 3]);
        for (i, expected) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
            let row = i / 3;
            let col = i % 3;
            assert!((arr[[row, col]] - *expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_raw_tensor_to_f32_dtype_mismatch() {
        let data: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0]; // only 2 ints
        let rt = RawTensor { data, dtype: 2, shape: vec![2, 3] };
        assert!(raw_tensor_to_f32(&rt).is_err());
    }

    #[test]
    fn test_raw_tensor_to_f32_unsupported_dtype() {
        let rt = RawTensor { data: vec![0, 0, 0, 0], dtype: 99, shape: vec![1] };
        assert!(raw_tensor_to_f32(&rt).is_err());
    }

    #[test]
    fn test_read_le_u32() {
        let mut buf: &[u8] = &[42, 0, 0, 0];
        assert_eq!(read_le_u32(&mut buf), Some(42));
        assert!(buf.is_empty());
        let mut buf: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(read_le_u32(&mut buf), Some(u32::MAX));
    }

    #[test]
    fn test_read_le_u64() {
        let mut buf: &[u8] = &[1, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(read_le_u64(&mut buf), Some(1));
        assert!(buf.is_empty());
    }

    #[test]
    fn test_read_bytes_varint() {
        let mut buf: &[u8] = &[0x03, b'a', b'b', b'c'];
        assert_eq!(read_bytes(&mut buf), Some(b"abc".as_ref()));
    }

    #[test]
    fn test_read_bytes_short() {
        let mut buf: &[u8] = &[0x05, b'a', b'b'];
        assert_eq!(read_bytes(&mut buf), None);
    }

    #[test]
    fn test_normalize_key_preserves_non_prefixed() {
        assert_eq!(normalize_key("embedding.weight"), "embedding.weight");
        assert_eq!(normalize_key("encoder.layer.0.attn.q_proj.bias"), "encoder.layer.0.attn.q_proj.bias");
    }

    #[test]
    fn test_maybe_transpose_weight_2d() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = ArrayD::<f32>::from_shape_vec(IxDyn(&[3, 4]), data.clone()).unwrap();
        let tensor = Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32);
        let result = maybe_transpose_weight(tensor, "linear.weight", true);
        assert_eq!(result.shape().len(), 2, "should remain 2D");
    }

    #[test]
    fn test_maybe_transpose_weight_3d_no_op() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let arr = ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 3, 4]), data.clone()).unwrap();
        let tensor = Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32);
        let result = maybe_transpose_weight(tensor, "conv.weight", true);
        assert_eq!(result.shape().len(), 3, "should remain 3D");
    }
}

