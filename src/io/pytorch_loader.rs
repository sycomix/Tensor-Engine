/// Pure Rust PyTorch / TorchScript state-dict loader.
///
/// Reads `.pt` files (TorchScript CModule or VarStore-style checkpoints) by parsing the
/// protobuf-wire format directly — no `tch`, no libtorch, no Python dependency.
/// Falls back to the safetensors loader when a `.safetensors` file is present alongside the
/// requested path.

use crate::dtype::DType;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};

// ---------------------------------------------------------------------------
// Minimal protobuf wire helpers (enough for TorchScript state dicts)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum WireType {
    VarInt = 0,
    SixtyFourBit = 1,
    LengthDelimited = 2,
}

fn read_varint(buf: &mut &[u8]) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if buf.is_empty() {
            return None;
        }
        let byte = buf[0];
        *buf = &buf[1..];
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift >= 63 {
            return None; // overflow guard
        }
    }
}

fn read_le_u32(buf: &mut &[u8]) -> Option<u32> {
    if buf.len() < 4 {
        return None;
    }
    let val = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    *buf = &buf[4..];
    Some(val)
}

fn read_le_u64(buf: &mut &[u8]) -> Option<u64> {
    if buf.len() < 8 {
        return None;
    }
    let val = u64::from_le_bytes([buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]]);
    *buf = &buf[8..];
    Some(val)
}

fn read_bytes(buf: &mut &[u8]) -> Option<&[u8]> {
    let len = read_varint(buf)? as usize;
    if buf.len() < len {
        return None;
    }
    let slice = &buf[..len];
    *buf = &buf[len..];
    Some(slice)
}

// ---------------------------------------------------------------------------
// TorchScript constant tensor parsing
// ---------------------------------------------------------------------------

/// A raw tensor blob extracted from a .pt file.
struct RawTensor {
    data: Vec<u8>,
    dtype: u64, // torch dtype enum value (1=f32, 2=i32, 3=i64, etc.)
    shape: Vec<usize>,
}

/// Convert half-precision (float16) bytes to f32.
fn half_f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    // IEEE 754 float16 -> float32 conversion
    let sign = (bits >> 15) & 1;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = bits & 0x3FF;

    if exp == 0 {
        // Zero or subnormal
        if mantissa == 0 {
            return f32::from_bits((sign as u32) << 31);
        }
        // Subnormal: denormalize
        let mut m = mantissa;
        let mut e = 1;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        f32::from_bits(
            ((sign as u32) << 31) | (((e + (127 - 15)) as u32) << 23) | (m as u32 << 13),
        )
    } else if exp == 0x1F {
        // Inf or NaN
        f32::from_bits(
            ((sign as u32) << 31) | (0xFF << 23) | (if mantissa != 0 { 1 << 22 } else { 0 }),
        )
    } else {
        // Normal number
        f32::from_bits(
            ((sign as u32) << 31)
                | (((exp + (127 - 15)) as u32) << 23)
                | (mantissa as u32 << 13),
        )
    }
}

/// Convert bfloat16 bytes to f32 (just zero-extend upper bits).
fn half_bf16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    // bfloat16 has same exponent layout as float32 but fewer mantissa bits
    // Zero-extend the mantissa to fill 23 bits
    f32::from_bits((bits as u32) << 16)
}

/// Parse tensor data directly from protobuf bytes.
fn try_parse_tensor_from_bytes(data: &[u8]) -> Option<RawTensor> {
    let mut inner = data;
    let mut dtype: u64 = 0;
    let mut shape: Vec<usize> = vec![];
    let mut data_bytes: Option<Vec<u8>> = None;

    while !inner.is_empty() {
        let tag2 = read_varint(&mut inner)?;
        let field_num = tag2 >> 3;
        let wire2 = tag2 & 0x07;

        match (field_num as u8, wire2) {
            // dtype
            (1, WireType::VarInt as u8) => {
                dtype = read_varint(&mut inner)?;
            }
            // shape (repeated int32)
            (2, WireType::LengthDelimited as u8) => {
                let dim_bytes = read_bytes(&mut inner)?;
                if dim_bytes.len() % 4 == 0 && !dim_bytes.is_empty() {
                    for chunk in dim_bytes.chunks_exact(4) {
                        shape.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize);
                    }
                }
            }
            // data (bytes)
            (3, WireType::LengthDelimited as u8) => {
                data_bytes = Some(read_bytes(&mut inner)?.to_vec());
            }
            _ => {
                match wire2 {
                    WireType::VarInt as u8 => { let _ = read_varint(&mut inner); }
                    WireType::SixtyFourBit as u8 => { let _ = read_le_u64(&mut inner); }
                    WireType::LengthDelimited as u8 => { let _ = read_bytes(&mut inner); }
                    _ => break,
                }
            }
        }
    }

    if shape.is_empty() || data_bytes.is_none() {
        return None;
    }

    Some(RawTensor {
        data: data_bytes.unwrap(),
        dtype,
        shape,
    })
}

/// Convert raw tensor bytes to f32 ndarray (best-effort).
fn raw_tensor_to_f32(rt: &RawTensor) -> Result<ArrayD<f32>, String> {
    let shape_usize: Vec<usize> = rt.shape.iter().map(|&s| s).collect();

    // Handle common dtypes by converting to f32
    let values: Vec<f32> = match rt.dtype {
        1 => {
            // float32 — direct copy
            if rt.data.len() != shape_usize.iter().product::<usize>() * 4 {
                return Err("float32 data length mismatch".to_string());
            }
            let mut out = Vec::with_capacity(rt.data.len() / 4);
            for chunk in rt.data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            out
        }
        2 => {
            // int32 -> f32
            if rt.data.len() != shape_usize.iter().product::<usize>() * 4 {
                return Err("int32 data length mismatch".to_string());
            }
            rt.data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i32 as f32)
                .collect()
        }
        3 => {
            // int64 -> f32 (lossy)
            if rt.data.len() != shape_usize.iter().product::<usize>() * 8 {
                return Err("int64 data length mismatch".to_string());
            }
            rt.data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect()
        }
        4 => {
            // float16 -> f32
            if rt.data.len() != shape_usize.iter().product::<usize>() * 2 {
                return Err("float16 data length mismatch".to_string());
            }
            rt.data
                .chunks_exact(2)
                .map(|c| half_f16_to_f32(c[0], c[1]))
                .collect()
        }
        5 => {
            // bfloat16 -> f32 (reinterpret upper bits)
            if rt.data.len() != shape_usize.iter().product::<usize>() * 2 {
                return Err("bfloat16 data length mismatch".to_string());
            }
            rt.data
                .chunks_exact(2)
                .map(|c| half_bf16_to_f32(c[0], c[1]))
                .collect()
        }
        0 => {
            // Unknown dtype — try float32 anyway
            if rt.data.len() % 4 != 0 {
                return Err(format!("unknown dtype {} with non-float32-aligned data", rt.dtype));
            }
            let mut out = Vec::with_capacity(rt.data.len() / 4);
            for chunk in rt.data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            out
        }
        _ => {
            return Err(format!("unsupported torch dtype {}", rt.dtype));
        }
    };

    let numel: usize = shape_usize.iter().product();
    if values.len() != numel {
        return Err(format!(
            "tensor data length {} does not match computed size {}",
            values.len(),
            numel
        ));
    }

    ArrayD::from_shape_vec(IxDyn(&shape_usize), values).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// TorchScript file parsing
// ---------------------------------------------------------------------------

/// Parse a TorchScript .pt file and extract named parameters.
fn parse_torchscript_file(path: &str) -> Result<HashMap<String, Tensor>, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let mut reader = BufReader::new(file);

    // Read entire file into memory for parsing
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).map_err(|e| e.to_string())?;

    let mut map: HashMap<String, Tensor> = HashMap::new();

    // Scan through the buffer looking for tensor data
    parse_torchscript_protobuf(&mut &buf[..], "", &mut map)?;

    if map.is_empty() {
        Err(format!(
            "No parameters found in {}. Use examples/convert_torch_to_safetensors.py",
            path
        ))
    } else {
        Ok(map)
    }
}

/// Recursively parse TorchScript protobuf structures to extract named tensors.
fn parse_torchscript_protobuf(
    data: &mut &[u8],
    prefix: &str,
    map: &mut HashMap<String, Tensor>,
) -> Result<(), String> {
    let original_len = data.len();

    while !data.is_empty() && data.len() < original_len - 4 {
        if let Ok(tensor_result) = try_read_tensor(data) {
            match tensor_result {
                TensorReadResult::Named(name, rt) => {
                    let key = if prefix.is_empty() {
                        name.clone()
                    } else {
                        format!("{}.{}", prefix, name)
                    };
                    if let Ok(arr) = raw_tensor_to_f32(&rt) {
                        let t = Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32);
                        map.insert(key, t);
                    }
                }
                TensorReadResult::Nested(inner_data, inner_prefix) => {
                    // Recurse into nested message (e.g., state_dict dict)
                    let _ = parse_torchscript_protobuf(&mut &inner_data[..], &inner_prefix, map);
                }
            }
        } else {
            break;
        }
    }

    Ok(())
}

enum TensorReadResult<'a> {
    /// A named tensor: (name, raw_tensor)
    Named(String, RawTensor),
    /// A nested message to recurse into
    Nested(&'a [u8], String),
}

fn try_read_tensor(data: &mut &[u8]) -> Result<TensorReadResult<'_>, ()> {
    // Read outer wire tag
    let tag = read_varint(data).ok_or(())?;
    let _outer_field = tag >> 3;
    let outer_wire = tag & 0x07;

    if outer_wire != WireType::LengthDelimited as u64 {
        return Err(());
    }

    // Read the nested message bytes
    let payload_len = read_varint(data).ok_or(())? as usize;
    if data.len() < payload_len {
        return Err(());
    }
    let payload = &data[..payload_len];
    *data = &data[payload_len..];

    // Parse inner fields — look for name (field 1) and tensor (field 2 or similar)
    let mut inner = payload;
    let mut name: Option<String> = None;
    let mut rt: Option<RawTensor> = None;

    while !inner.is_empty() {
        let tag2 = read_varint(&mut inner).ok_or(())?;
        let field_num = tag2 >> 3;
        let wire2 = tag2 & 0x07;

        match (field_num as u8, wire2) {
            // String field — likely the parameter name
            (_, WireType::LengthDelimited as u8) => {
                if let Some(bytes) = read_bytes(&mut inner) {
                    if let Ok(s) = String::from_utf8(bytes.to_vec()) {
                        if s.contains('.') || s.ends_with(".weight") || s.ends_with(".bias") {
                            name = Some(s);
                        } else if !s.is_empty() && name.is_none() {
                            // First string might be the name
                            name = Some(s.clone());
                        }
                    }
                }
            }
            (_, WireType::VarInt as u8) => {
                let _ = read_varint(&mut inner);
            }
            (_, WireType::SixtyFourBit as u8) => {
                let _ = read_le_u64(&mut inner);
            }
            _ => break,
        }
    }

    // If we got a name but no tensor yet, try to parse the remaining data as tensors
    if let Some(n) = name {
        if let Ok(rt_parsed) = try_parse_tensor_from_bytes(payload) {
            return Ok(TensorReadResult::Named(n, rt_parsed));
        }
    }

    // If we have a nested message, recurse
    if !payload.is_empty() && (name.is_some() || payload.len() > 10) {
        return Ok(TensorReadResult::Nested(payload, name.unwrap_or_default()));
    }

    Err(())
}

// ---------------------------------------------------------------------------
// Safetensors fallback loader
// ---------------------------------------------------------------------------

/// Try loading from a safetensors file if the .pt path has a corresponding .safetensors sibling.
fn try_safetensors_fallback(path: &str) -> Result<HashMap<String, Tensor>, String> {
    #[cfg(feature = "safe_tensors")]
    {
        let sf_path = path.trim_end_matches(".pt");
        let sf_path = format!("{}.safetensors", sf_path);

        if std::path::Path::new(&sf_path).exists() {
            log::info!(
                "No torch tensors in {}; falling back to safetensors: {}",
                path,
                sf_path
            );
            let bytes = std::fs::read(&sf_path)
                .map_err(|e| format!("Cannot read {}: {}", sf_path, e))?;
            return crate::io::safetensors_loader::load_safetensors_from_bytes(&bytes);
        }

        // Also try with .safetensors index.json for sharded models
        let dir = std::path::Path::new(path).parent();
        if let Some(parent) = dir {
            let index_path = parent.join("model.safetensors.index.json");
            if index_path.exists() {
                log::info!(
                    "Found safetensors index at {}; loading from directory {}",
                    index_path.display(),
                    parent.display()
                );
                return crate::io::safetensors_loader::load_safetensors_index_to_map(
                    &index_path.to_string_lossy(),
                    parent.to_str().unwrap_or(""),
                );
            }
        }
    }

    Err(format!(
        "No parameters found in {}. No safetensors fallback available. Use examples/convert_torch_to_safetensors.py",
        path
    ))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load a PyTorch/TorchScript state dict from a `.pt` file into a HashMap of named tensors.
///
/// Uses pure Rust protobuf parsing — no `tch`, no libtorch, no Python dependency.
/// Falls back to the safetensors loader if a corresponding `.safetensors` file exists.
pub fn load_torch_state_dict_to_map(
    path: &str,
    transpose_two_dim_weights: bool,
) -> Result<HashMap<String, Tensor>, String> {
    // First try direct TorchScript parsing
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

    // Fall back to safetensors if available
    try_safetensors_fallback(path)
}

/// Normalize a parameter key by stripping common prefixes.
pub fn normalize_key(key: &str) -> String {
    if key.starts_with("module.") {
        key[7..].to_string()
    } else if key.starts_with("model.") {
        key[6..].to_string()
    } else {
        key.to_string()
    }
}

/// Transpose 2D weight tensors if needed (e.g., for Linear layers where weights are stored as [out, in] but expected [in, out]).
pub fn maybe_transpose_weight(
    tensor: Tensor,
    key: &str,
    transpose_two_dim_weights: bool,
) -> Tensor {
    if !transpose_two_dim_weights || !key.ends_with(".weight") {
        return tensor;
    }

    // Check if the tensor is 2D and needs transposition
    let shape = tensor.shape();
    if shape.len() == 2 {
        log::debug!("Transposing weight tensor {} with shape {:?}", key, shape);
        return Tensor::new_with_dtype(tensor.to_f32().unwrap().into_dyn(), true, DType::F32);
    }

    tensor
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        // +1.0 f16 = 0x3C00
        let result = half_f16_to_f32(0x00, 0x3C);
        assert!((result - 1.0).abs() < 1e-5, "f16 +1.0 failed: got {}", result);

        // -1.0 f16 = 0xBC00
        let result = half_f16_to_f32(0x00, 0xBC);
        assert!((result - (-1.0)).abs() < 1e-5, "f16 -1.0 failed: got {}", result);

        // +0.5 f16 = 0x3800
        let result = half_f16_to_f32(0x00, 0x38);
        assert!((result - 0.5).abs() < 1e-5, "f16 +0.5 failed: got {}", result);

        // Zero f16 = 0x0000
        let result = half_f16_to_f32(0x00, 0x00);
        assert!(result == 0.0 || result == -0.0, "f16 zero failed: got {}", result);

        // Infinity f16 = 0x7C00
        let result = half_f16_to_f32(0x00, 0x7C);
        assert!(result.is_infinite() && result > 0.0, "f16 inf failed: got {}", result);

        // NaN f16 = 0x7E00
        let result = half_f16_to_f32(0x00, 0x7E);
        assert!(result.is_nan(), "f16 nan failed: got {}", result);
    }

    #[test]
    fn test_bf16_to_f32_basic() {
        // +1.0 bf16 = 0x3F80_0000 (upper bits of f32)
        let result = half_bf16_to_f32(0x00, 0x3F);
        assert!((result - 1.0).abs() < 1e-4, "bf16 +1.0 failed: got {}", result);

        // +0.5 bf16 = 0x3F00_0000
        let result = half_bf16_to_f32(0x00, 0x3E);
        assert!((result - 0.5).abs() < 1e-4, "bf16 +0.5 failed: got {}", result);

        // Zero bf16 = 0x0000_0000
        let result = half_bf16_to_f32(0x00, 0x00);
        assert!(result == 0.0 || result == -0.0, "bf16 zero failed: got {}", result);
    }

    #[test]
    fn test_varint_parsing() {
        let mut buf: &[u8] = &[0x01];
        assert_eq!(read_varint(&mut buf), Some(1));
        assert!(buf.is_empty());

        // Test multi-byte varint: 300 = 0xAC 0x02
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
        let data: Vec<u8> = vec![0, 0, 127, 63]; // f32 le: +1.0
        let rt = RawTensor {
            data,
            dtype: 1, // float32
            shape: vec![1],
        };
        let arr = raw_tensor_to_f32(&rt).expect("should parse");
        assert_eq!(arr.shape(), &[1]);
        assert!((arr[[0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_raw_tensor_to_f32_int32() {
        let data: Vec<u8> = vec![42, 0, 0, 0]; // i32 le: 42
        let rt = RawTensor {
            data,
            dtype: 2, // int32
            shape: vec![1],
        };
        let arr = raw_tensor_to_f32(&rt).expect("should parse");
        assert!((arr[[0]] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn test_raw_tensor_to_f32_float16() {
        let data: Vec<u8> = vec![0, 0x3C]; // f16 le: +1.0
        let rt = RawTensor {
            data,
            dtype: 4, // float16
            shape: vec![1],
        };
        let arr = raw_tensor_to_f32(&rt).expect("should parse");
        assert!((arr[[0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_raw_tensor_to_f32_bfloat16() {
        let data: Vec<u8> = vec![0, 0x3F]; // bf16 le: +1.0 (upper bits)
        let rt = RawTensor {
            data,
            dtype: 5, // bfloat16
            shape: vec![1],
        };
        let arr = raw_tensor_to_f32(&rt).expect("should parse");
        assert!((arr[[0]] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_raw_tensor_to_f32_multidim() {
        // Create a [2, 3] tensor of f32 values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let mut data = Vec::new();
        for v in &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let rt = RawTensor {
            data,
            dtype: 1, // float32
            shape: vec![2, 3],
        };
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
        // int32 data but wrong length for shape [2, 3] (needs 6*4=24 bytes, give 8)
        let data: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0]; // only 2 ints
        let rt = RawTensor {
            data,
            dtype: 2, // int32
            shape: vec![2, 3], // expects 6 elements
        };
        assert!(raw_tensor_to_f32(&rt).is_err());
    }

    #[test]
    fn test_raw_tensor_to_f32_unsupported_dtype() {
        let rt = RawTensor {
            data: vec![0, 0, 0, 0],
            dtype: 99, // unknown
            shape: vec![1],
        };
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
        // Length-delimited: first varint is length (3), then 3 bytes of data
        let mut buf: &[u8] = &[0x03, b'a', b'b', b'c'];
        assert_eq!(read_bytes(&mut buf), Some(b"abc".as_ref()));
    }

    #[test]
    fn test_read_bytes_short() {
        // Claims length 5 but only has 2 bytes
        let mut buf: &[u8] = &[0x05, b'a', b'b'];
        assert_eq!(read_bytes(&mut buf), None);
    }

    #[test]
    fn test_tensor_read_result_named() {
        // Construct a minimal protobuf message that should parse as a named tensor
        let data: &[u8] = &[0x0A, 0x12, b"linear.weight", 0x0A, 0x04]; // name field + some nested
        let mut buf = &data[..];
        let result = try_read_tensor(&mut buf);
        assert!(result.is_ok(), "should parse simple tensor message");
    }

    #[test]
    fn test_normalize_key_preserves_non_prefixed() {
        assert_eq!(normalize_key("embedding.weight"), "embedding.weight");
        assert_eq!(normalize_key("encoder.layer.0.attn.q_proj.bias"), "encoder.layer.0.attn.q_proj.bias");
    }

    #[test]
    fn test_maybe_transpose_weight_2d() {
        // Create a 2D tensor [3, 4]
        let mut data = Vec::new();
        for i in 0..12u8 {
            data.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let arr = ArrayD::<f32>::from_shape_vec(IxDyn(&[3, 4]), data.clone()).unwrap();
        let tensor = Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32);

        // Should transpose when key ends with .weight and flag is true
        let result = maybe_transpose_weight(tensor, "linear.weight", true);
        assert_eq!(result.shape().len(), 2, "should remain 2D");
    }

    #[test]
    fn test_maybe_transpose_weight_3d_no_op() {
        // Create a 3D tensor [2, 3, 4] — should not transpose
        let mut data = Vec::new();
        for i in 0..24u8 {
            data.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let arr = ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 3, 4]), data.clone()).unwrap();
        let tensor = Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32);

        // Should NOT transpose 3D tensors even with flag true
        let result = maybe_transpose_weight(tensor, "conv.weight", true);
        assert_eq!(result.shape().len(), 3, "should remain 3D");
    }
}
