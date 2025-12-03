#[cfg(feature = "safe_tensors")]
use tensor_engine::io::safetensors_loader::load_safetensors_from_bytes;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use tensor_engine::io::safetensors_loader::parse_safetensors_tensor;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use safetensors::tensor::Dtype as STDtype; // alias to avoid name conflicts
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use half::{f16, bf16};

#[cfg(feature = "safe_tensors")]
#[test]
fn test_safetensors_loader_invalid_bytes() {
    let bytes: Vec<u8> = vec![0, 1, 2, 3, 4];
    let res = load_safetensors_from_bytes(&bytes, true);
    assert!(res.is_err());
}

#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
#[test]
fn test_safetensors_parse_f32_and_f16() {
    // f32 test: 1.0, 2.0
    let bytes_f32: Vec<u8> = vec![1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
    let t = parse_safetensors_tensor(STDtype::F32, vec![2usize], &bytes_f32, false, Some("test.weight")).unwrap();
    let arr = t.lock().storage.to_f32_array();
    assert_eq!(arr.as_slice().unwrap(), &[1.0f32, 2.0f32]);

    // f16 test: use half::f16 to produce bytes slice
    let v = vec![f16::from_f32(1.0f32), f16::from_f32(2.0f32)];
    let mut bytes_f16: Vec<u8> = Vec::new();
    for x in v.iter() {
        bytes_f16.extend(&x.to_bits().to_le_bytes());
    }
    let t2 = parse_safetensors_tensor(STDtype::F16, vec![2usize], &bytes_f16, false, Some("test.weight")).unwrap();
    let arr2 = t2.lock().storage.to_f32_array();
    assert_eq!(arr2.as_slice().unwrap(), &[1.0f32, 2.0f32]);
    // bf16 test
    let v_bf = vec![bf16::from_f32(1.0f32), bf16::from_f32(2.0f32)];
    let mut bytes_bf16: Vec<u8> = Vec::new();
    for x in v_bf.iter() {
        bytes_bf16.extend(&x.to_bits().to_le_bytes());
    }
    let t3 = parse_safetensors_tensor(STDtype::BF16, vec![2usize], &bytes_bf16, false, Some("test.weight")).unwrap();
    let arr3 = t3.lock().storage.to_f32_array();
    assert_eq!(arr3.as_slice().unwrap(), &[1.0f32, 2.0f32]);
}
