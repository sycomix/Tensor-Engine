use tensor_engine::tensor::Tensor;
use tensor_engine::dtype::DType;
use ndarray::Array;

#[test]
fn test_astype_f8_roundtrip_and_dtype() {
    // Create a tensor with simple data
    let data = Array::from_shape_vec((2, 2), vec![1.0f32, -2.0f32, 3.0f32, -4.0f32])
        .expect("shape creation");
    let t = Tensor::new_with_dtype(data.into_dyn(), true, DType::F32);
    assert_eq!(t.dtype(), DType::F32);

    let t8 = t.astype(DType::F8);
    assert_eq!(t8.dtype(), DType::F8);
    // Ensure shape preserved
    assert_eq!(t.lock().storage.shape(), t8.lock().storage.shape());
}

#[test]
fn test_quantize_weights_rowwise_and_blockwise() {
    use ndarray::arr2;
    let data = arr2(&[[1.0f32, -2.0f32, 3.0f32, 0.5f32], [0.1f32, -0.2f32, 0.3f32, -0.4f32], [2.0f32, -1.0f32, 0.5f32, -0.25f32]]).into_dyn();
    let t = Tensor::new(data.clone(), false);
    let tr = t.quantize_weights(DType::I8Rowwise, None);
    assert_eq!(tr.dtype(), DType::I8Rowwise);
    let deq = tr.lock().storage.to_f32_array();
    assert_eq!(deq.shape(), t.lock().storage.shape());
    // check that dequantized values are close to original
    for (a, b) in data.iter().zip(deq.iter()) {
        assert!((a - b).abs() < 1e-2);
    }
    let tb = t.quantize_weights(DType::I8Blockwise, Some(2));
    assert_eq!(tb.dtype(), DType::I8Blockwise);
    let deqb = tb.lock().storage.to_f32_array();
    for (a, b) in data.iter().zip(deqb.iter()) {
        assert!((a - b).abs() < 1e-2);
    }
}

#[cfg(feature = "dtype_f16")]
#[test]
fn test_astype_f16_roundtrip_and_loss() {
    let data = Array::from_shape_vec((3,), vec![0.1f32, -0.25f32, 1.234f32]).expect("shape");
    let t = Tensor::new_with_dtype(data.into_dyn(), true, DType::F32);
    let t16 = t.astype(DType::F16);
    assert_eq!(t16.dtype(), DType::F16);
    assert_eq!(t.lock().storage.shape(), t16.lock().storage.shape());
    // Ensure numeric difference due to conversion exists (round-trip)
    assert!(t.lock().storage.to_f32_array() != t16.lock().storage.to_f32_array());
}

#[cfg(feature = "dtype_bf16")]
#[test]
fn test_astype_bf16_roundtrip_and_loss() {
    let data = Array::from_shape_vec((3,), vec![0.1234567f32, -0.5f32, 2.3456789f32]).expect("shape");
    let t = Tensor::new_with_dtype(data.into_dyn(), true, DType::F32);
    let tb = t.astype(DType::BF16);
    assert_eq!(tb.dtype(), DType::BF16);
    assert_eq!(t.lock().storage.shape(), tb.lock().storage.shape());
    assert!(t.lock().storage.to_f32_array() != tb.lock().storage.to_f32_array());
}
