use ndarray::{ArrayD, IxDyn};
use tensor_engine::dtype::DType;
use tensor_engine::tensor::Tensor;

fn tensor(shape: &[usize], values: Vec<f32>, requires_grad: bool) -> Tensor {
    Tensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
        requires_grad,
    )
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn quantized_matmul_matches_dequantized_weights_and_backpropagates_to_input() {
    let input = tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0], true);
    let weights = tensor(&[2, 3], vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0], true);
    let quantized = weights.quantize_weights(DType::I8Rowwise, None).unwrap();
    assert!(!quantized.lock().requires_grad);

    let dequantized = quantized.to_f32_array();
    let output = input.quantized_matmul(&quantized);
    let expected = input
        .to_f32_array()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()
        .dot(
            &dequantized
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap(),
        );
    assert_close(
        output.to_f32_array().as_slice().unwrap(),
        expected.as_slice().unwrap(),
        1e-6,
    );

    let loss_weights = tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false);
    output.mul(&loss_weights).sum().backward();
    let expected_grad = loss_weights
        .to_f32_array()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()
        .dot(
            &dequantized
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap()
                .t(),
        );
    assert_close(
        input.lock().grad.clone().unwrap().as_slice().unwrap(),
        expected_grad.as_slice().unwrap(),
        1e-6,
    );
}

#[test]
fn quantize_weights_is_inference_only_and_rejects_unsupported_requests() {
    let weights = tensor(&[2, 2], vec![0.25, -0.5, 1.0, 2.0], true);
    let before = weights.to_f32_array();
    assert!(weights.quantize_weights(DType::F16, None).is_err());
    assert!(weights
        .quantize_weights(DType::I8Blockwise, Some(0))
        .is_err());
    assert_eq!(weights.dtype(), DType::F32);
    assert_eq!(weights.to_f32_array(), before);

    for dtype in [DType::I8, DType::I8Rowwise, DType::I8Blockwise] {
        let quantized = weights.quantize_weights(dtype, Some(2)).unwrap();
        assert_eq!(quantized.dtype(), dtype);
        assert!(!quantized.lock().requires_grad);
        quantized.validate_quantized_weights_2d().unwrap();
    }
}

#[test]
fn floating_astype_preserves_identity_gradient_and_integer_cast_detaches() {
    let input = tensor(&[3], vec![0.25, -1.5, 3.75], true);
    let cast = input.astype(DType::F8);
    assert_eq!(cast.dtype(), DType::F8);
    let weights = tensor(&[3], vec![2.0, 3.0, 4.0], false);
    cast.mul(&weights).sum().backward();
    assert_eq!(
        input.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[2.0, 3.0, 4.0]
    );

    let integer = input.astype(DType::I8);
    assert_eq!(integer.dtype(), DType::I8);
    assert!(!integer.lock().requires_grad);
}

#[test]
fn quantized_matmul_rejects_nonquantized_and_incompatible_weights() {
    let input = tensor(&[2, 3], vec![1.0; 6], false);
    let plain = tensor(&[3, 2], vec![1.0; 6], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        input.quantized_matmul(&plain)
    }))
    .is_err());

    let incompatible = tensor(&[4, 2], vec![1.0; 8], false)
        .quantize_weights(DType::I8, None)
        .unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        input.quantized_matmul(&incompatible)
    }))
    .is_err());
}
