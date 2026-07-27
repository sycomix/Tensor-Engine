use ndarray::{ArrayD, IxDyn};
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

fn rotate_pairs(values: &[f32], angles: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(values.len());
    for (pair, &angle) in values.chunks_exact(2).zip(angles) {
        let (sin, cos) = angle.sin_cos();
        result.push(pair[0] * cos - pair[1] * sin);
        result.push(pair[1] * cos + pair[0] * sin);
    }
    result
}

fn inverse_rotate_pairs(values: &[f32], angles: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(values.len());
    for (pair, &angle) in values.chunks_exact(2).zip(angles) {
        let (sin, cos) = angle.sin_cos();
        result.push(pair[0] * cos + pair[1] * sin);
        result.push(-pair[0] * sin + pair[1] * cos);
    }
    result
}

#[test]
fn rope_honors_theta_scale_offset_and_uses_inverse_rotation_in_backward() {
    let input = tensor(
        &[1, 2, 4],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        true,
    );
    let output = input.rope(1, 16.0, 2.0, 3);

    let mut expected = rotate_pairs(&[1.0, 2.0, 3.0, 4.0], &[1.5, 0.375]);
    expected.extend(rotate_pairs(&[5.0, 6.0, 7.0, 8.0], &[2.0, 0.5]));
    assert_close(output.to_f32_array().as_slice().unwrap(), &expected, 1e-6);

    let weights = tensor(
        &[1, 2, 4],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        false,
    );
    output.mul(&weights).sum().backward();
    let mut expected_grad = inverse_rotate_pairs(&[1.0, 2.0, 3.0, 4.0], &[1.5, 0.375]);
    expected_grad.extend(inverse_rotate_pairs(&[5.0, 6.0, 7.0, 8.0], &[2.0, 0.5]));
    assert_close(
        input.lock().grad.clone().unwrap().as_slice().unwrap(),
        &expected_grad,
        1e-6,
    );
}

#[test]
fn rope_uses_the_penultimate_axis_as_sequence_for_rank_four() {
    let input = tensor(&[1, 1, 2, 2], vec![1.0, 0.0, 1.0, 0.0], false);
    let output = input.rope(1, 10.0, 1.0, 0);
    assert_close(
        output.to_f32_array().as_slice().unwrap(),
        &[1.0, 0.0, 1.0f32.cos(), 1.0f32.sin()],
        1e-6,
    );
}

#[test]
fn swiglu_forward_and_weighted_gradient_are_exact() {
    let input = tensor(&[1, 4], vec![0.0, 1.0, 2.0, 3.0], true);
    let output = input.swiglu();
    let sigmoid = 1.0 / (1.0 + (-1.0f32).exp());
    assert_close(
        output.to_f32_array().as_slice().unwrap(),
        &[0.0, 3.0 * sigmoid],
        1e-6,
    );

    let weights = tensor(&[1, 2], vec![4.0, 5.0], false);
    output.mul(&weights).sum().backward();
    let swish_prime = sigmoid + sigmoid * (1.0 - sigmoid);
    assert_close(
        input.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[4.0, 15.0 * swish_prime, 0.0, 5.0 * sigmoid],
        1e-6,
    );
}

#[test]
fn rope_and_swiglu_reject_invalid_public_inputs() {
    let input = tensor(&[1, 4], vec![1.0; 4], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        input.rope(0, 10_000.0, 1.0, 0)
    }))
    .is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        input.rope(1, 10_000.0, 0.0, 0)
    }))
    .is_err());

    let odd = tensor(&[2, 3], vec![1.0; 6], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| odd.swiglu())).is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        odd.rope(1, 10_000.0, 1.0, 0)
    }))
    .is_err());
}
