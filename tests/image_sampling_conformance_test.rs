use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

fn tensor(values: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor {
    Tensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
        requires_grad,
    )
}

fn assert_values(actual: &ArrayD<f32>, expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[test]
fn interpolate_nearest_forward_and_backward_are_exact() {
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], true);
    let output = input.interpolate((4, 4), "nearest".to_string(), false);

    assert_values(
        &output.to_f32_array(),
        &[
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ],
        0.0,
    );

    output.backward();
    assert_values(input.lock().grad.as_ref().unwrap(), &[4.0; 4], 0.0);
}

#[test]
fn interpolate_bilinear_align_corners_forward_and_backward_match_weights() {
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], true);
    let output = input.interpolate((3, 3), "bilinear".to_string(), true);

    assert_values(
        &output.to_f32_array(),
        &[1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0],
        1e-6,
    );

    output.backward();
    assert_values(input.lock().grad.as_ref().unwrap(), &[2.25; 4], 1e-6);
}

#[test]
fn grid_sample_bilinear_propagates_input_and_grid_gradients() {
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], true);
    let grid = tensor(vec![0.0, 0.0], &[1, 1, 1, 2], true);
    let output = input.grid_sample(&grid, "bilinear".to_string(), "zeros".to_string(), true);

    assert_values(&output.to_f32_array(), &[2.5], 1e-6);
    output.backward();
    assert_values(
        input.lock().grad.as_ref().unwrap(),
        &[0.25, 0.25, 0.25, 0.25],
        1e-6,
    );
    assert_values(grid.lock().grad.as_ref().unwrap(), &[0.5, 1.0], 1e-6);
}

#[test]
fn grid_sample_nearest_has_zero_grid_gradient() {
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], true);
    let grid = tensor(vec![1.0, -1.0], &[1, 1, 1, 2], true);
    let output = input.grid_sample(&grid, "nearest".to_string(), "zeros".to_string(), true);

    assert_values(&output.to_f32_array(), &[2.0], 0.0);
    output.backward();
    assert_values(
        input.lock().grad.as_ref().unwrap(),
        &[0.0, 1.0, 0.0, 0.0],
        0.0,
    );
    assert_values(grid.lock().grad.as_ref().unwrap(), &[0.0, 0.0], 0.0);
}

#[test]
fn grid_sample_distinguishes_zero_and_border_padding() {
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], false);
    let grid = tensor(vec![2.0, 2.0], &[1, 1, 1, 2], false);

    let zeros = input.grid_sample(&grid, "bilinear".to_string(), "zeros".to_string(), true);
    let border = input.grid_sample(&grid, "bilinear".to_string(), "border".to_string(), true);

    assert_values(&zeros.to_f32_array(), &[1.0], 1e-6);
    assert_values(&border.to_f32_array(), &[4.0], 1e-6);
}

#[test]
#[should_panic(expected = "grid_sample padding_mode must be 'zeros' or 'border'")]
fn grid_sample_rejects_unimplemented_reflection_padding() {
    let input = tensor(vec![1.0], &[1, 1, 1, 1], false);
    let grid = tensor(vec![0.0, 0.0], &[1, 1, 1, 2], false);
    let _ = input.grid_sample(
        &grid,
        "bilinear".to_string(),
        "reflection".to_string(),
        false,
    );
}
