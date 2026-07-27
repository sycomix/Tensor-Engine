use ndarray::{arr2, Array, ArrayD};
use tensor_engine::tensor::Tensor;

fn assert_close(actual: &ArrayD<f32>, expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[test]
fn rectangular_matmul_forward_and_backward_match_matrix_calculus() {
    let left = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(), true);
    let right = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).into_dyn(), true);
    let output = left.matmul(&right);

    assert_eq!(output.shape(), vec![2, 2]);
    assert_close(&output.to_f32_array(), &[22.0, 28.0, 49.0, 64.0], 0.0);
    output.sum().backward();
    assert_close(
        left.lock().grad.as_ref().unwrap(),
        &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0],
        0.0,
    );
    assert_close(
        right.lock().grad.as_ref().unwrap(),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0],
        0.0,
    );
}

#[test]
fn matmul_supports_nd_left_operand_with_matrix_rhs_and_gradients() {
    let left = Tensor::new(
        Array::from_shape_vec((2, 2, 3), (1..=12).map(|v| v as f32).collect())
            .unwrap()
            .into_dyn(),
        true,
    );
    let right = Tensor::new(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).into_dyn(), true);
    let output = left.matmul(&right);

    assert_eq!(output.shape(), vec![2, 2, 2]);
    assert_close(
        &output.to_f32_array(),
        &[4.0, 5.0, 10.0, 11.0, 16.0, 17.0, 22.0, 23.0],
        0.0,
    );
    output.sum().backward();
    assert_close(
        left.lock().grad.as_ref().unwrap(),
        &[1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        0.0,
    );
    assert_close(
        right.lock().grad.as_ref().unwrap(),
        &[22.0, 22.0, 26.0, 26.0, 30.0, 30.0],
        0.0,
    );
}

#[test]
fn matmul_supports_matching_nd_batch_prefixes() {
    let left = Tensor::new(Array::from_elem((2, 3, 2, 2), 2.0).into_dyn(), true);
    let right = Tensor::new(Array::from_elem((2, 3, 2, 1), 3.0).into_dyn(), true);
    let output = left.matmul(&right);

    assert_eq!(output.shape(), vec![2, 3, 2, 1]);
    assert!(output.to_f32_array().iter().all(|value| *value == 12.0));
    output.sum().backward();
    assert!(left
        .lock()
        .grad
        .as_ref()
        .unwrap()
        .iter()
        .all(|value| *value == 3.0));
    assert!(right
        .lock()
        .grad
        .as_ref()
        .unwrap()
        .iter()
        .all(|value| *value == 4.0));
}

#[test]
fn batched_matmul_keeps_batches_independent_in_forward_and_backward() {
    let left = Tensor::new(
        Array::from_shape_vec((2, 2, 2), vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 3.0])
            .unwrap()
            .into_dyn(),
        true,
    );
    let right = Tensor::new(
        Array::from_shape_vec((2, 2, 2), vec![5.0, 6.0, 7.0, 8.0, 4.0, 1.0, 2.0, 5.0])
            .unwrap()
            .into_dyn(),
        true,
    );
    let output = left.batched_matmul(&right);

    assert_close(
        &output.to_f32_array(),
        &[19.0, 22.0, 43.0, 50.0, 8.0, 2.0, 6.0, 15.0],
        0.0,
    );
    output.sum().backward();
    assert_close(
        left.lock().grad.as_ref().unwrap(),
        &[11.0, 15.0, 11.0, 15.0, 5.0, 7.0, 5.0, 7.0],
        0.0,
    );
    assert_close(
        right.lock().grad.as_ref().unwrap(),
        &[4.0, 4.0, 6.0, 6.0, 2.0, 2.0, 3.0, 3.0],
        0.0,
    );
}

#[test]
fn determinant_backward_uses_cofactors_for_singular_matrices() {
    let matrix = Tensor::new(arr2(&[[1.0, 2.0], [2.0, 4.0]]).into_dyn(), true);
    let determinant = matrix.det();

    assert_close(&determinant.to_f32_array(), &[0.0], 0.0);
    determinant.backward();
    assert_close(
        matrix.lock().grad.as_ref().unwrap(),
        &[4.0, -2.0, -2.0, 1.0],
        0.0,
    );
}

#[test]
fn batched_determinant_forward_and_backward_are_independent() {
    let matrices = Tensor::new(
        Array::from_shape_vec((2, 2, 2), vec![1.0, 2.0, 3.0, 4.0, 4.0, 7.0, 2.0, 6.0])
            .unwrap()
            .into_dyn(),
        true,
    );
    let determinants = matrices.det();

    assert_close(&determinants.to_f32_array(), &[-2.0, 10.0], 1e-6);
    determinants.sum().backward();
    assert_close(
        matrices.lock().grad.as_ref().unwrap(),
        &[4.0, -3.0, -2.0, 1.0, 6.0, -2.0, -7.0, 4.0],
        0.0,
    );
}

#[test]
fn inverse_forward_and_backward_match_closed_form_diagonal_case() {
    let matrix = Tensor::new(arr2(&[[2.0, 0.0], [0.0, 4.0]]).into_dyn(), true);
    let inverse = matrix.inv();

    assert_close(&inverse.to_f32_array(), &[0.5, 0.0, 0.0, 0.25], 0.0);
    inverse.sum().backward();
    assert_close(
        matrix.lock().grad.as_ref().unwrap(),
        &[-0.25, -0.125, -0.125, -0.0625],
        0.0,
    );
}

#[test]
#[should_panic(expected = "inv encountered a singular matrix")]
fn inverse_rejects_singular_matrices() {
    let matrix = Tensor::new(arr2(&[[1.0, 2.0], [2.0, 4.0]]).into_dyn(), false);
    let _ = matrix.inv();
}

#[test]
#[should_panic(expected = "matmul inner dimensions must match")]
fn matmul_rejects_incompatible_inner_dimensions() {
    let left = Tensor::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn(), false);
    let right = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(), false);
    let _ = left.matmul(&right);
}
