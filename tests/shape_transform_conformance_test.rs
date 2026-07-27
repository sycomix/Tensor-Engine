use ndarray::{arr2, Array, ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

fn assert_values(actual: &ArrayD<f32>, expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(*actual, *expected, "mismatch at flat index {index}");
    }
}

fn weighted_sum(output: &Tensor, weights: Vec<f32>) -> Tensor {
    let weights = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&output.shape()), weights).unwrap(),
        false,
    );
    output.mul(&weights).sum()
}

#[test]
fn reshape_preserves_logical_order_and_routes_gradients() {
    let input = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(), true);
    let output = input.reshape(vec![3, 2]).unwrap();

    assert_eq!(output.shape(), vec![3, 2]);
    assert_values(&output.to_f32_array(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    weighted_sum(&output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).backward();
    assert_values(
        input.lock().grad.as_ref().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
}

#[test]
fn transpose_is_differentiable_and_reverses_all_axes() {
    let input = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(), true);
    let output = input.transpose();

    assert_eq!(output.shape(), vec![3, 2]);
    assert_values(&output.to_f32_array(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    weighted_sum(&output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).backward();
    assert_values(
        input.lock().grad.as_ref().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
    );
}

#[test]
fn permute_routes_each_gradient_through_the_inverse_permutation() {
    let input = Tensor::new(
        Array::from_shape_vec((2, 2, 3), (1..=12).map(|value| value as f32).collect())
            .unwrap()
            .into_dyn(),
        true,
    );
    let output = input.permute(vec![2, 0, 1]);

    assert_eq!(output.shape(), vec![3, 2, 2]);
    assert_values(
        &output.to_f32_array(),
        &[
            1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0,
        ],
    );
    weighted_sum(&output, (1..=12).map(|value| value as f32).collect()).backward();
    assert_values(
        input.lock().grad.as_ref().unwrap(),
        &[
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ],
    );
}

#[test]
fn concat_splits_backward_gradient_at_each_input_boundary() {
    let left = Tensor::new(arr2(&[[1.0], [2.0]]).into_dyn(), true);
    let right = Tensor::new(arr2(&[[3.0, 4.0], [5.0, 6.0]]).into_dyn(), true);
    let output = Tensor::concat(&[left.clone(), right.clone()], 1);

    assert_values(&output.to_f32_array(), &[1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
    weighted_sum(&output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).backward();
    assert_values(left.lock().grad.as_ref().unwrap(), &[1.0, 4.0]);
    assert_values(right.lock().grad.as_ref().unwrap(), &[2.0, 3.0, 5.0, 6.0]);
}

#[test]
fn stack_removes_the_inserted_axis_during_backward() {
    let first = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(), true);
    let second = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]).into_dyn(), true);
    let output = Tensor::stack(&[first.clone(), second.clone()], 1);

    assert_eq!(output.shape(), vec![2, 2, 2]);
    assert_values(
        &output.to_f32_array(),
        &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0],
    );
    weighted_sum(&output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).backward();
    assert_values(first.lock().grad.as_ref().unwrap(), &[1.0, 2.0, 5.0, 6.0]);
    assert_values(second.lock().grad.as_ref().unwrap(), &[3.0, 4.0, 7.0, 8.0]);
}

#[test]
#[should_panic(expected = "permute axes must be unique")]
fn permute_rejects_duplicate_axes() {
    let input = Tensor::new(arr2(&[[1.0, 2.0]]).into_dyn(), false);
    let _ = input.permute(vec![0, 0]);
}

#[test]
#[should_panic(expected = "concat requires at least one tensor")]
fn concat_rejects_empty_input_lists() {
    let _ = Tensor::concat(&[], 0);
}
