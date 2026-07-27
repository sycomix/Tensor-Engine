use ndarray::{arr1, arr2, ArrayD};
use tensor_engine::tensor::Tensor;

fn assert_values(actual: &ArrayD<f32>, expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn comparisons_broadcast_and_have_zero_gradients() {
    let left = Tensor::new(arr2(&[[1.0], [3.0]]).into_dyn(), true);
    let right = Tensor::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn(), true);

    assert_values(
        &left.equal(&right).to_f32_array(),
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        0.0,
    );
    let greater = left.greater(&right);
    assert_values(
        &greater.to_f32_array(),
        &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        0.0,
    );
    assert_values(
        &left.less(&right).to_f32_array(),
        &[0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        0.0,
    );
    greater.sum().backward();
    assert_values(left.lock().grad.as_ref().unwrap(), &[0.0, 0.0], 0.0);
    assert_values(right.lock().grad.as_ref().unwrap(), &[0.0, 0.0, 0.0], 0.0);
}

#[test]
fn sum_axis_handles_negative_axes_keepdims_and_backward_broadcast() {
    let input = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(), true);
    let output = input.sum_axis(-1, true);
    let weights = Tensor::new(arr2(&[[2.0], [3.0]]).into_dyn(), false);

    assert_eq!(output.shape(), vec![2, 1]);
    assert_values(&output.to_f32_array(), &[6.0, 15.0], 0.0);
    output.mul(&weights).sum().backward();
    assert_values(
        input.lock().grad.as_ref().unwrap(),
        &[2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        0.0,
    );
}

#[test]
fn cumulative_sum_and_product_match_exact_reverse_mode_gradients() {
    let sum_input = Tensor::new(arr1(&[1.0, 2.0, 3.0]).into_dyn(), true);
    let sum_output = sum_input.cumsum(0);
    let weights = Tensor::new(arr1(&[1.0, 2.0, 3.0]).into_dyn(), false);
    assert_values(&sum_output.to_f32_array(), &[1.0, 3.0, 6.0], 0.0);
    sum_output.mul(&weights).sum().backward();
    assert_values(
        sum_input.lock().grad.as_ref().unwrap(),
        &[6.0, 5.0, 3.0],
        0.0,
    );

    let product_input = Tensor::new(arr1(&[2.0, 3.0, 4.0]).into_dyn(), true);
    let product_output = product_input.cumprod(0);
    assert_values(&product_output.to_f32_array(), &[2.0, 6.0, 24.0], 0.0);
    product_output.mul(&weights).sum().backward();
    assert_values(
        product_input.lock().grad.as_ref().unwrap(),
        &[43.0, 28.0, 18.0],
        0.0,
    );

    let zero_input = Tensor::new(arr1(&[0.0, 2.0, 3.0]).into_dyn(), true);
    zero_input.cumprod(0).sum().backward();
    assert_values(
        zero_input.lock().grad.as_ref().unwrap(),
        &[9.0, 0.0, 0.0],
        0.0,
    );
}

#[test]
fn cumulative_extrema_route_each_prefix_to_its_first_winner() {
    let max_input = Tensor::new(arr1(&[2.0, 1.0, 3.0, 3.0, 0.0]).into_dyn(), true);
    let max_output = max_input.cummax(0);
    assert_values(&max_output.to_f32_array(), &[2.0, 2.0, 3.0, 3.0, 3.0], 0.0);
    max_output.sum().backward();
    assert_values(
        max_input.lock().grad.as_ref().unwrap(),
        &[2.0, 0.0, 3.0, 0.0, 0.0],
        0.0,
    );

    let min_input = Tensor::new(arr1(&[2.0, 3.0, 1.0, 1.0, 4.0]).into_dyn(), true);
    let min_output = min_input.cummin(0);
    assert_values(&min_output.to_f32_array(), &[2.0, 2.0, 1.0, 1.0, 1.0], 0.0);
    min_output.sum().backward();
    assert_values(
        min_input.lock().grad.as_ref().unwrap(),
        &[2.0, 0.0, 3.0, 0.0, 0.0],
        0.0,
    );
}

#[test]
fn sort_and_argsort_use_stable_last_axis_ordering() {
    let input = Tensor::new(arr2(&[[3.0, 1.0, 2.0], [2.0, 2.0, 1.0]]).into_dyn(), true);
    let sorted = input.sort();
    assert_values(&sorted.to_f32_array(), &[1.0, 2.0, 3.0, 1.0, 2.0, 2.0], 0.0);
    assert_values(
        &input.argsort().to_f32_array(),
        &[1.0, 2.0, 0.0, 2.0, 0.0, 1.0],
        0.0,
    );
    sorted
        .mul(&Tensor::new(
            arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(),
            false,
        ))
        .sum()
        .backward();
    assert_values(
        input.lock().grad.as_ref().unwrap(),
        &[3.0, 1.0, 2.0, 5.0, 6.0, 4.0],
        0.0,
    );
}

#[test]
fn topk_routes_only_value_half_gradients_to_stable_winners() {
    let input = Tensor::new(arr2(&[[1.0, 5.0, 3.0, 5.0]]).into_dyn(), true);
    let output = input.topk(2);
    assert_values(&output.to_f32_array(), &[5.0, 5.0, 1.0, 3.0], 0.0);
    output
        .mul(&Tensor::new(
            arr2(&[[2.0, 3.0, 100.0, 100.0]]).into_dyn(),
            false,
        ))
        .sum()
        .backward();
    assert_values(
        input.lock().grad.as_ref().unwrap(),
        &[0.0, 2.0, 0.0, 3.0],
        0.0,
    );
}

#[test]
#[should_panic(expected = "topk k must not exceed the last dimension")]
fn topk_rejects_excessive_k() {
    let input = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), false);
    let _ = input.topk(3);
}
