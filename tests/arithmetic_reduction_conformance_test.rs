use ndarray::{arr1, arr2, Array, ArrayD};
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

fn broadcast_inputs() -> (ArrayD<f32>, ArrayD<f32>) {
    (
        Array::from_shape_vec((2, 1, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn(),
        Array::from_shape_vec((1, 4, 1), vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .into_dyn(),
    )
}

#[test]
fn add_and_sub_broadcast_mixed_ranks_with_reduced_gradients() {
    let (left_values, right_values) = broadcast_inputs();
    for subtract in [false, true] {
        let left = Tensor::new(left_values.clone(), true);
        let right = Tensor::new(right_values.clone(), true);
        let output = if subtract {
            left.sub(&right)
        } else {
            left.add(&right)
        };

        assert_eq!(output.shape(), vec![2, 4, 3]);
        let expected_first_batch = if subtract {
            vec![
                0.0, 1.0, 2.0, -1.0, 0.0, 1.0, -2.0, -1.0, 0.0, -3.0, -2.0, -1.0,
            ]
        } else {
            vec![2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0]
        };
        assert_close(
            &output
                .to_f32_array()
                .slice(ndarray::s![0, .., ..])
                .to_owned()
                .into_dyn(),
            &expected_first_batch,
            0.0,
        );
        output.sum().backward();
        assert_close(left.lock().grad.as_ref().unwrap(), &[4.0; 6], 0.0);
        assert_close(
            right.lock().grad.as_ref().unwrap(),
            &[if subtract { -6.0 } else { 6.0 }; 4],
            0.0,
        );
    }
}

#[test]
fn mul_broadcast_forward_and_backward_match_product_rule() {
    let (left_values, right_values) = broadcast_inputs();
    let left = Tensor::new(left_values, true);
    let right = Tensor::new(right_values, true);
    let output = left.mul(&right);

    assert_eq!(output.shape(), vec![2, 4, 3]);
    assert_close(
        &output
            .to_f32_array()
            .slice(ndarray::s![0, .., ..])
            .to_owned()
            .into_dyn(),
        &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0],
        0.0,
    );
    output.sum().backward();
    assert_close(left.lock().grad.as_ref().unwrap(), &[10.0; 6], 0.0);
    assert_close(right.lock().grad.as_ref().unwrap(), &[21.0; 4], 0.0);
}

#[test]
fn div_broadcast_forward_and_backward_match_quotient_rule() {
    let (left_values, right_values) = broadcast_inputs();
    let left = Tensor::new(left_values, true);
    let right = Tensor::new(right_values, true);
    let output = left.div(&right);

    assert_eq!(output.shape(), vec![2, 4, 3]);
    assert_close(
        &output
            .to_f32_array()
            .slice(ndarray::s![0, .., ..])
            .to_owned()
            .into_dyn(),
        &[
            1.0,
            2.0,
            3.0,
            0.5,
            1.0,
            1.5,
            1.0 / 3.0,
            2.0 / 3.0,
            1.0,
            0.25,
            0.5,
            0.75,
        ],
        1e-6,
    );
    output.sum().backward();
    let reciprocal_sum = 1.0 + 0.5 + 1.0 / 3.0 + 0.25;
    assert_close(
        left.lock().grad.as_ref().unwrap(),
        &[reciprocal_sum; 6],
        1e-6,
    );
    assert_close(
        right.lock().grad.as_ref().unwrap(),
        &[-21.0, -21.0 / 4.0, -21.0 / 9.0, -21.0 / 16.0],
        1e-6,
    );
}

#[test]
fn sum_and_mean_apply_exact_global_reduction_gradients() {
    let sum_input = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(), true);
    let mean_input = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(), true);

    let sum = sum_input.sum();
    let mean = mean_input.mean();
    assert_close(&sum.to_f32_array(), &[21.0], 0.0);
    assert_close(&mean.to_f32_array(), &[3.5], 0.0);
    sum.backward();
    mean.backward();
    assert_close(sum_input.lock().grad.as_ref().unwrap(), &[1.0; 6], 0.0);
    assert_close(
        mean_input.lock().grad.as_ref().unwrap(),
        &[1.0 / 6.0; 6],
        0.0,
    );
}

#[test]
fn max_and_min_split_exact_ties_but_not_nearby_values() {
    let max_input = Tensor::new(arr1(&[2.0, 2.0, 2.0 - 5e-7]).into_dyn(), true);
    let min_input = Tensor::new(arr1(&[-2.0, -2.0, -2.0 + 5e-7]).into_dyn(), true);

    max_input.max().backward();
    min_input.min().backward();
    assert_close(
        max_input.lock().grad.as_ref().unwrap(),
        &[0.5, 0.5, 0.0],
        0.0,
    );
    assert_close(
        min_input.lock().grad.as_ref().unwrap(),
        &[0.5, 0.5, 0.0],
        0.0,
    );
}
