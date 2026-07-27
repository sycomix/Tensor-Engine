use ndarray::{arr1, arr2, ArrayD, IxDyn};
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

fn stable_softmax(values: &[f32]) -> Vec<f32> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<_> = values.iter().map(|value| (value - maximum).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.into_iter().map(|value| value / total).collect()
}

#[test]
fn softmax_and_log_softmax_non_last_axis_match_jacobian_formulas() {
    let values = arr2(&[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]).into_dyn();
    let weights_values = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).into_dyn();
    let probabilities = stable_softmax(&[1.0, 2.0, 3.0]);

    let softmax_input = Tensor::new(values.clone(), true);
    let weights = Tensor::new(weights_values.clone(), false);
    let softmax = softmax_input.softmax(0);
    assert_close(
        &softmax.to_f32_array(),
        &[
            probabilities[0],
            1.0 / 3.0,
            probabilities[1],
            1.0 / 3.0,
            probabilities[2],
            1.0 / 3.0,
        ],
        1e-6,
    );
    softmax.mul(&weights).sum().backward();
    let first_dot = probabilities[0] + 3.0 * probabilities[1] + 5.0 * probabilities[2];
    let second_dot = 4.0;
    assert_close(
        softmax_input.lock().grad.as_ref().unwrap(),
        &[
            probabilities[0] * (1.0 - first_dot),
            (1.0 / 3.0) * (2.0 - second_dot),
            probabilities[1] * (3.0 - first_dot),
            (1.0 / 3.0) * (4.0 - second_dot),
            probabilities[2] * (5.0 - first_dot),
            (1.0 / 3.0) * (6.0 - second_dot),
        ],
        1e-6,
    );

    let log_input = Tensor::new(values, true);
    let log_softmax = log_input.log_softmax(0);
    let expected_log: Vec<_> = [
        probabilities[0],
        1.0 / 3.0,
        probabilities[1],
        1.0 / 3.0,
        probabilities[2],
        1.0 / 3.0,
    ]
    .into_iter()
    .map(f32::ln)
    .collect();
    assert_close(&log_softmax.to_f32_array(), &expected_log, 1e-6);
    log_softmax.mul(&weights).sum().backward();
    assert_close(
        log_input.lock().grad.as_ref().unwrap(),
        &[
            1.0 - probabilities[0] * 9.0,
            2.0 - (1.0 / 3.0) * 12.0,
            3.0 - probabilities[1] * 9.0,
            4.0 - (1.0 / 3.0) * 12.0,
            5.0 - probabilities[2] * 9.0,
            6.0 - (1.0 / 3.0) * 12.0,
        ],
        1e-6,
    );
}

#[test]
fn softmax_variants_are_stable_for_extreme_and_all_negative_infinite_rows() {
    for logarithmic in [false, true] {
        let input = Tensor::new(
            arr2(&[
                [10_000.0, 10_001.0, 9_999.0],
                [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
            ])
            .into_dyn(),
            true,
        );
        let output = if logarithmic {
            input.log_softmax(1)
        } else {
            input.softmax(1)
        };
        assert!(output.to_vec().iter().all(|value| value.is_finite()));
        if logarithmic {
            assert_close(
                &output
                    .to_f32_array()
                    .slice(ndarray::s![1, ..])
                    .to_owned()
                    .into_dyn(),
                &[-3.0f32.ln(); 3],
                1e-6,
            );
        } else {
            assert_close(
                &output
                    .to_f32_array()
                    .slice(ndarray::s![1, ..])
                    .to_owned()
                    .into_dyn(),
                &[1.0 / 3.0; 3],
                1e-6,
            );
        }
        output.sum().backward();
        assert!(input
            .lock()
            .grad
            .as_ref()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }
}

fn numerical_gradient<F>(values: &[f32], epsilon: f32, function: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let mut gradient = vec![0.0; values.len()];
    for index in 0..values.len() {
        let mut positive = values.to_vec();
        let mut negative = values.to_vec();
        positive[index] += epsilon;
        negative[index] -= epsilon;
        gradient[index] = (function(&positive) - function(&negative)) / (2.0 * epsilon);
    }
    gradient
}

#[test]
fn rmsnorm_non_last_axis_matches_finite_difference_gradients() {
    let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0];
    let gamma_values = vec![0.5, 1.5, -0.75];
    let weights = vec![1.0, -1.0, 0.5, 2.0, -0.5, 3.0];
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), x_values.clone()).unwrap(),
        true,
    );
    let gamma = Tensor::new(arr1(&gamma_values).into_dyn(), true);
    let output = x.rmsnorm(&gamma, 0, 1e-5);
    output
        .mul(&Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), weights.clone()).unwrap(),
            false,
        ))
        .sum()
        .backward();

    let evaluate = |x_data: &[f32], gamma_data: &[f32]| {
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), x_data.to_vec()).unwrap(),
            false,
        );
        let scale = Tensor::new(arr1(gamma_data).into_dyn(), false);
        input
            .rmsnorm(&scale, 0, 1e-5)
            .to_vec()
            .iter()
            .zip(&weights)
            .map(|(value, weight)| value * weight)
            .sum()
    };
    let numerical_x = numerical_gradient(&x_values, 1e-3, |values| evaluate(values, &gamma_values));
    let numerical_gamma =
        numerical_gradient(&gamma_values, 1e-3, |values| evaluate(&x_values, values));
    assert_close(x.lock().grad.as_ref().unwrap(), &numerical_x, 2e-3);
    assert_close(gamma.lock().grad.as_ref().unwrap(), &numerical_gamma, 2e-3);
}

#[test]
fn layer_norm_non_last_axis_matches_finite_difference_parameter_gradients() {
    let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0];
    let gamma_values = vec![0.5, 1.5, -0.75];
    let beta_values = vec![0.2, -0.3, 0.7];
    let weights = vec![1.0, -1.0, 0.5, 2.0, -0.5, 3.0];
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), x_values.clone()).unwrap(),
        true,
    );
    let gamma = Tensor::new(arr1(&gamma_values).into_dyn(), true);
    let beta = Tensor::new(arr1(&beta_values).into_dyn(), true);
    x.layer_norm(0, 1e-5, &gamma, &beta)
        .mul(&Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), weights.clone()).unwrap(),
            false,
        ))
        .sum()
        .backward();

    let evaluate = |x_data: &[f32], gamma_data: &[f32], beta_data: &[f32]| {
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), x_data.to_vec()).unwrap(),
            false,
        );
        let scale = Tensor::new(arr1(gamma_data).into_dyn(), false);
        let bias = Tensor::new(arr1(beta_data).into_dyn(), false);
        input
            .layer_norm(0, 1e-5, &scale, &bias)
            .to_vec()
            .iter()
            .zip(&weights)
            .map(|(value, weight)| value * weight)
            .sum()
    };
    let numerical_x = numerical_gradient(&x_values, 1e-3, |values| {
        evaluate(values, &gamma_values, &beta_values)
    });
    let numerical_gamma = numerical_gradient(&gamma_values, 1e-3, |values| {
        evaluate(&x_values, values, &beta_values)
    });
    let numerical_beta = numerical_gradient(&beta_values, 1e-3, |values| {
        evaluate(&x_values, &gamma_values, values)
    });
    assert_close(x.lock().grad.as_ref().unwrap(), &numerical_x, 3e-3);
    assert_close(gamma.lock().grad.as_ref().unwrap(), &numerical_gamma, 3e-3);
    assert_close(beta.lock().grad.as_ref().unwrap(), &numerical_beta, 3e-3);
}

#[test]
#[should_panic(expected = "softmax axis 2 is out of bounds")]
fn softmax_rejects_invalid_axes() {
    let input = Tensor::new(arr2(&[[1.0, 2.0]]).into_dyn(), false);
    let _ = input.softmax(2);
}
