use ndarray::{Array, ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

fn tensor(values: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor {
    Tensor::new(
        Array::from_shape_vec(IxDyn(shape), values).unwrap(),
        requires_grad,
    )
}

fn assert_close(actual: &ArrayD<f32>, expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[test]
fn batch_norm_with_params_training_matches_analytical_forward_and_backward() {
    let values = [1.0f32, 2.0, 4.0];
    let upstream = [0.5f32, -1.0, 2.0];
    let eps = 1e-5f32;
    let gamma_value = 1.5f32;
    let beta_value = -0.25f32;
    let momentum = 0.25f32;

    let x = tensor(values.to_vec(), &[3, 1, 1], true);
    let gamma = tensor(vec![gamma_value], &[1], true);
    let beta = tensor(vec![beta_value], &[1], true);
    let running_mean = tensor(vec![0.0], &[1], true);
    let running_var = tensor(vec![1.0], &[1], true);
    let weights = tensor(upstream.to_vec(), &[3, 1, 1], false);

    let output = x.batch_norm_with_params(
        &gamma,
        &beta,
        &running_mean,
        &running_var,
        momentum,
        eps,
        true,
    );

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();
    let normalized = values.map(|value| (value - mean) * inv_std);
    let expected_output = normalized.map(|value| value * gamma_value + beta_value);
    assert_close(&output.to_f32_array(), &expected_output, 1e-5);

    let running_mean_expected = momentum * mean;
    let unbiased_variance = variance * values.len() as f32 / (values.len() - 1) as f32;
    let running_var_expected = (1.0 - momentum) + momentum * unbiased_variance;
    assert_close(&running_mean.to_f32_array(), &[running_mean_expected], 1e-6);
    assert_close(&running_var.to_f32_array(), &[running_var_expected], 1e-6);

    output.mul(&weights).sum().backward();
    let sum_upstream = upstream.iter().sum::<f32>();
    let sum_upstream_normalized = upstream
        .iter()
        .zip(normalized.iter())
        .map(|(gradient, normalized)| gradient * normalized)
        .sum::<f32>();
    let n = values.len() as f32;
    let expected_x_grad = upstream
        .iter()
        .zip(normalized.iter())
        .map(|(gradient, normalized)| {
            gamma_value * inv_std / n
                * (n * gradient - sum_upstream - normalized * sum_upstream_normalized)
        })
        .collect::<Vec<_>>();

    assert_close(x.lock().grad.as_ref().unwrap(), &expected_x_grad, 1e-5);
    assert_close(
        gamma.lock().grad.as_ref().unwrap(),
        &[sum_upstream_normalized],
        1e-5,
    );
    assert_close(beta.lock().grad.as_ref().unwrap(), &[sum_upstream], 1e-6);
    assert_close(running_mean.lock().grad.as_ref().unwrap(), &[0.0], 0.0);
    assert_close(running_var.lock().grad.as_ref().unwrap(), &[0.0], 0.0);
}

#[test]
fn batch_norm_with_params_evaluation_uses_running_statistics() {
    let x = tensor(vec![3.0, 7.0], &[2, 1, 1], true);
    let gamma = tensor(vec![2.0], &[1], true);
    let beta = tensor(vec![-1.0], &[1], true);
    let running_mean = tensor(vec![3.0], &[1], false);
    let running_var = tensor(vec![4.0], &[1], false);

    let output =
        x.batch_norm_with_params(&gamma, &beta, &running_mean, &running_var, 0.1, 0.0, false);
    assert_close(&output.to_f32_array(), &[-1.0, 3.0], 1e-6);

    output.sum().backward();
    assert_close(x.lock().grad.as_ref().unwrap(), &[1.0, 1.0], 1e-6);
    assert_close(gamma.lock().grad.as_ref().unwrap(), &[2.0], 1e-6);
    assert_close(beta.lock().grad.as_ref().unwrap(), &[2.0], 1e-6);
    assert_close(&running_mean.to_f32_array(), &[3.0], 0.0);
    assert_close(&running_var.to_f32_array(), &[4.0], 0.0);
}

#[test]
fn batch_norm_with_params_matches_config_wrapper() {
    let x = tensor(vec![1.0, 3.0], &[2, 1, 1], false);
    let gamma = tensor(vec![0.75], &[1], false);
    let beta = tensor(vec![0.5], &[1], false);
    let running_mean_a = tensor(vec![0.0], &[1], false);
    let running_var_a = tensor(vec![1.0], &[1], false);
    let running_mean_b = tensor(vec![0.0], &[1], false);
    let running_var_b = tensor(vec![1.0], &[1], false);

    let direct = x.batch_norm_with_params(
        &gamma,
        &beta,
        &running_mean_a,
        &running_var_a,
        0.2,
        1e-5,
        true,
    );
    let configured = x.batch_norm(
        &gamma,
        &beta,
        &running_mean_b,
        &running_var_b,
        tensor_engine::tensor::BatchNormConfig {
            momentum: 0.2,
            eps: 1e-5,
            training: true,
        },
    );

    assert_close(&direct.to_f32_array(), configured.to_vec().as_slice(), 0.0);
    assert_close(
        &running_mean_a.to_f32_array(),
        running_mean_b.to_vec().as_slice(),
        0.0,
    );
    assert_close(
        &running_var_a.to_f32_array(),
        running_var_b.to_vec().as_slice(),
        0.0,
    );
}

#[test]
fn batch_norm_public_contract_rejects_invalid_parameters() {
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let channels = tensor(vec![1.0, 1.0], &[2], false);
    let zeros = tensor(vec![0.0, 0.0], &[2], false);
    let ones = tensor(vec![1.0, 1.0], &[2], false);

    let wrong_gamma = tensor(vec![1.0], &[1], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        x.batch_norm_with_params(&wrong_gamma, &zeros, &zeros, &ones, 0.1, 1e-5, true)
    }))
    .is_err());

    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        x.batch_norm_with_params(&channels, &zeros, &zeros, &ones, 1.1, 1e-5, true)
    }))
    .is_err());

    let negative_variance = tensor(vec![1.0, -1.0], &[2], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        x.batch_norm_with_params(
            &channels,
            &zeros,
            &zeros,
            &negative_variance,
            0.1,
            1e-5,
            false,
        )
    }))
    .is_err());
}
