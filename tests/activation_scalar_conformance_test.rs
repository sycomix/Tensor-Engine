use ndarray::{arr1, ArrayD};
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
fn pow_exp_log_and_sqrt_match_analytical_values_and_gradients() {
    let values = [0.25, 1.0, 4.0];

    let power_input = Tensor::new(arr1(&values).into_dyn(), true);
    let power = power_input.pow(2.5);
    assert_close(&power.to_f32_array(), &[0.03125, 1.0, 32.0], 1e-6);
    power.sum().backward();
    assert_close(
        power_input.lock().grad.as_ref().unwrap(),
        &[0.3125, 2.5, 20.0],
        1e-6,
    );

    let exp_input = Tensor::new(arr1(&[0.0, 1.0, -1.0]).into_dyn(), true);
    let exp = exp_input.exp();
    assert_close(
        &exp.to_f32_array(),
        &[1.0, std::f32::consts::E, 1.0 / std::f32::consts::E],
        1e-6,
    );
    exp.sum().backward();
    assert_close(
        exp_input.lock().grad.as_ref().unwrap(),
        &[1.0, std::f32::consts::E, 1.0 / std::f32::consts::E],
        1e-6,
    );

    let log_input = Tensor::new(arr1(&values).into_dyn(), true);
    let log = log_input.log();
    assert_close(&log.to_f32_array(), &[0.25f32.ln(), 0.0, 4.0f32.ln()], 1e-6);
    log.sum().backward();
    assert_close(
        log_input.lock().grad.as_ref().unwrap(),
        &[4.0, 1.0, 0.25],
        1e-6,
    );

    let sqrt_input = Tensor::new(arr1(&values).into_dyn(), true);
    let sqrt = sqrt_input.sqrt();
    assert_close(&sqrt.to_f32_array(), &[0.5, 1.0, 2.0], 0.0);
    sqrt.sum().backward();
    assert_close(
        sqrt_input.lock().grad.as_ref().unwrap(),
        &[1.0, 0.5, 0.25],
        0.0,
    );
}

#[test]
fn relu_sigmoid_and_tanh_match_boundary_and_gradient_contracts() {
    let relu_input = Tensor::new(arr1(&[-2.0, 0.0, 3.0]).into_dyn(), true);
    let relu = relu_input.relu();
    assert_close(&relu.to_f32_array(), &[0.0, 0.0, 3.0], 0.0);
    relu.sum().backward();
    assert_close(
        relu_input.lock().grad.as_ref().unwrap(),
        &[0.0, 0.0, 1.0],
        0.0,
    );

    let sigmoid_input = Tensor::new(arr1(&[-1.0, 0.0, 1.0]).into_dyn(), true);
    let sigmoid = sigmoid_input.sigmoid();
    let sigmoid_values = [
        1.0 / (1.0 + 1.0f32.exp()),
        0.5,
        1.0 / (1.0 + (-1.0f32).exp()),
    ];
    assert_close(&sigmoid.to_f32_array(), &sigmoid_values, 1e-6);
    sigmoid.sum().backward();
    assert_close(
        sigmoid_input.lock().grad.as_ref().unwrap(),
        &sigmoid_values.map(|value| value * (1.0 - value)),
        1e-6,
    );

    let tanh_input = Tensor::new(arr1(&[-1.0, 0.0, 1.0]).into_dyn(), true);
    let tanh = tanh_input.tanh();
    let tanh_values = [(-1.0f32).tanh(), 0.0, 1.0f32.tanh()];
    assert_close(&tanh.to_f32_array(), &tanh_values, 1e-6);
    tanh.sum().backward();
    assert_close(
        tanh_input.lock().grad.as_ref().unwrap(),
        &tanh_values.map(|value| 1.0 - value * value),
        1e-6,
    );
}

#[test]
fn gelu_and_silu_match_closed_form_approximations_and_gradients() {
    let values = [-1.0, 0.0, 2.0];
    let gelu_input = Tensor::new(arr1(&values).into_dyn(), true);
    let gelu = gelu_input.gelu();
    let coefficient = (2.0f32 / std::f32::consts::PI).sqrt();
    let gelu_forward = |x: f32| {
        let u = coefficient * (x + 0.044715 * x.powi(3));
        0.5 * x * (1.0 + u.tanh())
    };
    let gelu_derivative = |x: f32| {
        let u = coefficient * (x + 0.044715 * x.powi(3));
        let tanh_u = u.tanh();
        0.5 * (1.0 + tanh_u)
            + 0.5 * x * (1.0 - tanh_u * tanh_u) * coefficient * (1.0 + 3.0 * 0.044715 * x * x)
    };
    assert_close(&gelu.to_f32_array(), &values.map(gelu_forward), 1e-6);
    gelu.sum().backward();
    assert_close(
        gelu_input.lock().grad.as_ref().unwrap(),
        &values.map(gelu_derivative),
        1e-6,
    );

    let silu_input = Tensor::new(arr1(&values).into_dyn(), true);
    let silu = silu_input.silu();
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    assert_close(
        &silu.to_f32_array(),
        &values.map(|value| value * sigmoid(value)),
        1e-6,
    );
    silu.sum().backward();
    assert_close(
        silu_input.lock().grad.as_ref().unwrap(),
        &values.map(|value| {
            let probability = sigmoid(value);
            probability + value * probability * (1.0 - probability)
        }),
        1e-6,
    );
}

#[test]
fn clamp_preserves_autograd_and_uses_inclusive_boundary_gradients() {
    let input = Tensor::new(arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn(), true);
    let output = input.clamp(-1.0, 1.0);

    assert_close(&output.to_f32_array(), &[-1.0, -1.0, 0.0, 1.0, 1.0], 0.0);
    output.sum().backward();
    assert_close(
        input.lock().grad.as_ref().unwrap(),
        &[0.0, 1.0, 1.0, 1.0, 0.0],
        0.0,
    );
}

#[test]
fn sigmoid_tanh_and_silu_remain_finite_at_large_magnitudes() {
    for operation in ["sigmoid", "tanh", "silu"] {
        let input = Tensor::new(arr1(&[-100.0, 100.0]).into_dyn(), true);
        let output = match operation {
            "sigmoid" => input.sigmoid(),
            "tanh" => input.tanh(),
            _ => input.silu(),
        };
        assert!(output.to_vec().iter().all(|value| value.is_finite()));
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

#[test]
#[should_panic(expected = "clamp minimum must not exceed maximum")]
fn clamp_rejects_reversed_bounds() {
    let input = Tensor::new(arr1(&[0.0]).into_dyn(), false);
    let _ = input.clamp(1.0, -1.0);
}
