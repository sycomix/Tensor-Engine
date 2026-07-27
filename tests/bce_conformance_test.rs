use ndarray::{arr1, arr2, ArrayD};
use tensor_engine::tensor::Tensor;

fn assert_close(actual: &ArrayD<f32>, expected: &ArrayD<f32>, tolerance: f32) {
    assert_eq!(actual.shape(), expected.shape());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[test]
fn bce_forward_and_backward_match_analytical_values() {
    let probabilities = Tensor::new(arr2(&[[0.2, 0.8], [0.4, 0.7]]).into_dyn(), true);
    let targets = Tensor::new(arr2(&[[0.0, 1.0], [0.25, 0.75]]).into_dyn(), true);

    let loss = probabilities.binary_cross_entropy(&targets);
    let expected_loss = arr2(&[
        [-0.8f32.ln(), -0.8f32.ln()],
        [
            -(0.25 * 0.4f32.ln() + 0.75 * 0.6f32.ln()),
            -(0.75 * 0.7f32.ln() + 0.25 * 0.3f32.ln()),
        ],
    ])
    .into_dyn();
    assert_close(&loss.to_f32_array(), &expected_loss, 1e-6);

    loss.backward();
    let probability_grad = probabilities.lock().grad.clone().unwrap();
    let target_grad = targets.lock().grad.clone().unwrap();
    let expected_probability_grad = arr2(&[
        [(0.2 - 0.0) / (0.2 * 0.8), (0.8 - 1.0) / (0.8 * 0.2)],
        [(0.4 - 0.25) / (0.4 * 0.6), (0.7 - 0.75) / (0.7 * 0.3)],
    ])
    .into_dyn();
    let expected_target_grad = arr2(&[
        [(0.8f32 / 0.2).ln(), (0.2f32 / 0.8).ln()],
        [(0.6f32 / 0.4).ln(), (0.3f32 / 0.7).ln()],
    ])
    .into_dyn();
    assert_close(&probability_grad, &expected_probability_grad, 1e-5);
    assert_close(&target_grad, &expected_target_grad, 1e-5);
}

#[test]
fn bce_is_finite_at_probability_boundaries() {
    let probabilities = Tensor::new(arr1(&[0.0, 1.0]).into_dyn(), true);
    let targets = Tensor::new(arr1(&[0.0, 1.0]).into_dyn(), false);
    let loss = probabilities.binary_cross_entropy(&targets);

    assert!(loss.to_vec().iter().all(|value| value.is_finite()));
    loss.backward();
    assert!(probabilities
        .lock()
        .grad
        .as_ref()
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
}

#[test]
fn bce_with_logits_forward_and_backward_match_analytical_values() {
    let logits = Tensor::new(arr1(&[-2.0, 0.0, 3.0]).into_dyn(), true);
    let targets = Tensor::new(arr1(&[0.0, 0.5, 1.0]).into_dyn(), true);

    let loss = logits.binary_cross_entropy_with_logits(&targets);
    let expected_loss = arr1(&[
        (1.0 + (-2.0f32).exp()).ln(),
        2.0f32.ln(),
        (1.0 + (-3.0f32).exp()).ln(),
    ])
    .into_dyn();
    assert_close(&loss.to_f32_array(), &expected_loss, 1e-6);

    loss.backward();
    let logits_grad = logits.lock().grad.clone().unwrap();
    let target_grad = targets.lock().grad.clone().unwrap();
    let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
    let expected_logits_grad =
        arr1(&[sigmoid(-2.0), sigmoid(0.0) - 0.5, sigmoid(3.0) - 1.0]).into_dyn();
    let expected_target_grad = arr1(&[2.0, 0.0, -3.0]).into_dyn();
    assert_close(&logits_grad, &expected_logits_grad, 1e-6);
    assert_close(&target_grad, &expected_target_grad, 1e-6);
}

#[test]
fn bce_with_logits_is_stable_for_large_magnitudes() {
    let logits = Tensor::new(arr1(&[-100.0, 100.0]).into_dyn(), true);
    let targets = Tensor::new(arr1(&[0.0, 1.0]).into_dyn(), false);
    let loss = logits.binary_cross_entropy_with_logits(&targets);

    assert!(loss.to_vec().iter().all(|value| value.is_finite()));
    loss.backward();
    assert!(logits
        .lock()
        .grad
        .as_ref()
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
}
