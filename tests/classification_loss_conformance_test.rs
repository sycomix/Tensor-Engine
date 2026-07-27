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

fn stable_softmax(row: &[f32]) -> Vec<f32> {
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<_> = row.iter().map(|value| (value - maximum).exp()).collect();
    let total: f32 = exponentials.iter().sum();
    exponentials
        .into_iter()
        .map(|value| value / total)
        .collect()
}

#[test]
fn fused_cross_entropy_variants_match_analytical_index_loss_and_gradient() {
    let values = arr2(&[[1.0, 2.0, -1.0], [0.1, 0.2, 0.3]]).into_dyn();
    let targets = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), false);
    let logits_a = Tensor::new(values.clone(), true);
    let logits_b = Tensor::new(values.clone(), true);

    let loss_a = logits_a.cross_entropy_with_logits(&targets, 1);
    let loss_b = logits_b.softmax_cross_entropy_with_logits(&targets, -1);
    let soft_a = stable_softmax(&[1.0, 2.0, -1.0]);
    let soft_b = stable_softmax(&[0.1, 0.2, 0.3]);
    let expected_loss = -(soft_a[1].ln() + soft_b[2].ln()) / 2.0;
    assert_close(
        &loss_a.to_f32_array(),
        &ArrayD::from_elem(ndarray::IxDyn(&[]), expected_loss),
        1e-6,
    );
    assert_close(&loss_b.to_f32_array(), &loss_a.to_f32_array(), 1e-6);

    loss_a.backward();
    loss_b.backward();
    let expected_gradient = arr2(&[
        [soft_a[0] / 2.0, (soft_a[1] - 1.0) / 2.0, soft_a[2] / 2.0],
        [soft_b[0] / 2.0, soft_b[1] / 2.0, (soft_b[2] - 1.0) / 2.0],
    ])
    .into_dyn();
    assert_close(
        logits_a.lock().grad.as_ref().unwrap(),
        &expected_gradient,
        1e-6,
    );
    assert_close(
        logits_b.lock().grad.as_ref().unwrap(),
        &expected_gradient,
        1e-6,
    );
}

#[test]
fn cross_entropy_supports_non_last_class_axis_and_one_hot_targets() {
    let logits = Tensor::new(
        arr2(&[[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]]).into_dyn(),
        true,
    );
    let targets = Tensor::new(
        arr2(&[[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]).into_dyn(),
        false,
    );
    let loss = logits.cross_entropy_with_logits(&targets, 0);
    let first = stable_softmax(&[1.0, 2.0, -1.0]);
    let second = stable_softmax(&[0.0, 0.0, 0.0]);
    let expected_loss = -(first[1].ln() + second[0].ln()) / 2.0;

    assert!(
        (loss.to_vec()[0] - expected_loss).abs() <= 1e-6,
        "actual={}, expected={expected_loss}",
        loss.to_vec()[0]
    );
    loss.backward();
    let expected = arr2(&[
        [first[0] / 2.0, (second[0] - 1.0) / 2.0],
        [(first[1] - 1.0) / 2.0, second[1] / 2.0],
        [first[2] / 2.0, second[2] / 2.0],
    ])
    .into_dyn();
    assert_close(logits.lock().grad.as_ref().unwrap(), &expected, 1e-6);
}

#[test]
fn fused_cross_entropy_remains_finite_for_extreme_logits() {
    let targets = Tensor::new(arr1(&[0.0, 1.0]).into_dyn(), false);
    for fused in [false, true] {
        let logits = Tensor::new(
            arr2(&[[10_000.0, -10_000.0], [-10_000.0, 10_000.0]]).into_dyn(),
            true,
        );
        let loss = if fused {
            logits.softmax_cross_entropy_with_logits(&targets, 1)
        } else {
            logits.cross_entropy_with_logits(&targets, 1)
        };
        assert!(loss.to_vec()[0].is_finite());
        loss.backward();
        assert!(logits
            .lock()
            .grad
            .as_ref()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }
}

#[test]
fn nll_loss_index_and_one_hot_forms_match_forward_and_backward() {
    let log_probs_values = arr2(&[
        [(0.2f32).ln(), (0.3f32).ln(), (0.5f32).ln()],
        [(0.6f32).ln(), (0.1f32).ln(), (0.3f32).ln()],
    ])
    .into_dyn();
    let index_targets = Tensor::new(arr1(&[2.0, 0.0]).into_dyn(), false);
    let one_hot_targets = Tensor::new(arr2(&[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).into_dyn(), false);
    let index_input = Tensor::new(log_probs_values.clone(), true);
    let one_hot_input = Tensor::new(log_probs_values, true);
    let index_loss = index_input.nll_loss(&index_targets);
    let one_hot_loss = one_hot_input.nll_loss(&one_hot_targets);
    let expected_loss = -((0.5f32).ln() + (0.6f32).ln()) / 2.0;

    assert!((index_loss.to_vec()[0] - expected_loss).abs() <= 1e-6);
    assert_close(
        &one_hot_loss.to_f32_array(),
        &index_loss.to_f32_array(),
        1e-6,
    );
    index_loss.backward();
    one_hot_loss.backward();
    let expected_gradient = arr2(&[[0.0, 0.0, -0.5], [-0.5, 0.0, 0.0]]).into_dyn();
    assert_close(
        index_input.lock().grad.as_ref().unwrap(),
        &expected_gradient,
        0.0,
    );
    assert_close(
        one_hot_input.lock().grad.as_ref().unwrap(),
        &expected_gradient,
        0.0,
    );
}

#[test]
#[should_panic(expected = "target labels must be finite non-negative integers")]
fn cross_entropy_rejects_fractional_class_labels() {
    let logits = Tensor::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn(), false);
    let targets = Tensor::new(arr1(&[1.5]).into_dyn(), false);
    let _ = logits.cross_entropy_with_logits(&targets, 1);
}

#[test]
#[should_panic(expected = "axis 2 is out of bounds for rank 2")]
fn cross_entropy_rejects_invalid_axes() {
    let logits = Tensor::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn(), false);
    let targets = Tensor::new(arr1(&[1.0]).into_dyn(), false);
    let _ = logits.cross_entropy_with_logits(&targets, 2);
}
