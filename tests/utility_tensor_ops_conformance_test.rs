use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

fn tensor(shape: &[usize], values: Vec<f32>, requires_grad: bool) -> Tensor {
    Tensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
        requires_grad,
    )
}

#[test]
fn neg_and_scaled_ternary_have_exact_forward_and_weighted_gradients() {
    let input = tensor(&[4], vec![-2.0, -0.4, 0.2, 1.4], true);
    let negated = input.neg();
    assert_eq!(
        negated.to_f32_array().as_slice().unwrap(),
        &[2.0, 0.4, -0.2, -1.4]
    );
    negated
        .mul(&tensor(&[4], vec![1.0, 2.0, 3.0, 4.0], false))
        .sum()
        .backward();
    assert_eq!(
        input.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[-1.0, -2.0, -3.0, -4.0]
    );

    let input = tensor(&[4], vec![-2.0, -0.4, 0.2, 1.4], true);
    let ternary = input.ternary();
    assert_eq!(
        ternary.to_f32_array().as_slice().unwrap(),
        &[-1.0, 0.0, 0.0, 1.0]
    );
    ternary
        .mul(&tensor(&[4], vec![4.0, 3.0, 2.0, 1.0], false))
        .sum()
        .backward();
    assert_eq!(
        input.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[4.0, 3.0, 2.0, 1.0]
    );
}

#[test]
fn geglu_matches_its_documented_formula_and_backward() {
    let input = tensor(&[1, 4], vec![1.0, -0.5, 2.0, 3.0], true);
    let output = input.geglu();
    let gelu = |x: f32| 0.5 * x * (1.0 + (1.702 * x).tanh());
    let expected = vec![gelu(1.0) * 2.0, gelu(-0.5) * 3.0];
    assert_eq!(output.shape(), vec![1, 2]);
    for (actual, expected) in output.to_f32_array().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }

    output.sum().backward();
    let grad = input.lock().grad.clone().unwrap();
    let tanh_1 = (1.702_f32).tanh();
    let tanh_2 = (-0.5_f32 * 1.702).tanh();
    let gelu_prime =
        |x: f32, tanh_x: f32| 0.5 * (1.0 + tanh_x) + 0.5 * x * (1.0 - tanh_x * tanh_x) * 1.702;
    let expected_grad = [
        gelu_prime(1.0, tanh_1) * 2.0,
        gelu_prime(-0.5, tanh_2) * 3.0,
        gelu(1.0),
        gelu(-0.5),
    ];
    for (actual, expected) in grad.iter().zip(expected_grad) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn slice_channels_preserves_autograd_and_routes_only_selected_channels() {
    let input = tensor(&[1, 3, 1, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], true);
    let output = input.slice_channels(1, 2);
    assert_eq!(output.shape(), vec![1, 2, 1, 2]);
    assert_eq!(
        output.to_f32_array().as_slice().unwrap(),
        &[3.0, 4.0, 5.0, 6.0]
    );
    output
        .mul(&tensor(&[1, 2, 1, 2], vec![1.0, 2.0, 3.0, 4.0], false))
        .sum()
        .backward();
    assert_eq!(
        input.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn nearest_upsample_routes_each_output_gradient_to_its_source_pixel() {
    let input = tensor(&[1, 1, 1, 2], vec![10.0, 20.0], true);
    let output = input.upsample_nearest2d(2);
    assert_eq!(
        output.to_f32_array().as_slice().unwrap(),
        &[10.0, 10.0, 20.0, 20.0, 10.0, 10.0, 20.0, 20.0]
    );
    output
        .mul(&tensor(
            &[1, 1, 2, 4],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            false,
        ))
        .sum()
        .backward();
    assert_eq!(
        input.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[14.0, 22.0]
    );
}

#[test]
fn utility_ops_reject_invalid_public_inputs() {
    let empty = tensor(&[0], vec![], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| empty.ternary())).is_err());

    let rank_two = tensor(&[2, 2], vec![1.0; 4], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rank_two.slice_channels(0, 1)
    }))
    .is_err());

    let image = tensor(&[1, 2, 1, 1], vec![1.0; 2], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        image.slice_channels(1, 2)
    }))
    .is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        image.upsample_nearest2d(0)
    }))
    .is_err());
}
