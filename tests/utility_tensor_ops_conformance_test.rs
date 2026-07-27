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
