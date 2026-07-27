use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

fn tensor(shape: &[usize], values: Vec<f32>, requires_grad: bool) -> Tensor {
    Tensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
        requires_grad,
    )
}

fn weighted_loss(
    shape: &[usize],
    values: &[f32],
    output_weights: &ArrayD<f32>,
    operation: fn(&Tensor) -> Tensor,
) -> f32 {
    let input = tensor(shape, values.to_vec(), false);
    operation(&input)
        .mul(&Tensor::new(output_weights.clone(), false))
        .sum()
        .to_f32_array()[IxDyn(&[])]
}

fn assert_numerical_gradient(
    shape: &[usize],
    values: &[f32],
    output_weights: ArrayD<f32>,
    operation: fn(&Tensor) -> Tensor,
) {
    let input = tensor(shape, values.to_vec(), true);
    operation(&input)
        .mul(&Tensor::new(output_weights.clone(), false))
        .sum()
        .backward();
    let analytical = input.lock().grad.clone().unwrap();
    let epsilon = 1e-3;

    for index in 0..values.len() {
        let mut plus = values.to_vec();
        let mut minus = values.to_vec();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let numerical = (weighted_loss(shape, &plus, &output_weights, operation)
            - weighted_loss(shape, &minus, &output_weights, operation))
            / (2.0 * epsilon);
        assert!(
            (analytical.as_slice().unwrap()[index] - numerical).abs() < 2e-3,
            "gradient {index}: analytical {}, numerical {numerical}",
            analytical.as_slice().unwrap()[index]
        );
    }
}

#[test]
fn fft_family_weighted_gradients_match_finite_differences() {
    let delta = tensor(&[4], vec![1.0, 0.0, 0.0, 0.0], false).fft();
    assert_eq!(
        delta.to_f32_array().as_slice().unwrap(),
        &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    );
    assert_numerical_gradient(
        &[2, 3],
        &[0.2, -0.5, 1.0, 1.5, 0.25, -0.75],
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 2]),
            (1..=12).map(|value| value as f32 / 7.0).collect(),
        )
        .unwrap(),
        Tensor::fft,
    );
    assert_numerical_gradient(
        &[2, 3, 2],
        &[
            0.2, -0.1, 0.5, 0.3, -0.7, 0.9, 1.0, -0.4, 0.25, 0.75, -0.6, 0.8,
        ],
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, -2.0, 3.0, 0.5, 1.5, -0.75]).unwrap(),
        Tensor::ifft,
    );
    assert_numerical_gradient(
        &[4],
        &[0.2, -0.5, 1.0, 1.5],
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, -1.0, 0.5, 3.0, -2.0]).unwrap(),
        Tensor::rfft,
    );
    assert_numerical_gradient(
        &[3, 2],
        &[1.0, 0.0, 0.5, -0.25, -1.0, 0.0],
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, -2.0, 3.0, 0.5]).unwrap(),
        Tensor::irfft,
    );
}

#[test]
fn complex_pair_operations_have_exact_weighted_gradients() {
    let value = tensor(&[2, 2], vec![1.0, 2.0, 3.0, -4.0], true);
    let conjugated = value.complex_conj();
    assert_eq!(
        conjugated.to_f32_array().as_slice().unwrap(),
        &[1.0, -2.0, 3.0, 4.0]
    );
    let weights = tensor(&[2, 2], vec![2.0, 3.0, 4.0, 5.0], false);
    conjugated.mul(&weights).sum().backward();
    assert_eq!(
        value.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[2.0, -3.0, 4.0, -5.0]
    );

    let left = tensor(&[1, 2], vec![1.0, 2.0], true);
    let right = tensor(&[1, 2], vec![3.0, 4.0], true);
    let product = left.complex_mul(&right);
    assert_eq!(product.to_f32_array().as_slice().unwrap(), &[-5.0, 10.0]);
    product
        .mul(&tensor(&[1, 2], vec![5.0, 7.0], false))
        .sum()
        .backward();
    assert_eq!(
        left.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[43.0, 1.0]
    );
    assert_eq!(
        right.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[19.0, -3.0]
    );
}

#[test]
fn spectral_and_complex_public_contracts_reject_invalid_shapes() {
    let scalar = tensor(&[], vec![1.0], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scalar.fft())).is_err());

    let real = tensor(&[3], vec![1.0; 3], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| real.ifft())).is_err());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| real.complex_conj())).is_err()
    );

    let short_spectrum = tensor(&[1, 2], vec![1.0, 0.0], false);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { short_spectrum.irfft() }))
            .is_err()
    );

    let complex = tensor(&[2, 2], vec![1.0; 4], false);
    let mismatch = tensor(&[1, 2], vec![1.0; 2], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        complex.complex_mul(&mismatch)
    }))
    .is_err());
}
