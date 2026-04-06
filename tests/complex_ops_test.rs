use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

#[test]
fn test_complex_conj_forward() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![1.0, 2.0, 3.0, -4.0]).unwrap(),
        false,
    );

    let y = x.complex_conj();
    let ya = y.to_f32_array();
    let s = ya.as_slice().unwrap();

    assert!((s[0] - 1.0).abs() < 1e-6);
    assert!((s[1] + 2.0).abs() < 1e-6);
    assert!((s[2] - 3.0).abs() < 1e-6);
    assert!((s[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_complex_mul_forward() {
    // (1 + 2i) * (3 + 4i) = -5 + 10i
    let a = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[1, 2][..]), vec![1.0, 2.0]).unwrap(), false);
    let b = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[1, 2][..]), vec![3.0, 4.0]).unwrap(), false);

    let y = a.complex_mul(&b);
    let s = y.to_f32_array();
    let v = s.as_slice().unwrap();

    assert!((v[0] + 5.0).abs() < 1e-6);
    assert!((v[1] - 10.0).abs() < 1e-6);
}

#[test]
fn test_complex_mul_backward_runs() {
    let a = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[1, 2][..]), vec![1.0, 2.0]).unwrap(), true);
    let b = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[1, 2][..]), vec![3.0, 4.0]).unwrap(), true);

    let y = a.complex_mul(&b);
    let loss = y.sum();
    loss.backward();

    assert!(a.lock().grad.is_some());
    assert!(b.lock().grad.is_some());
}
