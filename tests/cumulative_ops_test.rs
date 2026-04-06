use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

#[test]
fn test_cumsum_forward_dim1() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        false,
    );

    let y = x.cumsum(1);
    let ya = y.to_f32_array();
    let s = ya.as_slice().unwrap();

    assert_eq!(ya.shape(), &[2, 3]);
    assert!((s[0] - 1.0).abs() < 1e-6);
    assert!((s[1] - 3.0).abs() < 1e-6);
    assert!((s[2] - 6.0).abs() < 1e-6);
    assert!((s[3] - 4.0).abs() < 1e-6);
    assert!((s[4] - 9.0).abs() < 1e-6);
    assert!((s[5] - 15.0).abs() < 1e-6);
}

#[test]
fn test_cumsum_backward_dim1() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4][..]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        true,
    );

    let y = x.cumsum(1);
    let loss = y.sum();
    loss.backward();

    let gx = x.lock().grad.clone().unwrap();
    let g = gx.as_slice().unwrap();

    assert!((g[0] - 4.0).abs() < 1e-6);
    assert!((g[1] - 3.0).abs() < 1e-6);
    assert!((g[2] - 2.0).abs() < 1e-6);
    assert!((g[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_cumprod_forward_dim1() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4][..]), vec![2.0, 3.0, 4.0, 5.0]).unwrap(),
        false,
    );

    let y = x.cumprod(1);
    let ya = y.to_f32_array();
    let s = ya.as_slice().unwrap();

    assert_eq!(ya.shape(), &[1, 4]);
    assert!((s[0] - 2.0).abs() < 1e-6);
    assert!((s[1] - 6.0).abs() < 1e-6);
    assert!((s[2] - 24.0).abs() < 1e-6);
    assert!((s[3] - 120.0).abs() < 1e-6);
}

#[test]
fn test_cumprod_backward_dim1() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3][..]), vec![2.0, 3.0, 4.0]).unwrap(),
        true,
    );

    let y = x.cumprod(1);
    let loss = y.sum();
    loss.backward();

    let gx = x.lock().grad.clone().unwrap();
    let g = gx.as_slice().unwrap();

    // y = [x0, x0*x1, x0*x1*x2] ; d sum(y)/dx = [1 + x1 + x1*x2, x0 + x0*x2, x0*x1]
    assert!((g[0] - (1.0 + 3.0 + 12.0)).abs() < 1e-6);
    assert!((g[1] - (2.0 + 8.0)).abs() < 1e-6);
    assert!((g[2] - 6.0).abs() < 1e-6);
}

#[test]
fn test_cummax_forward_and_backward_dim1() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4][..]), vec![2.0, 1.0, 3.0, 2.5]).unwrap(),
        true,
    );

    let y = x.cummax(1);
    let ya = y.to_f32_array();
    let s = ya.as_slice().unwrap();
    assert!((s[0] - 2.0).abs() < 1e-6);
    assert!((s[1] - 2.0).abs() < 1e-6);
    assert!((s[2] - 3.0).abs() < 1e-6);
    assert!((s[3] - 3.0).abs() < 1e-6);

    let loss = y.sum();
    loss.backward();
    let gx = x.lock().grad.clone().unwrap();
    let g = gx.as_slice().unwrap();

    // Running max arg indices: [0,0,2,2]
    assert!((g[0] - 2.0).abs() < 1e-6);
    assert!((g[1] - 0.0).abs() < 1e-6);
    assert!((g[2] - 2.0).abs() < 1e-6);
    assert!((g[3] - 0.0).abs() < 1e-6);
}

#[test]
fn test_cummin_forward_and_backward_dim1() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4][..]), vec![2.0, 1.0, 3.0, 0.5]).unwrap(),
        true,
    );

    let y = x.cummin(1);
    let ya = y.to_f32_array();
    let s = ya.as_slice().unwrap();
    assert!((s[0] - 2.0).abs() < 1e-6);
    assert!((s[1] - 1.0).abs() < 1e-6);
    assert!((s[2] - 1.0).abs() < 1e-6);
    assert!((s[3] - 0.5).abs() < 1e-6);

    let loss = y.sum();
    loss.backward();
    let gx = x.lock().grad.clone().unwrap();
    let g = gx.as_slice().unwrap();

    // Running min arg indices: [0,1,1,3]
    assert!((g[0] - 1.0).abs() < 1e-6);
    assert!((g[1] - 2.0).abs() < 1e-6);
    assert!((g[2] - 0.0).abs() < 1e-6);
    assert!((g[3] - 1.0).abs() < 1e-6);
}
