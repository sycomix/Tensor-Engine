use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

#[test]
fn test_det_forward_2x2() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        false,
    );

    let y = x.det();
    let s = y.to_f32_array();
    assert_eq!(s.shape(), &[] as &[usize]);
    assert!((s.iter().next().copied().unwrap_or(0.0) - (-2.0)).abs() < 1e-6);
}

#[test]
fn test_det_backward_2x2() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        true,
    );

    let y = x.det();
    y.backward();

    let gx = x.lock().grad.clone().unwrap();
    let g = gx.as_slice().unwrap();

    // d det(A) / dA = det(A) * A^{-T} = [[d, -c], [-b, a]] for 2x2.
    assert!((g[0] - 4.0).abs() < 1e-6);
    assert!((g[1] - (-3.0)).abs() < 1e-6);
    assert!((g[2] - (-2.0)).abs() < 1e-6);
    assert!((g[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_inv_forward_2x2() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![4.0, 7.0, 2.0, 6.0]).unwrap(),
        false,
    );

    let y = x.inv();
    let ya = y.to_f32_array();
    let s = ya.as_slice().unwrap();

    // inv([[4,7],[2,6]]) = 1/10 * [[6,-7],[-2,4]]
    assert!((s[0] - 0.6).abs() < 1e-6);
    assert!((s[1] - (-0.7)).abs() < 1e-6);
    assert!((s[2] - (-0.2)).abs() < 1e-6);
    assert!((s[3] - 0.4).abs() < 1e-6);
}

#[test]
fn test_inv_backward_sum_loss_diagonal_matrix() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![2.0, 0.0, 0.0, 4.0]).unwrap(),
        true,
    );

    let y = x.inv();
    let loss = y.sum();
    loss.backward();

    let gx = x.lock().grad.clone().unwrap();
    let g = gx.as_slice().unwrap();

    // dL/dA = -A^{-T} * 1 * A^{-T}; A^{-1}=diag(0.5,0.25)
    assert!((g[0] - (-0.25)).abs() < 1e-6);
    assert!((g[1] - (-0.125)).abs() < 1e-6);
    assert!((g[2] - (-0.125)).abs() < 1e-6);
    assert!((g[3] - (-0.0625)).abs() < 1e-6);
}

#[test]
fn test_det_and_inv_batched_forward() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2][..]),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 7.0, 2.0, 6.0],
        )
            .unwrap(),
        false,
    );

    let d = x.det();
    let da = d.to_f32_array();
    let ds = da.as_slice().unwrap();
    assert_eq!(da.shape(), &[2]);
    assert!((ds[0] - (-2.0)).abs() < 1e-6);
    assert!((ds[1] - 10.0).abs() < 1e-6);

    let inv = x.inv();
    let ia = inv.to_f32_array();
    assert_eq!(ia.shape(), &[2, 2, 2]);
}
