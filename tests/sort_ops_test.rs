use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

#[test]
fn test_sort_forward_last_dim_ascending() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 4][..]), vec![3.0, 1.0, 4.0, 2.0, -1.0, 5.0, 0.0, 2.0])
            .unwrap(),
        false,
    );

    let y = x.sort();
    let ya = y.to_f32_array();
    assert_eq!(ya.shape(), &[2, 4]);
    let s = ya.as_slice().unwrap();

    assert!((s[0] - 1.0).abs() < 1e-6);
    assert!((s[1] - 2.0).abs() < 1e-6);
    assert!((s[2] - 3.0).abs() < 1e-6);
    assert!((s[3] - 4.0).abs() < 1e-6);

    assert!((s[4] - -1.0).abs() < 1e-6);
    assert!((s[5] - 0.0).abs() < 1e-6);
    assert!((s[6] - 2.0).abs() < 1e-6);
    assert!((s[7] - 5.0).abs() < 1e-6);
}

#[test]
fn test_argsort_forward_last_dim_ascending() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4][..]), vec![3.0, 1.0, 4.0, 2.0]).unwrap(),
        false,
    );

    let idx = x.argsort();
    let ia = idx.to_f32_array();
    assert_eq!(ia.shape(), &[1, 4]);
    let s = ia.as_slice().unwrap();

    assert!((s[0] - 1.0).abs() < 1e-6);
    assert!((s[1] - 3.0).abs() < 1e-6);
    assert!((s[2] - 0.0).abs() < 1e-6);
    assert!((s[3] - 2.0).abs() < 1e-6);
}

#[test]
fn test_sort_backward_routes_gradients_to_original_positions() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4][..]), vec![3.0, 1.0, 4.0, 2.0]).unwrap(),
        true,
    );

    let y = x.sort();
    let loss = y.sum();
    loss.backward();

    let gx = x.lock().grad.clone().unwrap();
    let g = gx.as_slice().unwrap();
    for &v in g {
        assert!((v - 1.0).abs() < 1e-6);
    }
}
