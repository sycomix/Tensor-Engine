use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

#[test]
fn test_index_select_forward_dim1() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        false,
    );
    let indices = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 0.0]).unwrap(), false);

    let out = x.index_select(1, &indices);
    let vals = out.to_f32_array();
    let s = vals.as_slice().unwrap();

    assert_eq!(vals.shape(), &[2, 2]);
    assert!((s[0] - 3.0).abs() < 1e-6);
    assert!((s[1] - 1.0).abs() < 1e-6);
    assert!((s[2] - 6.0).abs() < 1e-6);
    assert!((s[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_index_select_backward_accumulates_repeated_indices() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        true,
    );
    let indices = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 1.0, 0.0]).unwrap(), false);

    let out = x.index_select(1, &indices);
    let loss = out.sum();
    loss.backward();

    let gx = x.lock().grad.clone().unwrap();
    let g = gx.as_slice().unwrap();

    assert!((g[0] - 1.0).abs() < 1e-6);
    assert!((g[1] - 2.0).abs() < 1e-6);
    assert!((g[2] - 0.0).abs() < 1e-6);
    assert!((g[3] - 1.0).abs() < 1e-6);
    assert!((g[4] - 2.0).abs() < 1e-6);
    assert!((g[5] - 0.0).abs() < 1e-6);
}
