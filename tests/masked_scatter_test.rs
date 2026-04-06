use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

#[test]
fn test_masked_scatter_forward_broadcast() {
    let base = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        false,
    );
    let mask = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3][..]), vec![1.0, 0.0, 1.0]).unwrap(),
        false,
    );
    let source = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4][..]), vec![9.0, 8.0, 7.0, 6.0]).unwrap(),
        false,
    );

    let out = base.masked_scatter(&mask, &source);
    let out_arr = out.to_f32_array();
    let vals = out_arr.as_slice().unwrap();

    assert!((vals[0] - 9.0).abs() < 1e-6);
    assert!((vals[1] - 2.0).abs() < 1e-6);
    assert!((vals[2] - 8.0).abs() < 1e-6);
    assert!((vals[3] - 7.0).abs() < 1e-6);
    assert!((vals[4] - 5.0).abs() < 1e-6);
    assert!((vals[5] - 6.0).abs() < 1e-6);
}

#[test]
fn test_masked_scatter_backward_routes_gradients() {
    let base = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        true,
    );
    let mask = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3][..]), vec![1.0, 0.0, 1.0]).unwrap(),
        false,
    );
    let source = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4][..]), vec![9.0, 8.0, 7.0, 6.0]).unwrap(),
        true,
    );

    let out = base.masked_scatter(&mask, &source);
    let loss = out.sum();
    loss.backward();

    let base_grad = base.lock().grad.clone().unwrap();
    let source_grad = source.lock().grad.clone().unwrap();

    let bg = base_grad.as_slice().unwrap();
    let sg = source_grad.as_slice().unwrap();

    assert!((bg[0] - 0.0).abs() < 1e-6);
    assert!((bg[1] - 1.0).abs() < 1e-6);
    assert!((bg[2] - 0.0).abs() < 1e-6);
    assert!((bg[3] - 0.0).abs() < 1e-6);
    assert!((bg[4] - 1.0).abs() < 1e-6);
    assert!((bg[5] - 0.0).abs() < 1e-6);

    assert!((sg[0] - 1.0).abs() < 1e-6);
    assert!((sg[1] - 1.0).abs() < 1e-6);
    assert!((sg[2] - 1.0).abs() < 1e-6);
    assert!((sg[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_masked_scatter_short_source_keeps_tail_and_grad_flows_to_base() {
    let base = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        true,
    );
    let mask = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3][..]), vec![1.0, 0.0, 1.0]).unwrap(),
        false,
    );
    let source = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2][..]), vec![10.0, 11.0]).unwrap(),
        true,
    );

    let out = base.masked_scatter(&mask, &source);
    let out_arr = out.to_f32_array();
    let vals = out_arr.as_slice().unwrap();

    assert!((vals[0] - 10.0).abs() < 1e-6);
    assert!((vals[1] - 2.0).abs() < 1e-6);
    assert!((vals[2] - 11.0).abs() < 1e-6);
    assert!((vals[3] - 4.0).abs() < 1e-6);
    assert!((vals[4] - 5.0).abs() < 1e-6);
    assert!((vals[5] - 6.0).abs() < 1e-6);

    let loss = out.sum();
    loss.backward();

    let base_grad = base.lock().grad.clone().unwrap();
    let source_grad = source.lock().grad.clone().unwrap();

    let bg = base_grad.as_slice().unwrap();
    let sg = source_grad.as_slice().unwrap();

    assert!((bg[0] - 0.0).abs() < 1e-6);
    assert!((bg[1] - 1.0).abs() < 1e-6);
    assert!((bg[2] - 0.0).abs() < 1e-6);
    assert!((bg[3] - 1.0).abs() < 1e-6);
    assert!((bg[4] - 1.0).abs() < 1e-6);
    assert!((bg[5] - 1.0).abs() < 1e-6);

    assert!((sg[0] - 1.0).abs() < 1e-6);
    assert!((sg[1] - 1.0).abs() < 1e-6);
}
