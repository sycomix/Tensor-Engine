use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

#[test]
fn test_where_select_forward_broadcast() {
    let cond = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 0.0]).unwrap(),
        false,
    );
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
        false,
    );
    let y = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
        false,
    );

    let out = Tensor::where_select(&cond, &x, &y);
    let out_data = out.to_f32_array();
    assert_eq!(out_data.shape(), &[2, 3]);

    let vals = out_data.as_slice().unwrap();
    assert!((vals[0] - 10.0).abs() < 1e-6);
    assert!((vals[1] - 20.0).abs() < 1e-6);
    assert!((vals[2] - 30.0).abs() < 1e-6);
    assert!((vals[3] - 1.0).abs() < 1e-6);
    assert!((vals[4] - 2.0).abs() < 1e-6);
    assert!((vals[5] - 3.0).abs() < 1e-6);
}

#[test]
fn test_where_select_backward_routes_gradients() {
    let cond = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        false,
    );
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 2.0, 2.0, 2.0]).unwrap(),
        true,
    );
    let y = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![3.0, 3.0, 3.0, 3.0]).unwrap(),
        true,
    );

    let out = Tensor::where_select(&cond, &x, &y);
    let loss = out.sum();
    loss.backward();

    let gx = x.lock().grad.clone().unwrap();
    let gy = y.lock().grad.clone().unwrap();

    let gx_vals = gx.as_slice().unwrap();
    let gy_vals = gy.as_slice().unwrap();

    assert!((gx_vals[0] - 1.0).abs() < 1e-6);
    assert!((gx_vals[1] - 0.0).abs() < 1e-6);
    assert!((gx_vals[2] - 0.0).abs() < 1e-6);
    assert!((gx_vals[3] - 1.0).abs() < 1e-6);

    assert!((gy_vals[0] - 0.0).abs() < 1e-6);
    assert!((gy_vals[1] - 1.0).abs() < 1e-6);
    assert!((gy_vals[2] - 1.0).abs() < 1e-6);
    assert!((gy_vals[3] - 0.0).abs() < 1e-6);
}

#[test]
fn test_masked_fill_forward() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        false,
    );
    let mask = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unwrap(),
        false,
    );

    let out = x.masked_fill(&mask, -5.0);
    let out_vals = out.to_f32_array();
    let vals = out_vals.as_slice().unwrap();

    assert!((vals[0] - 1.0).abs() < 1e-6);
    assert!((vals[1] + 5.0).abs() < 1e-6);
    assert!((vals[2] - 3.0).abs() < 1e-6);
    assert!((vals[3] + 5.0).abs() < 1e-6);
    assert!((vals[4] - 5.0).abs() < 1e-6);
    assert!((vals[5] + 5.0).abs() < 1e-6);
}
