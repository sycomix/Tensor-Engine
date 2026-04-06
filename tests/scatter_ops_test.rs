use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

#[test]
fn test_scatter_forward_dim1() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        false,
    );
    let index = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![2.0, 0.0, 1.0, 2.0]).unwrap(),
        false,
    );
    let src = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![9.0, 8.0, 7.0, 6.0]).unwrap(),
        false,
    );

    let out = x.scatter(1, &index, &src);
    let vals = out.to_f32_array();
    let s = vals.as_slice().unwrap();

    assert_eq!(vals.shape(), &[2, 3]);
    assert!((s[0] - 8.0).abs() < 1e-6);
    assert!((s[1] - 2.0).abs() < 1e-6);
    assert!((s[2] - 9.0).abs() < 1e-6);
    assert!((s[3] - 4.0).abs() < 1e-6);
    assert!((s[4] - 7.0).abs() < 1e-6);
    assert!((s[5] - 6.0).abs() < 1e-6);
}

#[test]
fn test_scatter_backward_basic() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        true,
    );
    let index = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![2.0, 0.0, 1.0, 2.0]).unwrap(),
        false,
    );
    let src = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![9.0, 8.0, 7.0, 6.0]).unwrap(),
        true,
    );

    let out = x.scatter(1, &index, &src);
    let loss = out.sum();
    loss.backward();

    let gx = x.lock().grad.clone().unwrap();
    let gs = src.lock().grad.clone().unwrap();
    let gxs = gx.as_slice().unwrap();
    let gss = gs.as_slice().unwrap();

    assert!((gxs[0] - 0.0).abs() < 1e-6);
    assert!((gxs[1] - 1.0).abs() < 1e-6);
    assert!((gxs[2] - 0.0).abs() < 1e-6);
    assert!((gxs[3] - 1.0).abs() < 1e-6);
    assert!((gxs[4] - 0.0).abs() < 1e-6);
    assert!((gxs[5] - 0.0).abs() < 1e-6);

    assert!((gss[0] - 1.0).abs() < 1e-6);
    assert!((gss[1] - 1.0).abs() < 1e-6);
    assert!((gss[2] - 1.0).abs() < 1e-6);
    assert!((gss[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_scatter_add_forward_dim1_with_collisions() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        false,
    );
    let index = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![1.0, 1.0, 0.0, 2.0]).unwrap(),
        false,
    );
    let src = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
        false,
    );

    let out = x.scatter_add(1, &index, &src);
    let vals = out.to_f32_array();
    let s = vals.as_slice().unwrap();

    assert_eq!(vals.shape(), &[2, 3]);
    assert!((s[0] - 1.0).abs() < 1e-6);
    assert!((s[1] - 32.0).abs() < 1e-6);
    assert!((s[2] - 3.0).abs() < 1e-6);
    assert!((s[3] - 34.0).abs() < 1e-6);
    assert!((s[4] - 5.0).abs() < 1e-6);
    assert!((s[5] - 46.0).abs() < 1e-6);
}

#[test]
fn test_scatter_add_backward_basic() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        true,
    );
    let index = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![1.0, 1.0, 0.0, 2.0]).unwrap(),
        false,
    );
    let src = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
        true,
    );

    let out = x.scatter_add(1, &index, &src);
    let loss = out.sum();
    loss.backward();

    let gx = x.lock().grad.clone().unwrap();
    let gs = src.lock().grad.clone().unwrap();
    let gxs = gx.as_slice().unwrap();
    let gss = gs.as_slice().unwrap();

    for &v in gxs {
        assert!((v - 1.0).abs() < 1e-6);
    }
    for &v in gss {
        assert!((v - 1.0).abs() < 1e-6);
    }
}
