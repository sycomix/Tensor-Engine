use crate::optim::{AdamW, Optimizer, SGD};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

#[test]
fn clip_gradients_scales_by_global_norm() {
    let p = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2][..]), vec![0.0, 0.0]).unwrap(),
        true,
    );

    {
        let mut lock = p.lock();
        lock.grad = Some(ArrayD::from_shape_vec(IxDyn(&[2][..]), vec![3.0, 4.0]).unwrap());
    }

    let mut opt = SGD::new(0.1, 0.0);
    opt.clip_gradients(std::slice::from_ref(&p), 1.0);

    let g = p.lock().grad.clone().unwrap();
    let gs = g.as_slice().unwrap();
    assert!((gs[0] - 0.6).abs() < 1e-6);
    assert!((gs[1] - 0.8).abs() < 1e-6);
}

#[test]
fn clip_grad_values_clamps_elementwise() {
    let p = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![0.0, 0.0, 0.0]).unwrap(),
        true,
    );

    {
        let mut lock = p.lock();
        lock.grad = Some(ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![2.0, -3.0, 0.5]).unwrap());
    }

    let mut opt = SGD::new(0.1, 0.0);
    opt.clip_grad_values(std::slice::from_ref(&p), 1.0);

    let g = p.lock().grad.clone().unwrap();
    let gs = g.as_slice().unwrap();
    assert!((gs[0] - 1.0).abs() < 1e-6);
    assert!((gs[1] - (-1.0)).abs() < 1e-6);
    assert!((gs[2] - 0.5).abs() < 1e-6);
}

#[test]
fn adamw_applies_decoupled_weight_decay_when_grad_zero() {
    let p = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![1.0]).unwrap(),
        true,
    );

    {
        let mut lock = p.lock();
        lock.grad = Some(ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![0.0]).unwrap());
    }

    let mut opt = AdamW::new(0.1, 0.0, 0.0, 1e-8, 0.1);
    opt.step(std::slice::from_ref(&p));

    let v = p.to_f32_array();
    let s = v.as_slice().unwrap();
    assert!((s[0] - 0.99).abs() < 1e-6);
}
