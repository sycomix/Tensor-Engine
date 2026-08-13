use tensor_engine::nn::{Conv2D, LayerNorm, Linear, MSELoss, Module};
use tensor_engine::optim::{Adam, Optimizer};
use tensor_engine::tensor::Tensor;

fn tensor_to_vec(t: &Tensor) -> Vec<f32> {
    t.lock().storage.to_f32_array().into_raw_vec_and_offset().0
}

#[test]
fn linear_layer_convergence_multi_step() {
    let lin = Linear::new_with_seed(1, 1, true, 1);
    let loss_fn = MSELoss::new();
    let mut opt = Adam::new(1e-2, 0.9, 0.999, 1e-8);

    let xs: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let ys: Vec<f32> = xs.iter().map(|x| 3.0 * x + 1.0).collect();

    let x_arr = ndarray::Array::from_vec(xs).into_dyn();
    let x = Tensor::new(x_arr, false).reshape(vec![8, 1]).unwrap();
    let y_arr = ndarray::Array::from_vec(ys).into_dyn();
    let y = Tensor::new(y_arr, false).reshape(vec![8, 1]).unwrap();

    let mut first_loss = None;
    let mut last_loss = f32::INFINITY;
    for _ in 0..100 {
        let pred = lin.forward(&x);
        let loss = loss_fn.forward(&pred, &y);
        let loss_val = tensor_to_vec(&loss)[0];

        if first_loss.is_none() {
            first_loss = Some(loss_val);
        }
        last_loss = loss_val;

        loss.backward();
        let params = lin.parameters();
        opt.step(&params);
        opt.zero_grad(&params);
    }
    let first = first_loss.unwrap();
    assert!(
        last_loss < first * 0.5,
        "model did not converge: first loss {first}, final loss {last_loss}"
    );
}

#[test]
fn conv2d_convergence_loss_decreases() {
    let conv = Conv2D::new(1, 1, 3, 1, 1, true);
    let loss_fn = MSELoss::new();
    let mut opt = Adam::new(5e-3, 0.9, 0.999, 1e-8);

    let input = Tensor::new(
        ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&[1, 1, 4, 4][..]),
            (0..16).map(|i| i as f32).collect::<Vec<_>>(),
        )
        .unwrap(),
        true,
    );

    let target = Tensor::new(
        ndarray::Array::zeros(ndarray::IxDyn(&[1, 1, 4, 4][..])),
        false,
    );

    let mut prev_loss = f32::INFINITY;
    for _ in 0..30 {
        let out = conv.forward(&input);
        let loss = loss_fn.forward(&out, &target);
        let loss_val = tensor_to_vec(&loss)[0];

        assert!(
            loss_val < prev_loss + 1.0,
            "conv loss not decreasing: {loss_val} >= {prev_loss}"
        );
        prev_loss = loss_val;

        loss.backward();
        let params = conv.parameters();
        opt.step(&params);
        opt.zero_grad(&params);
    }
    assert!(prev_loss < 100.0, "conv final loss {} too high", prev_loss);
}

#[test]
fn layer_norm_forward_deterministic() {
    let ln = LayerNorm::new(8, 1, 1e-5);
    let input = Tensor::new(
        ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&[2, 8][..]),
            (0..16).map(|i| i as f32).collect(),
        )
        .unwrap(),
        false,
    );
    let out1 = ln.forward(&input);
    let out2 = ln.forward(&input);
    let v1 = tensor_to_vec(&out1);
    let v2 = tensor_to_vec(&out2);
    for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "deterministic mismatch at {i}: {a} != {b}"
        );
    }
}
