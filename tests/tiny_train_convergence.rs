use ndarray::Array;
use tensor_engine::nn::Module;
use tensor_engine::nn::{Adam, Linear, MSELoss, Optimizer};
use tensor_engine::tensor::Tensor;

#[test]
fn tiny_linear_convergence() {
    // Fit y = 2*x + 1 with a single Linear layer
    let lin = Linear::new(1, 1, true);
    let x_arr = Array::from_vec(vec![0.0f32, 1.0, 2.0, 3.0]).into_dyn();
    let x = Tensor::new(x_arr.clone(), false).reshape(vec![4, 1]).unwrap();
    let y_arr = Array::from_vec(vec![1.0f32, 3.0, 5.0, 7.0]).into_dyn();
    let y = Tensor::new(y_arr.clone(), false).reshape(vec![4, 1]).unwrap();
    let loss_fn = MSELoss::new();
    // initial loss
    let pred = <dyn Module>::forward(&lin, &x);
    let loss0 = loss_fn.forward(&pred, &y);
    let loss0v = loss0.lock().storage.to_f32_array().iter().next().cloned().unwrap();
    // one training step
    loss0.backward();
    let mut opt = Adam::new(1e-2, 0.9, 0.999, 1e-8);
    let params = <dyn Module>::parameters(&lin);
    opt.clip_gradients(&params, 1.0);
    opt.step(&params);
    let pred2 = <dyn Module>::forward(&lin, &x);
    let loss1 = loss_fn.forward(&pred2, &y);
    let loss1v = loss1.lock().storage.to_f32_array().iter().next().cloned().unwrap();
    assert!(loss1v <= loss0v, "Loss must not increase after one optimization step: {} -> {}", loss0v, loss1v);
}
