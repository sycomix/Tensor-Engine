use ndarray::{arr1, Array2};
use tensor_engine::nn::{Linear, Module, Optimizer, Sequential, SGD};
use tensor_engine::tensor::Tensor;

#[test]
fn test_linear_forward() {
    let linear = Linear::new(2, 3, true);
    let input = Tensor::new(Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap().into_dyn(), false);
    let output = linear.forward(&input);

    // Check output shape
    assert_eq!(output.lock().data.shape(), &[1, 3]);
}

#[test]
fn test_sequential() {
    let seq = Sequential::new()
        .add(Linear::new(2, 3, true))
        .add(Linear::new(3, 1, false));

    let input = Tensor::new(Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap().into_dyn(), false);
    let output = seq.forward(&input);

    assert_eq!(output.lock().data.shape(), &[1, 1]);
}

#[test]
fn test_sgd_step() {
    let param = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), true);
    param.lock().grad = Some(arr1(&[0.1, 0.2]).into_dyn());

    let mut sgd = SGD::new(0.1, 0.0);
    sgd.step(&[param.clone()]);

    let data = param.lock().data.clone();
    // param -= lr * grad = [1, 2] - 0.1 * [0.1, 0.2] = [0.99, 1.98]
    assert!((data[0] - 0.99).abs() < 1e-6);
    assert!((data[1] - 1.98).abs() < 1e-6);
}

#[test]
fn test_zero_grad() {
    let param = Tensor::new(arr1(&[1.0]).into_dyn(), true);
    param.lock().grad = Some(arr1(&[1.0]).into_dyn());

    param.zero_grad();

    assert!(param.lock().grad.is_none());
}

#[test]
fn test_optimizer_zero_grad() {
    let param = Tensor::new(arr1(&[1.0]).into_dyn(), true);
    param.lock().grad = Some(arr1(&[1.0]).into_dyn());

    let mut sgd = SGD::new(0.1, 0.0);
    sgd.zero_grad(&[param.clone()]);

    assert!(param.lock().grad.is_none());
}