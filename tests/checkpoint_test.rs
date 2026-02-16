use tensor_engine::autograd;
use tensor_engine::tensor::Tensor;

#[test]
fn test_checkpoint_correctness_simple() {
    use ndarray::{ArrayD, IxDyn};
    // Define a simple function: y = (x * x) * x = x^3
    // We will compute gradients with and without checkpointing logic.
    // Note: Since 'checkpoint' creates a closed loop re-computation,
    // we strictly check if gradients match standard execution.

    let f = |inputs: &[Tensor]| -> Tensor {
        let x = &inputs[0];
        let y = x.mul(x);
        y.mul(x)
    };

    // Standard execution
    let data = ArrayD::from_shape_vec(IxDyn(&[1usize][..]), vec![2.0]).unwrap();
    let x_std = Tensor::new(data.clone(), true);
    let y_std = f(&[x_std.clone()][..]);
    y_std.backward();
    let grad_std = x_std.lock().grad.clone().unwrap()[[0]];

    // Checkpointed execution
    let x_cp = Tensor::new(data, true);
    let y_cp = autograd::checkpoint(f, &[x_cp.clone()][..]);
    y_cp.backward();
    let grad_cp = x_cp.lock().grad.clone().unwrap()[[0]];

    // Expected derivative of x^3 at x=2 is 3*x^2 = 3*4 = 12.
    println!("Standard grad: {}", grad_std);
    println!("Checkpoint grad: {}", grad_cp);

    assert!((grad_std - 12.0).abs() < 1e-5, "Standard grad incorrect");
    assert!((grad_cp - 12.0).abs() < 1e-5, "Checkpoint grad incorrect");
    assert!((grad_std - grad_cp).abs() < 1e-5, "Gradients mismatch");
}

#[test]
fn test_checkpoint_shared_inputs() {
    use ndarray::{ArrayD, IxDyn};
    // f(a, b) = a * b
    let f = |inputs: &[Tensor]| -> Tensor { inputs[0].mul(&inputs[1]) };

    let data_a = ArrayD::from_shape_vec(IxDyn(&[1usize][..]), vec![3.0]).unwrap();
    let data_b = ArrayD::from_shape_vec(IxDyn(&[1usize][..]), vec![4.0]).unwrap();

    let a = Tensor::new(data_a, true);
    let b = Tensor::new(data_b, true);

    let y = autograd::checkpoint(f, &[a.clone(), b.clone()][..]);
    y.backward();

    // d(a*b)/da = b = 4
    // d(a*b)/db = a = 3
    let grad_a = a.lock().grad.clone().unwrap()[[0]];
    let grad_b = b.lock().grad.clone().unwrap()[[0]];

    assert!((grad_a - 4.0).abs() < 1e-5);
    assert!((grad_b - 3.0).abs() < 1e-5);
}
