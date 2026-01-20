use tensor_engine::nn::{Linear, Module};
use tensor_engine::optim::{Optimizer, SGD};
use tensor_engine::tensor::Tensor;
use ndarray::ArrayD;

fn main() {
    // 1. Data Generation
    // y = 2x + 1
    let x_val = vec![1.0, 2.0, 3.0, 4.0];
    let y_val = vec![3.0, 5.0, 7.0, 9.0]; // 2*x + 1
    
    // Reshape to [4, 1]
    let x = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 1]), x_val).unwrap(),
        false
    );
    let y = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 1]), y_val).unwrap(),
        false
    );

    // 2. Model Definition
    // Linear(1, 1)
    let model = Linear::new(1, 1, true);
    
    // 3. Optimizer
    let mut optim = SGD::new(model.parameters(), 0.05);

    println!("Initial params:");
    for (name, p) in model.named_parameters("model") {
        println!("{}: {:?}", name, p.lock().storage.to_f32_array());
    }

    // 4. Training Loop
    for epoch in 0..1000 {
        // Zero grad
        optim.zero_grad();

        // Forward
        let out = model.forward(&x);

        // Loss (MSE)
        let diff = out.sub(&y);
        let sq_diff = diff.pow(2.0);
        let loss = sq_diff.mean();

        // Backward
        loss.backward();

        // Step
        optim.step();

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss.lock().storage.to_f32_array());
        }
    }

    println!("Final params:");
    for (name, p) in model.named_parameters("model") {
        println!("{}: {:?}", name, p.lock().storage.to_f32_array());
    }

    // Verify results
    // Expected: weight ~ 2.0, bias ~ 1.0
    let w = model.weight.lock().storage.to_f32_array();
    let b = model.bias.as_ref().unwrap().lock().storage.to_f32_array();
    println!("Final Weight: {}", w);
    println!("Final Bias: {}", b);

    let w_val = w[[0,0]];
    let b_val = b[[0]];
    
    assert!((w_val - 2.0).abs() < 0.1, "Weight did not converge");
    assert!((b_val - 1.0).abs() < 0.1, "Bias did not converge");
    println!("Verification Passed!");
}
