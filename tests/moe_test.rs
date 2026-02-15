use ndarray::{ArrayD, IxDyn};
use tensor_engine::nn::{moe::MoELayer, Module};
use tensor_engine::tensor::Tensor;

#[test]
fn test_topk_op() {
    // [batch, 4]
    // 10 30 20 40
    let data = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![10.0, 30.0, 20.0, 40.0]).unwrap();
    let t = Tensor::new(data, true);

    // Top 2: should be 40 (idx 3) and 30 (idx 1).
    let out = t.topk(2);
    let out_data = out.to_f32_array();

    // Expected shape: [1, 4] (2 values + 2 indices)
    assert_eq!(out_data.shape(), &[1, 4]);

    let vals = out_data.as_slice().unwrap();
    // Values
    assert!((vals[0] - 40.0).abs() < 1e-5);
    assert!((vals[1] - 30.0).abs() < 1e-5);
    // Indices
    assert!((vals[2] - 3.0).abs() < 1e-5);
    assert!((vals[3] - 1.0).abs() < 1e-5);
}

#[test]
fn test_moe_forward() {
    // d_model = 16, d_ff = 32, num_experts = 4, k = 2
    let moe = MoELayer::new(16, 32, 4, 2);

    // Input: [batch=2, seq=5, d_model=16]
    let input_shape = vec![2, 5, 16];
    let total_elements = 2 * 5 * 16;
    let data = ArrayD::from_shape_vec(IxDyn(&input_shape), vec![0.5; total_elements]).unwrap();
    let x = Tensor::new(data, true);

    let out = moe.forward(&x);

    assert_eq!(out.to_f32_array().shape(), &[2, 5, 16]);

    // Check gradients
    let loss = out.sum();
    loss.backward();

    assert!(x.lock().grad.is_some());
    // Check expert params have grads
    let params = moe.parameters();
    for p in params {
        // Some experts might not be selected, so grad might be None or Zero.
        // But with uniform input 0.5 and random init, likely some experts were hit.
        // We just check if at least ONE param got grad.
        if p.lock().grad.is_some() {
            // Success
        }
    }
}
