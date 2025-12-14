use ndarray::{Array, IxDyn};
use tensor_engine::nn::transformer_cleaned::TransformerBlock;
use tensor_engine::tensor::Tensor as TE;
use tensor_engine::tensor::Tensor as TE2;
// use tensor_engine::tensor::Tensor; // not needed explicitly

// Finite-difference numeric gradient helper
fn numeric_gradient<F>(f: F, x: &Array<f32, IxDyn>, h: f32) -> Array<f32, IxDyn>
where
    F: Fn(&Array<f32, IxDyn>) -> f32,
{
    let mut grad = Array::zeros(x.raw_dim());
    for i in 0..x.len() {
        let base = x.as_slice().unwrap()[i].abs();
        let h_local = h * (1.0 + base);
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus.as_slice_mut().unwrap()[i] += h_local;
        x_minus.as_slice_mut().unwrap()[i] -= h_local;
        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);
        grad.as_slice_mut().unwrap()[i] = (f_plus - f_minus) / (2.0 * h_local);
    }
    grad
}

#[test]
fn test_numeric_gradient_llama_linear1_weight() {
    let d_model = 4usize;
    let d_ff = 8usize; // linear1 has out = 2*d_ff
    let num_heads = 2usize;
    let kv_heads = 2usize;
    let mut block = TransformerBlock::new_llama_style(d_model, d_ff, num_heads, kv_heads, false, false);

    // Create a simple input with distinct values
    let arr = Array::from_shape_fn((1, 3, d_model), |(_, s, d)| s as f32 * 0.01 + d as f32 * 0.001).into_dyn();
    let x = TE::new(arr.clone(), true);

    // Initialize linear1 and linear2 weights with patterned values
    let w1 = Array::from_shape_fn((d_model, d_ff * 2), |(i, j)| (i as f32 * 0.01) + (j as f32 * 0.001)).into_dyn();
    block.linear1.weight = TE::new(w1.clone(), true);

    // Do analytic autograd: forward -> sum -> backward
    let y = block.forward_block(&x);
    let s = y.sum();
    s.backward();
    let grad_autograd = block.linear1.weight.lock().grad.clone().unwrap();

    // Numeric gradient w.r.t linear1.weight
    let w1_arr = w1;
    let f = |w_arr: &Array<f32, IxDyn>| {
        let mut bc = block.clone();
        bc.linear1.weight = TE2::new(w_arr.clone(), false);
        let tx = TE2::new(arr.clone(), false);
        let y2 = bc.forward_block(&tx);
        let s2 = y2.sum();
        let val = s2.lock().storage.to_f32_array().iter().next().cloned().unwrap();
        val
    };
    let grad_numeric = numeric_gradient(f, &w1_arr, 1e-3);

    // Compare shapes and approximate equality
    assert_eq!(grad_numeric.shape(), grad_autograd.shape());
    for (a, b) in grad_autograd.iter().zip(grad_numeric.iter()) {
        assert!((a - b).abs() < 5e-2);
    }
}
