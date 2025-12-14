use ndarray::Array;
use ndarray::IxDyn;
use tensor_engine::nn::transformer::{BiasFunction, MultiHeadAttention};
use tensor_engine::nn::Module;
use tensor_engine::tensor::Tensor;

#[test]
fn test_nl_oob_forward_affects_logits() {
    let dha = 8usize;
    let num_heads = 2usize;
    let mut mha = MultiHeadAttention::new_with_nl_oob(dha, num_heads, BiasFunction::Logarithmic, 1.0);
    let b = 1usize;
    let seq = 3usize;
    let d = dha;
    // Set non-zero weights for q/k/v so outputs differ
    mha.linear_q.weight = Tensor::new(Array::from_elem(IxDyn(&[d, d]), 0.5f32), false);
    mha.linear_k.weight = Tensor::new(Array::from_elem(IxDyn(&[d, d]), 0.5f32), false);
    let v_weights: Vec<f32> = (0..(d * d)).map(|i| 0.5f32 + (i as f32) * 0.01f32).collect();
    mha.linear_v.weight = Tensor::new(Array::from_shape_vec(IxDyn(&[d, d]), v_weights).unwrap(), false);
    mha.linear_o.weight = Tensor::new(Array::from_elem(IxDyn(&[d, d]), 0.5f32), false);
    // Set slopes to a non-uniform scale to ensure NL-OOB has effect
    mha.slopes = Some(Tensor::new(ndarray::Array::from_shape_vec(IxDyn(&[1, num_heads, 1, 1]), vec![2.0f32; num_heads]).unwrap(), true));

    let inp = Tensor::new(Array::from_elem(IxDyn(&[b, seq, d]), 0.5f32), false);
    // Distance matrix with non-uniform values so bias affects attention differently
    let mut dist_arr = ndarray::Array::zeros(IxDyn(&[seq, seq]));
    for i in 0..seq {
        for j in 0..seq {
            dist_arr[[i, j]] = (i as f32) * 10.0 + (j as f32);
        }
    }
    let dist = Tensor::new(dist_arr.into_dyn(), false);
    // Check q is non-zero
    let q = mha.linear_q.forward(&inp);
    let q_arr = q.lock().storage.to_f32_array();
    // sum of values should not be zero if weights are not zero
    let sum_q = q_arr.iter().fold(0f32, |acc, x| acc + *x);
    assert!(sum_q != 0.0);
    let out1 = mha.forward_impl(&inp);
    let out2 = mha.forward_with_distance(&inp, &dist);
    // The outputs should not be identical when NL-OOB is configured and a non-zero distance matrix is provided
    let a = out1.lock().storage.to_f32_array();
    let b = out2.lock().storage.to_f32_array();
    assert_ne!(a, b);
}
