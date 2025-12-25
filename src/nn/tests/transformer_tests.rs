use crate::nn::BiasFunction;
use crate::nn::MultiHeadAttention;
use crate::nn::TransformerBlock;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn transformer_block_forward_shape() {
    let b = 2;
    let seq = 4;
    let d_model = 8;
    let d_ff = 16;
    let num_heads = 2;
    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32 * 0.01).collect();
    let x = Tensor::new(Array::from_shape_vec((b, seq, d_model), x_data).unwrap().into_dyn(), true);
    let block = TransformerBlock::new(d_model, d_ff, num_heads).expect("create transformer block");
    let out = block.forward_block_no_cache(&x);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);
}

#[test]
fn mha_forward_with_distance_applies_penalty() {
    let b = 1usize;
    let seq = 3usize;
    let d_model = 4usize;
    let num_heads = 2usize;
    // input
    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| (i % 5) as f32).collect();
    let x = Tensor::new(ndarray::Array::from_shape_vec((b, seq, d_model), x_data).unwrap().into_dyn(), false);
    let mha = MultiHeadAttention::new(d_model, num_heads);
    let mha_nl = MultiHeadAttention::new_with_nl_oob(d_model, num_heads, BiasFunction::Logarithmic, 1.0);
    // Distance matrix: increasing distances
    let mut dist = Vec::new();
    for i in 0..seq {
        for j in 0..seq {
            let d = ((j as isize - i as isize).abs() as f32);
            dist.push(d);
        }
    }
    let dist_t = Tensor::new(ndarray::Array::from_shape_vec((seq, seq), dist).unwrap().into_dyn(), false);
    let out_base = mha.forward(&x);
    let out_nl = mha_nl.forward_with_distance(&x, &dist_t);
    assert_eq!(out_base.lock().storage.shape(), out_nl.lock().storage.shape());
    // Ensure outputs differ when NL-OOB applied
    let a = out_base.lock().storage.to_f32_array();
    let b_arr = out_nl.lock().storage.to_f32_array();
    assert!(a != b_arr, "Outputs with and without NL-OOB should differ");
}

#[test]
fn mha_slopes_are_learnable_and_receive_grad() {
    let b = 1usize;
    let seq = 3usize;
    let d_model = 4usize;
    let num_heads = 2usize;
    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| (i % 7) as f32 * 0.1).collect();
    let x = Tensor::new(ndarray::Array::from_shape_vec((b, seq, d_model), x_data).unwrap().into_dyn(), true);
    let mut mha_nl = MultiHeadAttention::new_with_nl_oob(d_model, num_heads, BiasFunction::Gaussian, 1.0);
    // Build distance matrix
    let mut dist = Vec::new();
    for i in 0..seq {
        for j in 0..seq {
            dist.push(((j as isize - i as isize).abs() as f32) + 1.0);
        }
    }
    let dist_t = Tensor::new(ndarray::Array::from_shape_vec((seq, seq), dist).unwrap().into_dyn(), false);
    // forward/backward
    let out = mha_nl.forward_with_distance(&x, &dist_t);
    let s = out.sum();
    s.backward();
    // slopes should have grad populated
    if let Some(slopes) = &mha_nl.slopes {
        let g = slopes.lock().grad.clone();
        assert!(g.is_some(), "slopes grad must be computed after backward");
    } else {
        panic!("slopes not initialized")
    }
}

#[test]
fn mha_forward_with_causal_masking() {
    let b = 1usize;
    let seq = 4usize;
    let d_model = 8usize;
    let num_heads = 2usize;
    // input with increasing tokens to ensure future tokens influence non-causal attention
    let mut x_data: Vec<f32> = Vec::new();
    for t in 0..seq {
        for _d in 0..d_model {
            x_data.push((t + 1) as f32);
        }
    }
    let x = Tensor::new(ndarray::Array::from_shape_vec((b, seq, d_model), x_data).unwrap().into_dyn(), false);
    let mha = MultiHeadAttention::new(d_model, num_heads);
    let out_base = mha.forward(&x);
    let out_causal = mha.forward_with_causal(&x, true, None);
    assert_eq!(out_base.lock().storage.shape(), out_causal.lock().storage.shape());
    let a = out_base.lock().storage.to_f32_array();
    let b_arr = out_causal.lock().storage.to_f32_array();
    assert!(a != b_arr, "Causal attention should differ from unrestricted attention");
}

#[test]
fn mha_forward_basic_runs() {
    let b = 1usize;
    let seq = 2usize;
    let d_model = 4usize;
    let num_heads = 2usize;
    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32).collect();
    let x = Tensor::new(ndarray::Array::from_shape_vec((b, seq, d_model), x_data).unwrap().into_dyn(), false);
    let mha = MultiHeadAttention::new(d_model, num_heads);
    let out = mha.forward(&x);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);
}
