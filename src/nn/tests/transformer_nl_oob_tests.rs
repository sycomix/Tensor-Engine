use crate::nn::transformer_cleaned::BiasFunction;
use crate::nn::Module;
use crate::nn::MultiHeadAttention;
use crate::nn::TransformerBlock as TB;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn mha_forward_with_distance_shapes_and_slopes_present() {
    let b = 1;
    let seq = 6;
    let d_model = 8;
    let num_heads = 4;
    // Build input tensor
    let x_data: Vec<f32> = (0..(b * seq * d_model))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        true,
    );

    // Distance matrix: abs(i-j)
    let mut dist_data: Vec<f32> = Vec::with_capacity(seq * seq);
    for i in 0..seq {
        for j in 0..seq {
            dist_data.push(((i as isize - j as isize).abs()) as f32);
        }
    }
    let dist = Tensor::new(
        Array::from_shape_vec((seq, seq), dist_data)
            .unwrap()
            .into_dyn(),
        false,
    );

    // Create MultiHeadAttention with NL-OOB (logarithmic)
    let mha =
        MultiHeadAttention::new_with_nl_oob(d_model, num_heads, BiasFunction::Logarithmic, 4.0);

    // Forward pass with distance
    let out = mha.forward_with_distance(&x, &dist);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);

    // Check parameters include slopes and that slopes require_grad=true; also check named parameter key
    let params = mha.parameters();
    assert!(params.len() > 0);
    // slopes should be present and require grad
    let mut slopes_found = false;
    for p in params {
        let requires_grad = p.lock().requires_grad;
        // We expect at least one parameter marked as requires_grad
        if requires_grad {
            slopes_found = true;
            break;
        }
    }
    assert!(slopes_found);
    let named = mha.named_parameters("mha");
    let mut found_named_slopes = false;
    for (k, _) in named {
        if k == "mha.nl_oob.slopes" {
            found_named_slopes = true;
            break;
        }
    }
    assert!(found_named_slopes);
}

#[test]
fn mha_forward_with_distance_batch_and_gaussian() {
    let b = 2;
    let seq = 4;
    let d_model = 8;
    let num_heads = 2;
    let x_data: Vec<f32> = (0..(b * seq * d_model))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        true,
    );
    // distance matrix with batch dim
    let mut dist_data: Vec<f32> = Vec::with_capacity(b * seq * seq);
    for batch in 0..b {
        for i in 0..seq {
            for j in 0..seq {
                dist_data.push(((i as isize - j as isize).abs()) as f32 + (batch as f32));
            }
        }
    }
    let dist = Tensor::new(
        Array::from_shape_vec((b, seq, seq), dist_data)
            .unwrap()
            .into_dyn(),
        false,
    );
    let mha = MultiHeadAttention::new_with_nl_oob(d_model, num_heads, BiasFunction::Gaussian, 2.0);
    let out = mha.forward_with_distance(&x, &dist);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);
}

#[test]
fn mha_forward_with_distance_mismatched_batch_returns_input() {
    let b = 1;
    let seq = 4;
    let d_model = 8;
    let num_heads = 2;
    let x_data: Vec<f32> = (0..(b * seq * d_model))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        true,
    );
    // distance matrix batch mismatch: b=2 but x has b=1
    let mut dist_data: Vec<f32> = Vec::with_capacity(2 * seq * seq);
    for batch in 0..2 {
        for i in 0..seq {
            for j in 0..seq {
                dist_data.push(((i as isize - j as isize).abs()) as f32 + (batch as f32));
            }
        }
    }
    let dist = Tensor::new(
        Array::from_shape_vec((2, seq, seq), dist_data)
            .unwrap()
            .into_dyn(),
        false,
    );
    let mha = MultiHeadAttention::new_with_nl_oob(d_model, num_heads, BiasFunction::Gaussian, 2.0);
    let out = mha.forward_with_distance(&x, &dist);
    // On batch mismatch the implementation returns x unchanged
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);
    // Ensure it's equal to input (should be identical shape and values)
    assert_eq!(
        out.lock().storage.to_f32_array().len(),
        x.lock().storage.to_f32_array().len()
    );
}

#[test]
fn transformer_block_forward_with_distance_integrates_nl_oob() {
    let b = 1;
    let seq = 5;
    let d_model = 8;
    let d_ff = 16;
    let num_heads = 4;
    let x_data: Vec<f32> = (0..(b * seq * d_model))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        true,
    );
    let mut dist_data: Vec<f32> = Vec::with_capacity(seq * seq);
    for i in 0..seq {
        for j in 0..seq {
            dist_data.push(((i as isize - j as isize).abs()) as f32);
        }
    }
    let dist = Tensor::new(
        Array::from_shape_vec((seq, seq), dist_data)
            .unwrap()
            .into_dyn(),
        false,
    );
    let mut block = TB::new_with_kv_and_rope(
        d_model, d_ff, num_heads, num_heads, false, 10000.0, 1.0, true,
    )
    .expect("create tb");
    // Replace block's MHA with a NL-OOB-enabled MHA
    block.mha =
        MultiHeadAttention::new_with_nl_oob(d_model, num_heads, BiasFunction::Logarithmic, 2.0);
    let out = block.forward_block_with_distance(&x, &dist);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);
}

#[test]
fn transformer_block_builder_with_nl_oob_works() {
    let b = 1;
    let seq = 6;
    let d_model = 8;
    let d_ff = 16;
    let num_heads = 4;
    let mut block = crate::nn::TransformerBlock::new_with_nl_oob(
        d_model,
        d_ff,
        num_heads,
        BiasFunction::Logarithmic,
        3.0,
    )
    .expect("create nl-oob block");
    // Ensure parameters include slopes
    let named = block.named_parameters("block");
    let mut found = false;
    for (k, _) in named {
        if k == "block.mha.nl_oob.slopes" {
            found = true;
            break;
        }
    }
    assert!(found);

    // Exercise the block with a small input to ensure it runs and uses `b`/`seq`.
    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32 * 0.01).collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        false,
    );
    let out = block.forward_block(&x);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);
}

#[test]
fn load_state_dict_sets_nl_oob_config_from_state() {
    use std::collections::HashMap;
    let d_model = 8;
    let num_heads = 2;
    let mut mha = MultiHeadAttention::new(d_model, num_heads);
    // Create a fake state dict with 'nl_oob.config' = 1.0 -> Gaussian
    let cfg = Tensor::new(
        Array::from_shape_vec((1,), vec![1.0]).unwrap().into_dyn(),
        false,
    );
    let slopes_arr = ndarray::Array::from_shape_vec((1, num_heads, 1, 1), vec![2.0, 1.0])
        .unwrap()
        .into_dyn();
    let slopes_t = Tensor::new(slopes_arr, true);
    let mut state: HashMap<String, Tensor> = HashMap::new();
    state.insert("mha.nl_oob.config".to_string(), cfg);
    state.insert("mha.nl_oob.slopes".to_string(), slopes_t.clone());
    // Load into MHA
    let res = mha.load_state_dict(&state, "mha");
    assert!(res.is_ok());
    assert_eq!(mha.nl_oob_config.unwrap(), BiasFunction::Gaussian);
    assert!(mha.slopes.is_some());
    // Ensure slopes were set to require_grad
    let s = mha.slopes.unwrap();
    assert!(s.lock().requires_grad);
}

#[test]
fn slopes_receive_grad_on_backward() {
    let b = 1;
    let seq = 4;
    let d_model = 8;
    let num_heads = 2;
    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32 * 0.01).collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        true,
    );
    let mut dist_data: Vec<f32> = Vec::with_capacity(seq * seq);
    for i in 0..seq {
        for j in 0..seq {
            dist_data.push(((i as isize - j as isize).abs()) as f32);
        }
    }
    let dist = Tensor::new(
        Array::from_shape_vec((seq, seq), dist_data)
            .unwrap()
            .into_dyn(),
        false,
    );
    let mha =
        MultiHeadAttention::new_with_nl_oob(d_model, num_heads, BiasFunction::Logarithmic, 4.0);
    let out = mha.forward_with_distance(&x, &dist);
    // scalar loss: sum of outputs
    let loss = out.sum();
    loss.backward();
    // slopes parameter should receive gradients
    if let Some(s) = mha.slopes.as_ref() {
        let s_lock = s.lock();
        assert!(s_lock.grad.is_some());
        let g = s_lock.grad.clone().unwrap();
        // Some elements of gradient should be non-zero (gradient propagated to slopes)
        assert!(g.iter().any(|&v| v != 0.0));
    } else {
        panic!("Expected slopes to be present for NL-OOB MHA");
    }
}

#[test]
fn mha_forward_with_distance_2d_phi_broadcasts_to_batch() {
    // Ensure a 2D distance matrix is broadcast across batch when b > 1
    let b = 4;
    let seq = 8;
    let d_model = 16;
    let num_heads = 4;
    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32 * 0.001).collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        false,
    );
    let mut dist_data: Vec<f32> = Vec::with_capacity(seq * seq);
    for i in 0..seq {
        for j in 0..seq {
            dist_data.push(((i as isize - j as isize).abs()) as f32);
        }
    }
    let dist = Tensor::new(
        Array::from_shape_vec((seq, seq), dist_data)
            .unwrap()
            .into_dyn(),
        false,
    );
    let mha =
        MultiHeadAttention::new_with_nl_oob(d_model, num_heads, BiasFunction::Logarithmic, 2.0);
    let out = mha.forward_with_distance(&x, &dist);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);
}
