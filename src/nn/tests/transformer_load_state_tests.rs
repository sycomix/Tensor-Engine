use super::super::super::tensor::Tensor;
use super::super::MultiHeadAttention;
use ndarray::Array;
use std::collections::HashMap;

#[test]
fn load_state_expands_kv_heads_for_k_proj_weight() {
    let d_model = 8usize;
    let num_heads = 4usize;
    let kv_heads = 2usize;

    let mut mha = MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, false);

    // head_dim = d_model / num_heads = 2, expected_k_rows = kv_heads * head_dim = 4
    let head_dim = d_model / num_heads;
    let rows = kv_heads * head_dim; // 4
    let cols = d_model; // 8

    // Create a simple pattern matrix with rows x cols
    let mut data = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            data.push((r * cols + c) as f32 + 1.0);
        }
    }
    let arr = Array::from_shape_vec((rows, cols), data.clone())
        .unwrap()
        .into_dyn();
    let t = Tensor::new(arr, false);

    let mut state: HashMap<String, Tensor> = HashMap::new();
    // Use the key format expected by the special-case branch
    state.insert("mha.mha.linear_k.weight".to_string(), t.clone());

    let res = mha.load_state_dict(&state, "mha");
    assert!(res.is_ok());

    // After loading, linear_k.weight should be expanded to [num_heads * head_dim, cols] == [d_model, d_model]
    let shape = mha.linear_k.weight.lock().storage.shape().to_vec();
    assert_eq!(shape, vec![d_model, d_model]);

    // Check that expanded weight contains repeated groups: original row block 0 should be repeated
    // Original rows represent kv_heads groups; when expanded, group 0 should occupy rows 0 and 1 (if repeat=2)
    let expanded = mha.linear_k.weight.lock().storage.to_f32_array();
    // Compare a value from original first group and corresponding expanded position
    let orig_val = data[0]; // row 0, col 0
    let expanded_val = expanded[[0, 0]];
    assert_eq!(orig_val, expanded_val);
}

#[test]
fn load_state_transposes_k_proj_key_when_needed() {
    let d_model = 8usize;
    let num_heads = 4usize;
    let mut mha = MultiHeadAttention::new(d_model, num_heads);

    // Create a weight with shape (rows != d_model, cols == d_model) to trigger transpose branch
    let rows = 4usize;
    let cols = d_model;
    let mut data = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            data.push((r * cols + c) as f32 + 10.0);
        }
    }
    let arr = Array::from_shape_vec((rows, cols), data.clone())
        .unwrap()
        .into_dyn();
    let t = Tensor::new(arr, false);

    let mut state: HashMap<String, Tensor> = HashMap::new();
    // Use the simpler k_proj key that triggers later transpose logic
    state.insert("mha.k_proj.weight".to_string(), t.clone());

    let res = mha.load_state_dict(&state, "mha");
    assert!(res.is_ok());

    // After load, the code checks shape and transposes if shape[0] != d_model && shape[1] == d_model
    let shape = mha.linear_k.weight.lock().storage.shape().to_vec();
    // Expect transposition happened so shape becomes [d_model, rows]
    assert_eq!(shape, vec![d_model, rows]);

    // Verify that value at [c, r] equals original [r, c]
    let loaded = mha.linear_k.weight.lock().storage.to_f32_array();
    assert_eq!(loaded[[0, 0]], data[0]);
    assert_eq!(loaded[[1, 2]], data[2 * cols + 1]);
}
