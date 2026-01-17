use super::super::super::tensor::Tensor;
use super::super::{Module, TransformerBlock};
use ndarray::Array;
use std::collections::HashMap;

#[test]
fn transformer_loads_llama_style_keys_and_mlps() {
    let d_model = 8usize;
    let d_ff = 4usize; // small for test
    let num_heads = 2usize;
    let kv_heads = 2usize;
    let mut block = TransformerBlock::new_llama_style(
        d_model, d_ff, num_heads, kv_heads, false, false, 10000.0, 1.0,
    )
    .expect("create llama block");

    // Create fake state dict following LLaMA naming: input_layernorm/post_attention_layernorm and mlp.* keys
    let mut state: HashMap<String, Tensor> = HashMap::new();

    // weights for self_attn projections
    let q = Array::from_shape_vec(
        (d_model, d_model),
        (0..(d_model * d_model)).map(|i| i as f32 + 1.0).collect(),
    )
    .unwrap()
    .into_dyn();
    let k = Array::from_shape_vec(
        (d_model, d_model),
        (0..(d_model * d_model)).map(|i| i as f32 + 10.0).collect(),
    )
    .unwrap()
    .into_dyn();
    let v = Array::from_shape_vec(
        (d_model, d_model),
        (0..(d_model * d_model)).map(|i| i as f32 + 20.0).collect(),
    )
    .unwrap()
    .into_dyn();
    let o = Array::from_shape_vec(
        (d_model, d_model),
        (0..(d_model * d_model)).map(|i| i as f32 + 30.0).collect(),
    )
    .unwrap()
    .into_dyn();

    state.insert(
        "block.self_attn.mha.linear_q.weight".to_string(),
        Tensor::new(q, false),
    );
    state.insert(
        "block.self_attn.mha.linear_k.weight".to_string(),
        Tensor::new(k, false),
    );
    state.insert(
        "block.self_attn.mha.linear_v.weight".to_string(),
        Tensor::new(v, false),
    );
    state.insert(
        "block.self_attn.mha.linear_o.weight".to_string(),
        Tensor::new(o, false),
    );

    // LLaMA-style layernorm gammas
    let gamma_attn = Tensor::new(
        Array::from_shape_vec((d_model,), vec![2.0f32; d_model])
            .unwrap()
            .into_dyn(),
        false,
    );
    let gamma_ffn = Tensor::new(
        Array::from_shape_vec((d_model,), vec![3.0f32; d_model])
            .unwrap()
            .into_dyn(),
        false,
    );
    state.insert(
        "block.input_layernorm.weight".to_string(),
        gamma_attn.clone(),
    );
    state.insert(
        "block.post_attention_layernorm.weight".to_string(),
        gamma_ffn.clone(),
    );

    // MLP gate_proj + down_proj: gate/down each size (d_ff, d_model) so concatenation should make linear1 weight of shape (2*d_ff, d_model)
    let gate = Array::from_shape_vec(
        (d_ff, d_model),
        (0..(d_ff * d_model)).map(|i| i as f32 + 40.0).collect(),
    )
    .unwrap()
    .into_dyn();
    let down = Array::from_shape_vec(
        (d_ff, d_model),
        (0..(d_ff * d_model)).map(|i| i as f32 + 60.0).collect(),
    )
    .unwrap()
    .into_dyn();
    let up = Array::from_shape_vec(
        (d_model, d_ff),
        (0..(d_model * d_ff)).map(|i| i as f32 + 80.0).collect(),
    )
    .unwrap()
    .into_dyn();

    state.insert(
        "block.mlp.gate_proj.weight".to_string(),
        Tensor::new(gate, false),
    );
    state.insert(
        "block.mlp.down_proj.weight".to_string(),
        Tensor::new(down, false),
    );
    state.insert(
        "block.mlp.up_proj.weight".to_string(),
        Tensor::new(up, false),
    );

    // Now load
    let res = block.load_state_dict(&state, "block");
    assert!(res.is_ok(), "load_state_dict should succeed");

    // Check that RMS gamma parameters were set
    assert!(block.rms_attn_gamma.is_some());
    assert!(block.rms_ffn_gamma.is_some());

    // Check that linear1 weight was set to the concatenation (2*d_ff, d_model)
    let w1_shape = block.linear1.weight.lock().storage.shape().to_vec();
    assert_eq!(w1_shape, vec![2 * d_ff, d_model]);

    // Check that linear2 weight matches up_proj shape
    let w2_shape = block.linear2.weight.lock().storage.shape().to_vec();
    assert_eq!(w2_shape, vec![d_model, d_ff]);
}
