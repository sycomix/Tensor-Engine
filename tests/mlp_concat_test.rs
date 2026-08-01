use ndarray::IxDyn;
use std::collections::HashMap;
use tensor_engine::nn::transformer::{TransformerBlock, TransformerConfig};
use tensor_engine::tensor::Tensor;

#[test]
fn test_gate_up_transposed_concat() {
    let mut t = TransformerBlock::new_llama_style(TransformerConfig {
        d_model: 3072,
        d_ff: 8192,
        num_heads: 24,
        kv_heads: 24,
        use_rope: true,
        rope_theta: 10000.0,
        rope_scale: 1.0,
        bias: false,
        ffn_activation: Default::default(),
        parallel_residual: false,
        attn_logit_softcap: None,
        final_logit_softcap: None,
    })
    .expect("create llama-style block");
    // gate saved shape [3072,8192], up saved shape [3072,8192]
    let gate = Tensor::new(ndarray::Array::zeros(IxDyn(&[3072, 8192][..])), false);
    let up = Tensor::new(ndarray::Array::zeros(IxDyn(&[3072, 8192][..])), false);
    // down saved shape [8192,3072]
    let down = Tensor::new(ndarray::Array::zeros(IxDyn(&[8192, 3072][..])), false);
    let mut state: HashMap<String, Tensor> = HashMap::new();
    state.insert(".mlp.gate_proj.weight".to_string(), gate);
    state.insert(".mlp.up_proj.weight".to_string(), up);
    state.insert(".mlp.down_proj.weight".to_string(), down);
    let res = t.load_state_dict_impl(&state, "model.layers.0");
    assert!(res.is_ok());
    let lin1_shape = t
        .linear1
        .as_f32()
        .unwrap()
        .weight
        .lock()
        .storage
        .shape()
        .to_vec();
    // linear1 = concat(gate_proj, up_proj) along axis=1 => [3072, 8192+8192] = [3072, 16384]
    assert_eq!(lin1_shape, vec![3072, 16384]);
    let lin2_shape = t
        .linear2
        .as_f32()
        .unwrap()
        .weight
        .lock()
        .storage
        .shape()
        .to_vec();
    // linear2 = down_proj => [8192, 3072]
    assert_eq!(lin2_shape, vec![8192, 3072]);
}
