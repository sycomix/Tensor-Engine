use std::collections::HashMap;
use tensor_engine::nn::TransformerBlock;
use tensor_engine::tensor::Tensor;
use ndarray::IxDyn;

#[test]
fn test_gate_down_transposed_concat() {
    // Simulate lin1 expected shape (3072, 16384) -> r=3072, c=16384
    let mut t = TransformerBlock::new_llama_style(3072, 8192, 24, 24, true, false);
    // gate saved shape [3072,8192]
    let gate = Tensor::new(ndarray::Array::zeros(IxDyn(&[3072, 8192])), false);
    // down saved shape transposed [8192,3072]
    let down = Tensor::new(ndarray::Array::zeros(IxDyn(&[8192, 3072])), false);
    let mut state: HashMap<String, Tensor> = HashMap::new();
    state.insert(".mlp.gate_proj.weight".to_string(), gate);
    state.insert(".mlp.down_proj.weight".to_string(), down);
    // Use prefix "model.layers.0"
    let res = t.load_state_dict_impl(&state, "model.layers.0");
    assert!(res.is_ok());
    let lin1_shape = t.linear1.weight.lock().storage.shape().to_vec();
    assert_eq!(lin1_shape, vec![3072, 16384]);
}
