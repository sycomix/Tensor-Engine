use tensor_engine::nn::Llama;
use tensor_engine::nn::Module;
use tensor_engine::tensor::Tensor;
use std::collections::HashMap;
use ndarray::IxDyn;

#[test]
fn test_embed_transpose_fix() {
    let mut m = Llama::new(128256, 3072, 2, 8192, 24, 8);
    // Simulate embedding saved transposed [d_model, vocab]
    let emb = Tensor::new(ndarray::Array::zeros(IxDyn(&[3072, 128256])), false);
    let mut state: HashMap<String, Tensor> = HashMap::new();
    state.insert("model.embed_tokens.weight".to_string(), emb);
    let res = m.load_state_dict(&state, "model");
    assert!(res.is_ok());
    let shape = m.embed_tokens.lock().storage.shape().to_vec();
    assert_eq!(shape, vec![128256, 3072]);
}
