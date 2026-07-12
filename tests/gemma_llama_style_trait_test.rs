use ndarray::IxDyn;
use tensor_engine::nn::transformer::Gemma;
use tensor_engine::nn::LlamaStyleModel;
use tensor_engine::tensor::Tensor;

#[test]
fn gemma_llama_style_trait_single_token_delegates_to_inherent_methods() {
    let mut model = Gemma::new(32, 8, 1, 16, 2, 1, 1.0).expect("Gemma::new failed");
    let model_trait: &mut dyn LlamaStyleModel = &mut model;

    model_trait
        .init_kv_caches(4)
        .expect("Gemma trait init_kv_caches failed");

    let token = Tensor::new(ndarray::Array::from_elem(IxDyn(&[1]), 0.0f32), false);
    let logits = model_trait
        .forward_single_token(&token, None)
        .expect("Gemma trait forward_single_token failed");
    assert_eq!(logits.lock().storage.shape().to_vec(), vec![1, 1, 32]);

    model_trait.reset_kv_caches();

    let logits_after_reset = model_trait
        .forward_single_token(&token, None)
        .expect("Gemma trait forward_single_token after reset failed");
    assert_eq!(
        logits_after_reset.lock().storage.shape().to_vec(),
        vec![1, 1, 32]
    );
}
