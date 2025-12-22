use tensor_engine::nn::Llama;
use tensor_engine::tensor::Tensor;
use ndarray::IxDyn;

#[test]
fn test_llama_forward_small() {
    // Tiny Llama config
    let vocab_size = 1024usize;
    let d_model = 64usize;
    let num_layers = 2usize;
    let d_ff = 256usize;
    let num_heads = 4usize;
    let kv_heads = 2usize; // different from num_heads to exercise expansion

    let model = Llama::new(vocab_size, d_model, num_layers, d_ff, num_heads, kv_heads).expect("Llama::new failed");
    // Create token input: batch=1, seq=3
    // Test 2D token ids (batch=1)
    let token_ids = Tensor::new(ndarray::Array::from_elem(IxDyn(&[1, 3]), 0f32).into_dyn(), false);
    use tensor_engine::nn::Module;
    let logits = model.forward(&token_ids);
    let shape = logits.lock().storage.shape().to_vec();
    assert_eq!(shape, vec![1, 3, vocab_size]);

    // Test 1D token ids (single sequence)
    let token_ids_1d = Tensor::new(ndarray::Array::from_vec(vec![0f32, 0f32, 0f32]).into_dyn(), false);
    let logits1 = model.forward(&token_ids_1d);
    let shape1 = logits1.lock().storage.shape().to_vec();
    assert_eq!(shape1, vec![3, vocab_size]);
}
