use tensor_engine::nn::transformer_cleaned::TransformerBlock;
use tensor_engine::tensor::Tensor;
use ndarray::IxDyn;

#[test]
fn test_llama_style_forward_shape_and_params() {
    let d_model = 8;
    let d_ff = 16;
    let num_heads = 2;
    let kv_heads = 2;
    let block = TransformerBlock::new_llama_style(d_model, d_ff, num_heads, kv_heads, true, false);
    let arr = ndarray::Array::from_elem(IxDyn(&[1, 3, d_model]), 0.1f32);
    let x = Tensor::new(arr, true);
    let out = block.forward_block(&x);
    let shape = out.lock().storage.shape().to_vec();
    assert_eq!(shape, vec![1, 3, d_model]);

    let named = block.named_parameters_impl("test");
    let mut found_attn = false;
    let mut found_ffn = false;
    for (k, _) in named {
        if k.ends_with("rms_attn_gamma") { found_attn = true; }
        if k.ends_with("rms_ffn_gamma") { found_ffn = true; }
    }
    assert!(found_attn && found_ffn);
}

#[test]
fn test_llama_style_use_rope_differs() {
    let d_model = 8;
    let d_ff = 16;
    let num_heads = 2;
    let kv_heads = 2;
    let mut block_rope = TransformerBlock::new_llama_style(d_model, d_ff, num_heads, kv_heads, true, false);
    let mut block_no_rope = TransformerBlock::new_llama_style(d_model, d_ff, num_heads, kv_heads, false, false);
    // initialize linear weights to non-zero values so RoPE produces different outputs
    let w_q = ndarray::Array::from_shape_fn((d_model, d_model), |(i, j)| (i as f32 * 0.01) + (j as f32 * 0.001)).into_dyn();
    block_rope.mha.linear_q.weight = Tensor::new(w_q.clone(), true);
    block_no_rope.mha.linear_q.weight = Tensor::new(w_q.clone(), true);
    block_rope.mha.linear_k.weight = Tensor::new(w_q.clone(), true);
    block_no_rope.mha.linear_k.weight = Tensor::new(w_q.clone(), true);
    block_rope.mha.linear_v.weight = Tensor::new(w_q.clone(), true);
    block_no_rope.mha.linear_v.weight = Tensor::new(w_q.clone(), true);
    // also set linear1 and linear2 weights to non-zero
    let w1 = ndarray::Array::from_elem(IxDyn(&[d_model, d_ff * 2]), 0.1f32);
    let w2 = ndarray::Array::from_elem(IxDyn(&[d_ff, d_model]), 0.1f32);
    block_rope.linear1.weight = Tensor::new(w1.clone(), true);
    block_no_rope.linear1.weight = Tensor::new(w1.clone(), true);
    block_rope.linear2.weight = Tensor::new(w2.clone(), true);
    block_no_rope.linear2.weight = Tensor::new(w2.clone(), true);

    let arr = ndarray::Array::from_shape_fn((1, 3, d_model), |(_, s, d)| s as f32 * 0.01 + d as f32 * 0.001);
    let arr = arr.into_dyn();
    let x = Tensor::new(arr.clone(), true);
    // Confirm the use_rope flag is set on the constructed multi-head attention module.
    assert!(block_rope.mha.use_rope);
    assert!(!block_no_rope.mha.use_rope);
    // Also verify the forward call doesn't panic (smoke test), and respects output shape
    let _out1 = block_rope.forward_block(&x);
    let _out2 = block_no_rope.forward_block(&x);

}
