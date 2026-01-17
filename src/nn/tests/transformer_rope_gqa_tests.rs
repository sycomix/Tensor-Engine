use crate::nn::TransformerBlock;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn transformer_block_rope_and_gqa_shapes() {
    let b = 1;
    let seq = 4;
    let d_model = 8;
    let d_ff = 16;
    let num_heads = 4;
    let kv_heads = 2;
    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32 * 0.01).collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        true,
    );
    // With RoPE on and GQA
    let block = TransformerBlock::new_with_kv_and_rope(
        d_model, d_ff, num_heads, kv_heads, true, 500000.0, 1.0, true,
    )
    .expect("create block with kv+rope");
    let out = block.forward_block_no_cache(&x);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);

    // Without RoPE
    let block2 = TransformerBlock::new_with_kv_and_rope(
        d_model, d_ff, num_heads, kv_heads, false, 500000.0, 1.0, true,
    )
    .expect("create block without rope");
    let out2 = block2.forward_block_no_cache(&x);
    assert_eq!(out2.lock().storage.shape(), &[b, seq, d_model]);
}
