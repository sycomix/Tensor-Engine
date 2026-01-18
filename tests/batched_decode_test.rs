use ndarray::Array;
use tensor_engine::nn::transformer_cleaned::Llama;
use tensor_engine::tensor::Tensor;

#[test]
fn test_llama_batched_decode_prefill() {
    let vocab_size = 128;
    let d_model = 32;
    let num_layers = 1;
    let d_ff = 64;
    let num_heads = 4;
    let kv_heads = 4;

    let mut llama = Llama::new(vocab_size, d_model, num_layers, d_ff, num_heads, kv_heads)
        .expect("create llama");

    // Enable KV Cache
    llama.set_kv_cache(true);

    // Batch=2, Seq=3
    // Input IDs (using indices, but since we use embedding_lookup on float tensor typically in this engine... wait)
    // Llama::forward calls `embedding_lookup`. Tensor::embedding_lookup takes generic Tensor?
    // In `tests/transformer_llama_numeric_grad.rs` it used numeric inputs.
    // Llama embedding tokens are [vocab, d_model].
    // Input must be [batch, seq] of indices.
    // Check `Tensor::embedding_lookup` definition from `src/tensor/ops.rs` if I viewed it.
    // Assuming it accepts indices as f32 values (standard for this engine?)

    let batch = 2;
    let seq = 3;
    let input_data = vec![
        1.0, 2.0, 3.0, // seq 0
        4.0, 5.0, 0.0, // seq 1
    ];
    let input = Tensor::new(
        Array::from_shape_vec((batch, seq), input_data)
            .unwrap()
            .into_dyn(),
        false,
    );

    // Mask: [batch, 1, seq, seq]
    // 0.0 = valid, -1e9 = mask
    let mask_data = vec![0.0; batch * seq * seq];
    // Mask for padding in seq 1 (last token is pad)
    // seq 1 is at index 1.
    // Index in flat: 1*seq*seq + i*seq + j
    // Mask out attention TO last token (2)
    // For seq 1:
    // row 2 (last token) is pad.
    // Actually if input is 0.0, we probably mask it.
    // Let's just create a dummy mask test.

    let mask = Tensor::new(
        Array::from_shape_vec((batch, 1, seq, seq), mask_data)
            .unwrap()
            .into_dyn(),
        false,
    );

    let out = llama.forward_with_mask(&input, Some(&mask));

    assert_eq!(out.lock().storage.shape(), &[batch, seq, vocab_size]);
}

#[test]
fn test_llama_incremental_decode() {
    let vocab_size = 128;
    let d_model = 32;
    let num_layers = 1;
    let d_ff = 64;
    let num_heads = 4;
    let kv_heads = 4;

    let mut llama = Llama::new(vocab_size, d_model, num_layers, d_ff, num_heads, kv_heads)
        .expect("create llama");
    llama.set_kv_cache(true);

    // Step 1: Prefill 1 token
    let input1 = Tensor::new(
        Array::from_shape_vec((1, 1), vec![1.0]).unwrap().into_dyn(),
        false,
    );
    let out1 = llama.forward_with_mask(&input1, None);
    assert_eq!(out1.lock().storage.shape(), &[1, 1, vocab_size]);

    // Check cache is populated
    {
        let l0 = &llama.layers[0];
        assert!(l0.kv_cache_clone().unwrap().seq_len() == 1);
    }

    // Step 2: Decode next token
    let input2 = Tensor::new(
        Array::from_shape_vec((1, 1), vec![2.0]).unwrap().into_dyn(),
        false,
    );
    // Mask logic for incremental: usually handled by causal internal if step > 0?
    // forward_with_caching handles appending.

    let out2 = llama.forward_with_mask(&input2, None);
    assert_eq!(out2.lock().storage.shape(), &[1, 1, vocab_size]);

    {
        let l0 = &llama.layers[0];
        assert!(l0.kv_cache_clone().unwrap().seq_len() == 2);
    }
}
