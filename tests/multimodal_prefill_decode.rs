use ndarray::Array;
use tensor_engine::nn::{MultimodalLLM, VisionTransformer};
use tensor_engine::tensor::Tensor;

#[test]
fn prefill_and_decode_step() {
    let b = 1usize;
    let c = 3usize;
    let h = 8usize;
    let w = 8usize;
    let patch_size = 2usize;
    let d_model = 16usize;
    let d_ff = 32usize;
    let num_heads = 2usize;
    let depth = 1usize;
    let max_len = 16usize;
    let vocab = 32usize;
    let seq = 4usize;

    let vit = VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, max_len)
        .expect("create vision transformer");
    let mut model = MultimodalLLM::new(vit, vocab, d_model, d_ff, num_heads, depth)
        .expect("create multimodal model");

    // Create random input
    let img_data = vec![0.5f32; b * c * h * w];
    let images = Tensor::new(
        Array::from_shape_vec((b, c, h, w), img_data)
            .unwrap()
            .into_dyn(),
        false,
    );

    let ids_data: Vec<f32> = vec![1.0; b * seq];
    let ids = Tensor::new(
        Array::from_shape_vec((b, seq), ids_data)
            .unwrap()
            .into_dyn(),
        false,
    );

    let mem = model
        .prefill(&images, Some(&ids))
        .expect("prefill should succeed");
    assert_eq!(mem.prefill_image_tokens > 0, true);
    // New token(s) to decode step
    let new_ids_data: Vec<f32> = vec![2.0f32; b * 1];
    let new_ids = Tensor::new(
        Array::from_shape_vec((b, 1), new_ids_data)
            .unwrap()
            .into_dyn(),
        false,
    );
    let (logits, new_mem) = model
        .decode_step(&mem, &new_ids)
        .expect("decode_step should succeed");
    let out_shape = logits.lock().storage.shape();
    let patches_per_dim = (h / patch_size) * (w / patch_size);
    let expected_seq = patches_per_dim + seq + 1usize; // image patches + prefix + new tokens
    assert_eq!(out_shape, vec![b, expected_seq, vocab]);
    assert_eq!(new_mem.prefill_image_tokens, mem.prefill_image_tokens);
}
