use ndarray::Array;
use tensor_engine::nn::{MultimodalLLM, VisionTransformer};
use tensor_engine::tensor::Tensor;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > tol {
            return false;
        }
    }
    true
}

#[test]
fn prefill_incremental_matches_full_decode() {
    let b = 1usize;
    let c = 3usize;
    let h = 8usize;
    let w = 8usize;
    let patch_size = 2usize;
    let d_model = 16usize;
    let d_ff = 32usize;
    let num_heads = 2usize;
    let depth = 2usize;
    let max_len = 16usize;
    let vocab = 32usize;
    let seq = 3usize; // prefix length

    let vit = VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, max_len)
        .expect("create vision transformer");
    let mut model = MultimodalLLM::new(vit, vocab, d_model, d_ff, num_heads, depth)
        .expect("create multimodal model");

    // Set non-zero embeddings so outputs are non-trivial
    let emb = Array::from_shape_fn((vocab, d_model), |(i, j)| {
        (i as f32 * 0.01) + (j as f32 * 0.001)
    })
    .into_dyn();
    model.text_embedding = Tensor::new(emb, true);

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

    // Prefill -> should include per_layer_kv
    let mem = model
        .prefill(&images, Some(&ids))
        .expect("prefill should succeed");
    assert!(mem.per_layer_kv.is_some());
    let num_caches = mem.per_layer_kv.as_ref().unwrap().len();
    assert_eq!(num_caches, depth);

    // Now do incremental decoding of N new tokens
    let new_tokens = vec![2usize, 3usize, 4usize];
    let mut mem_incr = mem.clone();
    let mut last_logits_incr = None;
    for &t in &new_tokens {
        let token = Tensor::new(
            Array::from_shape_vec((b, 1), vec![t as f32; b * 1])
                .unwrap()
                .into_dyn(),
            false,
        );
        let (logits, new_mem) = model
            .decode_step(&mem_incr, &token)
            .expect("decode_step should succeed");
        mem_incr = new_mem;
        last_logits_incr = Some(logits);
    }
    let last_logits_incr = last_logits_incr.unwrap();

    // Full decode with all tokens at once (prefix + new_tokens)
    let mut full_ids_vec: Vec<f32> = ids.lock().storage.to_f32_array().iter().cloned().collect();
    for &t in &new_tokens {
        full_ids_vec.push(t as f32);
    }
    let full_ids = Tensor::new(
        Array::from_shape_vec((b, seq + new_tokens.len()), full_ids_vec)
            .unwrap()
            .into_dyn(),
        false,
    );
    // NOTE: model.forward requires &mut self
    let full_logits = model.forward(&images, &full_ids);

    let a = last_logits_incr.lock().storage.to_f32_array();
    let b_arr = full_logits.lock().storage.to_f32_array();
    let af = a.iter().cloned().collect::<Vec<f32>>();
    let bf = b_arr.iter().cloned().collect::<Vec<f32>>();

    assert!(
        approx_eq(&af, &bf, 1e-4),
        "incremental logits should match full decode"
    );
}
