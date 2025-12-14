use crate::nn::{MultimodalLLM, VisionTransformer};
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};

#[test]
fn multimodal_forward_shape() {
    let b = 1usize;
    let c = 3usize;
    let h = 8usize;
    let w = 8usize;
    let patch_size = 2usize;
    let d_model = 16usize;
    let d_ff = 32usize;
    let num_heads = 4usize;
    let depth = 2usize;
    let max_len = 128usize;
    let vocab = 100usize;
    let seq = 6usize;
    let vit = VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, max_len);
    let model = MultimodalLLM::new(vit, vocab, d_model, d_ff, num_heads, depth);

    let img_data = vec![1.0f32; b * c * h * w];
    let images = Tensor::new(Array::from_shape_vec((b, c, h, w), img_data).unwrap().into_dyn(), false);
    // build input ids tensor (float indices)
    let ids_data: Vec<f32> = vec![1.0; b * seq];
    let ids = Tensor::new(Array::from_shape_vec((b, seq), ids_data).unwrap().into_dyn(), false);
    let logits = model.forward(&images, &ids);
    let out_shape = logits.lock().storage.shape();
    let patches_per_dim = (h / patch_size) * (w / patch_size);
    assert_eq!(out_shape, vec![b, patches_per_dim + seq, vocab]);
}

#[test]
fn multimodal_causal_masking_text_unaffected_by_future_text() {
    let b = 1usize;
    let c = 3usize;
    let h = 1usize;
    let w = 1usize;
    let patch_size = 1usize;
    let d_model = 4usize;
    let d_ff = 8usize;
    let num_heads = 1usize;
    let depth = 1usize;
    let max_len = 8usize;
    let vocab = d_model; // make head identity by matching dims
    let seq = 2usize;
    let vit = VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, max_len);
    let mut model = MultimodalLLM::new(vit, vocab, d_model, d_ff, num_heads, depth);

    // Set text embeddings for indices 1 and 2
    // Embedding rows: idx 0 unused, idx 1 = t0, idx 2 = t1 variants
    let t0 = ndarray::Array::from_shape_vec((d_model,), vec![1.0, 0.0, 0.0, 0.0]).unwrap().into_dyn();
    let t1a = ndarray::Array::from_shape_vec((d_model,), vec![0.0, 1.0, 0.0, 0.0]).unwrap().into_dyn();
    let t1b = ndarray::Array::from_shape_vec((d_model,), vec![0.0, 2.0, 0.0, 0.0]).unwrap().into_dyn();
    // Build embedding matrix of shape [vocab, d_model]
    let mut emb_mat = ndarray::Array::zeros(ndarray::IxDyn(&[vocab, d_model]));
    emb_mat.slice_mut(ndarray::s![1, ..]).assign(&t0.into_dimensionality().unwrap());
    emb_mat.slice_mut(ndarray::s![2, ..]).assign(&t1a.into_dimensionality().unwrap());
    // Set the model text embedding weight to this base
    model.text_embedding = crate::tensor::Tensor::new(emb_mat.into_dyn(), true);

    // Make linear_q/k/v and linear_o identity for each decoder block
    for blk in &mut model.decoder_blocks {
        let dim = d_model;
        let id = ndarray::Array::eye(dim).into_dyn();
        blk.mha.linear_q.weight = crate::tensor::Tensor::new(id.clone(), false);
        blk.mha.linear_k.weight = crate::tensor::Tensor::new(id.clone(), false);
        blk.mha.linear_v.weight = crate::tensor::Tensor::new(id.clone(), false);
        blk.mha.linear_o.weight = crate::tensor::Tensor::new(id.clone(), false);
    }

    // Make head an identity mapping (vocab == d_model)
    model.head.weight = crate::tensor::Tensor::new(ndarray::Array::eye(d_model).into_dyn(), false);

    // Input images: zeros
    let img_data = vec![0.0f32; b * c * h * w];
    let images = Tensor::new(ndarray::Array::from_shape_vec((b, c, h, w), img_data).unwrap().into_dyn(), false);

    // Build two inputs that only differ in the last text token
    let ids_a: Vec<f32> = vec![1.0, 2.0];
    let ids_b: Vec<f32> = vec![1.0, 3.0];
    // For ids_b, set embedding row 3 to t1b
    let mut emb_mat_b = model.text_embedding.lock().storage.clone();
    emb_mat_b[[3, 0]] = 0.0; // ensure size
    // Replace row 3
    let mut em = model.text_embedding.lock();
    // resize matrix if needed
    if em.storage.shape()[0] <= 3 {
        // expand
        let mut new_emb = ndarray::Array::zeros(ndarray::IxDyn(&[4, d_model]));
        new_emb.slice_mut(ndarray::s![..vocab, ..]).assign(&em.storage);
        em.storage = new_emb.into_dyn();
    }
    drop(em);
    // Now write row 3
    {
        let mut em = model.text_embedding.lock();
        for i in 0..d_model { em.storage[[3, i]] = t1b[i]; }
    }

    let ids_a_t = Tensor::new(ndarray::Array::from_shape_vec((b, seq), ids_a.clone()).unwrap().into_dyn(), false);
    let ids_b_t = Tensor::new(ndarray::Array::from_shape_vec((b, seq), ids_b.clone()).unwrap().into_dyn(), false);

    let logits_a = model.forward(&images, &ids_a_t);
    let logits_b = model.forward(&images, &ids_b_t);
    let arr_a = logits_a.lock().storage.clone();
    let arr_b = logits_b.lock().storage.clone();
    // Position index of first text token is Np (=1)
    let pos_text = 1usize;
    // Compare outputs at pos_text across both variants; they should be identical because the future text token differs but is masked
    for v in 0..vocab {
        let val_a = arr_a[[0, pos_text, v]];
        let val_b = arr_b[[0, pos_text, v]];
        assert!((val_a - val_b).abs() < 1e-4, "text pos outputs differ despite causal masking");
    }
    // Check that image token outputs differ (index 0), since image tokens can attend to the changed future text token and should reflect the change
    let mut any_diff = false;
    for v in 0..vocab {
        let val_a = arr_a[[0, 0, v]];
        let val_b = arr_b[[0, 0, v]];
        if (val_a - val_b).abs() > 1e-6 {
            any_diff = true;
            break;
        }
    }
    assert!(any_diff, "image token outputs did not change, expected to be affected by text token change");
}
