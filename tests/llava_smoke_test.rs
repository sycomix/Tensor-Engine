use ndarray::Array;
use tensor_engine::nn::{Adam, CrossEntropyLogitsLoss, Module, MultimodalLLM, Optimizer, VisionTransformer};
use tensor_engine::tensor::Tensor;

#[test]
fn llava_tiny_train_step() {
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

    let vit = VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, max_len);
    let model = MultimodalLLM::new(vit, vocab, d_model, d_ff, num_heads, depth);

    // Create random input
    let img_data = vec![1.0f32; b * c * h * w];
    let images = Tensor::new(Array::from_shape_vec((b, c, h, w), img_data).unwrap().into_dyn(), false);

    let ids_data: Vec<f32> = vec![1.0; b * seq];
    let ids = Tensor::new(Array::from_shape_vec((b, seq), ids_data).unwrap().into_dyn(), false);

    // Forward
    let logits = model.forward(&images, &ids);

    // Check shape
    let out_shape = logits.lock().storage.shape();
    let patches_per_dim = (h / patch_size) * (w / patch_size);
    assert_eq!(out_shape, vec![b, patches_per_dim + seq, vocab]);

    // Prepare target labels for the text tokens: a small numeric list
    let labels_vec: Vec<i64> = vec![1; b * seq];
    // Loss: use CrossEntropyLogitsLoss forward_from_labels with axis=2 (vocab axis)
    let cel = CrossEntropyLogitsLoss::new();
    let labels_arr = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[labels_vec.len()]), labels_vec.clone()).unwrap();
    let labels = tensor_engine::labels::Labels::new(labels_arr);
    let loss = cel.forward_from_labels(&logits, &labels, 2);

    // Backprop and optimizer step
    loss.backward();
    let mut opt = Adam::new(1e-3, 0.9, 0.999, 1e-8);
    let params = model.parameters();
    opt.step(&params);

    // Ensure parameters are present
    assert!(!params.is_empty());
}
