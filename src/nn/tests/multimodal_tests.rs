use crate::nn::{VisionTransformer, MultimodalLLM};
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
