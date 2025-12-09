use crate::nn::{VisionTransformer, PatchEmbed};
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};

#[test]
fn test_patch_embed_and_vit_forward_shape() {
    let b = 1usize;
    let c = 3usize;
    let h = 8usize;
    let w = 8usize;
    let patch_size = 2usize;
    let d_model = 16usize;
    let d_ff = 32usize;
    let num_heads = 4usize;
    let depth = 2usize;
    let max_len = 64usize;
    // create image data (NCHW)
    let data = vec![1.0f32; b * c * h * w];
    let img = Tensor::new(Array::from_shape_vec((b, c, h, w), data).unwrap().into_dyn(), false);

    let vit = VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, max_len);
    let out = vit.forward(&img);
    let shape = out.lock().storage.shape();
    // patches per dim = h/patch_size * w/patch_size
    let patches_per_dim = (h / patch_size) * (w / patch_size);
    assert_eq!(shape, vec![b, patches_per_dim, d_model]);
}
