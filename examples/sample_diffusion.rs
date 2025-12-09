use tensor_engine::nn::{UNetModel, TimestepEmbedding, DDPMScheduler};
use tensor_engine::tensor::Tensor;
use ndarray::Array;

fn main() {
    let in_ch = 4usize;
    let base_channels = 4usize;
    let depth = 2usize;
    let d_model = 8usize;

    let unet = UNetModel::new(in_ch, base_channels, depth);
    let temb = TimestepEmbedding::new(d_model, d_model);
    let scheduler = DDPMScheduler::new_linear(100, 0.0001, 0.02);

    let b = 1usize;
    let c = in_ch;
    let h = 8usize;
    let w = 8usize;
    // start from noise
    let mut x_t = Tensor::new(Array::from_shape_vec((b, c, h, w), vec![0.0f32; b*c*h*w]).unwrap().into_dyn(), false);

    // Sampling loop: reverse from T-1 down to 0
    for t in (0..scheduler.num_train_timesteps).rev() {
        let t_tensor = Tensor::new(Array::from_shape_vec((b, 1), vec![t as f32]).unwrap().into_dyn(), false);
        let t_emb = temb.forward(&t_tensor);
        let _eps = unet.forward(&x_t, &t_emb);
        // Posterior step: for demo, compute x_{t-1} mean via scheduler.step using model preds (unused _eps ignored)
        let mean = scheduler.step(&unet, &x_t, t);
        x_t = mean; // no noise for simplicity
    }
    println!("Sampled shape: {:?}", x_t.lock().storage.shape());
}
