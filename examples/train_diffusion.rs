use ndarray::Array;
use tensor_engine::nn::{DDPMScheduler, TimestepEmbedding, UNetModel};
use tensor_engine::tensor::Tensor;

fn main() {
    let in_ch = 4usize;
    let base_channels = 4usize;
    let depth = 2usize;
    let d_model = 8usize;
    let unet = UNetModel::new(in_ch, base_channels, depth);
    let temb = TimestepEmbedding::new(d_model, d_model);
    let scheduler = DDPMScheduler::new_linear(100, 0.0001, 0.02);

    // Synthetic data: single image of shape [1, C, H, W]
    let b = 1usize;
    let c = in_ch;
    let h = 8usize;
    let w = 8usize;
    let data = vec![0.5f32; b * c * h * w];
    let x_start = Tensor::new(Array::from_shape_vec((b, c, h, w), data).unwrap().into_dyn(), false);

    // Training loop skeleton: sample random t and noise, compute x_t via q_sample
    let t = 10usize;
    let noise = Tensor::new(Array::from_shape_vec((b, c, h, w), vec![0.01f32; b * c * h * w]).unwrap().into_dyn(), false);
    let x_t = scheduler.q_sample(&x_start, t, &noise);
    println!("x_t shape: {:?}", x_t.lock().storage.shape());
    let t_tensor = Tensor::new(Array::from_shape_vec((b, 1), vec![t as f32]).unwrap().into_dyn(), false);
    let t_embed = temb.forward(&t_tensor);
    // Model forward
    let pred = unet.forward(&x_t, &t_embed);
    println!("Predicted shape: {:?}", pred.lock().storage.shape());
}
