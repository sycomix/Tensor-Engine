use crate::tensor::Tensor;
use ndarray::IxDyn;
use ndarray::Array;

#[cfg(feature = "vision")]
pub fn load_image_to_tensor(path: &str, resize: Option<(u32, u32)>) -> Result<Tensor, String> {
    use image::GenericImageView;
    let img = image::open(path).map_err(|e| format!("failed to open image {}: {}", path, e))?;
    let img = if let Some((w, h)) = resize { img.resize(w, h, image::imageops::FilterType::Triangle) } else { img };
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut data: Vec<f32> = Vec::with_capacity((w * h * 3) as usize);
    // Convert to NCHW
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let p = rgb.get_pixel(x, y)[c];
                data.push((p as f32) / 255.0);
            }
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&[1, 3, h as usize, w as usize]), data).map_err(|e| format!("ndarray shape creation failed: {}", e))?;
    Ok(Tensor::new(arr.into_dyn(), false))
}

#[cfg(not(feature = "vision"))]
pub fn load_image_to_tensor(_path: &str, _resize: Option<(u32, u32)>) -> Result<Tensor, String> {
    Err("vision feature not enabled; compile with --features vision".to_string())
}
