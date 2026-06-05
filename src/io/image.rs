//! Image augmentation operations for data preprocessing.
//!
//! Implements common image transformations used in computer vision training:
//! random cropping, flipping, rotation, color jittering, normalization, etc.

use crate::tensor::Tensor;
use image::GenericImageView;
use ndarray::{ArrayD, IxDyn};
use rand::Rng;
use std::f32::consts::PI;

/// Load an image file to a tensor [C, H, W] with values in [0, 1].
/// Supports PNG and JPEG formats via the `image` crate.
#[cfg(feature = "vision")]
pub fn load_image_to_tensor(path: &str, resize: Option<(u32, u32)>) -> Result<Tensor, String> {
    let img = image::open(path).map_err(|e| format!("Failed to open image {}: {}", path, e))?;
    let (w, h) = img.dimensions();
    let img = if let Some((rw, rh)) = resize {
        img.resize(rw, rh, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let (w, h) = img.dimensions();
    let c = 3u32;
    let mut data = Vec::with_capacity((c * h * w) as usize);
    for y in 0..h {
        for x in 0..w {
            let pixel = img.get_pixel(x, y);
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    let arr = ArrayD::from_shape_vec(IxDyn(&[c as usize, h as usize, w as usize]), data)
        .map_err(|e| format!("Failed to create tensor from image: {}", e))?;
    Ok(Tensor::new(arr.into_dyn(), false))
}

/// Load an image file to a tensor [C, H, W] with values in [0, 1].
/// Returns an error if the `vision` feature is not enabled.
#[cfg(not(feature = "vision"))]
pub fn load_image_to_tensor(_path: &str, _resize: Option<(u32, u32)>) -> Result<Tensor, String> {
    Err("load_image_to_tensor requires the 'vision' feature to be enabled".to_string())
}

/// Image augmentation configuration
#[derive(Clone)]
pub struct AugmentationConfig {
    /// Random crop size (height, width)
    pub crop_size: Option<(usize, usize)>,
    /// Random horizontal flip probability
    pub hflip_prob: f32,
    /// Random vertical flip probability
    pub vflip_prob: f32,
    /// Random rotation range in degrees
    pub rotation_range: f32,
    /// Random brightness adjustment range
    pub brightness_range: f32,
    /// Random contrast adjustment range
    pub contrast_range: f32,
    /// Random saturation adjustment range
    pub saturation_range: f32,
    /// Random hue adjustment range in radians
    pub hue_range: f32,
    /// Random Gaussian noise std
    pub noise_std: f32,
    /// Random Gaussian blur kernel size
    pub blur_kernel_size: usize,
    /// Image normalization parameters
    pub normalize: bool,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        AugmentationConfig {
            crop_size: None,
            hflip_prob: 0.5,
            vflip_prob: 0.5,
            rotation_range: 10.0,
            brightness_range: 0.2,
            contrast_range: 0.2,
            saturation_range: 0.2,
            hue_range: 0.1,
            noise_std: 0.0,
            blur_kernel_size: 3,
            normalize: true,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }
}

/// Image augmentor applies transformations to image tensors.
pub struct ImageAugmentor {
    config: AugmentationConfig,
}

impl ImageAugmentor {
    /// Create a new augmentor with the given configuration.
    pub fn new(config: AugmentationConfig) -> Self {
        ImageAugmentor { config }
    }

    /// Create a default augmentor (ImageNet-style).
    pub fn new_default() -> Self {
        ImageAugmentor {
            config: AugmentationConfig::default(),
        }
    }

    /// Apply all augmentations to an image tensor [C, H, W].
    pub fn augment(&self, image: &Tensor) -> Tensor {
        let mut result = image.clone();

        // Apply augmentations in order
        if let Some((h, w)) = self.config.crop_size {
            result = self.random_crop(&result, h, w);
        }
        if self.config.hflip_prob > 0.0 && rand::random::<f32>() < self.config.hflip_prob {
            result = self.horizontal_flip(&result);
        }
        if self.config.vflip_prob > 0.0 && rand::random::<f32>() < self.config.vflip_prob {
            result = self.vertical_flip(&result);
        }
        if self.config.rotation_range > 0.0 {
            result = self.rotate(&result, self.random_angle());
        }
        result = self.color_jitter(&result);
        result = self.add_noise(&result);
        if self.config.normalize {
            result = self.normalize(&result);
        }

        result
    }

    /// Random crop from an image.
    fn random_crop(&self, image: &Tensor, crop_h: usize, crop_w: usize) -> Tensor {
        let arr = image.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();
        if shape.len() != 3 {
            return image.clone();
        }

        let c = shape[0];
        let h = shape[1];
        let w = shape[2];

        if crop_h > h || crop_w > w {
            return image.clone();
        }

        let mut rng = rand::rng();
        let start_y = rng.random_range(0..(h - crop_h + 1));
        let start_x = rng.random_range(0..(w - crop_w + 1));

        let mut cropped = ArrayD::<f32>::zeros(IxDyn(&[c, crop_h, crop_w][..]));
        for ch in 0..c {
            for y in 0..crop_h {
                for x in 0..crop_w {
                    cropped[[ch, y, x]] = arr[[ch, start_y + y, start_x + x]];
                }
            }
        }

        Tensor::new(cropped.into_dyn(), false)
    }

    /// Horizontal flip.
    fn horizontal_flip(&self, image: &Tensor) -> Tensor {
        let arr = image.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();
        if shape.len() != 3 {
            return image.clone();
        }

        let c = shape[0];
        let h = shape[1];
        let w = shape[2];

        let mut flipped = ArrayD::<f32>::zeros(IxDyn(&[c, h, w][..]));
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    flipped[[ch, y, x]] = arr[[ch, y, w - 1 - x]];
                }
            }
        }

        Tensor::new(flipped.into_dyn(), false)
    }

    /// Vertical flip.
    fn vertical_flip(&self, image: &Tensor) -> Tensor {
        let arr = image.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();
        if shape.len() != 3 {
            return image.clone();
        }

        let c = shape[0];
        let h = shape[1];
        let w = shape[2];

        let mut flipped = ArrayD::<f32>::zeros(IxDyn(&[c, h, w][..]));
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    flipped[[ch, y, x]] = arr[[ch, h - 1 - y, x]];
                }
            }
        }

        Tensor::new(flipped.into_dyn(), false)
    }

    /// Rotate image by angle (in degrees).
    fn rotate(&self, image: &Tensor, angle_deg: f32) -> Tensor {
        let arr = image.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();
        if shape.len() != 3 {
            return image.clone();
        }

        let c = shape[0];
        let h = shape[1];
        let w = shape[2];
        let angle_rad = angle_deg * PI / 180.0;

        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let center_y = (h as f32) / 2.0;
        let center_x = (w as f32) / 2.0;

        let mut rotated = ArrayD::<f32>::zeros(IxDyn(&[c, h, w][..]));
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    // Translate to origin
                    let dx = x as f32 - center_x;
                    let dy = y as f32 - center_y;

                    // Rotate
                    let rx = dx * cos_a - dy * sin_a + center_x;
                    let ry = dx * sin_a + dy * cos_a + center_y;

                    // Clamp and bilinear interpolate
                    let y0 = ry.floor() as isize;
                    let y1 = (y0 + 1).clamp(0, (h as isize) - 1);
                    let x0 = rx.floor() as isize;
                    let x1 = (x0 + 1).clamp(0, (w as isize) - 1);

                    let fy = ry - y0 as f32;
                    let fx = rx - x0 as f32;

                    let v00 = if y0 >= 0 && y0 < h as isize && x0 >= 0 && x0 < w as isize {
                        arr[[ch, y0 as usize, x0 as usize]]
                    } else {
                        0.0
                    };
                    let v01 = if y0 >= 0 && y0 < h as isize && x1 >= 0 && x1 < w as isize {
                        arr[[ch, y0 as usize, x1 as usize]]
                    } else {
                        0.0
                    };
                    let v10 = if y1 >= 0 && y1 < h as isize && x0 >= 0 && x0 < w as isize {
                        arr[[ch, y1 as usize, x0 as usize]]
                    } else {
                        0.0
                    };
                    let v11 = if y1 >= 0 && y1 < h as isize && x1 >= 0 && x1 < w as isize {
                        arr[[ch, y1 as usize, x1 as usize]]
                    } else {
                        0.0
                    };

                    let val = (1.0 - fy) * (1.0 - fx) * v00
                        + (1.0 - fy) * fx * v01
                        + fy * (1.0 - fx) * v10
                        + fy * fx * v11;

                    rotated[[ch, y, x]] = val;
                }
            }
        }

        Tensor::new(rotated.into_dyn(), false)
    }

    /// Color jitter: random brightness, contrast, saturation, hue.
    fn color_jitter(&self, image: &Tensor) -> Tensor {
        let arr = image.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();
        if shape.len() != 3 {
            return image.clone();
        }

        let c = shape[0];
        let h = shape[1];
        let w = shape[2];

        let mut jittered = arr.clone();

        // Brightness
        if self.config.brightness_range > 0.0 {
            let factor = 1.0 + (rand::random::<f32>() * 2.0 - 1.0) * self.config.brightness_range;
            for i in 0..jittered.len() {
                jittered[i] *= factor;
            }
        }

        // Contrast
        if self.config.contrast_range > 0.0 {
            let factor = 1.0 + (rand::random::<f32>() * 2.0 - 1.0) * self.config.contrast_range;
            for i in 0..jittered.len() {
                jittered[i] *= factor;
            }
        }

        // Saturation (per channel)
        if self.config.saturation_range > 0.0 && c >= 3 {
            for ch in 0..3.min(c) {
                let factor =
                    1.0 + (rand::random::<f32>() * 2.0 - 1.0) * self.config.saturation_range;
                for y in 0..h {
                    for x in 0..w {
                        jittered[[ch, y, x]] *= factor;
                    }
                }
            }
        }

        Tensor::new(jittered.into_dyn(), false)
    }

    /// Add Gaussian noise.
    fn add_noise(&self, image: &Tensor) -> Tensor {
        if self.config.noise_std <= 0.0 {
            return image.clone();
        }

        let arr = image.lock().storage.to_f32_array();
        let mut noisy = arr.clone();

        for i in 0..noisy.len() {
            let noise = rand::random::<f32>() * 2.0 - 1.0;
            noisy[i] += noise * self.config.noise_std;
        }

        Tensor::new(noisy.into_dyn(), false)
    }

    /// Normalize image with mean and std.
    fn normalize(&self, image: &Tensor) -> Tensor {
        let arr = image.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();
        if shape.len() != 3 {
            return image.clone();
        }

        let c = shape[0];
        let mut normalized = ArrayD::<f32>::zeros(IxDyn(&shape));

        for ch in 0..c.min(self.config.mean.len()) {
            let mean = self.config.mean[ch];
            let std = if ch < self.config.std.len() {
                self.config.std[ch]
            } else {
                1.0
            };

            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    normalized[[ch, y, x]] = (arr[[ch, y, x]] - mean) / std;
                }
            }
        }

        Tensor::new(normalized.into_dyn(), false)
    }

    /// Generate a random angle within the configured range.
    fn random_angle(&self) -> f32 {
        (rand::random::<f32>() * 2.0 - 1.0) * self.config.rotation_range
    }
}

/// Random erasing augmentation.
/// Randomly erases rectangular regions of the image.
pub struct RandomErasing {
    /// Probability of applying erasing
    pub probability: f32,
    /// Minimum area ratio
    pub min_area_ratio: f32,
    /// Maximum area ratio
    pub max_area_ratio: f32,
    /// Aspect ratio range
    pub aspect_ratio_range: (f32, f32),
    /// Fill value
    pub fill_value: f32,
}

impl RandomErasing {
    /// Create a new random erasing augmentation.
    pub fn new(
        probability: f32,
        min_area_ratio: f32,
        max_area_ratio: f32,
        aspect_ratio_range: (f32, f32),
        fill_value: f32,
    ) -> Self {
        RandomErasing {
            probability,
            min_area_ratio,
            max_area_ratio,
            aspect_ratio_range,
            fill_value,
        }
    }

    /// Apply random erasing to an image tensor [C, H, W].
    pub fn apply(&self, image: &Tensor) -> Tensor {
        if rand::random::<f32>() > self.probability {
            return image.clone();
        }

        let arr = image.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();
        if shape.len() != 3 {
            return image.clone();
        }

        let c = shape[0];
        let h = shape[1];
        let w = shape[2];
        let area = h * w;

        let mut rng = rand::rng();
        let mut erase_area = (area as f32
            * (self.min_area_ratio
                + rng.random::<f32>() * (self.max_area_ratio - self.min_area_ratio)))
            as usize;
        // Ensure we erase at least one pixel for small images / ratios
        if erase_area == 0 {
            erase_area = 1;
        }
        let aspect_ratio = self.aspect_ratio_range.0
            + rng.random::<f32>() * (self.aspect_ratio_range.1 - self.aspect_ratio_range.0);
        let mut erase_h = (erase_area as f32 * aspect_ratio).sqrt() as usize;
        let mut erase_w = (erase_area as f32 / aspect_ratio).sqrt() as usize;

        // Ensure at least a 1x1 erase region
        if erase_h == 0 {
            erase_h = 1;
        }
        if erase_w == 0 {
            erase_w = 1;
        }

        let erase_h = erase_h.min(h);
        let erase_w = erase_w.min(w);

        let y1 = rng.random_range(0..(h - erase_h + 1));
        let x1 = rng.random_range(0..(w - erase_w + 1));

        let mut erased = arr.clone();
        for ch in 0..c {
            for y in y1..(y1 + erase_h) {
                for x in x1..(x1 + erase_w) {
                    erased[[ch, y, x]] = self.fill_value;
                }
            }
        }

        Tensor::new(erased.into_dyn(), false)
    }
}

/// Color jitter augmentation.
pub struct ColorJitter {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
}

impl ColorJitter {
    /// Create a new color jitter augmentation.
    pub fn new(brightness: f32, contrast: f32, saturation: f32, hue: f32) -> Self {
        ColorJitter {
            brightness,
            contrast,
            saturation,
            hue,
        }
    }

    /// Apply color jitter to an image tensor [C, H, W].
    pub fn apply(&self, image: &Tensor) -> Tensor {
        let arr = image.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();
        if shape.len() != 3 {
            return image.clone();
        }

        let mut result = arr.clone();

        if self.brightness > 0.0 {
            let factor = 1.0 + (rand::random::<f32>() * 2.0 - 1.0) * self.brightness;
            for v in result.iter_mut() {
                *v *= factor;
            }
        }

        if self.contrast > 0.0 {
            let factor = 1.0 + (rand::random::<f32>() * 2.0 - 1.0) * self.contrast;
            for v in result.iter_mut() {
                *v *= factor;
            }
        }

        if self.saturation > 0.0 {
            let factor = 1.0 + (rand::random::<f32>() * 2.0 - 1.0) * self.saturation;
            for v in result.iter_mut() {
                *v *= factor;
            }
        }

        if self.hue > 0.0 {
            // Simple hue shift: add a constant to all channels
            let shift = (rand::random::<f32>() * 2.0 - 1.0) * self.hue;
            for v in result.iter_mut() {
                *v += shift;
            }
        }

        Tensor::new(result.into_dyn(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizontal_flip() {
        let config = AugmentationConfig::default();
        let augmentor = ImageAugmentor::new(config);
        let image_data = vec![1.0, 2.0, 3.0, 4.0]; // [1, 2, 2]
        let image = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), image_data).unwrap(),
            false,
        );
        let flipped = augmentor.horizontal_flip(&image);
        let flipped_arr = flipped.lock().storage.to_f32_array();
        // Original: [[1, 2], [3, 4]]
        // Flipped: [[2, 1], [4, 3]]
        assert!((flipped_arr[[0, 0, 0]] - 2.0).abs() < 1e-6);
        assert!((flipped_arr[[0, 0, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vertical_flip() {
        let config = AugmentationConfig::default();
        let augmentor = ImageAugmentor::new(config);
        let image_data = vec![1.0, 2.0, 3.0, 4.0]; // [1, 2, 2]
        let image = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), image_data).unwrap(),
            false,
        );
        let flipped = augmentor.vertical_flip(&image);
        let flipped_arr = flipped.lock().storage.to_f32_array();
        // Original: [[1, 2], [3, 4]]
        // Flipped: [[3, 4], [1, 2]]
        assert!((flipped_arr[[0, 0, 0]] - 3.0).abs() < 1e-6);
        assert!((flipped_arr[[0, 1, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let config = AugmentationConfig {
            normalize: true,
            mean: vec![0.5],
            std: vec![0.5],
            ..Default::default()
        };
        let augmentor = ImageAugmentor::new(config);
        let image_data = vec![0.5, 1.0, 1.5, 2.0]; // [1, 2, 2]
        let image = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), image_data).unwrap(),
            false,
        );
        let normalized = augmentor.normalize(&image);
        let norm_arr = normalized.lock().storage.to_f32_array();
        // (0.5 - 0.5) / 0.5 = 0.0
        // (1.0 - 0.5) / 0.5 = 1.0
        // (1.5 - 0.5) / 0.5 = 2.0
        // (2.0 - 0.5) / 0.5 = 3.0
        assert!((norm_arr[[0, 0, 0]] - 0.0).abs() < 1e-6);
        assert!((norm_arr[[0, 0, 1]] - 1.0).abs() < 1e-6);
        assert!((norm_arr[[0, 1, 0]] - 2.0).abs() < 1e-6);
        assert!((norm_arr[[0, 1, 1]] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_random_erasing() {
        let erasing = RandomErasing::new(1.0, 0.1, 0.2, (0.5, 2.0), 0.0);
        let image_data = vec![1.0, 2.0, 3.0, 4.0]; // [1, 2, 2]
        let image = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), image_data).unwrap(),
            false,
        );
        let erased = erasing.apply(&image);
        let erased_arr = erased.lock().storage.to_f32_array();
        // At least some values should be 0.0 (erased)
        let erased_count = erased_arr.iter().filter(|&&v| v == 0.0).count();
        assert!(erased_count > 0);
    }
}
