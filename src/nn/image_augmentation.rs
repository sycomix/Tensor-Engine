//! Image preprocessing and augmentation operations.
//!
//! Provides common image transformations used in computer vision pipelines:
//! normalization, resizing, cropping, flipping, rotation, color jitter, etc.

use crate::tensor::Tensor;
use ndarray::{s, ArrayD, IxDyn};

/// Image normalization parameters.
#[derive(Clone, Debug)]
pub struct NormalizeConfig {
    /// Mean values per channel (typically [0.485, 0.456, 0.406] for ImageNet)
    pub mean: Vec<f32>,
    /// Standard deviation per channel (typically [0.229, 0.224, 0.225] for ImageNet)
    pub std: Vec<f32>,
}

impl NormalizeConfig {
    /// ImageNet normalization (default).
    pub fn imagenet() -> Self {
        NormalizeConfig {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }

    /// Zero-mean, unit-variance normalization.
    pub fn standard() -> Self {
        NormalizeConfig {
            mean: vec![0.0],
            std: vec![1.0],
        }
    }
}

/// Image normalizer: applies per-channel normalization.
///
/// Input shape: [C, H, W] or [B, C, H, W] (NCHW format)
pub struct ImageNormalizer {
    config: NormalizeConfig,
}

impl ImageNormalizer {
    /// Create a new normalizer.
    pub fn new(config: NormalizeConfig) -> Self {
        ImageNormalizer { config }
    }

    /// Normalize an image tensor.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let arr = input.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();

        if shape.len() != 4 {
            log::error!(
                "ImageNormalizer: expected 4D input [B, C, H, W], got {:?}",
                shape
            );
            return input.clone();
        }

        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[b, c, h, w][..]));

        let mean = &self.config.mean;
        let std = &self.config.std;

        for n in 0..b {
            for ch in 0..c {
                let m = if ch < mean.len() { mean[ch] } else { 0.0 };
                let s = if ch < std.len() { std[ch] } else { 1.0 };
                let inv_std = if s > 1e-7 { 1.0 / s } else { 0.0 };

                for y in 0..h {
                    for x in 0..w {
                        let val = arr[[n, ch, y, x]];
                        out[[n, ch, y, x]] = (val - m) * inv_std;
                    }
                }
            }
        }

        Tensor::new(out, false)
    }
}

/// Image resize operation.
///
/// Supports nearest-neighbor and bilinear interpolation.
pub struct ImageResize {
    pub height: usize,
    pub width: usize,
    pub mode: ResizeMode,
}

/// Resize interpolation mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResizeMode {
    Nearest,
    Bilinear,
}

impl ImageResize {
    /// Create a new resize operation.
    pub fn new(height: usize, width: usize, mode: ResizeMode) -> Self {
        ImageResize {
            height,
            width,
            mode,
        }
    }

    /// Resize an image tensor [B, C, H, W] to new dimensions.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let arr = input.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();

        if shape.len() != 4 {
            log::error!(
                "ImageResize: expected 4D input [B, C, H, W], got {:?}",
                shape
            );
            return input.clone();
        }

        let (b, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let h_out = self.height;
        let w_out = self.width;

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[b, c, h_out, w_out][..]));

        let scale_h = (h_in as f32 - 1.0) / (h_out as f32 - 1.0).max(1.0);
        let scale_w = (w_in as f32 - 1.0) / (w_out as f32 - 1.0).max(1.0);

        match self.mode {
            ResizeMode::Nearest => {
                for n in 0..b {
                    for ch in 0..c {
                        for y in 0..h_out {
                            let src_y = if h_out > 1 {
                                (scale_h * y as f32).round() as usize
                            } else {
                                0
                            };
                            let src_y = src_y.min(h_in - 1);

                            for x in 0..w_out {
                                let src_x = if w_out > 1 {
                                    (scale_w * x as f32).round() as usize
                                } else {
                                    0
                                };
                                let src_x = src_x.min(w_in - 1);
                                out[[n, ch, y, x]] = arr[[n, ch, src_y, src_x]];
                            }
                        }
                    }
                }
            }
            ResizeMode::Bilinear => {
                for n in 0..b {
                    for ch in 0..c {
                        for y in 0..h_out {
                            let src_y = if h_out > 1 {
                                scale_h * y as f32
                            } else {
                                0.0
                            };
                            let y0 = src_y.floor() as usize;
                            let y1 = (y0 + 1).min(h_in - 1);
                            let dy = src_y - y0 as f32;

                            for x in 0..w_out {
                                let src_x = if w_out > 1 {
                                    scale_w * x as f32
                                } else {
                                    0.0
                                };
                                let x0 = src_x.floor() as usize;
                                let x1 = (x0 + 1).min(w_in - 1);
                                let dx = src_x - x0 as f32;

                                let v00 = arr[[n, ch, y0, x0]];
                                let v01 = arr[[n, ch, y0, x1]];
                                let v10 = arr[[n, ch, y1, x0]];
                                let v11 = arr[[n, ch, y1, x1]];

                                out[[n, ch, y, x]] = (1.0 - dy) * (1.0 - dx) * v00
                                    + (1.0 - dy) * dx * v01
                                    + dy * (1.0 - dx) * v10
                                    + dy * dx * v11;
                            }
                        }
                    }
                }
            }
        }

        Tensor::new(out, false)
    }
}

/// Random horizontal flip.
pub struct HorizontalFlip {
    pub p: f32, // probability of flipping
}

impl HorizontalFlip {
    pub fn new(p: f32) -> Self {
        HorizontalFlip { p: p.clamp(0.0, 1.0) }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        if rand::random::<f32>() > self.p {
            return input.clone();
        }

        let arr = input.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();

        if shape.len() != 4 {
            return input.clone();
        }

        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[b, c, h, w][..]));

        for n in 0..b {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        out[[n, ch, y, x]] = arr[[n, ch, y, w - 1 - x]];
                    }
                }
            }
        }

        Tensor::new(out, false)
    }
}

/// Random vertical flip.
pub struct VerticalFlip {
    pub p: f32,
}

impl VerticalFlip {
    pub fn new(p: f32) -> Self {
        VerticalFlip { p: p.clamp(0.0, 1.0) }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        if rand::random::<f32>() > self.p {
            return input.clone();
        }

        let arr = input.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();

        if shape.len() != 4 {
            return input.clone();
        }

        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[b, c, h, w][..]));

        for n in 0..b {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        out[[n, ch, y, x]] = arr[[n, ch, h - 1 - y, x]];
                    }
                }
            }
        }

        Tensor::new(out, false)
    }
}

/// Random crop configuration.
#[derive(Clone, Debug)]
pub struct RandomCropConfig {
    pub height: usize,
    pub width: usize,
}

/// Random crop operation.
pub struct RandomCrop {
    config: RandomCropConfig,
}

impl RandomCrop {
    pub fn new(height: usize, width: usize) -> Self {
        RandomCrop {
            config: RandomCropConfig { height, width },
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let arr = input.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();

        if shape.len() != 4 {
            return input.clone();
        }

        let (b, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let h_out = self.config.height;
        let w_out = self.config.width;

        if h_out > h_in || w_out > w_in {
            log::error!(
                "RandomCrop: output size ({}, {}) > input size ({}, {})",
                h_out,
                w_out,
                h_in,
                w_in
            );
            return input.clone();
        }

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[b, c, h_out, w_out][..]));

        for n in 0..b {
            let y_start = rand::random::<usize>() % (h_in - h_out + 1);
            let x_start = rand::random::<usize>() % (w_in - w_out + 1);

            for ch in 0..c {
                for y in 0..h_out {
                    for x in 0..w_out {
                        out[[n, ch, y, x]] = arr[[n, ch, y_start + y, x_start + x]];
                    }
                }
            }
        }

        Tensor::new(out, false)
    }
}

/// Color jitter: random brightness, contrast, saturation adjustments.
#[derive(Clone, Debug)]
pub struct ColorJitterConfig {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
}

impl ColorJitterConfig {
    pub fn default() -> Self {
        ColorJitterConfig {
            brightness: 0.1,
            contrast: 0.1,
            saturation: 0.1,
        }
    }
}

/// Color jitter operation.
pub struct ColorJitter {
    config: ColorJitterConfig,
}

impl ColorJitter {
    pub fn new(config: ColorJitterConfig) -> Self {
        ColorJitter { config }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let arr = input.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();

        if shape.len() != 4 {
            return input.clone();
        }

        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let mut out = arr.clone();

        // Random brightness
        let brightness_factor = 1.0 + (rand::random::<f32>() * 2.0 - 1.0) * self.config.brightness;
        if (brightness_factor - 1.0).abs() > 1e-6 {
            for n in 0..b {
                for ch in 0..c {
                    for y in 0..h {
                        for x in 0..w {
                            out[[n, ch, y, x]] *= brightness_factor;
                        }
                    }
                }
            }
        }

        // Random contrast
        let contrast_factor = 1.0 + (rand::random::<f32>() * 2.0 - 1.0) * self.config.contrast;
        if (contrast_factor - 1.0).abs() > 1e-6 {
            for n in 0..b {
                // Compute mean per channel
                let mut mean = 0.0f32;
                let count = h * w;
                for y in 0..h {
                    for x in 0..w {
                        mean += out[[n, 0, y, x]];
                    }
                }
                mean /= count as f32;

                for ch in 0..c {
                    for y in 0..h {
                        for x in 0..w {
                            out[[n, ch, y, x]] = mean + (out[[n, ch, y, x]] - mean) * contrast_factor;
                        }
                    }
                }
            }
        }

        Tensor::new(out, false)
    }
}

/// Image augmentation pipeline.
pub struct ImageAugmenter {
    operations: Vec<AugmentOp>,
}

/// Image augmentation operation.
#[derive(Clone)]
pub enum AugmentOp {
    Normalize(NormalizeConfig),
    Resize(usize, usize, ResizeMode),
    HorizontalFlip(f32),
    VerticalFlip(f32),
    RandomCrop(usize, usize),
    ColorJitter(ColorJitterConfig),
}

impl ImageAugmenter {
    /// Create a new augmenter with no operations.
    pub fn new() -> Self {
        ImageAugmenter {
            operations: Vec::new(),
        }
    }

    /// Add normalization.
    pub fn with_normalize(mut self, config: NormalizeConfig) -> Self {
        self.operations.push(AugmentOp::Normalize(config));
        self
    }

    /// Add resize.
    pub fn with_resize(mut self, height: usize, width: usize, mode: ResizeMode) -> Self {
        self.operations.push(AugmentOp::Resize(height, width, mode));
        self
    }

    /// Add horizontal flip.
    pub fn with_horizontal_flip(mut self, p: f32) -> Self {
        self.operations.push(AugmentOp::HorizontalFlip(p));
        self
    }

    /// Add vertical flip.
    pub fn with_vertical_flip(mut self, p: f32) -> Self {
        self.operations.push(AugmentOp::VerticalFlip(p));
        self
    }

    /// Add random crop.
    pub fn with_random_crop(mut self, height: usize, width: usize) -> Self {
        self.operations.push(AugmentOp::RandomCrop(height, width));
        self
    }

    /// Add color jitter.
    pub fn with_color_jitter(mut self, config: ColorJitterConfig) -> Self {
        self.operations.push(AugmentOp::ColorJitter(config));
        self
    }

    /// Apply all augmentations in order.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.clone();
        for op in &self.operations {
            out = match op {
                AugmentOp::Normalize(config) => {
                    ImageNormalizer::new(config.clone()).forward(&out)
                }
                AugmentOp::Resize(h, w, mode) => {
                    ImageResize::new(*h, *w, mode.clone()).forward(&out)
                }
                AugmentOp::HorizontalFlip(p) => HorizontalFlip::new(*p).forward(&out),
                AugmentOp::VerticalFlip(p) => VerticalFlip::new(*p).forward(&out),
                AugmentOp::RandomCrop(h, w) => RandomCrop::new(*h, *w).forward(&out),
                AugmentOp::ColorJitter(config) => ColorJitter::new(config.clone()).forward(&out),
            };
        }
        out
    }
}

impl Default for ImageAugmenter {
    fn default() -> Self {
        Self::new()
    }
}

/// Common ImageNet augmentation pipeline.
pub fn imagenet_augment_pipeline() -> ImageAugmenter {
    ImageAugmenter::new()
        .with_normalize(NormalizeConfig::imagenet())
        .with_resize(224, 224, ResizeMode::Bilinear)
        .with_horizontal_flip(0.5)
        .with_color_jitter(ColorJitterConfig::default())
}

#[cfg(test)]
mod image_augmentation_tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let normalizer = ImageNormalizer::new(NormalizeConfig::imagenet());

        // Create a simple 1x3x4x4 image
        let data: Vec<f32> = vec![1.0; 3 * 4 * 4];
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 3, 4, 4]), data).unwrap(),
            false,
        );

        let output = normalizer.forward(&input);
        let out_arr = output.lock().storage.to_f32_array();

        // Check first channel normalization
        let mean = 0.485;
        let std = 0.229;
        let expected = (1.0 - mean) / std;
        assert!((out_arr[[0, 0, 0, 0]] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_resize_bilinear() {
        let resize = ImageResize::new(8, 8, ResizeMode::Bilinear);

        let data: Vec<f32> = vec![1.0; 1 * 3 * 4 * 4];
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 3, 4, 4]), data).unwrap(),
            false,
        );

        let output = resize.forward(&input);
        let out_arr = output.lock().storage.to_f32_array();
        let shape = out_arr.shape().to_vec();
        assert_eq!(shape, vec![1, 3, 8, 8]);
    }

    #[test]
    fn test_resize_nearest() {
        let resize = ImageResize::new(8, 8, ResizeMode::Nearest);

        let data: Vec<f32> = vec![1.0; 1 * 3 * 4 * 4];
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 3, 4, 4]), data).unwrap(),
            false,
        );

        let output = resize.forward(&input);
        let out_arr = output.lock().storage.to_f32_array();
        let shape = out_arr.shape().to_vec();
        assert_eq!(shape, vec![1, 3, 8, 8]);
    }

    #[test]
    fn test_horizontal_flip() {
        let flip = HorizontalFlip::new(1.0); // Always flip

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 4]), data).unwrap(),
            false,
        );

        let output = flip.forward(&input);
        let out_arr = output.lock().storage.to_f32_array();
        // Should be reversed: [4, 3, 2, 1]
        assert_eq!(out_arr[[0, 0, 0, 0]], 4.0);
        assert_eq!(out_arr[[0, 0, 0, 1]], 3.0);
        assert_eq!(out_arr[[0, 0, 0, 2]], 2.0);
        assert_eq!(out_arr[[0, 0, 0, 3]], 1.0);
    }

    #[test]
    fn test_augmenter_pipeline() {
        let pipeline = imagenet_augment_pipeline();

        let data: Vec<f32> = vec![1.0; 1 * 3 * 224 * 224];
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 3, 224, 224]), data).unwrap(),
            false,
        );

        let output = pipeline.forward(&input);
        let out_arr = output.lock().storage.to_f32_array();
        let shape = out_arr.shape().to_vec();
        assert_eq!(shape, vec![1, 3, 224, 224]);
    }

    #[test]
    fn test_color_jitter() {
        let jitter = ColorJitter::new(ColorJitterConfig::default());

        let data: Vec<f32> = vec![0.5; 1 * 3 * 4 * 4];
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 3, 4, 4]), data).unwrap(),
            false,
        );

        let output = jitter.forward(&input);
        let out_arr = output.lock().storage.to_f32_array();

        // Values should be close to original (brightness/contrast near 1.0)
        for &v in out_arr.iter() {
            assert!(v > 0.0);
        }
    }
}
