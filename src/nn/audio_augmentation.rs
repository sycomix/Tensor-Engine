//! Audio augmentation operations.
//!
//! Provides common audio transformations for data augmentation:
//! noise injection, speed perturbation, time stretching, pitch shifting,
//! volume normalization, and more.

use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use rand::Rng;

/// Audio augmentation configuration.
#[derive(Clone, Debug)]
pub struct AudioAugmentConfig {
    /// Maximum noise level (SNR in dB)
    pub max_noise_db: f32,
    /// Speed perturbation range (e.g., 0.9 to 1.1)
    pub speed_range: (f32, f32),
    /// Pitch shift range (in semitones)
    pub pitch_range: (f32, f32),
    /// Maximum volume change (dB)
    pub max_volume_db: f32,
    /// Whether to apply random time stretching
    pub do_time_stretch: bool,
    /// Time stretch factor range
    pub time_stretch_range: (f32, f32),
}

impl Default for AudioAugmentConfig {
    fn default() -> Self {
        AudioAugmentConfig {
            max_noise_db: 20.0,
            speed_range: (0.9, 1.1),
            pitch_range: (-2.0, 2.0),
            max_volume_db: 6.0,
            do_time_stretch: true,
            time_stretch_range: (0.9, 1.1),
        }
    }
}

/// Audio augmentator.
pub struct AudioAugmenter {
    config: AudioAugmentConfig,
}

impl AudioAugmenter {
    /// Create a new audio augmentator.
    pub fn new(config: AudioAugmentConfig) -> Self {
        AudioAugmenter { config }
    }

    /// Apply all augmentations to a waveform.
    pub fn forward(&self, waveform: &Tensor) -> Tensor {
        let mut out = waveform.clone();

        // Apply augmentations in order
        out = self.add_noise(&out);
        out = self.apply_speed_perturbation(&out);
        out = self.apply_pitch_shift(&out);
        out = self.apply_volume_change(&out);
        if self.config.do_time_stretch {
            out = self.apply_time_stretch(&out);
        }

        out
    }

    /// Add random Gaussian noise to the waveform.
    fn add_noise(&self, waveform: &Tensor) -> Tensor {
        let mut rng = rand::rng();
        let snr_db = rng.random_range(0.0..=self.config.max_noise_db);
        let snr = 10.0_f32.powf(-snr_db / 20.0);

        let data = waveform.lock().storage.to_f32_array();
        let shape = data.shape().to_vec();
        let mut out = data.clone();

        // Compute signal power
        let signal_power: f32 = data.iter().map(|v| v * v).sum::<f32>() / data.len() as f32;
        let noise_power = signal_power * (snr * snr);
        let noise_std = noise_power.sqrt();

        if noise_std > 1e-10 {
            for v in out.iter_mut() {
                let noise = rng.random_range(-1.0..1.0) * noise_std;
                *v += noise;
            }
        }

        Tensor::new(
            match ArrayD::from_shape_vec(IxDyn(&shape), out) {
                Ok(v) => v,
                Err(_) => ArrayD::zeros(IxDyn(&shape)),
            },
            false,
        )
    }

    /// Apply speed perturbation (time scaling).
    fn apply_speed_perturbation(&self, waveform: &Tensor) -> Tensor {
        let mut rng = rand::rng();
        let speed = rng.random_range(self.config.speed_range.0..=self.config.speed_range.1);

        if (speed - 1.0).abs() < 1e-6 {
            return waveform.clone();
        }

        let data = waveform.lock().storage.to_f32_array();
        let n_samples = data.len();
        let new_len = (n_samples as f32 / speed) as usize;

        if new_len == 0 {
            return waveform.clone();
        }

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[new_len][..]));

        for i in 0..new_len {
            let src_pos = (i as f32 * speed) as usize;
            if src_pos < n_samples {
                out[[i]] = data[[src_pos]];
            }
        }

        Tensor::new(out, false)
    }

    /// Apply pitch shift (simplified via resampling).
    fn apply_pitch_shift(&self, waveform: &Tensor) -> Tensor {
        let mut rng = rand::rng();
        let semitones = rng.random_range(self.config.pitch_range.0..=self.config.pitch_range.1);

        if (semitones).abs() < 1e-6 {
            return waveform.clone();
        }

        // Pitch shift factor: 2^(semitones / 12)
        let factor = 2.0_f32.powf(semitones / 12.0);
        let data = waveform.lock().storage.to_f32_array();
        let n_samples = data.len();
        let new_len = (n_samples as f32 / factor) as usize;

        if new_len == 0 || new_len == n_samples {
            return waveform.clone();
        }

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[new_len][..]));

        for i in 0..new_len {
            let src_pos = (i as f32 * factor) as usize;
            if src_pos < n_samples {
                out[[i]] = data[[src_pos]];
            }
        }

        Tensor::new(out, false)
    }

    /// Apply random volume change.
    fn apply_volume_change(&self, waveform: &Tensor) -> Tensor {
        let mut rng = rand::rng();
        let db_change = rng.random_range(-self.config.max_volume_db..=self.config.max_volume_db);
        let factor = 10.0_f32.powf(db_change / 20.0);

        if (factor - 1.0).abs() < 1e-6 {
            return waveform.clone();
        }

        let data = waveform.lock().storage.to_f32_array();
        let shape = data.shape().to_vec();
        let mut out = data.clone();

        for v in out.iter_mut() {
            *v *= factor;
        }

        Tensor::new(
            match ArrayD::from_shape_vec(IxDyn(&shape), out) {
                Ok(v) => v,
                Err(_) => ArrayD::zeros(IxDyn(&shape)),
            },
            false,
        )
    }

    /// Apply time stretching.
    fn apply_time_stretch(&self, waveform: &Tensor) -> Tensor {
        let mut rng = rand::rng();
        let stretch_factor =
            rng.random_range(self.config.time_stretch_range.0..=self.config.time_stretch_range.1);

        if (stretch_factor - 1.0).abs() < 1e-6 {
            return waveform.clone();
        }

        let data = waveform.lock().storage.to_f32_array();
        let n_samples = data.len();
        let new_len = (n_samples as f32 * stretch_factor) as usize;

        if new_len == 0 || new_len == n_samples {
            return waveform.clone();
        }

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[new_len][..]));

        for i in 0..new_len {
            let src_pos = (i as f32 / stretch_factor) as usize;
            if src_pos < n_samples {
                out[[i]] = data[[src_pos]];
            }
        }

        Tensor::new(out, false)
    }

    /// Get the configuration.
    pub fn config(&self) -> &AudioAugmentConfig {
        &self.config
    }
}

/// Audio normalization utilities.
pub struct AudioNormalizer {
    /// Target RMS level in dB
    pub target_rms_db: f32,
}

impl AudioNormalizer {
    /// Create a new audio normalizer.
    pub fn new(target_rms_db: f32) -> Self {
        AudioNormalizer { target_rms_db }
    }

    /// Normalize audio to target RMS level.
    pub fn normalize(&self, waveform: &Tensor) -> Tensor {
        let data = waveform.lock().storage.to_f32_array();
        let shape = data.shape().to_vec();

        // Compute current RMS
        let rms: f32 = (data.iter().map(|v| v * v).sum::<f32>() / data.len() as f32).sqrt();

        if rms < 1e-10 {
            return waveform.clone();
        }

        // Target RMS
        let target_rms = 10.0_f32.powf(self.target_rms_db / 20.0);
        let scale = target_rms / rms;

        let mut out = data.clone();
        for v in out.iter_mut() {
            *v *= scale;
        }

        Tensor::new(
            match ArrayD::from_shape_vec(IxDyn(&shape), out) {
                Ok(v) => v,
                Err(_) => ArrayD::zeros(IxDyn(&shape)),
            },
            false,
        )
    }

    /// Peak normalize audio to target peak value.
    pub fn peak_normalize(&self, waveform: &Tensor, target_peak: f32) -> Tensor {
        let data = waveform.lock().storage.to_f32_array();
        let shape = data.shape().to_vec();

        let peak = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        if peak < 1e-10 {
            return waveform.clone();
        }

        let scale = target_peak / peak;
        let mut out = data.clone();
        for v in out.iter_mut() {
            *v *= scale;
        }

        Tensor::new(
            match ArrayD::from_shape_vec(IxDyn(&shape), out.into_raw_vec()) {
                Ok(v) => v,
                Err(_) => ArrayD::zeros(IxDyn(&shape)),
            },
            false,
        )
    }
}

/// Audio data augmentation pipeline.
pub struct AudioAugmentationPipeline {
    augmenters: Vec<AudioAugmenter>,
    normalizer: Option<AudioNormalizer>,
}

impl AudioAugmentationPipeline {
    /// Create a new pipeline.
    pub fn new() -> Self {
        AudioAugmentationPipeline {
            augmenters: Vec::new(),
            normalizer: None,
        }
    }

    /// Add an augmenter to the pipeline.
    pub fn with_augmenter(mut self, augmenter: AudioAugmenter) -> Self {
        self.augmenters.push(augmenter);
        self
    }

    /// Add a normalizer to the pipeline.
    pub fn with_normalizer(mut self, normalizer: AudioNormalizer) -> Self {
        self.normalizer = Some(normalizer);
        self
    }

    /// Apply all augmentations in order.
    pub fn forward(&self, waveform: &Tensor) -> Tensor {
        let mut out = waveform.clone();
        for aug in &self.augmenters {
            out = aug.forward(&out);
        }
        if let Some(normalizer) = &self.normalizer {
            out = normalizer.normalize(&out);
        }
        out
    }

    /// Apply augmentations to a batch of waveforms.
    pub fn forward_batch(&self, waveforms: &[Tensor]) -> Vec<Tensor> {
        waveforms.iter().map(|w| self.forward(w)).collect()
    }
}

impl Default for AudioAugmentationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Common audio augmentation presets.
impl AudioAugmenter {
    /// Standard speech augmentation: noise + speed perturbation.
    pub fn speech_standard() -> Self {
        AudioAugmenter::new(AudioAugmentConfig {
            max_noise_db: 20.0,
            speed_range: (0.9, 1.1),
            pitch_range: (-2.0, 2.0),
            max_volume_db: 6.0,
            do_time_stretch: false,
            time_stretch_range: (0.9, 1.1),
        })
    }

    /// Music augmentation: pitch + time stretch.
    pub fn music_standard() -> Self {
        AudioAugmenter::new(AudioAugmentConfig {
            max_noise_db: 30.0,
            speed_range: (1.0, 1.0),
            pitch_range: (-1.0, 1.0),
            max_volume_db: 3.0,
            do_time_stretch: true,
            time_stretch_range: (0.95, 1.05),
        })
    }
}

#[cfg(test)]
mod audio_augmentation_tests {
    use super::*;

    #[test]
    fn test_add_noise() {
        let config = AudioAugmentConfig {
            max_noise_db: 20.0,
            ..AudioAugmentConfig::default()
        };
        let augmenter = AudioAugmenter::new(config);

        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();
        let waveform = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1600]), samples).unwrap(),
            false,
        );

        let result = augmenter.add_noise(&waveform);
        let result_data = result.lock().storage.to_f32_array();
        assert_eq!(result_data.len(), 1600);
    }

    #[test]
    fn test_speed_perturbation() {
        let config = AudioAugmentConfig {
            speed_range: (0.9, 1.1),
            ..AudioAugmentConfig::default()
        };
        let augmenter = AudioAugmenter::new(config);

        let samples: Vec<f32> = (0..1600).map(|i| i as f32).collect();
        let waveform = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1600]), samples).unwrap(),
            false,
        );

        let result = augmenter.apply_speed_perturbation(&waveform);
        let result_data = result.lock().storage.to_f32_array();
        // Length should be approximately 1600 / speed
        assert!(result_data.len() > 0);
    }

    #[test]
    fn test_volume_change() {
        let config = AudioAugmentConfig {
            max_volume_db: 6.0,
            ..AudioAugmentConfig::default()
        };
        let augmenter = AudioAugmenter::new(config);

        let samples: Vec<f32> = vec![0.5; 100];
        let waveform = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[100]), samples).unwrap(),
            false,
        );

        let result = augmenter.apply_volume_change(&waveform);
        let result_data = result.lock().storage.to_f32_array();
        assert_eq!(result_data.len(), 100);
    }

    #[test]
    fn test_audio_normalizer() {
        let normalizer = AudioNormalizer::new(-20.0);

        let samples: Vec<f32> = vec![1.0; 1000];
        let waveform = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1000]), samples).unwrap(),
            false,
        );

        let result = normalizer.normalize(&waveform);
        let result_data = result.lock().storage.to_f32_array();
        assert_eq!(result_data.len(), 1000);
    }

    #[test]
    fn test_audio_augmentation_pipeline() {
        let pipeline = AudioAugmentationPipeline::new()
            .with_augmenter(AudioAugmenter::speech_standard())
            .with_normalizer(AudioNormalizer::new(-20.0));

        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.005).sin()).collect();
        let waveform = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1600]), samples).unwrap(),
            false,
        );

        let result = pipeline.forward(&waveform);
        let result_data = result.lock().storage.to_f32_array();
        assert!(result_data.len() > 0);
    }

    #[test]
    fn test_peak_normalize() {
        let normalizer = AudioNormalizer::new(-20.0);

        let samples: Vec<f32> = vec![10.0; 100];
        let waveform = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[100]), samples).unwrap(),
            false,
        );

        let result = normalizer.peak_normalize(&waveform, 1.0);
        let result_data = result.lock().storage.to_f32_array();

        // All values should be normalized to ~1.0
        for &v in result_data.iter() {
            assert!((v.abs() - 1.0) < 1e-5);
        }
    }
}
