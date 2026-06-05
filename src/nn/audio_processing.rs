use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

/// Mel-spectrogram computation.
///
/// Converts a waveform (1D audio signal) into a mel-spectrogram (2D time-frequency representation).
/// Uses the existing FFT infrastructure in ops.rs.
///
/// # Arguments
/// * `waveform` - Input audio waveform of shape [T] (samples)
/// * `sample_rate` - Sample rate of the audio in Hz
/// * `n_fft` - Number of FFT components (window size)
/// * `hop_length` - Number of samples between successive frames
/// * `n_mels` - Number of mel frequency bins
/// * `f_min` - Minimum frequency (Hz)
/// * `f_max` - Maximum frequency (Hz). If None, use Nyquist (sample_rate / 2)
pub struct MelSpectrogram {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
    sample_rate: usize,
    mel_basis: Vec<f32>,
}

impl MelSpectrogram {
    /// Create a new MelSpectrogram processor.
    pub fn new(sample_rate: usize, n_fft: usize, hop_length: usize, n_mels: usize) -> Self {
        let f_max = (sample_rate / 2) as f32;
        let mel_basis = Self::create_mel_basis(
            n_fft,
            n_mels,
            f_min_default(sample_rate),
            f_max,
            sample_rate,
        );
        MelSpectrogram {
            n_fft,
            hop_length,
            n_mels,
            f_min: f_min_default(sample_rate),
            f_max,
            sample_rate,
            mel_basis,
        }
    }

    fn new_with_params(
        sample_rate: usize,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        f_min: f32,
        f_max: f32,
    ) -> Self {
        let mel_basis = Self::create_mel_basis(n_fft, n_mels, f_min, f_max, sample_rate);
        MelSpectrogram {
            n_fft,
            hop_length,
            n_mels,
            f_min,
            f_max,
            sample_rate,
            mel_basis,
        }
    }

    /// Create mel filter bank matrix.
    /// Returns a matrix of shape [n_mels, n_fft//2 + 1]
    fn create_mel_basis(
        n_fft: usize,
        n_mels: usize,
        f_min: f32,
        f_max: f32,
        sample_rate: usize,
    ) -> Vec<f32> {
        let n_fft_plus_1 = n_fft / 2 + 1;
        let mut mel = vec![0.0f32; n_mels * n_fft_plus_1];

        let f_points = (0..=n_mels + 1)
            .map(|i| f_min + (f_max - f_min) * (i as f32) / ((n_mels + 1) as f32))
            .collect::<Vec<f32>>();

        let freqs = (0..n_fft_plus_1)
            .map(|f| (f as f32) * (sample_rate as f32) / (n_fft as f32))
            .collect::<Vec<f32>>();

        let f_diffs = (1..f_points.len())
            .map(|i| f_points[i] - f_points[i - 1])
            .collect::<Vec<f32>>();

        for m in 0..n_mels {
            for k in 0..n_fft_plus_1 {
                let f = freqs[k];
                let f_prev = f_points[m];
                let f_curr = f_points[m + 1];
                let f_next = f_points[m + 2];

                let mut weight = 0.0f32;
                if f >= f_prev && f <= f_curr {
                    weight += (f - f_prev) / f_diffs[m];
                }
                if f >= f_curr && f <= f_next {
                    weight += (f_next - f) / f_diffs[m + 1];
                }
                mel[m * n_fft_plus_1 + k] = weight;
            }
        }

        mel
    }

    /// Compute mel-spectrogram from a waveform tensor [T].
    /// Returns tensor of shape [n_mels, T'] where T' = (T - n_fft) / hop_length + 1
    pub fn forward(&self, waveform: &Tensor) -> Tensor {
        let audio = waveform.lock().storage.to_f32_array();
        let n_samples = audio.len();

        if n_samples < self.n_fft {
            log::error!(
                "MelSpectrogram: waveform length {} < n_fft {}",
                n_samples,
                self.n_fft
            );
            let empty = ArrayD::zeros(IxDyn(&[self.n_mels, 1]));
            return Tensor::new(empty, false);
        }

        // Compute STFT frame by frame
        let n_frames = (n_samples - self.n_fft) / self.hop_length + 1;
        let n_freq_bins = self.n_fft / 2 + 1;

        let mut spectrogram = ArrayD::<f32>::zeros(IxDyn(&[n_freq_bins, n_frames][..]));

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let end = start + self.n_fft;

            // Extract frame with Hann window
            let mut frame = vec![0.0f32; self.n_fft];
            for i in 0..self.n_fft {
                let hann = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (self.n_fft as f32 - 1.0))
                            .cos());
                frame[i] = audio[start + i] * hann;
            }

            // Compute magnitude spectrum via RFFT
            let mag = Self::compute_rfft_magnitude(&frame, self.n_fft);

            for k in 0..n_freq_bins {
                spectrogram[[k, frame_idx]] = mag[k];
            }
        }

        // Apply mel filter bank: mel @ spectrogram -> [n_mels, n_frames]
        let mut mel_spec = ArrayD::<f32>::zeros(IxDyn(&[self.n_mels, n_frames][..]));
        for m in 0..self.n_mels {
            for k in 0..n_freq_bins {
                let weight = self.mel_basis[m * n_freq_bins + k];
                if weight > 0.0 {
                    for f in 0..n_frames {
                        mel_spec[[m, f]] += weight * spectrogram[[k, f]];
                    }
                }
            }
        }

        // Take log magnitude (with epsilon for stability)
        let eps = 1e-10;
        mel_spec.mapv_inplace(|v| (v.max(eps)).log10());

        Tensor::new(mel_spec, false)
    }

    /// Compute magnitude spectrum from a frame using DFT (simplified RFFT).
    fn compute_rfft_magnitude(frame: &[f32], n_fft: usize) -> Vec<f32> {
        let n_freq_bins = n_fft / 2 + 1;
        let mut magnitude = vec![0.0f32; n_freq_bins];

        for k in 0..n_freq_bins {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for t in 0..n_fft {
                let theta = 2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n_fft as f32);
                re += frame[t] * theta.cos();
                im -= frame[t] * theta.sin();
            }
            magnitude[k] = (re * re + im * im).sqrt();
        }

        magnitude
    }
}

/// STFT (Short-time Fourier Transform) operation.
///
/// Computes the STFT of a waveform, returning complex magnitude and phase.
pub struct STFT {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
}

impl STFT {
    /// Create a new STFT processor.
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        let window = Self::create_hann_window(n_fft);
        STFT {
            n_fft,
            hop_length,
            window,
        }
    }

    fn create_hann_window(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos()))
            .collect()
    }

    /// Compute STFT from waveform [T].
    /// Returns tensor of shape [n_freq_bins, n_frames, 2] where last dim is [real, imag]
    pub fn forward(&self, waveform: &Tensor) -> Tensor {
        let audio = waveform.lock().storage.to_f32_array();
        let n_samples = audio.len();

        if n_samples < self.n_fft {
            log::error!("STFT: waveform length {} < n_fft {}", n_samples, self.n_fft);
            let empty = ArrayD::zeros(IxDyn(&[0, 0, 2]));
            return Tensor::new(empty, false);
        }

        let n_frames = (n_samples - self.n_fft) / self.hop_length + 1;
        let n_freq_bins = self.n_fft / 2 + 1;

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n_freq_bins, n_frames, 2][..]));

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;

            // Apply window
            let mut frame = vec![0.0f32; self.n_fft];
            for i in 0..self.n_fft {
                frame[i] = audio[start + i] * self.window[i];
            }

            // Compute DFT for positive frequencies only
            for k in 0..n_freq_bins {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for t in 0..self.n_fft {
                    let theta =
                        2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (self.n_fft as f32);
                    re += frame[t] * theta.cos();
                    im -= frame[t] * theta.sin();
                }
                out[[k, frame_idx, 0]] = re;
                out[[k, frame_idx, 1]] = im;
            }
        }

        Tensor::new(out, false)
    }

    /// Compute magnitude from STFT output [n_freq, n_frames, 2].
    pub fn magnitude(&self, stft: &Tensor) -> Tensor {
        let data = stft.lock().storage.to_f32_array();
        let shape = data.shape().to_vec();
        if shape.len() != 3 || shape[2] != 2 {
            log::error!("STFT::magnitude: expected last dim = 2, got {:?}", shape);
            return stft.clone();
        }

        let (n_freq, n_frames, _) = (shape[0], shape[1], shape[2]);
        let mut mag = ArrayD::<f32>::zeros(IxDyn(&[n_freq, n_frames][..]));

        for k in 0..n_freq {
            for f in 0..n_frames {
                let re = data[[k, f, 0]];
                let im = data[[k, f, 1]];
                mag[[k, f]] = (re * re + im * im).sqrt();
            }
        }

        Tensor::new(mag, false)
    }

    /// Compute phase from STFT output [n_freq, n_frames, 2].
    pub fn phase(&self, stft: &Tensor) -> Tensor {
        let data = stft.lock().storage.to_f32_array();
        let shape = data.shape().to_vec();
        if shape.len() != 3 || shape[2] != 2 {
            log::error!("STFT::phase: expected last dim = 2, got {:?}", shape);
            return stft.clone();
        }

        let (n_freq, n_frames, _) = (shape[0], shape[1], shape[2]);
        let mut phase = ArrayD::<f32>::zeros(IxDyn(&[n_freq, n_frames][..]));

        for k in 0..n_freq {
            for f in 0..n_frames {
                let re = data[[k, f, 0]];
                let im = data[[k, f, 1]];
                phase[[k, f]] = re.atan2(im);
            }
        }

        Tensor::new(phase, false)
    }
}

/// Inverse STFT (iSTFT) to reconstruct waveform from STFT representation.
pub struct ISTFT {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
}

impl ISTFT {
    /// Create a new ISTFT processor.
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        let window = STFT::create_hann_window(n_fft);
        ISTFT {
            n_fft,
            hop_length,
            window,
        }
    }

    /// Reconstruct waveform from STFT output [n_freq, n_frames, 2].
    pub fn forward(&self, stft: &Tensor) -> Tensor {
        let data = stft.lock().storage.to_f32_array();
        let shape = data.shape().to_vec();
        if shape.len() != 3 || shape[2] != 2 {
            log::error!("ISTFT: expected last dim = 2, got {:?}", shape);
            let empty = ArrayD::zeros(IxDyn(&[0]));
            return Tensor::new(empty, false);
        }

        let (n_freq, n_frames, _) = (shape[0], shape[1], shape[2]);
        let n_fft = n_freq * 2 - 2;
        let n_samples = (n_frames - 1) * self.hop_length + n_fft;

        let mut waveform = vec![0.0f32; n_samples];
        let mut window_sum = vec![0.0f32; n_samples];

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;

            // Extract real and imaginary parts
            let mut re = vec![0.0f32; n_freq];
            let mut im = vec![0.0f32; n_freq];
            for k in 0..n_freq {
                re[k] = data[[k, frame_idx, 0]];
                im[k] = data[[k, frame_idx, 1]];
            }

            // Inverse DFT
            for t in 0..n_fft {
                let mut val = 0.0f32;
                for k in 0..n_freq {
                    let theta =
                        2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n_fft as f32);
                    val += re[k] * theta.cos() - im[k] * theta.sin();
                }
                val /= n_fft as f32;
                if start + t < n_samples {
                    waveform[start + t] += val * self.window[t];
                    window_sum[start + t] += self.window[t] * self.window[t];
                }
            }
        }

        // Normalize by window overlap
        for t in 0..n_samples {
            if window_sum[t] > 1e-8 {
                waveform[t] /= window_sum[t];
            }
        }

        let out = match ArrayD::from_shape_vec(IxDyn(&[n_samples]), waveform) {
            Ok(v) => v,
            Err(e) => {
                log::error!("ISTFT: shape construction failed: {}", e);
                ArrayD::zeros(IxDyn(&[0]))
            }
        };

        Tensor::new(out, false)
    }
}

/// Helper: default f_min based on sample rate
fn f_min_default(sample_rate: usize) -> f32 {
    if sample_rate >= 16000 {
        0.0
    } else {
        125.0
    }
}

#[cfg(test)]
mod mel_spectrogram_tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_mel_spectrogram_basic() {
        let sample_rate = 16000;
        let n_fft = 400;
        let hop_length = 160;
        let n_mels = 80;

        let mel = MelSpectrogram::new(sample_rate, n_fft, hop_length, n_mels);

        // Create a simple sine wave
        let duration = 1.0; // 1 second
        let n_samples = sample_rate;
        let freq = 440.0; // A4 note
        let mut samples = Vec::with_capacity(n_samples);
        for t in 0..n_samples {
            let t_f = t as f32 / sample_rate as f32;
            samples.push((2.0 * std::f32::consts::PI * freq * t_f).sin() * 0.5);
        }

        let waveform = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[n_samples]), samples).unwrap(),
            false,
        );

        let spec = mel.forward(&waveform);
        let shape = spec.lock().storage.shape();
        assert_eq!(shape[0], n_mels);
        assert!(shape[1] > 0);
    }

    #[test]
    fn test_stft_forward() {
        let stft = STFT::new(400, 160);

        let n_samples = 1600;
        let samples: Vec<f32> = (0..n_samples).map(|i| (i as f32 * 0.01).sin()).collect();
        let waveform = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[n_samples]), samples).unwrap(),
            false,
        );

        let result = stft.forward(&waveform);
        let shape = result.lock().storage.shape();
        assert_eq!(shape[2], 2); // real + imag
        assert!(shape[0] > 0);
        assert!(shape[1] > 0);
    }

    #[test]
    fn test_stft_magnitude() {
        let stft = STFT::new(400, 160);

        let n_samples = 1600;
        let samples: Vec<f32> = (0..n_samples).map(|i| (i as f32 * 0.01).sin()).collect();
        let waveform = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[n_samples]), samples).unwrap(),
            false,
        );

        let stft_result = stft.forward(&waveform);
        let mag = stft.magnitude(&stft_result);
        let shape = mag.lock().storage.shape();
        assert_eq!(shape[0], 201); // n_fft/2 + 1
        assert!(shape[1] > 0);
    }

    #[test]
    fn test_istft_reconstruction() {
        let n_fft = 400;
        let hop_length = 160;

        let stft_proc = STFT::new(n_fft, hop_length);
        let istft_proc = ISTFT::new(n_fft, hop_length);

        let n_samples = 3200;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (i as f32 * 0.005).sin() * 0.3)
            .collect();
        let waveform = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[n_samples]), samples).unwrap(),
            false,
        );

        let stft_result = stft_proc.forward(&waveform);
        let reconstructed = istft_proc.forward(&stft_result);
        let recon_shape = reconstructed.lock().storage.shape();
        assert_eq!(recon_shape[0], n_samples);
    }
}
