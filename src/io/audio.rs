#[cfg(feature = "audio")]
use crate::tensor::Tensor;
#[cfg(feature = "audio")]
use hound;
#[cfg(feature = "audio")]
use ndarray::{ArrayD, IxDyn};
#[cfg(feature = "audio")]
use std::f32::consts::PI;

#[cfg(feature = "audio")]
pub fn load_wav_to_tensor(path: &str) -> Result<(Tensor, u32), String> {
    let reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let rate = spec.sample_rate;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0f32))
            .collect(),
        hound::SampleFormat::Int => {
            // convert to f32 based on bits per sample
            let bits = spec.bits_per_sample;
            if bits == 16 {
                reader
                    .into_samples::<i16>()
                    .map(|s| s.unwrap_or(0) as f32 / i16::MAX as f32)
                    .collect()
            } else if bits == 24 {
                reader
                    .into_samples::<i32>()
                    .map(|s| s.unwrap_or(0) as f32 / (2i32.pow(23) as f32))
                    .collect()
            } else {
                return Err(format!("Unsupported bits per sample: {}", bits));
            }
        }
    };

    // For simplicity, assume mono for now; if multi-channel, take first channel
    // If channels >1, samples are interleaved
    let channels = spec.channels as usize;
    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|chunk| chunk[0])
            .collect::<Vec<f32>>()
    };

    let len = mono.len();
    let mut flat = Vec::with_capacity(len);
    for s in mono.iter() {
        flat.push(*s);
    }
    let arr = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1, 1, len]), flat)
        .map_err(|e| e.to_string())?;
    Ok((Tensor::new(arr.into_dyn(), false), rate))
}

#[cfg(feature = "audio")]
pub fn write_wav_from_tensor(t: &Tensor, path: &str, sample_rate: u32) -> Result<(), String> {
    let arr = t.lock().storage.to_f32_array();
    // Expect shape [1,1,T] or [T]
    let shape = arr.shape().to_vec();
    let data: Vec<f32> = if shape.len() == 3 {
        // N,C,L
        arr.into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| e.to_string())?
            .iter()
            .cloned()
            .collect()
    } else if shape.len() == 1 {
        arr.into_dimensionality::<ndarray::Ix1>()
            .map_err(|e| e.to_string())?
            .iter()
            .cloned()
            .collect()
    } else {
        return Err(format!("Unsupported tensor shape for wav: {:?}", shape));
    };

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
    for &v in data.iter() {
        let s = (v * i16::MAX as f32) as i16;
        writer.write_sample(s).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    Ok(())
}

/// Mel-spectrogram configuration matching torchaudio's defaults.
#[derive(Clone)]
pub struct MelSpectrogramConfig {
    pub sample_rate: u32,
    pub window_length: usize,
    pub hop_length: usize,
    pub mel_bins: usize,
    pub f_min: f32,
    pub f_max: Option<f32>,
    pub normalize: bool,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        MelSpectrogramConfig {
            sample_rate: 16000,
            window_length: 400, // 25ms at 16kHz
            hop_length: 160,    // 10ms at 16kHz
            mel_bins: 80,
            f_min: 0.0,
            f_max: None,
            normalize: false,
        }
    }
}

/// Compute mel-spectrogram from a waveform tensor.
/// Input shape: [1, 1, T] or [T]
/// Output shape: [mel_bins, num_frames]
pub fn mel_spectrogram(waveform: &Tensor, config: &MelSpectrogramConfig) -> Result<Tensor, String> {
    let arr = waveform.lock().storage.to_f32_array();
    let samples: Vec<f32> = if arr.ndim() == 3 {
        arr.into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| format!("mel_spectrogram: invalid shape: {}", e))?
            .iter()
            .cloned()
            .collect()
    } else if arr.ndim() == 1 {
        arr.iter().cloned().collect()
    } else {
        return Err(format!(
            "mel_spectrogram: expected 1D or 3D input, got {}D",
            arr.ndim()
        ));
    };

    let n_fft = config.window_length;
    let hop = config.hop_length;
    let num_frames = ((samples.len() - n_fft) / hop) + 1;
    if num_frames <= 0 {
        return Err("mel_spectrogram: signal too short for window size".to_string());
    }

    let f_max = config.f_max.unwrap_or(config.sample_rate as f32 / 2.0);
    let mel_matrix = compute_mel_filterbank(n_fft, config.f_min, f_max, config.mel_bins, config.sample_rate)?;

    // Compute STFT
    let stft = compute_stft(&samples, n_fft, hop)?;
    // stft shape: [n_fft/2+1, num_frames, 2] (real, imag)

    // Compute magnitude spectrogram
    let mut magnitude = ArrayD::<f32>::zeros(IxDyn(&[n_fft / 2 + 1, num_frames]));
    for f in 0..n_fft / 2 + 1 {
        for t in 0..num_frames {
            let re = stft[[f, t, 0]];
            let im = stft[[f, t, 1]];
            magnitude[[f, t]] = (re * re + im * im).sqrt();
        }
    }

    // Apply mel filterbank: mel @ magnitude
    let mel_bins = config.mel_bins;
    let mut mel_spec = ArrayD::<f32>::zeros(IxDyn(&[mel_bins, num_frames]));
    for m in 0..mel_bins {
        for t in 0..num_frames {
            let mut val = 0.0f32;
            for f in 0..n_fft / 2 + 1 {
                val += mel_matrix[m][f] * magnitude[[f, t]];
            }
            // Log scale: log(1 + mel)
            mel_spec[[m, t]] = (1.0 + val).ln();
        }
    }

    // Normalize if requested
    if config.normalize {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for v in mel_spec.iter() {
            min = min.min(*v);
            max = max.max(*v);
        }
        let range = max - min;
        if range > 1e-8 {
            for v in mel_spec.iter_mut() {
                *v = ((*v - min) / range) * 2.0 - 1.0;
            }
        }
    }

    Ok(Tensor::new(mel_spec.into_dyn(), false))
}

/// Compute mel filterbank matrix.
/// Returns shape [mel_bins, n_fft/2+1]
fn compute_mel_filterbank(
    n_fft: usize,
    f_min: f32,
    f_max: f32,
    mel_bins: usize,
    sample_rate: u32,
) -> Result<Vec<Vec<f32>>, String> {
    let n_fft_plus_1 = n_fft / 2 + 1;
    let fft_freqs: Vec<f32> = (0..n_fft_plus_1)
        .map(|i| (i as f32) * sample_rate as f32 / (n_fft as f32))
        .collect();

    // Convert frequencies to mel scale
    let f_min_mel = 2595.0 * (1.0 + (f_min / 700.0).log10());
    let f_max_mel = 2595.0 * (1.0 + (f_max / 700.0).log10());

    // Compute mel points
    let mel_points: Vec<f32> = (0..mel_bins + 2)
        .map(|i| {
            f_min_mel + (i as f32) * (f_max_mel - f_min_mel) / (mel_bins + 1) as f32
        })
        .collect();

    // Convert mel points back to Hz
    let mel_to_hz: Vec<f32> = mel_points
        .iter()
        .map(|m| 700.0 * (1.0 + m / 2595.0).ln() - 1.0)
        .collect();

    // Compute filterbank
    let mut filterbank = vec![vec![0.0f32; n_fft_plus_1]; mel_bins];
    for m in 0..mel_bins {
        let f0 = mel_to_hz[m];
        let f1 = mel_to_hz[m + 1];
        let f2 = mel_to_hz[m + 2];

        for (i, &fft_f) in fft_freqs.iter().enumerate() {
            if f0 <= fft_f && fft_f < f1 {
                filterbank[m][i] = (fft_f - f0) / (f1 - f0);
            } else if f1 <= fft_f && fft_f < f2 {
                filterbank[m][i] = (f2 - fft_f) / (f2 - f1);
            }
        }
    }

    Ok(filterbank)
}

/// Compute Short-Time Fourier Transform (STFT).
/// Input: waveform [T]
/// Output: [n_fft/2+1, num_frames, 2] where last axis is [real, imag]
pub fn stft(
    waveform: &[f32],
    n_fft: usize,
    hop_length: usize,
    window_fn: Option<fn(usize) -> Vec<f32>>,
) -> Result<ArrayD<f32>, String> {
    let num_frames = ((waveform.len() - n_fft) / hop_length) + 1;
    if num_frames <= 0 {
        return Err("stft: signal too short for window size".to_string());
    }

    // Apply window
    let window = window_fn.unwrap_or(|n| hann_window(n))[n_fft];

    let mut out = ArrayD::<f32>::zeros(IxDyn(&[n_fft / 2 + 1, num_frames, 2]));

    for t in 0..num_frames {
        let start = t * hop_length;
        if start + n_fft > waveform.len() {
            break;
        }

        // Apply window and compute FFT for this frame
        let frame: Vec<f32> = (0..n_fft)
            .map(|i| waveform[start + i] * window[i])
            .collect();

        // Compute DFT (only positive frequencies)
        for k in 0..n_fft / 2 + 1 {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            let theta = 2.0 * PI * (k as f32) / (n_fft as f32);
            for n in 0..n_fft {
                let angle = theta * (n as f32);
                re += frame[n] * angle.cos();
                im -= frame[n] * angle.sin();
            }
            // Normalize by window energy
            let norm = (n_fft as f32).sqrt();
            out[[k, t, 0]] = re / norm;
            out[[k, t, 1]] = im / norm;
        }
    }

    Ok(out)
}

/// Compute inverse STFT (overlap-add).
/// Input: stft [n_fft/2+1, num_frames, 2]
/// Output: waveform [T_out]
pub fn istft(
    stft: &ArrayD<f32>,
    n_fft: usize,
    hop_length: usize,
    window_fn: Option<fn(usize) -> Vec<f32>>,
) -> Result<Vec<f32>, String> {
    let num_frames = stft.shape()[1];
    let window = window_fn.unwrap_or(|n| hann_window(n));
    let win_len = window.len();

    // Estimate output length
    let out_len = (num_frames - 1) * hop_length + n_fft;
    let mut waveform = vec![0.0f32; out_len];
    let mut window_sum = vec![0.0f32; out_len];

    for t in 0..num_frames {
        let start = t * hop_length;
        if start + win_len > out_len {
            break;
        }

        // Compute inverse DFT for this frame
        let mut frame = vec![0.0f32; n_fft];
        for n in 0..n_fft {
            let mut val = 0.0f32;
            for k in 0..n_fft / 2 + 1 {
                let re = stft[[k, t, 0]];
                let im = stft[[k, t, 1]];
                let angle = 2.0 * PI * (k as f32) * (n as f32) / (n_fft as f32);
                val += re * angle.cos() - im * angle.sin();
            }
            frame[n] = val / (n_fft as f32);
        }

        // Overlap-add with window
        for i in 0..win_len {
            if start + i < out_len {
                waveform[start + i] += frame[i] * window[i];
                window_sum[start + i] += window[i] * window[i];
            }
        }
    }

    // Normalize by window sum
    for i in 0..out_len {
        if window_sum[i] > 1e-8 {
            waveform[i] /= window_sum[i];
        }
    }

    Ok(waveform)
}

/// Compute Hann window of given length.
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f32) / (n as f32 - 1.0)).cos())).collect()
}

/// Compute spectrogram magnitude from waveform.
/// Convenience wrapper around stft that returns magnitude only.
pub fn spectrogram(
    waveform: &[f32],
    n_fft: usize,
    hop_length: usize,
) -> Result<ArrayD<f32>, String> {
    let stft_out = stft(waveform, n_fft, hop_length, None)?;
    let (freq_bins, num_frames, _) = (stft_out.shape()[0], stft_out.shape()[1], stft_out.shape()[2]);
    let mut magnitude = ArrayD::<f32>::zeros(IxDyn(&[freq_bins, num_frames]));
    for f in 0..freq_bins {
        for t in 0..num_frames {
            let re = stft_out[[f, t, 0]];
            let im = stft_out[[f, t, 1]];
            magnitude[[f, t]] = (re * re + im * im).sqrt();
        }
    }
    Ok(magnitude)
}
