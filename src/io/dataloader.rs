use crate::io::audio::{load_wav_to_tensor};
use crate::tensor::Tensor;
use std::path::{Path, PathBuf};
use std::fs;
use log::info;
#[cfg(feature = "rubato")]
use rubato::{FftFixedIn, Resampler};
#[cfg(feature = "rubato")]
use num_integer::gcd;
/// Simple linear resampling. Not high-quality but useful for example/training.
pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate {
        return samples.to_vec();
    }
    let src_len = samples.len();
    if src_len == 0 {
        return vec![];
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len_f = (src_len as f64) * ratio;
    let out_len = (out_len_f.round() as usize).max(1);
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) / ratio;
        let left = src_pos.floor() as usize;
        let right = if left + 1 < src_len { left + 1 } else { left };
        let frac = src_pos - left as f64;
        let lval = samples[left];
        let rval = samples[right];
        let v = (1.0 - frac as f32) * lval + (frac as f32) * rval;
        out.push(v);
    }
    out
}

/// High quality resampling using `rubato` when available. Falls back to resample_linear if the feature isn't enabled.
#[cfg(feature = "rubato")]
pub fn resample_high_quality(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    // rubato expects f32 slices and operates in frames; for mono audio we can use single channel.
    // SincFixedIn expects inputs as Vec<Vec<f32>> (one Vec per channel), and outputs similarly.
    let src_rate_f = src_rate as f64;
    let dst_rate_f = dst_rate as f64;
    if src_rate_f == dst_rate_f { return samples.to_vec(); }
    // Use FFT-based fixed resampler for high quality.
    // compute gcd to reduce rate ratio and make chunk_size multiples of input rate units
    let g = gcd(src_rate as usize, dst_rate as usize);
    let in_rate_unit = (src_rate as usize) / g;
    let out_rate_unit = (dst_rate as usize) / g;
    let raw_chunk = 1024usize.min(samples.len().max(1));
    let chunk_size = if raw_chunk >= in_rate_unit { raw_chunk - (raw_chunk % in_rate_unit) } else { in_rate_unit };
    info!("resample_high_quality: src_rate={} dst_rate={} gcd={} in_unit={} out_unit={} chunk_size={}", src_rate, dst_rate, g, in_rate_unit, out_rate_unit, chunk_size);
    let sub_chunks = 1usize;
    let mut resampler = match FftFixedIn::<f32>::new(src_rate as usize, dst_rate as usize, chunk_size, sub_chunks, 1) {
        Ok(r) => r,
        Err(e) => { log::warn!("resample_high_quality: rubato resampler init failed: {} - falling back to linear resampling", e); return resample_linear(samples, src_rate, dst_rate); }
    };
    let in_frames: Vec<Vec<f32>> = vec![samples.to_vec()];
    let out_frames = match resampler.process(&in_frames, None) {
        Ok(f) => f,
        Err(e) => { log::warn!("resample_high_quality: rubato process failed: {} - falling back to linear resampling", e); return resample_linear(samples, src_rate, dst_rate); }
    };
    // flatten first channel
    out_frames[0].clone()
}

#[cfg(not(feature = "rubato"))]
pub fn resample_high_quality(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    // Run the fast linear resampler as fallback.
    resample_linear(samples, src_rate, dst_rate)
}
use ndarray::Ix3;

pub struct WavDataLoader {
    files: Vec<PathBuf>,
    pub sample_rate: u32,
    pub chunk_len: usize,
    pub batch_size: usize,
    pub resample: bool,
}

impl WavDataLoader {
    pub fn new<P: AsRef<Path>>(dir: P, sample_rate: u32, chunk_len: usize, batch_size: usize, resample: bool) -> Result<Self, String> {
        let mut files = Vec::new();
        let dirp = dir.as_ref();
        if !dirp.exists() {
            return Err(format!("Directory not found: {}", dirp.display()));
        }
        for entry in fs::read_dir(dirp).map_err(|e| format!("{}", e))? {
            let path = entry.map_err(|e| format!("{}", e))?.path();
            if let Some(ext) = path.extension() {
                if ext == "wav" || ext == "WAV" {
                    files.push(path);
                }
            }
        }
        if files.is_empty() {
            return Err(format!("No wav files found in {}", dirp.display()));
        }
        info!("WavDataLoader created: dir={} files_found={} sample_rate={} chunk_len={} batch_size={} resample={}", dirp.display(), files.len(), sample_rate, chunk_len, batch_size, resample);
        Ok(WavDataLoader { files, sample_rate, chunk_len, batch_size, resample })
    }

    pub fn num_batches(&self) -> usize {
        (self.files.len() + self.batch_size - 1) / self.batch_size
    }

    /// Load a batch by batch index. Pads or truncates audio to chunk_len.
    pub fn load_batch(&self, batch_idx: usize) -> Result<Vec<Tensor>, String> {
        let start = batch_idx * self.batch_size;
        let mut out = Vec::new();
        info!("WavDataLoader loading batch={} start_idx={} batch_size={} total_files={}", batch_idx, start, self.batch_size, self.files.len());
        for i in 0..self.batch_size {
            let idx = start + i;
            if idx >= self.files.len() { break; }
            let p = &self.files[idx];
            let p_str = match p.to_str() { Some(s) => s, None => return Err(format!("Invalid path string: {}", p.display())) };
            let (t, rate) = load_wav_to_tensor(p_str)?;
            let mut arr_owned = t.lock().storage.to_f32_array().into_dyn();
            if rate != self.sample_rate {
                if self.resample {
                    // perform resampling (rubato high-quality if available, otherwise linear fallback)
                    let flat = match arr_owned.into_dimensionality::<Ix3>() {
                        Ok(f) => f,
                        Err(e) => { log::error!("WavDataLoader::load_batch: expected 3D array for audio but conversion failed: {}", e); return Err(format!("WavDataLoader::load_batch: expected 3D array")); }
                    };
                    // flatten channel & batch dims: assume [1,1,L]
                    let mut samples = Vec::new();
                    for j in 0..flat.shape()[2] {
                        samples.push(flat[[0,0,j]]);
                    }
                    let resampled = resample_high_quality(&samples, rate, self.sample_rate);
                    let mut flat2 = vec![0.0f32; 1 * 1 * resampled.len()];
                    for (j, v) in resampled.iter().enumerate() { flat2[j] = *v; }
                    arr_owned = match ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1,1,resampled.len()]), flat2) {
                        Ok(a) => a.into_dyn(),
                        Err(e) => { log::error!("WavDataLoader::load_batch: failed to construct array for resampled audio: {}", e); return Err(format!("WavDataLoader: failed to create resampled array: {}", e)); }
                    };
                } else {
                    return Err(format!("Sample rate mismatch: expected {} got {} for file: {}", self.sample_rate, rate, p.display()));
                }
            }
            // Ensure shape [1,1,L]
            let arr = arr_owned;
            let len = match arr.shape().last() { Some(&l) => l, None => { log::error!("WavDataLoader::load_batch: audio tensor shape missing length dimension"); return Err("WavDataLoader::load_batch: audio tensor shape missing length".to_string()); } };
            let len_usize = len as usize;
            if len_usize > self.chunk_len {
                // trim center
                let start_idx = (len_usize - self.chunk_len) / 2;
                let slice = arr.slice(ndarray::s![0..1, 0..1, start_idx..start_idx + self.chunk_len]).to_owned();
                out.push(Tensor::new(slice.into_dyn(), false));
            } else if len_usize < self.chunk_len {
                // zero pad
                let mut flat = vec![0.0f32; 1 * 1 * self.chunk_len];
                for j in 0..len_usize {
                    flat[j] = arr[[0,0,j]];
                }
                let arr2 = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1,1,self.chunk_len]), flat).map_err(|e| e.to_string())?;
                out.push(Tensor::new(arr2.into_dyn(), false));
            } else {
                out.push(t.clone());
            }
        }
        info!("WavDataLoader loaded {} tensors for batch {}", out.len(), batch_idx);
        Ok(out)
    }
} 
