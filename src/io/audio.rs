#[cfg(feature = "audio")]
use hound;
#[cfg(feature = "audio")]
use crate::tensor::Tensor;

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
    let mut flat = Vec::with_capacity((1 * 1 * len) as usize);
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
