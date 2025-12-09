use tensor_engine::nn::{AudioEncoder, AudioDecoder, Module, RVQ, MSELoss, Adam, Optimizer};
use tensor_engine::tensor::Tensor;
#[cfg(feature = "audio")]
use tensor_engine::io::dataloader::WavDataLoader;
#[cfg(feature = "audio")]
use std::env;

fn main() {
    println!("Audio codec training example (skeleton)");
    let enc = AudioEncoder::new(1, 8, 3); // hidden=8, 3 layers => channels grow
    let dec = AudioDecoder::new(8 * (1 << 2), 8, 3); // input channels equal enc last out
    let mut rvq = RVQ::new(8, 8, 2); // num_codes=8, dim=8, 2 levels
    // avoid dead codes by reinitializing empty codebook entries; adjust update frequency as needed
    rvq.set_reinit_empty_codes(true);
    rvq.set_ema_update_every(1);
    let mse = MSELoss::new();

    // Synthetic data: sine wave
    let len = 512usize;
    let mut data = Vec::with_capacity(len);
    for i in 0..len {
        let t = i as f32 / len as f32 * std::f32::consts::PI * 2.0 * 4.0; // 4 cycles
        data.push((t).sin() * 0.5);
    }
    let arr = ndarray::Array::from_shape_vec((1, 1, len), data).unwrap().into_dyn();
    let input = Tensor::new(arr, false);

    // gather parameters (RVQ codebooks are trainable and can be included)
    let mut params = enc.parameters();
    params.extend(dec.parameters());
    params.extend(rvq.parameters());
    let mut opt = Adam::new(1e-3, 0.9, 0.999, 1e-8);

    // attempt to load audio dataset if TRAIN_AUDIO_DIR is set (only when compiled with audio feature)
    #[cfg(feature = "audio")]
    let mut use_real_data = false;
    #[cfg(feature = "audio")]
    let mut loader: Option<WavDataLoader> = None;
    #[cfg(feature = "audio")]
    if let Ok(dir) = env::var("TRAIN_AUDIO_DIR") {
        match WavDataLoader::new(dir, 22050, 512, 4, true) {
            Ok(dl) => { loader = Some(dl); use_real_data = true; }
            Err(e) => log::warn!("Failed to initialize WavDataLoader: {}", e),
        }
    }

    for epoch in 0..5 {
        // If dataset available, loop over batches
            #[cfg(feature = "audio")]
            if use_real_data {
            let dl = loader.as_ref().unwrap();
            let num_batches = dl.num_batches();
            for bidx in 0..num_batches {
                let batch = dl.load_batch(bidx).expect("Failed to load batch");
                // stack batch to tensor: [B,1,L]
                let t = Tensor::stack(&batch, 0);
                // forward & rest same as synthetic
                let encoded = enc.forward(&t); // [B, C, L]
                let encoded_perm = encoded.permute(vec![0, 2, 1]); // [B, L, C]
                let enc_shape = encoded_perm.lock().storage.shape().to_vec();
                let shape = enc_shape.clone();
                let indices = rvq.quantize(&encoded_perm);
                let deq = rvq.dequantize(&indices, &shape).expect("Dequantize failed");
                let deq_perm = deq.permute(vec![0, 2, 1]); // back to NCL
                let decoded = dec.forward(&deq_perm);
                let loss = mse.forward(&decoded, &t);
                let commit = mse.forward(&encoded_perm, &deq);
                let total = loss.add(&commit.mul(&Tensor::new(ndarray::arr0(0.25).into_dyn(), false)));
                total.backward();
                opt.step(&params);
                opt.zero_grad(&params);
                rvq.update_ema(&encoded_perm, &indices, 0.999).unwrap();
                let total_arr = total.lock().storage.to_f32_array();
                let sum: f32 = total_arr.sum();
                let mean = sum / (total_arr.len() as f32);
                println!("Epoch {} batch {}: loss={}", epoch, bidx, mean);
            }
            continue;
        }

        // Synthetic data training
        // Forward
        let encoded = enc.forward(&input); // [1, C, L]
        let encoded_perm = encoded.permute(vec![0, 2, 1]); // [1, L, C]
        let enc_shape = encoded_perm.lock().storage.shape().to_vec();
        let shape = enc_shape.clone();
        let indices = rvq.quantize(&encoded_perm);
        let deq = rvq.dequantize(&indices, &shape).expect("Dequantize failed");
        let deq_perm = deq.permute(vec![0, 2, 1]); // back to NCL
        let decoded = dec.forward(&deq_perm);

        let loss = mse.forward(&decoded, &input);
        // commit loss: MSE between encoded (flattened) and dequantized
        let commit = mse.forward(&encoded_perm, &deq);
        let total = loss.add(&commit.mul(&Tensor::new(ndarray::arr0(0.25).into_dyn(), false)));
        // Backprop
        total.backward();
        // Optimizer step
        opt.step(&params);
        opt.zero_grad(&params);
        // Update codebooks using EMA based on assignments
        if let Err(e) = rvq.update_ema(&encoded_perm, &indices, 0.999) {
            log::error!("Failed to update RVQ EMA: {}", e);
        }

        let total_arr = total.lock().storage.to_f32_array();
        let sum: f32 = total_arr.sum();
        let mean = sum / (total_arr.len() as f32);
        let commit_arr = commit.lock().storage.to_f32_array();
        let commit_sum: f32 = commit_arr.sum();
        let commit_mean = commit_sum / (commit_arr.len() as f32);
        println!("Epoch {}: loss={}, commit={}", epoch, mean, commit_mean);
    }
    println!("Done training example.");
}
