fn main() {
    println!("Text to audio example (skeleton)");
    #[cfg(feature = "audio")]
    {
        println!("Audio feature enabled: hound available");
        use tensor_engine::nn::{AudioDecoder, RVQ, Module};
        use tensor_engine::io::audio::write_wav_from_tensor;

        // Create decoder & RVQ
        let dec = AudioDecoder::new(8 * (1 << 2), 8, 3);
        let rvq = RVQ::new(8, 8, 2);
        // Simple random codebook indices for a short sequence
        let l = 128usize;
        let mut indices: Vec<Vec<usize>> = Vec::new();
        for _ in 0..rvq.levels {
            let mut v = Vec::new();
            for _ in 0..(1 * l) {
                v.push(0usize); // choose code 0 for deterministic output
            }
            indices.push(v);
        }
        // dequantize: shape [1, L, C]
        let shape = vec![1usize, l, rvq.dim];
        let deq = rvq.dequantize(&indices, &shape).unwrap();
        // permute to [1,C,L]
        let deq_perm = deq.permute(vec![0, 2, 1]);
        let decoded = dec.forward(&deq_perm);
        // write to wav
        match write_wav_from_tensor(&decoded, "out.wav", 22050) {
            Ok(_) => println!("WAV written to out.wav"),
            Err(e) => println!("Error writing WAV: {}", e),
        }
    }
    #[cfg(not(feature = "audio"))]
    {
        println!("Audio feature not enabled. Build with --features audio to enable hound.");
    }
}
