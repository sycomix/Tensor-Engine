use tensor_engine::nn::{VisionTransformer, MultimodalLLM, AudioEncoder};
use tensor_engine::tensor::Tensor;
use ndarray::Array;

#[test]
fn test_audio_prefill_and_decode() {
    let _b = 1usize; let in_ch = 1usize; let l = 64usize; let vocab = 32usize;
    let d_model = 16usize; let d_ff = 32usize; let num_heads = 2usize; let depth = 1usize;
    let vit = VisionTransformer::new(3, 2, d_model, d_ff, num_heads, depth, 128);
    let mut model = MultimodalLLM::new(vit, vocab, d_model, d_ff, num_heads, depth);
    let audio_enc = AudioEncoder::new(in_ch, 8, 2);
    model.set_audio_encoder(audio_enc);
    // audio tensor: [1, 1, L]
    let audio = Tensor::new(Array::from_elem((1, in_ch, l), 0.5f32).into_dyn(), false);
    let mem = model.prefill_audio(&audio, None).expect("prefill_audio should succeed");
    assert!(mem.prefill_image_tokens > 0);
    let new_id = Tensor::new(Array::from_elem((1,1), 2.0f32).into_dyn(), false);
    let (logits, _new_mem) = model.decode_step(&mem, &new_id).expect("decode_step should succeed");
    let shape = logits.lock().storage.shape();
    assert_eq!(shape.len(), 3);
}
