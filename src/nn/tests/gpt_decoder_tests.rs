#![cfg(test)]

use crate::nn::GPTDecoder;
use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn gpt_decoder_forward_shape() {
    let vocab_size = 64;
    let d_model = 16;
    let num_layers = 2;
    let d_ff = 32;
    let num_heads = 4;
    let max_seq_len = 12;

    let model = GPTDecoder::new(vocab_size, d_model, num_layers, d_ff, num_heads, max_seq_len)
        .expect("create GPTDecoder");

    let b = 2;
    let seq = 7;
    let ids: Vec<f32> = vec![1.0; b * seq];
    let input = Tensor::new(
        Array::from_shape_vec((b, seq), ids).unwrap().into_dyn(),
        false,
    );

    let out = model.forward(&input);
    assert_eq!(out.lock().storage.shape(), &[b, seq, vocab_size]);
}

#[test]
fn gpt_decoder_rejects_too_long_sequence() {
    let vocab_size = 64;
    let d_model = 16;
    let num_layers = 1;
    let d_ff = 32;
    let num_heads = 4;
    let max_seq_len = 4;

    let model = GPTDecoder::new(vocab_size, d_model, num_layers, d_ff, num_heads, max_seq_len)
        .expect("create GPTDecoder");

    let b = 1;
    let seq = 6;
    let input = Tensor::new(
        Array::from_shape_vec((b, seq), vec![1.0; b * seq])
            .unwrap()
            .into_dyn(),
        false,
    );

    let out = model.forward(&input);
    assert_eq!(out.lock().storage.shape(), &[0]);
}
