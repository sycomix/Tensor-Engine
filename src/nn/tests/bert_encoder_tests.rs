#![cfg(test)]

use crate::nn::BERTEncoder;
use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn bert_encoder_forward_shape() {
    let vocab_size = 128;
    let d_model = 16;
    let num_layers = 2;
    let d_ff = 32;
    let num_heads = 4;
    let max_seq_len = 12;

    let model = BERTEncoder::new(vocab_size, d_model, num_layers, d_ff, num_heads, max_seq_len)
        .expect("create BERTEncoder");

    let b = 2;
    let seq = 7;
    let input = Tensor::new(
        Array::from_shape_vec((b, seq), vec![1.0; b * seq])
            .unwrap()
            .into_dyn(),
        false,
    );

    let out = model.forward(&input);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);
}

#[test]
fn bert_encoder_pooled_output_shape() {
    let model = BERTEncoder::new(64, 12, 1, 24, 3, 8).expect("create BERTEncoder");

    let b = 3;
    let seq = 5;
    let input = Tensor::new(
        Array::from_shape_vec((b, seq), vec![2.0; b * seq])
            .unwrap()
            .into_dyn(),
        false,
    );

    let encoded = model.forward(&input);
    let pooled = model.pooled_output(&encoded);
    assert_eq!(pooled.lock().storage.shape(), &[b, 12]);
}

#[test]
fn bert_encoder_rejects_too_long_sequence() {
    let model = BERTEncoder::new(64, 12, 1, 24, 3, 4).expect("create BERTEncoder");

    let input = Tensor::new(
        Array::from_shape_vec((1, 6), vec![1.0; 6]).unwrap().into_dyn(),
        false,
    );

    let out = model.forward(&input);
    assert_eq!(out.lock().storage.shape(), &[0]);
}
