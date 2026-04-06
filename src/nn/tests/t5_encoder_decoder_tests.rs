#![cfg(test)]

use crate::nn::Module;
use crate::nn::T5EncoderDecoder;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn t5_encoder_decoder_forward_seq2seq_shape() {
    let model = T5EncoderDecoder::new(128, 16, 2, 32, 4, 2).expect("create T5EncoderDecoder");

    let b = 2;
    let enc_seq = 6;
    let dec_seq = 4;
    let enc_ids = Tensor::new(
        Array::from_shape_vec((b, enc_seq), vec![1.0; b * enc_seq])
            .unwrap()
            .into_dyn(),
        false,
    );
    let dec_ids = Tensor::new(
        Array::from_shape_vec((b, dec_seq), vec![2.0; b * dec_seq])
            .unwrap()
            .into_dyn(),
        false,
    );

    let out = model.forward_seq2seq(&enc_ids, &dec_ids, None, None, None);
    assert_eq!(out.lock().storage.shape(), &[b, dec_seq, 128]);
}

#[test]
fn t5_encoder_decoder_module_forward_compat_shape() {
    let model = T5EncoderDecoder::new(64, 12, 1, 24, 3, 3).expect("create T5EncoderDecoder");
    let b = 1;
    let seq = 5;
    let ids = Tensor::new(
        Array::from_shape_vec((b, seq), vec![3.0; b * seq])
            .unwrap()
            .into_dyn(),
        false,
    );

    let out = model.forward(&ids);
    assert_eq!(out.lock().storage.shape(), &[b, seq, 64]);
}

#[test]
fn t5_encoder_decoder_rejects_invalid_head_ratio() {
    let res = T5EncoderDecoder::new(64, 12, 1, 24, 6, 4);
    assert!(res.is_err(), "expected constructor error for invalid head ratio");
}
