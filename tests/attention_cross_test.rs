use tensor_engine::nn::transformer::{TransformerBlock, EncoderDecoderTransformer};
use tensor_engine::nn::Module;
use tensor_engine::tensor::Tensor;
use ndarray::Array;
use ndarray::IxDyn;

#[test]
fn test_cross_attention_forward_shape() {
    // Build small encoder-decoder
    let enc_block = TransformerBlock::new(8, 16, 2);
    let dec_block = TransformerBlock::new(8, 16, 2);
    let ed = EncoderDecoderTransformer::new(vec![enc_block], vec![dec_block]);
    let b = 1usize;
    let seq = 4usize;
    let d = 8usize;
    let inp = Tensor::new(Array::from_elem(IxDyn(&[b, seq, d]), 1.0f32), false);
    let out = ed.forward(&inp);
    let out_shape = out.lock().storage.shape();
    assert_eq!(out_shape, vec![b, seq, d]);
}
