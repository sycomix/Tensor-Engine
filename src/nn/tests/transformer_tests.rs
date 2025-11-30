use crate::nn::TransformerBlock;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn transformer_block_forward_shape() {
    let b = 2;
    let seq = 4;
    let d_model = 8;
    let d_ff = 16;
    let num_heads = 2;
    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32 * 0.01).collect();
    let x = Tensor::new(Array::from_shape_vec((b, seq, d_model), x_data).unwrap().into_dyn(), true);
    let block = TransformerBlock::new(d_model, d_ff, num_heads);
    let out = block.forward_block(&x);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);
}
