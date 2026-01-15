use ndarray::ArrayD;
use ndarray::IxDyn;
use tensor_engine::ops::EmbeddingLookup;
use tensor_engine::ops::Operation;
use tensor_engine::tensor::Tensor;

#[test]
fn test_embedding_out_of_bounds_is_handled() {
    // embedding vocab 10, dim 4
    let emb = Tensor::new(ndarray::Array::zeros(IxDyn(&[10, 4])), false);
    // index equal to vocab (out of bounds)
    let idx = Tensor::new(ndarray::arr0(10.0f32).into_dyn(), false);
    let op = EmbeddingLookup::new();
    let mut out = ArrayD::<f32>::zeros(IxDyn(&[1, 4]));
    op.forward(&[emb, idx], &mut out);
    // output should remain zeros due to out-of-bounds
    assert!(out.iter().all(|&v| v == 0.0f32));
}
