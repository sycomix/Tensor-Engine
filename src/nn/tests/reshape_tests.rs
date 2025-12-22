use crate::nn::transformer::reshape_for_multihead;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn reshape_for_multihead_returns_err_on_size_mismatch() {
    // create a tensor with only 24 elements
    let t = Tensor::new(Array::from_shape_vec((2, 3, 4), vec![0.0f32; 24]).unwrap().into_dyn(), false);
    // request reshape that requires 48 elements (2*3*4*2)
    let res = reshape_for_multihead(&t, 2, 3, 4, 2);
    assert!(res.is_err(), "Expected reshape_for_multihead to return Err on size mismatch");
}
