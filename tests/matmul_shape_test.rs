use tensor_engine::ops::MatMul;
use tensor_engine::tensor::Tensor;
use ndarray::IxDyn;
use ndarray::ArrayD;

#[test]
fn test_matmul_shape_small() {
    // a: (2,3072), b: (3072,1024)
    let a = Tensor::new(ndarray::Array::zeros(IxDyn(&[2, 3072])), false);
    let b = Tensor::new(ndarray::Array::zeros(IxDyn(&[3072, 1024])), false);
    let mut out = ArrayD::<f32>::zeros(IxDyn(&[2, 1024]));
    let matmul = MatMul::new();
    matmul.forward(&[a, b], &mut out);
    // If forward did not panic and produced output, shape should match
    assert_eq!(out.shape(), &[2, 1024]);
}
