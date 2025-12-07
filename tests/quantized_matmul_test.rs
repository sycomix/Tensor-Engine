use tensor_engine::tensor::Tensor;
use tensor_engine::dtype::DType;
use ndarray::{Array, IxDyn};

#[test]
fn test_quantized_matmul_basic() {
    // Input 2x3 times 3x4 => 2x4
    let a = Tensor::new(Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32; 6]).unwrap(), false);
    let w = Tensor::new_with_dtype(Array::from_shape_vec(IxDyn(&[3, 4]), vec![1.0f32; 12]).unwrap(), false, DType::I8);
    let out = a.quantized_matmul(&w);
    let shap = out.lock().storage.shape();
    assert_eq!(shap, vec![2, 4]);
}
