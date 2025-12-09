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
    // numerical check: out should equal 3.0 for all entries in this simple case
    let out_arr = out.lock().storage.to_f32_array();
    for v in out_arr.iter() {
        assert!((*v - 3.0).abs() < 1e-5);
    }
}

#[test]
fn test_quantized_matmul_rowwise() {
    let a = Tensor::new(Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32; 6]).unwrap(), false);
    let w = Tensor::new(Array::from_shape_vec(IxDyn(&[3, 4]), vec![1.0f32; 12]).unwrap(), false);
    let qw = w.quantize_weights(tensor_engine::dtype::DType::I8Rowwise, None).expect("quantize weights rowwise");
    let out = a.quantized_matmul(&qw);
    let shap = out.lock().storage.shape();
    assert_eq!(shap, vec![2, 4]);
    // verify numerical equality with matmul on dequantized version
    let deq = qw.lock().storage.to_f32_array();
    let plain = a.matmul(&Tensor::new(deq, false));
    assert_eq!(plain.lock().storage.shape(), shap);
    let out_arr = out.lock().storage.to_f32_array();
    let plain_arr = plain.lock().storage.to_f32_array();
    assert_eq!(out_arr.shape(), plain_arr.shape());
    for (x, y) in out_arr.iter().zip(plain_arr.iter()) {
        assert!((x - y).abs() < 1e-4);
    }
}

#[test]
fn test_quantized_matmul_blockwise() {
    let a = Tensor::new(Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32; 6]).unwrap(), false);
    let w = Tensor::new(Array::from_shape_vec(IxDyn(&[3, 4]), vec![1.0f32; 12]).unwrap(), false);
    let qw = w.quantize_weights(tensor_engine::dtype::DType::I8Blockwise, Some(2)).expect("quantize weights blockwise");
    let out = a.quantized_matmul(&qw);
    let shap = out.lock().storage.shape();
    assert_eq!(shap, vec![2, 4]);
    let deq = qw.lock().storage.to_f32_array();
    let plain = a.matmul(&Tensor::new(deq, false));
    assert_eq!(plain.lock().storage.shape(), shap);
    let out_arr = out.lock().storage.to_f32_array();
    let plain_arr = plain.lock().storage.to_f32_array();
    assert_eq!(out_arr.shape(), plain_arr.shape());
    for (x, y) in out_arr.iter().zip(plain_arr.iter()) {
        assert!((x - y).abs() < 1e-4);
    }
}
