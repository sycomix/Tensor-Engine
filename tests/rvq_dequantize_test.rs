use ndarray::Array;
use tensor_engine::nn::RVQ;
use tensor_engine::tensor::Tensor;

#[test]
fn test_rvq_quantize_dequantize_roundtrip() {
    let rvq = RVQ::new(8, 4, 2);
    // Create simple input: 3 vectors of dim 4
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let arr = Array::from_shape_vec((3, 4), data).unwrap().into_dyn();
    let t = Tensor::new(arr, false);

    let indices = rvq.quantize(&t);
    assert_eq!(indices.len(), 2);
    assert_eq!(indices[0].len(), 3);

    let deq = rvq.dequantize(&indices, &[3, 4]).expect("dequantize should succeed");
    let deq_shape = deq.lock().storage.to_f32_array().shape().to_vec();
    assert_eq!(deq_shape, vec![3, 4]);
}
