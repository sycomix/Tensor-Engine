use tensor_engine::nn::RVQ;
use tensor_engine::tensor::Tensor;

#[test]
fn test_rvq_zero_codebook_quantize() {
    let rvq = RVQ::new(4, 2, 2);
    let arr = ndarray::Array::zeros(ndarray::IxDyn(&[2, 2]));
    let t = Tensor::new(arr.into_dyn(), false);
    let inds = rvq.quantize(&t);
    assert_eq!(inds.len(), 2);
    // Each level should return 2 indices (flattened input entries)
    assert_eq!(inds[0].len(), 2);
    assert_eq!(inds[1].len(), 2);
    assert!(inds[0].iter().all(|&x| x == 0));
    assert!(inds[1].iter().all(|&x| x == 0));
}
