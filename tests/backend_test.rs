use ndarray::Array2;
use tensor_engine::backend::{Backend, CudaBackend};
use tensor_engine::tensor::Tensor;

#[test]
fn test_cuda_backend_matmul_matches_ndarray_dot() {
    let a = Array2::from_shape_vec((2, 3), vec![1f32, 2., 3., 4., 5., 6.]).unwrap().into_dyn();
    let b = Array2::from_shape_vec((3, 2), vec![1f32, 2., 3., 4., 5., 6.]).unwrap().into_dyn();
    let ta = Tensor::new(a.clone(), false);
    let tb = Tensor::new(b.clone(), false);
    let cb = CudaBackend {};
    let result = cb.matmul(&ta, &tb);
    assert!(result.is_some());
    let r = result.unwrap();
    let expected = a.into_dimensionality::<ndarray::Ix2>().unwrap().dot(&b.into_dimensionality::<ndarray::Ix2>().unwrap());
    assert_eq!(r.shape(), vec![2usize, 2usize]);
    let r2 = r.into_dimensionality::<ndarray::Ix2>().unwrap();
    assert_eq!(r2, expected);
}
