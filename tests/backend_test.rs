#![cfg(feature = "backend_cuda")]
use ndarray::Array2;
use tensor_engine::backend::{Backend, CpuBackend};

#[test]
fn test_cpu_backend_matmul_matches_ndarray_dot() {
    let a = Array2::from_shape_vec((2, 3), vec![1f32, 2., 3., 4., 5., 6.]).unwrap();
    let b = Array2::from_shape_vec((3, 2), vec![1f32, 2., 3., 4., 5., 6.]).unwrap();
    let expected = a.dot(&b);
    let cb = CpuBackend::default();
    let result = cb.matmul(&a.into_dyn(), &b.into_dyn());
    assert!(result.is_some());
    let r = result.unwrap();
    assert_eq!(r.shape(), vec![2usize, 2usize]);
    let r2 = r.into_dimensionality::<ndarray::Ix2>().unwrap();
    assert_eq!(r2, expected);
}
