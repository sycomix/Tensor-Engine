use ndarray::Array2;
use tensor_engine::tensor::Tensor;

fn approx_eq(a: &ndarray::ArrayD<f32>, b: &ndarray::ArrayD<f32>, tol: f32) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    let aslice = a.as_slice().unwrap();
    let bslice = b.as_slice().unwrap();
    for (x, y) in aslice.iter().zip(bslice.iter()) {
        if (x - y).abs() > tol {
            return false;
        }
    }
    true
}

#[test]
fn test_blas_matmul_matches_ndarray_for_various_shapes() {
    // Test several shapes and contiguity patterns
    let shapes = vec![(1usize, 1usize, 1usize), (2, 2, 2), (3, 2, 4)];
    for (m, k, n) in shapes {
        let mut a_vals = Vec::new();
        let mut b_vals = Vec::new();
        for i in 0..(m * k) {
            a_vals.push((i as f32) + 1.0);
        }
        for i in 0..(k * n) {
            b_vals.push((i as f32) + 1.0);
        }
        let a_arr = Array2::from_shape_vec((m, k), a_vals)
            .unwrap()
            .into_dyn();
        let b_arr = Array2::from_shape_vec((k, n), b_vals)
            .unwrap()
            .into_dyn();
        let a_t = Tensor::new(a_arr.clone(), false);
        let b_t = Tensor::new(b_arr.clone(), false);
        let c = a_t.matmul(&b_t);
        let expected = a_arr.into_dimensionality::<ndarray::Ix2>().unwrap().dot(
            &b_arr.into_dimensionality::<ndarray::Ix2>().unwrap(),
        ).into_dyn();
        assert!(approx_eq(&c.lock().data, &expected, 1e-5));

        // Non-contiguous inputs are not explicitly tested here; the main goal is numeric parity.
    }
}
