#[cfg(all(feature = "backend_metal", target_os = "macos"))]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use tensor_engine::backend::metal::MetalBackend;
    use tensor_engine::backend::Backend;
    use tensor_engine::tensor::Tensor;

    fn assert_approx_eq(a: &ArrayD<f32>, b: &ArrayD<f32>, epsilon: f32) {
        assert_eq!(a.shape(), b.shape());
        let a_iter = a.iter();
        let b_iter = b.iter();
        for (v1, v2) in a_iter.zip(b_iter) {
            let diff = (v1 - v2).abs();
            if diff > epsilon {
                panic!("Mismatch: {} != {} (diff {})", v1, v2, diff);
            }
        }
    }

    #[test]
    fn test_metal_initialization() {
        let backend = MetalBackend::new();
        assert!(
            backend.is_ok(),
            "Failed to initialize Metal backend: {:?}",
            backend.err()
        );
    }

    #[test]
    fn test_metal_matmul_naive() {
        let backend = MetalBackend::new().expect("Failed to init backend");

        // 2x3 matrix
        let data_a =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor_a = Tensor::new(data_a.clone(), false);

        // 3x2 matrix
        let data_b =
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]).unwrap();
        let tensor_b = Tensor::new(data_b.clone(), false);

        // Expected result (manually calculated):
        // [1*7+2*9+3*2, 1*8+2*1+3*3] = [7+18+6, 8+2+9] = [31, 19]
        // [4*7+5*9+6*2, 4*8+5*1+6*3] = [28+45+12, 32+5+18] = [85, 55]
        let expected_data =
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![31.0, 19.0, 85.0, 55.0]).unwrap();

        let result_opts = backend.matmul(&tensor_a, &tensor_b);
        assert!(result_opts.is_some(), "Matmul returned None");

        let result = result_opts.unwrap();
        assert_approx_eq(&result, &expected_data, 1e-4);
    }
}
