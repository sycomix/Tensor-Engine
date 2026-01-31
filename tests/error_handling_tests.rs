#[cfg(test)]
mod error_handling_tests {
    use tensor_engine::error::TensorError;
    use tensor_engine::tensor::Tensor;

    #[test]
    fn test_shape_validation_success() {
        let a = Tensor::new(ndarray::arr1(&[2.0, 3.0]).into_dyn(), true);
        let b = Tensor::new(ndarray::arr1(&[3.0, 2.0]).into_dyn(), true);

        let result = tensor_engine::error::shape_validation::validate_binary_shapes(
            &a,
            &b,
            "test operation",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_shape_validation_failure() {
        let a = Tensor::new(ndarray::arr1(&[2.0, 3.0]).into_dyn(), true);
        let b = Tensor::new(ndarray::arr1(&[3.0, 2.0, 1.0]).into_dyn(), true);

        let result = tensor_engine::error::shape_validation::validate_binary_shapes(
            &a,
            &b,
            "test operation",
        );
        assert!(result.is_err());

        if let Err(TensorError::ShapeMismatch { .. }) = result {
            // Test passes - mismatched shapes detected
        } else {
            panic!("Test should have detected shape mismatch but error type mismatch");
        }
    }
}
