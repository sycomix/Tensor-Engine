//! Tests for the GPU/CPU backend abstraction layer.

#[cfg(test)]
mod tests {
    use crate::backend::{get_global_backend, set_cpu_backend};
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn test_default_backend_is_cpu() {
        let backend = get_global_backend();
        assert_eq!(backend.name(), "CPU");
    }

    #[test]
    fn test_cpu_backend_matmul_2d() {
        set_cpu_backend().expect("Failed to set CPU backend");
        let backend = get_global_backend();

        // Create simple 2x3 and 3x4 matrices
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x4

        let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), a_data).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[3, 4]), b_data).unwrap();

        let result = backend.matmul(&a, &b);
        assert!(result.is_some(), "Matmul should succeed for valid shapes");

        let result = result.unwrap();
        assert_eq!(result.shape(), &[2, 4]);

        // Expected matrix values are computed explicitly below for both rows.
        // Row 0: [1*1+2*3+3*5, 1*2+2*4+3*6, 1*3+2*5+3*7, 1*4+2*6+3*8] = [22, 28, 34, 40]
        // Row 1: [4*1+5*3+6*5, 4*2+5*4+6*6, 4*3+5*5+6*7, 4*4+5*6+6*8] = [49, 64, 79, 94]
        let expected = vec![22.0, 28.0, 34.0, 40.0, 49.0, 64.0, 79.0, 94.0];
        
        for (i, &val) in result.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-5, "Mismatch at index {}: got {}, expected {}", i, val, expected[i]);
        }
    }

    #[test]
    fn test_cpu_backend_matmul_3d() {
        set_cpu_backend().expect("Failed to set CPU backend");
        let backend = get_global_backend();

        // Create batch of 2x matrices: [batch=2, m=2, k=3] and [batch=2, k=3, n=4]
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 2x2x3
        let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]; // 2x3x4

        let a = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), a_data).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), b_data).unwrap();

        let result = backend.matmul(&a, &b);
        assert!(result.is_some(), "Matmul should succeed for valid 3D shapes");

        let result = result.unwrap();
        assert_eq!(result.shape(), &[2, 2, 4]);
    }

    #[test]
    fn test_cpu_backend_matmul_invalid_shapes() {
        set_cpu_backend().expect("Failed to set CPU backend");
        let backend = get_global_backend();

        // Invalid: 2x3 and 2x4 (inner dimensions don't match)
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), a_data).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[2, 4]), b_data).unwrap();

        let result = backend.matmul(&a, &b);
        assert!(result.is_none(), "Matmul should fail for invalid shapes");
    }

    #[test]
    fn test_cpu_backend_memory_info() {
        set_cpu_backend().expect("Failed to set CPU backend");
        let backend = get_global_backend();

        let (used, total) = backend.memory_info();
        assert!(total > 0, "Total memory should be positive");
        assert!(used <= total, "Used memory should not exceed total");
    }
}
