#[cfg(feature = "backend_wgpu")]
use ndarray::Array;

#[cfg(feature = "backend_wgpu")]
use tensor_engine::tensor::Tensor;

#[test]
#[cfg(feature = "backend_wgpu")]
fn test_wgpu_matmul_simple() {
    // 1. Initialize Backend
    // FAILS if no GPU adapter found (panic in new())
    tensor_engine::backend::set_wgpu_backend().expect("Failed to set wgpu backend");

    // 2. Create Input Tensors
    let a_data = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn(); // 2x3

    let b_data = Array::from_shape_vec((3, 2), vec![0.5, 0.1, 0.5, 0.1, 0.5, 0.1])
        .unwrap()
        .into_dyn(); // 3x2

    let a = Tensor::new(a_data, false);
    let b = Tensor::new(b_data, false);

    // 3. Compute
    // This should route to WgpuBackend::matmul -> GPU
    // Expected Result:
    // [1*0.5 + 2*0.5 + 3*0.5, 1*0.1 + 2*0.1 + 3*0.1] = [3.0, 0.6]
    // [4*0.5 + 5*0.5 + 6*0.5, 4*0.1 + 5*0.1 + 6*0.1] = [7.5, 1.5]
    let c = a.matmul(&b);

    let c_lock = c.lock();
    let c_arr = c_lock.storage.to_f32_array();

    println!("GPU Result: \n{:?}", c_arr);

    assert_eq!(c_arr.shape(), &[2, 2]);

    let expected = Array::from_shape_vec((2, 2), vec![3.0, 0.6, 7.5, 1.5])
        .unwrap()
        .into_dyn();

    // Allow small epsilon for float variations
    let diff = &c_arr - &expected;
    let max_diff = diff
        .mapv(|x| x.abs())
        .iter()
        .cloned()
        .fold(0. / 0., f32::max);

    assert!(max_diff < 1e-5, "Result mismatch! Max diff: {}", max_diff);
}
