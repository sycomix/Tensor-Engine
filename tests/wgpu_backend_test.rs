#[cfg(feature = "backend_wgpu")]
use ndarray::Array;

#[cfg(feature = "backend_wgpu")]
use tensor_engine::backend::wgpu::WgpuBackend;
#[cfg(feature = "backend_wgpu")]
use tensor_engine::backend::{Backend, CpuBackend};

#[test]
#[cfg(feature = "backend_wgpu")]
fn test_wgpu_matmul_simple() {
    let backend = match WgpuBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            println!("Skipping WGPU matmul test: {}", err);
            return;
        }
    };

    // 2. Create Input Tensors
    let a_data = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn(); // 2x3

    let b_data = Array::from_shape_vec((3, 2), vec![0.5, 0.1, 0.5, 0.1, 0.5, 0.1])
        .unwrap()
        .into_dyn(); // 3x2

    let c_arr = backend
        .matmul(&a_data, &b_data)
        .expect("WGPU backend should execute 2D f32 matmul");

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

#[test]
#[cfg(feature = "backend_wgpu")]
fn test_wgpu_matmul_matches_cpu_rectangular() {
    let backend = match WgpuBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            println!("Skipping WGPU rectangular matmul test: {}", err);
            return;
        }
    };
    let cpu = CpuBackend::default();

    let a = Array::from_shape_vec(
        (3, 4),
        vec![
            1.0, -2.0, 3.5, 4.0, 0.5, 2.0, -1.0, 8.0, 7.0, 0.0, 1.0, -3.0,
        ],
    )
    .unwrap()
    .into_dyn();
    let b = Array::from_shape_vec(
        (4, 5),
        vec![
            0.25, 1.0, -1.0, 2.0, 0.0, 3.0, -0.5, 0.5, 1.0, 2.0, -2.0, 4.0, 1.5, -1.5, 0.25, 0.75,
            2.5, -3.0, 0.0, 1.0,
        ],
    )
    .unwrap()
    .into_dyn();

    let gpu_result = backend
        .matmul(&a, &b)
        .expect("WGPU backend should execute rectangular matmul");
    let cpu_result = cpu
        .matmul(&a, &b)
        .expect("CPU backend should execute rectangular matmul");

    assert_eq!(gpu_result.shape(), cpu_result.shape());
    let max_diff = (&gpu_result - &cpu_result)
        .mapv(|value| value.abs())
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "WGPU and CPU matmul diverged; max_diff={}",
        max_diff
    );
}

#[test]
#[cfg(feature = "backend_wgpu")]
fn test_wgpu_batched_matmul_matches_cpu() {
    let backend = match WgpuBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            println!("Skipping WGPU batched matmul test: {}", err);
            return;
        }
    };
    let cpu = CpuBackend::default();

    let a = Array::from_shape_vec(
        (2, 2, 3),
        vec![
            1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.25, 0.5, 0.75, -2.0, 3.0, 1.0,
        ],
    )
    .unwrap()
    .into_dyn();
    let b = Array::from_shape_vec(
        (2, 3, 2),
        vec![
            0.5, 1.0, -1.0, 2.0, 3.0, -0.5, 1.5, -2.0, 0.0, 4.0, -3.0, 0.25,
        ],
    )
    .unwrap()
    .into_dyn();

    let gpu_result = backend
        .matmul(&a, &b)
        .expect("WGPU backend should execute batched matmul");
    let cpu_result = cpu
        .matmul(&a, &b)
        .expect("CPU backend should execute batched matmul");

    assert_eq!(gpu_result.shape(), &[2, 2, 2]);
    assert_eq!(gpu_result.shape(), cpu_result.shape());
    let max_diff = (&gpu_result - &cpu_result)
        .mapv(|value| value.abs())
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "WGPU and CPU batched matmul diverged; max_diff={}",
        max_diff
    );
}
