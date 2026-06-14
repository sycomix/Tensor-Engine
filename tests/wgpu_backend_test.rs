#[cfg(feature = "backend_wgpu")]
use ndarray::Array;

#[cfg(feature = "backend_wgpu")]
use tensor_engine::backend::wgpu::WgpuBackend;
#[cfg(feature = "backend_wgpu")]
use tensor_engine::backend::{ActivationKind, Backend, CpuBackend};

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

#[test]
#[cfg(feature = "backend_wgpu")]
fn test_wgpu_softmax_last_axis_rows_sum_to_one() {
    let backend = match WgpuBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            println!("Skipping WGPU softmax test: {}", err);
            return;
        }
    };

    let input = Array::from_shape_vec(
        (2, 2, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0, 0.5, 0.5, 0.5, 0.5, 10.0, 0.0, -10.0, 5.0,
        ],
    )
    .unwrap()
    .into_dyn();

    let output = backend
        .softmax(&input, 2)
        .expect("WGPU backend should execute last-axis softmax");
    assert_eq!(output.shape(), &[2, 2, 4]);

    for row in output.lanes(ndarray::Axis(2)) {
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax row sum should be 1.0, got {} for {:?}",
            sum,
            row
        );
        assert!(
            row.iter().all(|value| value.is_finite() && *value >= 0.0),
            "softmax row should contain finite non-negative probabilities: {:?}",
            row
        );
    }
}

#[test]
#[cfg(feature = "backend_wgpu")]
fn test_wgpu_softmax_all_negative_infinity_becomes_uniform() {
    let backend = match WgpuBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            println!("Skipping WGPU softmax all -inf test: {}", err);
            return;
        }
    };

    let input = Array::from_shape_vec(
        (2, 4),
        vec![
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            -2.0,
            -1.0,
            0.0,
            1.0,
        ],
    )
    .unwrap()
    .into_dyn();

    let output = backend
        .softmax(&input, 1)
        .expect("WGPU backend should execute softmax with all -inf rows");
    let uniform_row = output.index_axis(ndarray::Axis(0), 0);
    for value in uniform_row.iter() {
        assert!(
            (*value - 0.25).abs() < 1e-6,
            "all -inf softmax row should become uniform, got {}",
            value
        );
    }
}

#[test]
#[cfg(feature = "backend_wgpu")]
fn test_wgpu_rms_norm_matches_reference() {
    let backend = match WgpuBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            println!("Skipping WGPU RMSNorm test: {}", err);
            return;
        }
    };

    let input = Array::from_shape_vec(
        (2, 3, 4),
        vec![
            1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5, -1.0, -1.0, 2.0, 2.0, 4.0, 3.0, 2.0, 1.0,
            -3.0, 0.0, 3.0, 6.0, 0.25, -0.5, 0.75, -1.0,
        ],
    )
    .unwrap()
    .into_dyn();
    let weight = Array::from_shape_vec(4, vec![1.0, 0.5, -1.5, 2.0])
        .unwrap()
        .into_dyn();
    let eps = 1e-5;

    let gpu_result = backend
        .rms_norm(&input, &weight, eps, 2)
        .expect("WGPU backend should execute last-axis RMSNorm");
    let mut expected = input.clone();
    for mut row in expected.lanes_mut(ndarray::Axis(2)) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
        let denom = (mean_square + eps).sqrt();
        for (col, value) in row.iter_mut().enumerate() {
            *value = (*value / denom) * weight[col];
        }
    }

    assert_eq!(gpu_result.shape(), expected.shape());
    let max_diff = (&gpu_result - &expected)
        .mapv(|value| value.abs())
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "WGPU and reference RMSNorm diverged; max_diff={}",
        max_diff
    );
}

#[test]
#[cfg(feature = "backend_wgpu")]
fn test_wgpu_layer_norm_matches_reference() {
    let backend = match WgpuBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            println!("Skipping WGPU LayerNorm test: {}", err);
            return;
        }
    };

    let input = Array::from_shape_vec(
        (2, 3, 4),
        vec![
            1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5, -1.0, -1.0, 2.0, 2.0, 4.0, 3.0, 2.0, 1.0,
            -3.0, 0.0, 3.0, 6.0, 0.25, -0.5, 0.75, -1.0,
        ],
    )
    .unwrap()
    .into_dyn();
    let weight = Array::from_shape_vec(4, vec![1.0, 0.5, -1.5, 2.0])
        .unwrap()
        .into_dyn();
    let bias = Array::from_shape_vec(4, vec![0.1, -0.2, 0.3, -0.4])
        .unwrap()
        .into_dyn();
    let eps = 1e-5;

    let gpu_result = backend
        .layer_norm(&input, &weight, &bias, eps, 2)
        .expect("WGPU backend should execute last-axis LayerNorm");
    let mut expected = input.clone();
    for mut row in expected.lanes_mut(ndarray::Axis(2)) {
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        let variance = row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / row.len() as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();
        for (col, value) in row.iter_mut().enumerate() {
            *value = ((*value - mean) * inv_std) * weight[col] + bias[col];
        }
    }

    assert_eq!(gpu_result.shape(), expected.shape());
    let max_diff = (&gpu_result - &expected)
        .mapv(|value| value.abs())
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "WGPU and reference LayerNorm diverged; max_diff={}",
        max_diff
    );
}

#[test]
#[cfg(feature = "backend_wgpu")]
fn test_wgpu_unary_activations_match_reference() {
    let backend = match WgpuBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            println!("Skipping WGPU unary activation test: {}", err);
            return;
        }
    };

    let input = Array::from_shape_vec(
        (2, 2, 4),
        vec![
            -6.0, -2.0, -0.5, 0.0, 0.25, 0.75, 1.0, 2.0, -1.5, 3.0, 4.0, -4.0, 5.0, -0.25, 0.5, 1.5,
        ],
    )
    .unwrap()
    .into_dyn();

    let cases: &[(ActivationKind, fn(f32) -> f32, f32)] = &[
        (ActivationKind::Relu, |value| value.max(0.0), 1e-6),
        (
            ActivationKind::Sigmoid,
            |value| 1.0 / (1.0 + (-value).exp()),
            1e-6,
        ),
        (ActivationKind::Tanh, |value| value.tanh(), 1e-6),
        (
            ActivationKind::Gelu,
            |value| {
                let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
                let inner = sqrt_2_over_pi * (value + 0.044715 * value * value * value);
                0.5 * value * (1.0 + inner.tanh())
            },
            1e-5,
        ),
        (
            ActivationKind::Silu,
            |value| value / (1.0 + (-value).exp()),
            1e-6,
        ),
    ];

    for (kind, reference_fn, tolerance) in cases {
        let gpu_result = backend
            .unary_activation(&input, *kind)
            .expect("WGPU backend should execute unary activation");
        let expected = input.mapv(*reference_fn);
        assert_eq!(gpu_result.shape(), input.shape());
        let max_diff = (&gpu_result - &expected)
            .mapv(|value| value.abs())
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff <= *tolerance,
            "WGPU activation {:?} diverged; max_diff={} tolerance={}",
            kind,
            max_diff,
            tolerance
        );
    }
}
