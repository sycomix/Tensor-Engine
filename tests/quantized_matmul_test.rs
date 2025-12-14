use tensor_engine::tensor::Tensor;
use tensor_engine::dtype::DType;
use ndarray::{Array, Array2, Ix2, IxDyn};

fn manual_dequantize_i8(bytes: &[i8], scale: f32, rows: usize, cols: usize) -> Array2<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            out[idx] = bytes[idx] as f32 * scale;
        }
    }
    Array2::from_shape_vec((rows, cols), out).expect("manual_dequantize_i8 shape")
}

fn manual_dequantize_rowwise(
    bytes: &[i8],
    scales: &[f32],
    rows: usize,
    cols: usize,
) -> Array2<f32> {
    assert_eq!(scales.len(), rows, "rowwise scales must match rows");
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let s = scales[r];
        for c in 0..cols {
            let idx = r * cols + c;
            out[idx] = bytes[idx] as f32 * s;
        }
    }
    Array2::from_shape_vec((rows, cols), out).expect("manual_dequantize_rowwise shape")
}

fn manual_dequantize_blockwise(
    bytes: &[i8],
    scales: &[f32],
    rows: usize,
    cols: usize,
    block_size: usize,
) -> Array2<f32> {
    assert!(block_size > 0, "block_size must be > 0");
    let blocks_per_row = (cols + block_size - 1) / block_size;
    assert_eq!(
        scales.len(),
        rows * blocks_per_row,
        "blockwise scales must be rows*blocks_per_row"
    );
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let block_idx = c / block_size;
            let s = scales[r * blocks_per_row + block_idx];
            out[idx] = bytes[idx] as f32 * s;
        }
    }
    Array2::from_shape_vec((rows, cols), out).expect("manual_dequantize_blockwise shape")
}

#[test]
fn test_quantized_matmul_basic() {
    // Input 2x3 times 3x4 => 2x4
    let a = Tensor::new(Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32; 6]).unwrap(), false);
    let w = Tensor::new_with_dtype(Array::from_shape_vec(IxDyn(&[3, 4]), vec![1.0f32; 12]).unwrap(), false, DType::I8);
    let out = a.quantized_matmul(&w);
    let shap = out.lock().storage.shape();
    assert_eq!(shap, vec![2, 4]);
    // numerical check: out should equal 3.0 for all entries in this simple case
    let out_arr = out.lock().storage.to_f32_array();
    for v in out_arr.iter() {
        assert!((*v - 3.0).abs() < 1e-5);
    }
}

#[test]
fn test_quantized_matmul_i8_nontrivial_scale_matches_manual_dequant() {
    // Use non-trivial magnitudes to ensure per-tensor scale is not 1.0.
    // A: 3x4, W: 4x5
    let a_vals: Vec<f32> = vec![
        0.2, -0.1, 0.3, 1.2, // row 0
        -0.7, 0.4, 0.0, 0.9, // row 1
        1.5, -0.2, 0.8, -1.1, // row 2
    ];
    let w_vals: Vec<f32> = (0..(4 * 5))
        .map(|i| {
            // deterministic pattern, ranges roughly [-3, 3]
            let x = (i as i32 - 10) as f32;
            (x / 3.0).sin() * 2.5 + (x / 11.0)
        })
        .collect();

    let a = Tensor::new(Array::from_shape_vec(IxDyn(&[3, 4]), a_vals).unwrap(), false);
    let w_f32 = Tensor::new(Array::from_shape_vec(IxDyn(&[4, 5]), w_vals).unwrap(), false);
    let qw = Tensor::new_with_dtype(w_f32.lock().storage.to_f32_array(), false, DType::I8);

    let out = a.quantized_matmul(&qw);
    assert_eq!(out.lock().storage.shape(), vec![3, 5]);

    // Build a manual dequantized matrix from the underlying storage.
    let (bytes, scale, shape) = match &qw.lock().storage {
        tensor_engine::dtype::TensorStorage::I8(b, s, sh) => (b.clone(), *s, sh.clone()),
        other => panic!("expected TensorStorage::I8, got {other:?}"),
    };
    assert_eq!(shape, vec![4, 5]);
    let w_manual = manual_dequantize_i8(&bytes, scale, 4, 5);

    let a2 = a
        .lock()
        .storage
        .to_f32_array()
        .into_dimensionality::<Ix2>()
        .unwrap();
    let expected = a2.dot(&w_manual);

    let out2 = out
        .lock()
        .storage
        .to_f32_array()
        .into_dimensionality::<Ix2>()
        .unwrap();
    for (x, y) in out2.iter().zip(expected.iter()) {
        assert!((x - y).abs() < 1e-4, "mismatch: {x} vs {y}");
    }
}

#[test]
fn test_quantized_matmul_rowwise() {
    let a = Tensor::new(Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32; 6]).unwrap(), false);
    let w = Tensor::new(Array::from_shape_vec(IxDyn(&[3, 4]), vec![1.0f32; 12]).unwrap(), false);
    let qw = w.quantize_weights(tensor_engine::dtype::DType::I8Rowwise, None).expect("quantize weights rowwise");
    let out = a.quantized_matmul(&qw);
    let shap = out.lock().storage.shape();
    assert_eq!(shap, vec![2, 4]);
    // verify numerical equality with matmul on dequantized version
    let deq = qw.lock().storage.to_f32_array();
    let plain = a.matmul(&Tensor::new(deq, false));
    assert_eq!(plain.lock().storage.shape(), shap);
    let out_arr = out.lock().storage.to_f32_array();
    let plain_arr = plain.lock().storage.to_f32_array();
    assert_eq!(out_arr.shape(), plain_arr.shape());
    for (x, y) in out_arr.iter().zip(plain_arr.iter()) {
        assert!((x - y).abs() < 1e-4);
    }
}

#[test]
fn test_quantized_matmul_rowwise_nontrivial_scales_match_manual_dequant() {
    // A: 2x4, W: 4x7
    // Construct W so each row has distinct magnitude -> distinct row scales.
    let a_vals: Vec<f32> = vec![0.9, -0.3, 0.2, 1.1, -0.5, 0.7, -1.2, 0.4];
    let mut w_vals: Vec<f32> = Vec::with_capacity(4 * 7);
    for r in 0..4 {
        for c in 0..7 {
            let base = (r as f32 + 1.0) * 0.35;
            let v = ((c as f32 + 1.0) * base).sin() * (1.0 + r as f32 * 2.0);
            w_vals.push(v);
        }
    }

    let a = Tensor::new(Array::from_shape_vec(IxDyn(&[2, 4]), a_vals).unwrap(), false);
    let w = Tensor::new(Array::from_shape_vec(IxDyn(&[4, 7]), w_vals).unwrap(), false);
    let qw = w
        .quantize_weights(tensor_engine::dtype::DType::I8Rowwise, None)
        .expect("quantize weights rowwise");

    let out = a.quantized_matmul(&qw);
    assert_eq!(out.lock().storage.shape(), vec![2, 7]);

    let (bytes, scales, shape) = match &qw.lock().storage {
        tensor_engine::dtype::TensorStorage::I8Rowwise(b, s, sh) => (b.clone(), s.clone(), sh.clone()),
        other => panic!("expected TensorStorage::I8Rowwise, got {other:?}"),
    };
    assert_eq!(shape, vec![4, 7]);
    let w_manual = manual_dequantize_rowwise(&bytes, &scales, 4, 7);

    let a2 = a
        .lock()
        .storage
        .to_f32_array()
        .into_dimensionality::<Ix2>()
        .unwrap();
    let expected = a2.dot(&w_manual);

    let out2 = out
        .lock()
        .storage
        .to_f32_array()
        .into_dimensionality::<Ix2>()
        .unwrap();
    for (x, y) in out2.iter().zip(expected.iter()) {
        assert!((x - y).abs() < 1e-4, "mismatch: {x} vs {y}");
    }
}

#[test]
fn test_quantized_matmul_blockwise() {
    let a = Tensor::new(Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32; 6]).unwrap(), false);
    let w = Tensor::new(Array::from_shape_vec(IxDyn(&[3, 4]), vec![1.0f32; 12]).unwrap(), false);
    let qw = w.quantize_weights(tensor_engine::dtype::DType::I8Blockwise, Some(2)).expect("quantize weights blockwise");
    let out = a.quantized_matmul(&qw);
    let shap = out.lock().storage.shape();
    assert_eq!(shap, vec![2, 4]);
    let deq = qw.lock().storage.to_f32_array();
    let plain = a.matmul(&Tensor::new(deq, false));
    assert_eq!(plain.lock().storage.shape(), shap);
    let out_arr = out.lock().storage.to_f32_array();
    let plain_arr = plain.lock().storage.to_f32_array();
    assert_eq!(out_arr.shape(), plain_arr.shape());
    for (x, y) in out_arr.iter().zip(plain_arr.iter()) {
        assert!((x - y).abs() < 1e-4);
    }
}

#[test]
fn test_quantized_matmul_blockwise_nontrivial_scales_match_manual_dequant() {
    // Choose cols not divisible by block_size to stress the last partial block.
    // A: 3x5, W: 5x7, block_size=3 => blocks_per_row=3
    let a_vals: Vec<f32> = vec![
        0.2, 0.7, -1.1, 0.0, 0.4, // row 0
        1.3, -0.6, 0.9, 0.5, -0.2, // row 1
        -0.8, 0.1, 0.3, -1.4, 0.6, // row 2
    ];
    let mut w_vals: Vec<f32> = Vec::with_capacity(5 * 7);
    for r in 0..5 {
        for c in 0..7 {
            // Make different column blocks have visibly different magnitude.
            let block = c / 3;
            let mag = match block {
                0 => 0.25,
                1 => 1.5,
                _ => 3.0,
            };
            let v = ((r as f32 + 1.0) * (c as f32 + 0.5)).cos() * mag;
            w_vals.push(v);
        }
    }

    let a = Tensor::new(Array::from_shape_vec(IxDyn(&[3, 5]), a_vals).unwrap(), false);
    let w = Tensor::new(Array::from_shape_vec(IxDyn(&[5, 7]), w_vals).unwrap(), false);
    let qw = w
        .quantize_weights(tensor_engine::dtype::DType::I8Blockwise, Some(3))
        .expect("quantize weights blockwise");

    let out = a.quantized_matmul(&qw);
    assert_eq!(out.lock().storage.shape(), vec![3, 7]);

    let (bytes, scales, shape, block_size) = match &qw.lock().storage {
        tensor_engine::dtype::TensorStorage::I8Blockwise(b, s, sh, bs) => {
            (b.clone(), s.clone(), sh.clone(), *bs)
        }
        other => panic!("expected TensorStorage::I8Blockwise, got {other:?}"),
    };
    assert_eq!(shape, vec![5, 7]);
    assert_eq!(block_size, 3);
    let w_manual = manual_dequantize_blockwise(&bytes, &scales, 5, 7, block_size);

    let a2 = a
        .lock()
        .storage
        .to_f32_array()
        .into_dimensionality::<Ix2>()
        .unwrap();
    let expected = a2.dot(&w_manual);

    let out2 = out
        .lock()
        .storage
        .to_f32_array()
        .into_dimensionality::<Ix2>()
        .unwrap();
    for (x, y) in out2.iter().zip(expected.iter()) {
        assert!((x - y).abs() < 1e-4, "mismatch: {x} vs {y}");
    }
}
