use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// Configuration for AWQ (Activation-aware Weight Quantization).
/// Currently a placeholder for future config params like group size.
#[derive(Debug, Clone, Copy)]
pub struct AwqConfig {
    pub group_size: usize,
    pub zero_point: bool,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            zero_point: true,
        }
    }
}

/// Unpack a 4-bit packed tensor (u8) into an f32 tensor.
///
/// `packed`: A Tensor with dtype U8 (or U32 interpreted as bytes) containing packed 4-bit values.
///           Each byte contains two 4-bit values: [low_nibble, high_nibble].
///           Typically shape is [rows, cols / 2] (or cols/8 if packed in u32).
///           For this scaffold, we assume the input is a flat byte buffer or byte-aligned tensor.
///
/// `shape`: The target shape of the unpacked tensor (float).
///          Must verify that packed_size * 2 >= target_size.
///
/// Returns: Tensor (F32) with `shape`.
pub fn unpack_4bit_u8(packed: &Tensor, shape: &[usize]) -> Tensor {
    // 1. Get raw bytes from packed tensor
    // For now assuming storage is byte-accessible (f32, u8, etc) - we cast to u8 slice.
    // Real implementation should probably enforce U8 or I8 storage type.
    let lock = packed.lock();
    let data = match &lock.storage {
        crate::dtype::TensorStorage::U8(arr) => arr.as_slice().unwrap(),
        _ => {
            // Fallback: strictly we expect U8 for packed 4bit.
            // If we passed in F32 (simulating bytes), cast it.
            // For strict correctness in this scaffold, let's just abort/panic or return empty if not U8.
            // But to be robust for the test usage (where we might create U8 tensor), we rely on U8 storage support.
            // If U8 storage isn't fully exposed in DType yet, we might fallback to F32 storage reinterpreted,
            // but let's assume U8 storage exists (it does in `dtype.rs`).
            panic!("unpack_4bit_u8 expects TensorStorage::U8");
        }
    };

    let target_len: usize = shape.iter().product();
    if data.len() * 2 < target_len {
        panic!(
            "unpack_4bit_u8: Packed buffer too small. Bytes: {}, Target Elements: {}",
            data.len(),
            target_len
        );
    }

    let mut unpacked = Vec::with_capacity(target_len);

    // unpack loop
    let mut added = 0;
    for &byte in data {
        if added >= target_len {
            break;
        }

        // Lower 4 bits (first element)
        let low = byte & 0x0F;
        unpacked.push(low as f32);
        added += 1;

        if added >= target_len {
            break;
        }

        // Upper 4 bits (second element)
        let high = (byte >> 4) & 0x0F;
        unpacked.push(high as f32);
        added += 1;
    }

    let arr =
        ArrayD::from_shape_vec(IxDyn(shape), unpacked).expect("Shape mismatch in unpack_4bit");
    Tensor::new(arr, false)
}

/// Dequantize 4-bit packed weights using affine quantization: w = (q - z) * s
///
/// # Arguments
/// * `packed`: (N, K/2) packed u8 tensor [low|high]
/// * `scales`: (N, K/G) f32 scales (G=group_size, usually 128)
/// * `zeros`: (N, K/G) f32 zeros (already unpacked/converted to f32 for simplicity)
/// * `group_size`: The group size (e.g. 128)
/// * `target_shape`: (N, K)
///
/// Returns: (N, K) F32 Tensor.
pub fn awq_dequantize_affine(
    packed: &Tensor,
    scales: &Tensor,
    zeros: &Tensor,
    group_size: usize,
    target_shape: &[usize],
) -> Result<Tensor, String> {
    // 1. Unpack weights -> (N, K)
    let unpacked_q = unpack_4bit_u8(packed, target_shape);
    let q_arr = unpacked_q.lock().storage.to_f32_array();

    // 2. Expand scales/zeros to match (N, K)
    // Scales shape: (N, K/G) -> repeat G times inner dim
    let s_arr = scales.lock().storage.to_f32_array();
    let z_arr = zeros.lock().storage.to_f32_array();

    // Validation
    let n_rows = target_shape[0];
    let k_cols = target_shape[1];
    if s_arr.ndim() != 2 || z_arr.ndim() != 2 {
        return Err(format!(
            "Scales/Zeros must be 2D, got dim {:?} / {:?}",
            s_arr.ndim(),
            z_arr.ndim()
        ));
    }
    if s_arr.shape()[0] != n_rows || s_arr.shape()[1] * group_size != k_cols {
        return Err(format!(
            "Shape mismatch: target=({},{}), scales=({},{}), group={}",
            n_rows,
            k_cols,
            s_arr.shape()[0],
            s_arr.shape()[1],
            group_size
        ));
    }

    // 3. Compute in-place or new tensor
    // w = (q - z) * s
    // We iterate rows and groups for efficiency to avoid massive allocations for expansion
    let mut out_data = Vec::with_capacity(n_rows * k_cols);

    // s_arr / z_arr are [N, K_groups]
    // q_arr is [N, K]

    // Using iterators or raw indexing.
    // Since everything is row-major (Standard layout in ArrayD generally), we can iterate linearly if carefully handling groups.
    let q_slice = q_arr.as_slice().ok_or("q_arr not contiguous")?;
    let s_slice = s_arr.as_slice().ok_or("scales not contiguous")?;
    let z_slice = z_arr.as_slice().ok_or("zeros not contiguous")?;

    let k_groups = k_cols / group_size;

    for r in 0..n_rows {
        let row_offset_q = r * k_cols;
        let row_offset_sz = r * k_groups;

        for g in 0..k_groups {
            // Get scale/zero for this group
            let s = s_slice[row_offset_sz + g];
            let z = z_slice[row_offset_sz + g];

            // Apply to all elements in the group
            let group_offset = g * group_size;
            for i in 0..group_size {
                let q_val = q_slice[row_offset_q + group_offset + i];
                let w_val = (q_val - z) * s;
                out_data.push(w_val);
            }
        }
    }

    let arr = ArrayD::from_shape_vec(IxDyn(target_shape), out_data)
        .map_err(|e| format!("Output shape creation failed: {}", e))?;

    Ok(Tensor::new(arr, false))
}
