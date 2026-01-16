use std::slice;

/// PRODUCTION CRITICAL: BF16 to FP32 Expansion (Rust Implementation)
///
/// # Safety
/// This function is marked `unsafe` because it accepts raw C pointers.
/// It is the caller's responsibility to ensure:
/// 1. `src` points to a valid buffer of `count` u16 elements.
/// 2. `dst` points to a valid buffer of `count` f32 elements.
/// 3. The memory regions do not overlap.
#[no_mangle]
pub unsafe extern "C" fn convert_bf16_to_f32_buffer(
    src: *const u16,
    dst: *mut f32,
    count: usize,
) {
    // 1. Validate Pointers
    if src.is_null() || dst.is_null() {
        return;
    }

    // 2. Reconstruct Slices from Raw Pointers
    let src_slice = slice::from_raw_parts(src, count);
    let dst_slice = slice::from_raw_parts_mut(dst, count);

    // 3. The Conversion Loop
    for (i, &bf16_val) in src_slice.iter().enumerate() {
        let val_u32 = bf16_val as u32;
        let bits = val_u32 << 16;
        dst_slice[i] = f32::from_bits(bits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_one_point_zero() {
        let input: [u16; 1] = [0x3F80];
        let mut output: [f32; 1] = [0.0];

        unsafe {
            convert_bf16_to_f32_buffer(input.as_ptr(), output.as_mut_ptr(), 1);
        }

        assert_ne!(output[0], 16256.0, "CRITICAL: numeric cast bug");
        assert_eq!(output[0], 1.0, "Conversion failed: Expected 1.0");
    }

    #[test]
    fn test_bf16_negative_numbers() {
        let input: [u16; 1] = [0xC000];
        let mut output: [f32; 1] = [0.0];

        unsafe {
            convert_bf16_to_f32_buffer(input.as_ptr(), output.as_mut_ptr(), 1);
        }

        assert_eq!(output[0], -2.0);
    }
}
