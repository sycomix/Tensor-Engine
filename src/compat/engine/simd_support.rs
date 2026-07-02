use core::arch::x86_64::*;
use half::f16;

pub type I32x8 = __m256i;
pub type F32x8 = __m256;
pub type I16x8 = __m128i;

/* ------------------ */
/* Loading and storing things */
/* ------------------ */

#[inline]
pub unsafe fn load_i16x8(ptr: *const I16x8) -> I16x8 {
    unsafe { _mm_loadu_si128(ptr) }
}

#[inline]
pub unsafe fn store_i16x8(ptr: *mut I16x8, a: I16x8) {
    unsafe { _mm_storeu_si128(ptr, a) }
}

#[inline]
pub fn load_f32x8(ptr: *const F32x8) -> F32x8 {
    unsafe { _mm256_loadu_ps(ptr as *const f32) }
}

#[inline]
pub fn store_f32x8(ptr: *mut F32x8, a: F32x8) {
    unsafe { _mm256_storeu_ps(ptr as *mut f32, a) }
}

#[inline]
pub unsafe fn gather_f32x8(ptr: *const f32, indices: I32x8) -> F32x8 {
    unsafe { _mm256_i32gather_ps(ptr, indices, 1) }
}

/* ------------------ */
/* Conversions        */
/* ------------------ */

#[inline]
pub fn i16x8_as_f16_to_f32x8(a: I16x8) -> F32x8 {
    if is_x86_feature_detected!("f16c") {
        unsafe { i16x8_as_f16_to_f32x8_f16c(a) }
    } else {
        i16x8_as_f16_to_f32x8_fallback(a)
    }
}

#[target_feature(enable = "f16c")]
unsafe fn i16x8_as_f16_to_f32x8_f16c(a: I16x8) -> F32x8 {
    _mm256_cvtph_ps(a)
}

fn i16x8_as_f16_to_f32x8_fallback(a: I16x8) -> F32x8 {
    unsafe {
        let mut u16_arr: [u16; 8] = [0u16; 8];
        _mm_storeu_si128(u16_arr.as_mut_ptr() as *mut __m128i, a);
        let mut f32_arr: [f32; 8] = [0.0f32; 8];
        for i in 0..8 {
            f32_arr[i] = f16::from_bits(u16_arr[i]).to_f32();
        }
        _mm256_loadu_ps(f32_arr.as_ptr())
    }
}

#[inline]
pub fn f32x8_to_i16x8_as_f16(a: F32x8) -> I16x8 {
    if is_x86_feature_detected!("f16c") {
        unsafe { f32x8_to_i16x8_as_f16_f16c(a) }
    } else {
        f32x8_to_i16x8_as_f16_fallback(a)
    }
}

#[target_feature(enable = "f16c")]
unsafe fn f32x8_to_i16x8_as_f16_f16c(a: F32x8) -> I16x8 {
    _mm256_cvtps_ph(a, 0)
}

fn f32x8_to_i16x8_as_f16_fallback(a: F32x8) -> I16x8 {
    unsafe {
        let mut f32_arr: [f32; 8] = [0.0f32; 8];
        _mm256_storeu_ps(f32_arr.as_mut_ptr(), a);
        let mut u16_arr: [u16; 8] = [0u16; 8];
        for i in 0..8 {
            u16_arr[i] = f16::from_f32(f32_arr[i]).to_bits();
        }
        _mm_loadu_si128(u16_arr.as_ptr() as *const __m128i)
    }
}

/*
 * Constants, creating from constants
 */

pub fn f32x8_zero() -> F32x8 {
    unsafe { _mm256_setzero_ps() }
}

pub fn i16x8_zero() -> I16x8 {
    unsafe { _mm_setzero_si128() }
}

pub fn f32x8_singleton(value: f32) -> F32x8 {
    unsafe { _mm256_set1_ps(value) }
}

pub fn i32x8_from_values(
    val0: i32,
    val1: i32,
    val2: i32,
    val3: i32,
    val4: i32,
    val5: i32,
    val6: i32,
    val7: i32,
) -> I32x8 {
    unsafe { _mm256_set_epi32(val0, val1, val2, val3, val4, val5, val6, val7) }
}

/*
 * Operations
 */

// FMA

// a * b + c
pub fn fma_f32x8(a: F32x8, b: F32x8, c: F32x8) -> F32x8 {
    if is_x86_feature_detected!("fma") {
        unsafe { fma_f32x8_fma(a, b, c) }
    } else {
        unsafe { _mm256_add_ps(_mm256_mul_ps(a, b), c) }
    }
}

#[target_feature(enable = "fma")]
unsafe fn fma_f32x8_fma(a: F32x8, b: F32x8, c: F32x8) -> F32x8 {
    _mm256_fmadd_ps(a, b, c)
}

// Horizontal sums

#[inline]
pub fn horizontal_sum_f32x8(mut ymm: __m256) -> f32 {
    unsafe {
        let ymm2 = _mm256_permute2f128_ps(ymm, ymm, 1);
        ymm = _mm256_add_ps(ymm, ymm2);
        ymm = _mm256_hadd_ps(ymm, ymm);
        ymm = _mm256_hadd_ps(ymm, ymm);
        _mm256_cvtss_f32(ymm)
    }
}

#[inline]
pub fn horizontal_sum_and_f32_to_f16(mut ymm: __m256) -> f16 {
    unsafe {
        let ymm2 = _mm256_permute2f128_ps(ymm, ymm, 1);
        ymm = _mm256_add_ps(ymm, ymm2);
        ymm = _mm256_hadd_ps(ymm, ymm);
        ymm = _mm256_hadd_ps(ymm, ymm);
        f16::from_f32(_mm256_cvtss_f32(ymm))
    }
}
