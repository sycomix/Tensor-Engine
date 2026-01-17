#![cfg(not(feature = "openblas"))]
use std::slice;

// Provide a minimal, safe CBLAS SGEMM fallback when OpenBLAS is not available.
// This is intentionally simple and not optimized. It ensures the dynamic symbol
// `cblas_sgemm` is defined so builds that expect the symbol will still run.

#[no_mangle]
pub extern "C" fn cblas_sgemm(
    _order: i32, // CBLAS_ORDER
    transa: i32, // CBLAS_TRANSPOSE
    transb: i32, // CBLAS_TRANSPOSE
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    // Safety: caller must ensure pointers are valid; this mirrors the CBLAS contract.
    if a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    unsafe {
        let a_slice = slice::from_raw_parts(a, lda * if transa == 101 /* CblasRowMajor */ { k } else { m });
        let b_slice = slice::from_raw_parts(b, ldb * if transb == 101 /* CblasRowMajor */ { n } else { k });
        let c_slice = slice::from_raw_parts_mut(c, ldc * n);

        // Initialize c with beta
        for row in 0..m {
            for col in 0..n {
                let idx = row * ldc + col;
                c_slice[idx] = c_slice[idx] * beta;
            }
        }

        // Naive triple loop
        for i in 0..m {
            for p in 0..k {
                let a_ip = if transa == 112 /* CblasNoTrans */ {
                    // A is m x k, row-major
                    a_slice[i * lda + p]
                } else {
                    // A is k x m, transposed
                    a_slice[p * lda + i]
                };
                let a_ip = a_ip * alpha;
                for j in 0..n {
                    let b_pj = if transb == 112 /* CblasNoTrans */ {
                        // B is k x n
                        b_slice[p * ldb + j]
                    } else {
                        // B is n x k, transposed
                        b_slice[j * ldb + p]
                    };
                    let idx = i * ldc + j;
                    c_slice[idx] += a_ip * b_pj;
                }
            }
        }
    }
}
