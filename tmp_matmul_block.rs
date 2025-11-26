impl Operation for MatMul {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a_lock = inputs[0].lock();
        let b_lock = inputs[1].lock();
        let a: ArrayView2<f32> = a_lock
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("MatMul expects 2D left operand");
        let b: ArrayView2<f32> = b_lock
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("MatMul expects 2D right operand");
        #[cfg(all(feature = "openblas", not(target_os = "windows")))]
        {
            let detected = detect_blas_order();
            let detected = detect_blas_order();
            // Using CBLAS sgemm; both arrays are in row-major (C order)
            // Ensure we use contiguous row-major owned arrays for BLAS
            let a_owned = a.to_owned();
            let b_owned = b.to_owned();
            let m = a_owned.nrows() as i32;
            let k = a_owned.ncols() as i32;
            let n = b_owned.ncols() as i32;
            let a_slice = a_owned
                .as_slice()
                .expect("MatMul requires contiguous data for BLAS");
            let b_slice = b_owned
                .as_slice()
                .expect("MatMul requires contiguous data for BLAS");
            let mut c_vec = vec![0f32; (m as usize) * (n as usize)];
            // No-op: dimensions are validated below.
            match detected {
                Some(CBLAS_ORDER::CblasRowMajor) => unsafe {
                    cblas_sys::cblas_sgemm(
                        CBLAS_ORDER::CblasRowMajor,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        m,
                        n,
                        k,
                        1.0,
                        a_slice.as_ptr(),
                        k,
                        b_slice.as_ptr(),
                        n,
                        0.0,
                        c_vec.as_mut_ptr(),
                        n,
                    );
                },
                Some(CBLAS_ORDER::CblasColMajor) => {
                    // For ColumnMajor, build column-major buffers and call with ColumnMajor
                    let mut a_col_vec: Vec<f32> = Vec::with_capacity((m as usize) * (k as usize));
                    for col in 0..(k as usize) {
                        for row in 0..(m as usize) {
                            a_col_vec.push(a_owned[[row, col]]);
                        }
                    }
                    let mut b_col_vec: Vec<f32> = Vec::with_capacity((k as usize) * (n as usize));
                    for col in 0..(n as usize) {
                        for row in 0..(k as usize) {
                            b_col_vec.push(b_owned[[row, col]]);
                        }
                    }
                    unsafe {
                        cblas_sys::cblas_sgemm(
                            CBLAS_ORDER::CblasColMajor,
                            CBLAS_TRANSPOSE::CblasNoTrans,
                            CBLAS_TRANSPOSE::CblasNoTrans,
                            m,
                            n,
                            k,
                            1.0,
                            a_col_vec.as_ptr(),
                            m,
                            b_col_vec.as_ptr(),
                            k,
                            0.0,
                            c_vec.as_mut_ptr(),
                            m,
                        );
                    }
                    // convert column-major `c_vec` to row-major vector
                    let mut c_from_col = vec![0f32; (m as usize) * (n as usize)];
                    for row in 0..(m as usize) {
                        for col in 0..(n as usize) {
                            c_from_col[row * (n as usize) + col] = c_vec[col * (m as usize) + row];
                        }
                    }
                    c_vec = c_from_col;
                }
                None => {
                    // Fallback to ndarray
                    *output = a_owned.dot(&b_owned).into_dyn();
                    return;
                }
            }
            *output = ArrayD::from_shape_vec(IxDyn(&[m as usize, n as usize]), c_vec.clone())
                .expect("Failed to create matmul output array");

            // Quick check: verify the BLAS result equals ndarray's dot product. If not, try ColumnMajor or fall back to ndarray.
            let expected = a_owned.dot(&b_owned).into_dyn();
            if !approx_eq_arrayd(&expected, output) {
                log::warn!("BLAS RowMajor matmul result differs from ndarray dot; trying ColumnMajor fallback");
                // Try ColumnMajor by building column-major buffers and calling cblas with ColumnMajor
                let mut a_col_vec: Vec<f32> = Vec::with_capacity((m as usize) * (k as usize));
                for col in 0..(k as usize) {
                    for row in 0..(m as usize) {
                        a_col_vec.push(a_owned[[row, col]]);
                    }
                }
                let mut b_col_vec: Vec<f32> = Vec::with_capacity((k as usize) * (n as usize));
                for col in 0..(n as usize) {
                    for row in 0..(k as usize) {
                        b_col_vec.push(b_owned[[row, col]]);
                    }
                }
                let mut c_col_vec = vec![0f32; (m as usize) * (n as usize)];
                unsafe {
                    cblas_sys::cblas_sgemm(
                        CBLAS_ORDER::CblasColMajor,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        m,
                        n,
                        k,
                        1.0,
                        a_col_vec.as_ptr(),
                        m,
                        b_col_vec.as_ptr(),
                        k,
                        0.0,
                        c_col_vec.as_mut_ptr(),
                        m,
                    );
                }
                // Convert c_col_vec (column-major) to row-major vector
                let mut c_from_col = vec![0f32; (m as usize) * (n as usize)];
                for row in 0..(m as usize) {
                    for col in 0..(n as usize) {
                        // column-major index is col*m + row
                        c_from_col[row * (n as usize) + col] = c_col_vec[col * (m as usize) + row];
                    }
                }
                let cm = ArrayD::from_shape_vec(IxDyn(&[m as usize, n as usize]), c_from_col)
                    .expect("Failed to create matmul output array from column-major conversion");
                if approx_eq_arrayd(&expected, &cm) {
                    *output = cm;
                } else {
                    log::warn!("BLAS results differ (both RowMajor and ColumnMajor); falling back to ndarray dot");
                    *output = expected;
                }
            }
            return;
        }
        #[cfg(any(not(feature = "openblas"), target_os = "windows"))]
        {
            *output = a.dot(&b).into_dyn();
        }
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a_lock = inputs[0].lock();
        let b_lock = inputs[1].lock();
        let a: ArrayView2<f32> = a_lock
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("MatMul expects 2D left operand");
        let b: ArrayView2<f32> = b_lock
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("MatMul expects 2D right operand");
        let output_grad: ArrayView2<f32> = output_grad
            .view()
            .into_dimensionality::<Ix2>()
            .expect("MatMul expects 2D output grad");

        #[cfg(all(feature = "openblas", not(target_os = "windows")))]
        {
            let og: ArrayView2<f32> = output_grad
                .view()
                .into_dimensionality::<Ix2>()
                .expect("MatMul expects 2D output grad");
            // Because cblas requires contiguous row-major memory, make owned copies
            let og_owned = og.to_owned();
            let og_slice = og_owned
                .as_slice()
                .expect("MatMul output grad requires contiguous data");
            let a_owned = a.to_owned();
            let b_owned = b.to_owned();
            let a_slice = a_owned
                .as_slice()
                .expect("MatMul requires contiguous data for BLAS");
            let b_slice = b_owned
                .as_slice()
                .expect("MatMul requires contiguous data for BLAS");
            // Derive shape dims from the original inputs
            let m = a_owned.nrows() as i32;
            let k = a_owned.ncols() as i32;
            let n = b_owned.ncols() as i32;
            // grad_a = og @ b.T -> (m x n) @ (n x k) = (m x k)
            let mut grad_a_vec = vec![0f32; (m as usize) * (k as usize)];
            let detected = detect_blas_order();
            match detected {
                Some(CBLAS_ORDER::CblasRowMajor) => unsafe {
                    cblas_sys::cblas_sgemm(
                        CBLAS_ORDER::CblasRowMajor,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        CBLAS_TRANSPOSE::CblasTrans,
                        m,
                        k,
                        n,
                        1.0,
                        og_slice.as_ptr(),
                        n,
                        b_slice.as_ptr(),
                        n,
                        0.0,
                        grad_a_vec.as_mut_ptr(),
                        k,
                    );
                },
                Some(CBLAS_ORDER::CblasColMajor) => {
                    // Build column-major buffers for og and b
                    let og_owned = og_slice.to_vec();
                    let b_owned_vec = b_slice.to_vec();
                    // og (m x n) column-major vector
                    let mut og_col_vec = vec![0f32; (m as usize) * (n as usize)];
                    for col in 0..(n as usize) {
                        for row in 0..(m as usize) {
                            og_col_vec[col * (m as usize) + row] =
                                og_owned[row * (n as usize) + col];
                        }
                    }
                    // b (k x n) column-major vector
                    let mut b_col_vec = vec![0f32; (k as usize) * (n as usize)];
                    for col in 0..(n as usize) {
                        for row in 0..(k as usize) {
                            b_col_vec[col * (k as usize) + row] = b_owned[row * (n as usize) + col];
                        }
                    }
                    let mut grad_a_col_vec = vec![0f32; (m as usize) * (k as usize)];
                    unsafe {
                        cblas_sys::cblas_sgemm(
                            CBLAS_ORDER::CblasColMajor,
                            CBLAS_TRANSPOSE::CblasNoTrans,
                            CBLAS_TRANSPOSE::CblasTrans,
                            m,
                            k,
                            n,
                            1.0,
                            og_col_vec.as_ptr(),
                            m,
                            b_col_vec.as_ptr(),
                            k,
                            0.0,
                            grad_a_col_vec.as_mut_ptr(),
                            m,
                        );
                    }
                    // Convert column-major grad_a to row-major
                    for row in 0..(m as usize) {
                        for col in 0..(k as usize) {
                            grad_a_vec[row * (k as usize) + col] =
                                grad_a_col_vec[col * (m as usize) + row];
                        }
                    }
                }
                None => {
                    // fall back to ndarray if detection fails
                    let grad_a = output_grad.dot(&b.t()).into_dyn();
                    let grad_b = a.t().dot(&output_grad).into_dyn();
                    return vec![grad_a, grad_b];
                }
            }
            if cfg!(debug_assertions) {
                log::debug!(
                    "SGEMM backward grad_a params: m={}, k={}, n={}, lda={}, ldb={}, ldc={}",
                    m,
                    k,
                    n,
                    n,
                    n,
                    k
                );
            }
            unsafe {
                cblas_sys::cblas_sgemm(
                    CBLAS_ORDER::CblasRowMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    CBLAS_TRANSPOSE::CblasTrans,
                    m,
                    k,
                    n,
                    1.0,
                    og_slice.as_ptr(),
                    n,
                    b_slice.as_ptr(),
                    n, // since not transposed in memory b is k x n row-major but when transposed we use n
                    0.0,
                    grad_a_vec.as_mut_ptr(),
                    k,
                );
            }
            let grad_a = ArrayD::from_shape_vec(IxDyn(&[m as usize, k as usize]), grad_a_vec)
                .expect("Failed to create grad_a array");

            // grad_b = a.T @ og -> (k x m) @ (m x n) = (k x n)
            let mut grad_b_vec = vec![0f32; (k as usize) * (n as usize)];
            if cfg!(debug_assertions) {
                log::debug!(
                    "SGEMM backward grad_b params: k={}, n={}, m={}, lda={}, ldb={}, ldc={}",
                    k,
                    n,
                    m,
                    k,
                    n,
                    n
                );
            }
            let detected2 = detect_blas_order();
            match detected2 {
                Some(CBLAS_ORDER::CblasRowMajor) => unsafe {
                    cblas_sys::cblas_sgemm(
                        CBLAS_ORDER::CblasRowMajor,
                        CBLAS_TRANSPOSE::CblasTrans,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        k,
                        n,
                        m,
                        1.0,
                        a_slice.as_ptr(),
                        k,
                        og_slice.as_ptr(),
                        n,
                        0.0,
                        grad_b_vec.as_mut_ptr(),
                        n,
                    );
                },
                Some(CBLAS_ORDER::CblasColMajor) => {
                    // Build column-major buffers for a and og
                    let a_owned_vec = a_slice.to_vec();
                    let og_owned_vec = og_slice.to_vec();
                    let mut a_col_vec = vec![0f32; (k as usize) * (m as usize)];
                    for col in 0..(m as usize) {
                        for row in 0..(k as usize) {
                            a_col_vec[col * (k as usize) + row] =
                                a_owned_vec[row * (m as usize) + col];
                        }
                    }
                    let mut og_col_vec = vec![0f32; (m as usize) * (n as usize)];
                    for col in 0..(n as usize) {
                        for row in 0..(m as usize) {
                            og_col_vec[col * (m as usize) + row] =
                                og_owned_vec[row * (n as usize) + col];
                        }
                    }
                    let mut grad_b_col_vec = vec![0f32; (k as usize) * (n as usize)];
                    unsafe {
                        cblas_sys::cblas_sgemm(
                            CBLAS_ORDER::CblasColMajor,
                            CBLAS_TRANSPOSE::CblasTrans,
                            CBLAS_TRANSPOSE::CblasNoTrans,
                            k,
                            n,
                            m,
                            1.0,
                            a_col_vec.as_ptr(),
                            k,
                            og_col_vec.as_ptr(),
                            m,
                            0.0,
                            grad_b_col_vec.as_mut_ptr(),
                            k,
                        );
                    }
                    // convert col-major grad_b to row-major
                    for row in 0..(k as usize) {
                        for col in 0..(n as usize) {
                            grad_b_vec[row * (n as usize) + col] =
                                grad_b_col_vec[col * (k as usize) + row];
                        }
                    }
                }
                None => {
                    // already handled above; keep defensive fallback
                }
            }
            let grad_b = ArrayD::from_shape_vec(IxDyn(&[k as usize, n as usize]), grad_b_vec)
                .expect("Failed to create grad_b array");
            return vec![grad_a, grad_b];
        }
        #[cfg(any(not(feature = "openblas"), target_os = "windows"))]
        {
            // Fallback to ndarray-based computation for backward on platforms where BLAS may be unstable
            let grad_a = output_grad.dot(&b.t()).into_dyn();
            let grad_b = a.t().dot(&output_grad).into_dyn();
            return vec![grad_a, grad_b];
        }
        // NOTE: no-op - non-openblas or Windows fallback already handled above
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The ReLU activation function.
pub struct ReLU;

