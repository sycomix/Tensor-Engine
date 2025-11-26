#[cfg(all(feature = "openblas", unix))]
fn main() {
    use cblas_sys::{cblas_sgemm, CBLAS_ORDER, CBLAS_TRANSPOSE};
    use ndarray::arr2;

    // Simple 2x2 matrices
    let a = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[5.0f32, 6.0], [7.0, 8.0]]);

    let m = a.nrows() as i32;
    let k = a.ncols() as i32;
    let n = b.ncols() as i32;

    let a_owned = a.to_owned();
    let b_owned = b.to_owned();
    let a_slice = a_owned.as_slice().expect("A must be contiguous row-major");
    let b_slice = b_owned.as_slice().expect("B must be contiguous row-major");

    println!("MatMul dims: m={}, k={}, n={}", m, k, n);
    println!("lda={}, ldb={}, ldc={}", k, n, n);

    let mut c_vec = vec![0f32; (m as usize) * (n as usize)];

    unsafe {
        cblas_sgemm(
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
    }

    println!("Result (row-major vector): {:?}", c_vec);
    // Convert to 2D for display
    let c_matrix: Vec<Vec<f32>> = c_vec.chunks(n as usize).map(|row| row.to_vec()).collect();
    println!("Result matrix: {:?}", c_matrix);
}

#[cfg(any(not(feature = "openblas"), not(unix)))]
fn main() {
    println!("OpenBLAS feature not enabled; build with --features openblas to run this example.");
}
