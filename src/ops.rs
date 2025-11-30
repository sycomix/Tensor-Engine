use crate::tensor::Tensor;
#[cfg(all(feature = "openblas", not(target_os = "windows")))]
use cblas_sys::{self, CBLAS_ORDER, CBLAS_TRANSPOSE};
#[cfg(all(feature = "openblas", not(target_os = "windows")))]
use ndarray::Array2;
use ndarray::{s, ArrayD, ArrayView2, Axis, Ix2, IxDyn, SliceInfo, SliceInfoElem};
use rand::Rng;
use std::any::Any;
#[cfg(all(feature = "openblas", not(target_os = "windows")))]
use std::sync::OnceLock;

// Helper: reduce `grad` to `target_shape` by summing over broadcasted axes.
fn reduce_grad_to_shape(grad: &ArrayD<f32>, target_shape: &[usize]) -> ArrayD<f32> {
    // If shapes already equal, return clone
        if grad.shape() == target_shape {
        return grad.clone();
    }

    let mut res = grad.clone();
    let grad_ndim = res.ndim();
    let target_ndim = target_shape.len();
    // If grad has fewer dims than target, pad with ones on the left
    if grad_ndim < target_ndim {
        // reshape with leading ones
        let mut new_shape = vec![1; target_ndim - grad_ndim];
        new_shape.extend_from_slice(res.shape());
        res = match res.to_shape(IxDyn(&new_shape)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("reduce_grad_to_shape: Broadcast reshape failed: {}", e);
                return ArrayD::zeros(IxDyn(target_shape));
            }
        };
    }

    let grad_ndim = res.ndim();
    let dim_diff = grad_ndim as isize - target_ndim as isize;
    // Sum over axes where target dimension is 1 or axis doesn't exist in target
    for axis in (0..grad_ndim).rev() {
        let axis_idx = axis as isize;
        let target_dim = if axis_idx - dim_diff >= 0 {
            target_shape[(axis_idx - dim_diff) as usize]
        } else {
            1
        };
        if res.shape()[axis] != target_dim {
            // sum over axis
            res = res.sum_axis(Axis(axis));
        }
    }

    // Finally, reshape to the target_shape
    if res.shape() != target_shape {
        res = match res.to_shape(IxDyn(target_shape)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("reduce_grad_to_shape: Final reshape to target shape failed: {}", e);
                return ArrayD::zeros(IxDyn(target_shape));
            }
        };
    }
    res
}

// Helper: permute axes so that `axis` becomes the last axis.
fn permute_to_last(a: &ArrayD<f32>, axis: usize) -> (ArrayD<f32>, Option<Vec<usize>>) {
    let ndim = a.ndim();
    if axis == ndim - 1 {
        return (a.clone(), None);
    }
    let mut perm: Vec<usize> = (0..ndim).collect();
    let axis_val = perm.remove(axis);
    perm.push(axis_val);
    let permuted = a.view().permuted_axes(perm.clone()).to_owned();
    (permuted, Some(perm))
}

fn permute_back(a: ArrayD<f32>, perm: &Vec<usize>) -> ArrayD<f32> {
    // compute inverse permutation
    let ndim = perm.len();
    let mut inv = vec![0usize; ndim];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    a.view().permuted_axes(inv).to_owned()
}

/// A trait for operations that can be performed on tensors.
pub trait Operation: Send + Sync {
    /// Performs the forward pass of the operation.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The input tensors.
    /// * `output` - A mutable reference to the output tensor\'s data.
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>);

    /// Performs the backward pass of the operation.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The input tensors.
    /// * `output_grad` - The gradient of the output tensor.
    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>>;

    /// Returns the operation as a `&dyn Any`.
    fn as_any(&self) -> &dyn Any;
}

/// Reshape operation: changes tensor shape but keeps elements order
pub struct Reshape {
    pub shape: Vec<usize>,
}

impl Reshape {
    pub fn new(shape: Vec<usize>) -> Self {
        Reshape { shape }
    }
}

impl Operation for Reshape {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
           let a = inputs[0].to_f32_array();
        let a_clone = a.clone();
        match a_clone.to_shape(self.shape.clone()) {
            Ok(s) => *output = s.to_owned().into_dyn(),
            Err(e) => {
                log::error!("Reshape forward: invalid shape: {}", e);
                *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
                return;
            }
        }
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
           let in_shape = inputs[0].lock().storage.shape();
        let og_clone = output_grad.clone();
        let g = match og_clone.to_shape(IxDyn(&in_shape)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("Reshape backward: invalid shape: {}", e);
                return vec![ArrayD::zeros(IxDyn(&in_shape))];
            }
        };
        vec![g]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Sum operation: sums all elements to a scalar
pub struct Sum;

impl Operation for Sum {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
           let a = inputs[0].to_f32_array();
        let s = a.sum();
        *output = ArrayD::from_elem(IxDyn(&[]), s);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
           let a_shape = inputs[0].lock().storage.shape();
        // output_grad is scalar; expand to input shape
        let val = *output_grad
            .iter()
            .next()
            .expect("Expected scalar output_grad");
        let grad = ArrayD::from_elem(IxDyn(&a_shape), val);
        vec![grad]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Mean operation: computes mean over all elements to a scalar
pub struct Mean;

impl Operation for Mean {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
           let a = inputs[0].to_f32_array();
        let mean = a.sum() / (a.len() as f32);
        *output = ArrayD::from_elem(IxDyn(&[]), mean);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
           let a_shape = inputs[0].lock().storage.shape();
        let val = *output_grad
            .iter()
            .next()
            .expect("Expected scalar output_grad");
        let input_len = inputs[0].lock().storage.shape().iter().product::<usize>() as f32;
        let grad = ArrayD::from_elem(IxDyn(&a_shape), val / input_len);
        vec![grad]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The addition operation.
pub struct Add;

impl Operation for Add {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
           let a = inputs[0].to_f32_array();
           let b = inputs[1].to_f32_array();
        *output = a + b;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
           let a_shape = inputs[0].lock().storage.shape();
           let b_shape = inputs[1].lock().storage.shape();
        let grad_a = reduce_grad_to_shape(output_grad, &a_shape);
        let grad_b = reduce_grad_to_shape(output_grad, &b_shape);
        vec![grad_a, grad_b]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The multiplication operation.
pub struct Mul;

impl Operation for Mul {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
           let a = inputs[0].to_f32_array();
           let b = inputs[1].to_f32_array();
                *output = &a * &b;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
           let a = inputs[0].to_f32_array();
           let b = inputs[1].to_f32_array();
                let grad_a = (&b * output_grad).to_owned();
                let grad_b = (&a * output_grad).to_owned();
        let grad_a = reduce_grad_to_shape(&grad_a, a.shape());
        let grad_b = reduce_grad_to_shape(&grad_b, b.shape());
        vec![grad_a, grad_b]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The subtraction operation.
pub struct Sub;

impl Operation for Sub {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
           let a = inputs[0].to_f32_array();
           let b = inputs[1].to_f32_array();
        *output = a - b;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
           let a_shape = inputs[0].lock().storage.shape();
           let b_shape = inputs[1].lock().storage.shape();
        let grad_a = reduce_grad_to_shape(output_grad, &a_shape);
        let grad_b = reduce_grad_to_shape(&(-output_grad), &b_shape);
        vec![grad_a, grad_b]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The division operation.
pub struct Div;

impl Operation for Div {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
           let a = inputs[0].to_f32_array();
           let b = inputs[1].to_f32_array();
                *output = &a / &b;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
           let a = inputs[0].to_f32_array();
           let b = inputs[1].to_f32_array();
                let grad_a = (output_grad / &b).to_owned();
                let grad_b = (-&a * output_grad / (&b * &b)).to_owned();
        let grad_a = reduce_grad_to_shape(&grad_a, a.shape());
        let grad_b = reduce_grad_to_shape(&grad_b, b.shape());
        vec![grad_a, grad_b]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The power operation.
pub struct Pow(pub f32);

impl Operation for Pow {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
           let a = inputs[0].to_f32_array();
                *output = a.mapv(|x| x.powf(self.0));
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
           let a = inputs[0].to_f32_array();
                vec![(&*output_grad * a.mapv(|x| self.0 * x.powf(self.0 - 1.0))).to_owned()]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The matrix multiplication operation.
pub struct MatMul;
#[cfg(all(feature = "openblas", not(target_os = "windows")))]
static BLAS_ORDER_DETECTION: OnceLock<Option<CBLAS_ORDER>> = OnceLock::new();

#[cfg(all(feature = "openblas", not(target_os = "windows")))]
fn detect_blas_order() -> Option<CBLAS_ORDER> {
    if let Some(v) = BLAS_ORDER_DETECTION.get() {
        return *v;
    }
    // Try a small 2x2 matrix to detect BLAS expectations
    let a = match Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]) {
        Ok(v) => v,
        Err(e) => {
            log::error!("MatMul blas detection: Failed to create test 2x2 array 'a': {}", e);
            BLAS_ORDER_DETECTION.set(None).ok();
            return None;
        }
    };
    let b = match Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]) {
        Ok(v) => v,
        Err(e) => {
            log::error!("MatMul blas detection: Failed to create test 2x2 array 'b': {}", e);
            BLAS_ORDER_DETECTION.set(None).ok();
            return None;
        }
    };
    let expected = a.dot(&b);
    // RowMajor test
    let a_row = a.to_owned();
    let b_row = b.to_owned();
    let mut c_row = vec![0f32; 4];
    // Use contiguous slices if available, otherwise clone to make them contiguous for BLAS pointer use.
    let a_row_owned;
    let a_ptr = if let Some(slice) = a_row.as_slice() {
        slice.as_ptr()
    } else {
        log::warn!("MatMul blas detection: a_row not contiguous; cloning fallback");
        a_row_owned = a_row.clone();
        if let Some(s) = a_row_owned.as_slice() {
            s.as_ptr()
        } else {
            log::error!("MatMul blas detection: cloned a_row unexpectedly not contiguous; aborting detection");
            BLAS_ORDER_DETECTION.set(None).ok();
            return None;
        }
    };
    let b_row_owned;
    let b_ptr = if let Some(slice) = b_row.as_slice() {
        slice.as_ptr()
    } else {
        log::warn!("MatMul blas detection: b_row not contiguous; cloning fallback");
        b_row_owned = b_row.clone();
        if let Some(s) = b_row_owned.as_slice() {
            s.as_ptr()
        } else {
            log::error!("MatMul blas detection: cloned b_row unexpectedly not contiguous; aborting detection");
            BLAS_ORDER_DETECTION.set(None).ok();
            return None;
        }
    };
    unsafe {
        cblas_sys::cblas_sgemm(
            CBLAS_ORDER::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            2,
            2,
            2,
            1.0,
            a_ptr,
            2,
            b_ptr,
            2,
            0.0,
            c_row.as_mut_ptr(),
            2,
        );
    }
    let c_row_arr = match Array2::from_shape_vec((2, 2), c_row.clone()) {
        Ok(v) => v,
        Err(e) => {
            log::error!("MatMul blas detection: Failed to create C result array for RowMajor BLAS detection: {}", e);
            BLAS_ORDER_DETECTION.set(None).ok();
            return None;
        }
    };
    if c_row_arr == expected {
        BLAS_ORDER_DETECTION
            .set(Some(CBLAS_ORDER::CblasRowMajor))
            .ok();
        return Some(CBLAS_ORDER::CblasRowMajor);
    }
    // ColumnMajor test
    // Build column-major buffers
    let mut a_col_vec = vec![];
    for col in 0..2 {
        for row in 0..2 {
            a_col_vec.push(a[[row, col]]);
        }
    }
    let mut b_col_vec = vec![];
    for col in 0..2 {
        for row in 0..2 {
            b_col_vec.push(b[[row, col]]);
        }
    }
    let mut c_col_vec = vec![0f32; 4];
    unsafe {
        cblas_sys::cblas_sgemm(
            CBLAS_ORDER::CblasColMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            2,
            2,
            2,
            1.0,
            a_col_vec.as_ptr(),
            2,
            b_col_vec.as_ptr(),
            2,
            0.0,
            c_col_vec.as_mut_ptr(),
            2,
        );
    }
    // Convert column-major c_col_vec to row-major order
    let mut c_converted = vec![0f32; 4];
    for row in 0..2 {
        for col in 0..2 {
            c_converted[row * 2 + col] = c_col_vec[col * 2 + row];
        }
    }
    let c_col_arr = match Array2::from_shape_vec((2, 2), c_converted) {
        Ok(v) => v,
        Err(e) => {
            log::error!("MatMul blas detection: Failed to create C result array for ColumnMajor BLAS detection: {}", e);
            BLAS_ORDER_DETECTION.set(None).ok();
            return None;
        }
    };
    if c_col_arr == expected {
        BLAS_ORDER_DETECTION
            .set(Some(CBLAS_ORDER::CblasColMajor))
            .ok();
        return Some(CBLAS_ORDER::CblasColMajor);
    }
    BLAS_ORDER_DETECTION.set(None).ok();
    None
}

#[allow(dead_code)]
fn approx_eq_arrayd(a: &ArrayD<f32>, b: &ArrayD<f32>) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    let a_s = match a_slice {
        Some(s) => s,
        None => {
            log::error!("approx_eq_arrayd: left array is not contiguous, cannot compare");
            return false;
        }
    };
    let b_s = match b_slice {
        Some(s) => s,
        None => {
            log::error!("approx_eq_arrayd: right array is not contiguous, cannot compare");
            return false;
        }
    };
    for (x, y) in a_s.iter().zip(b_s.iter()) {
        if (x - y).abs() > 1e-5 {
            return false;
        }
    }
    true
}

impl Operation for MatMul {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a_lock = inputs[0].lock();
        let b_lock = inputs[1].lock();
            // Convert storage to f32 arrays for computation
            let a_owned = a_lock.storage.to_f32_array();
            let b_owned = b_lock.storage.to_f32_array();
            let a: ArrayView2<f32> = match a_owned
                .view()
                .into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        log::error!("MatMul forward: left operand is not 2D: {}", e);
                        *output = ArrayD::zeros(IxDyn(&[0]));
                        return;
                    }
            };
            let b: ArrayView2<f32> = match b_owned
                .view()
                .into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        log::error!("MatMul forward: right operand is not 2D: {}", e);
                        *output = ArrayD::zeros(IxDyn(&[0]));
                        return;
                    }
            };
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
            let a_slice = match a_owned.as_slice() {
                Some(s) => s,
                None => {
                    log::warn!("MatMul forward: a_owned not contiguous; falling back to ndarray dot");
                    *output = a_owned.dot(&b_owned).into_dyn();
                    return;
                }
            };
            let b_slice = match b_owned.as_slice() {
                Some(s) => s,
                None => {
                    log::warn!("MatMul forward: b_owned not contiguous; falling back to ndarray dot");
                    *output = a_owned.dot(&b_owned).into_dyn();
                    return;
                }
            };
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
            *output = match ArrayD::from_shape_vec(IxDyn(&[m as usize, n as usize]), c_vec.clone()) {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!("MatMul forward: Failed to create matmul output array: {}", e);
                    *output = a_owned.dot(&b_owned).into_dyn();
                    return;
                }
            };

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
                let cm = match ArrayD::from_shape_vec(IxDyn(&[m as usize, n as usize]), c_from_col) {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!("MatMul forward: Failed to create matmul output array from column-major conversion: {}", e);
                        *output = a_owned.dot(&b_owned).into_dyn();
                        return;
                    }
                };
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
        let a_owned = a_lock.storage.to_f32_array();
        let b_owned = b_lock.storage.to_f32_array();
        let a: ArrayView2<f32> = match a_owned
            .view()
            .into_dimensionality::<Ix2>() {
                Ok(v) => v,
                Err(e) => {
                    log::error!("MatMul backward: left operand is not 2D: {}", e);
                    let grad_a = ArrayD::zeros(IxDyn(&[0]));
                    let grad_b = ArrayD::zeros(IxDyn(&[0]));
                    return vec![grad_a, grad_b];
                }
            };
        let b: ArrayView2<f32> = match b_owned
            .view()
            .into_dimensionality::<Ix2>() {
                Ok(v) => v,
                Err(e) => {
                    log::error!("MatMul backward: right operand is not 2D: {}", e);
                    let grad_a = ArrayD::zeros(IxDyn(&[0]));
                    let grad_b = ArrayD::zeros(IxDyn(&[0]));
                    return vec![grad_a, grad_b];
                }
            };
        let output_grad: ArrayView2<f32> = match output_grad.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("MatMul backward: output_grad is not 2D: {}", e);
                let grad_a = ArrayD::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::zeros(IxDyn(&[0]));
                return vec![grad_a, grad_b];
            }
        };

        #[cfg(all(feature = "openblas", not(target_os = "windows")))]
        {
            let og: ArrayView2<f32> = match output_grad.view().into_dimensionality::<Ix2>() {
                Ok(v) => v,
                Err(e) => {
                    log::error!("MatMul backward: output_grad is not 2D: {}", e);
                    let grad_a = ArrayD::zeros(IxDyn(&[0]));
                    let grad_b = ArrayD::zeros(IxDyn(&[0]));
                    return vec![grad_a, grad_b];
                }
            };
            // Because cblas requires contiguous row-major memory, make owned copies
            let og_owned = og.to_owned();
            let og_slice = match og_owned.as_slice() {
                Some(s) => s,
                None => {
                    log::warn!("MatMul backward: output_grad not contiguous; falling back to ndarray path");
                    let grad_a = output_grad.dot(&b.t()).into_dyn();
                    let grad_b = a.t().dot(&output_grad).into_dyn();
                    return vec![grad_a, grad_b];
                }
            };
            let a_owned = a.to_owned();
            let b_owned = b.to_owned();
            let a_slice = match a_owned.as_slice() {
                Some(s) => s,
                None => {
                    log::warn!("MatMul backward: a not contiguous; fallback to ndarray path");
                    let grad_a = output_grad.dot(&b.t()).into_dyn();
                    let grad_b = a.t().dot(&output_grad).into_dyn();
                    return vec![grad_a, grad_b];
                }
            };
            let b_slice = match b_owned.as_slice() {
                Some(s) => s,
                None => {
                    log::warn!("MatMul backward: b not contiguous; fallback to ndarray path");
                    let grad_a = output_grad.dot(&b.t()).into_dyn();
                    let grad_b = a.t().dot(&output_grad).into_dyn();
                    return vec![grad_a, grad_b];
                }
            };
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
            let grad_a = match ArrayD::from_shape_vec(IxDyn(&[m as usize, k as usize]), grad_a_vec) {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!("MatMul backward: Failed to create grad_a array: {}", e);
                    let grad_a = output_grad.dot(&b.t()).into_dyn();
                    let grad_b = a.t().dot(&output_grad).into_dyn();
                    return vec![grad_a, grad_b];
                }
            };

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
            let grad_b = match ArrayD::from_shape_vec(IxDyn(&[k as usize, n as usize]), grad_b_vec) {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!("MatMul backward: Failed to create grad_b array: {}", e);
                    let grad_a = ArrayD::from_elem(IxDyn(&[m as usize, k as usize]), f32::NAN);
                    let grad_b = output_grad.clone();
                    return vec![grad_a, grad_b];
                }
            };
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

/// Ternary quantization operation projecting inputs to {-1, 0, 1} with STE backward.
pub struct Ternary;

impl Operation for Ternary {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        log::debug!("[Ternary] forward start");
        let a = inputs[0].to_f32_array();
        let eps = 1e-6f32;
        let mean_abs = a.mapv(|x| x.abs()).sum() / (a.len() as f32);
        log::debug!("[Ternary] mean_abs = {}", mean_abs);
        let scale = mean_abs + eps;
        let a_scaled = a.mapv(|x| x / scale);
        let rounded = a_scaled.mapv(|x| x.round().max(-1.0).min(1.0));
        *output = rounded.mapv(|x| x * mean_abs);
        log::debug!("[Ternary] forward done");
    }

    fn backward(&self, _inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        log::debug!("[Ternary] backward called");
        // Straight-through estimator: pass gradients unchanged
        vec![output_grad.clone()]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The ReLU activation function.
pub struct ReLU;

impl Operation for ReLU {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        *output = a.mapv(|x| x.max(0.0));
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        vec![output_grad * a.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The sigmoid activation function.
pub struct Sigmoid;

impl Operation for Sigmoid {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        *output = a.mapv(|x| 1.0 / (1.0 + (-x).exp()));
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let sigmoid_a = a.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        vec![output_grad * (sigmoid_a.clone() * (1.0 - sigmoid_a))]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The tanh activation function.
pub struct Tanh;

impl Operation for Tanh {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        *output = a.mapv(|x| x.tanh());
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let tanh_a = a.mapv(|x| x.tanh());
        vec![output_grad * (1.0 - tanh_a.mapv(|x| x.powi(2)))]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The natural logarithm operation element-wise
pub struct Log;

impl Operation for Log {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        *output = a.mapv(|x| x.ln());
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        // d/dx ln(x) = 1/x
        vec![output_grad / a]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// LogSoftmax operation (stable): computes log(softmax(x)) along axis
pub struct LogSoftmax {
    pub axis: usize,
}

impl LogSoftmax {
    pub fn new(axis: usize) -> Self {
        LogSoftmax { axis }
    }
}

impl Operation for LogSoftmax {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        let axis = if self.axis >= x.ndim() {
            x.ndim() - 1
        } else {
            self.axis
        };
        // stable log-softmax: x - logsumexp(x)
        // permute the axis to the last axis then operate on that axis
        let (mut out, perm_opt) = permute_to_last(&x, axis);
        let last_axis = out.ndim() - 1;
        for mut lane in out.lanes_mut(Axis(last_axis)) {
            let max = lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in lane.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            let logsum = sum.ln();
            for v in lane.iter_mut() {
                *v = (*v).ln() - logsum; // This is (v - max).ln() - logsum; actually we want log(exp(x-max)/sum) = (x-max) - ln(sum)
            }
        }
        // permute back if necessary
        if let Some(ref perm) = perm_opt {
            *output = permute_back(out, perm);
        } else {
            *output = out;
        }
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        let axis = if self.axis >= x.ndim() {
            x.ndim() - 1
        } else {
            self.axis
        };
        let (mut s, perm_opt) = permute_to_last(&output_grad, axis);
        let last_axis = s.ndim() - 1;
        // compute softmax from x
        for mut lane in s.lanes_mut(Axis(last_axis)) {
            let max = lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in lane.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in lane.iter_mut() {
                *v = *v / sum;
            }
            // Debug: verify normalization sums to 1.0
            // eprintln!("softmax row normalized sum: {}", lane.iter().sum::<f32>());
        }
        // grad_input = grad_output - softmax * sum(grad_output) along axis
        let (p_output_grad, _) = permute_to_last(&output_grad, axis);
        let mut grad_in = p_output_grad.clone();
        for ((mut g_lane, s_lane), og_lane) in grad_in
            .lanes_mut(Axis(last_axis))
            .into_iter()
            .zip(s.lanes(Axis(last_axis)).into_iter())
            .zip(p_output_grad.lanes(Axis(last_axis)).into_iter())
        {
            let mut sum = 0.0f32;
            for v in og_lane.iter() {
                sum += *v;
            }
            for (gi, &si) in g_lane.iter_mut().zip(s_lane.iter()) {
                *gi = *gi - si * sum;
            }
        }
        if let Some(ref perm) = perm_opt {
            vec![permute_back(grad_in, perm)]
        } else {
            vec![grad_in]
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Softmax operation (numerically stable), forward and backward on axis
pub struct Softmax {
    pub axis: usize,
}

impl Softmax {
    pub fn new(axis: usize) -> Self {
        Softmax { axis }
    }
}

impl Operation for Softmax {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        let axis = if self.axis >= x.ndim() {
            x.ndim() - 1
        } else {
            self.axis
        };
        // permute axis to last and compute softmax on last axis
        let (mut out, perm_opt) = permute_to_last(&x, axis);
        let last_axis = out.ndim() - 1;
        for mut lane in out.lanes_mut(Axis(last_axis)) {
            let max = lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in lane.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in lane.iter_mut() {
                *v = *v / sum;
            }
        }
        if let Some(ref perm) = perm_opt {
            *output = permute_back(out, perm);
        } else {
            *output = out;
        }
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        let axis = if self.axis >= x.ndim() {
            x.ndim() - 1
        } else {
            self.axis
        };
        // compute softmax y first, on permuted axis
        let (mut y, perm_opt) = permute_to_last(&x, axis);
        let last_axis = y.ndim() - 1;
        for mut lane in y.axis_iter_mut(Axis(last_axis)) {
            let max = lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in lane.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in lane.iter_mut() {
                *v = *v / sum;
            }
        }
        // grad = y * (grad_out - sum(grad_out * y))
        let (p_output_grad, _) = permute_to_last(&output_grad, axis);
        let mut grad_in = p_output_grad.clone();
        for ((mut g_lane, y_lane), og_lane) in grad_in
            .lanes_mut(Axis(last_axis))
            .into_iter()
            .zip(y.lanes(Axis(last_axis)).into_iter())
            .zip(p_output_grad.lanes(Axis(last_axis)).into_iter())
        {
            let mut s = 0.0f32;
            for (og, &yy) in og_lane.iter().zip(y_lane.iter()) {
                s += og * yy;
            }
            for (gi, &yy) in g_lane.iter_mut().zip(y_lane.iter()) {
                *gi = yy * (*gi - s);
            }
        }
        if let Some(ref perm) = perm_opt {
            vec![permute_back(grad_in, perm)]
        } else {
            vec![grad_in]
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Cross-entropy with logits operation (numerically stable using log-softmax)
/// Inputs: logits (N, C) and targets (either 1D class indices (N) or 2D one-hot (N, C))
pub struct CrossEntropyLogits {
    pub axis: usize,
}

/// Layer Normalization: normalizes across the last axis (or given axis) with learnable gain (gamma) and bias (beta).
///
/// # Expected inputs and parameters
///
/// - `inputs[0]` (x): the input tensor to normalize. The operation normalizes across the given `axis` (default: last axis).
/// - `inputs[1]` (gamma): per-feature learnable gain. Must be either shape `[features]` (1D) or a broadcastable shape to the last axis.
/// - `inputs[2]` (beta): per-feature learnable bias. Must be either shape `[features]` (1D) or a broadcastable shape to the last axis.
///
/// The operation computes normalized = (x - mean) / sqrt(var + eps) per row (where `row` means everything except the `axis`)
/// and applies `y = normalized * gamma + beta`. The `gamma` and `beta` parameters are applied per-feature along the axis.
pub struct LayerNorm {
    pub axis: usize,
    pub eps: f32,
    // Cache normalized values and inv_std per row for backward.
    cache: std::sync::Mutex<Option<(ArrayD<f32>, ArrayD<f32>)>>,
}

impl LayerNorm {
    pub fn new(axis: usize, eps: f32) -> Self {
        LayerNorm {
            axis,
            eps,
            cache: std::sync::Mutex::new(None),
        }
    }
}

impl Operation for LayerNorm {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        let gamma = inputs[1].to_f32_array();
        let beta = inputs[2].to_f32_array();
        let axis = if self.axis >= x.ndim() {
            x.ndim() - 1
        } else {
            self.axis
        };
        let (xp, perm_opt) = permute_to_last(&x, axis);
        let shape = xp.shape().to_vec();
        let ndim = xp.ndim();
        let nrows = shape.iter().take(ndim - 1).product::<usize>();
        let features = shape[ndim - 1];
        // reshape to 2D
        let x2 = xp
            .to_shape((nrows, features))
            .expect("Reshape to 2D for LayerNorm failed")
            .to_owned();

        // compute per-row mean and var
        let mut normalized = x2.clone();
        let mut inv_std = ArrayD::zeros(IxDyn(&[nrows, 1]));
        for (mut row, i) in normalized.rows_mut().into_iter().zip(0..nrows) {
            let mean = row.mean().unwrap_or_else(|| {
                log::error!("LayerNorm forward: encountered empty row while computing mean; defaulting to 0.0");
                0.0f32
            });
            // compute variance
            let mut var = 0.0f32;
            for v in row.iter() {
                var += (*v - mean) * (*v - mean);
            }
            var /= features as f32;
            let is = 1.0 / (var + self.eps).sqrt();
            for v in row.iter_mut() {
                *v = (*v - mean) * is;
            }
            inv_std[[i, 0]] = is;
        }

        // apply gamma and beta: gamma and beta expected shape [features] or broadcast
        let mut out2 = normalized.clone();
        // broadcast gamma/beta per row
        for (mut row, _) in out2.rows_mut().into_iter().zip(0..nrows) {
            for (j, v) in row.iter_mut().enumerate() {
                let g = if gamma.ndim() == 1 {
                    if let Some(slice) = gamma.as_slice() {
                        slice[j]
                    } else {
                        gamma[[j]]
                    }
                } else {
                    gamma[[j]]
                };
                let b = if beta.ndim() == 1 {
                    if let Some(slice) = beta.as_slice() {
                        slice[j]
                    } else {
                        beta[[j]]
                    }
                } else {
                    beta[[j]]
                };
                *v = *v * g + b;
            }
        }

        // store normalized and inv_std in cache for backward
        let mut lock = match self.cache.lock() {
            Ok(l) => l,
            Err(poisoned) => {
                log::error!("Failed to acquire LayerNorm cache lock: {:?}", poisoned);
                // If we cannot acquire the cache lock, avoid panicking: leave cache unchanged and proceed without caching.
                // While caching is disabled, continue forward but do not store cache.
                // Note: we return early if necessary by not attempting to write to the cache.
                // We cannot proceed with cache write; return early, skipping cache write.
                // No-op: allow function to continue without caching
                // Using `None` here; just continue
                return;
            }
        };
        *lock = Some((normalized.into_dyn(), inv_std.into_dyn()));

        // reshape back and permute back
        let out_perm = match out2.into_dyn().to_shape(IxDyn(&shape)) {
            Ok(o) => o.to_owned(),
            Err(e) => {
                log::error!("LayerNorm forward reshape back failed: {}", e);
                return;
            }
        };
        if let Some(ref perm) = perm_opt {
            *output = permute_back(out_perm, perm);
        } else {
            *output = out_perm;
        }
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // inputs: x, gamma, beta
        let x = inputs[0].to_f32_array();
        let gamma = inputs[1].to_f32_array();
        let _beta = inputs[2].to_f32_array(); // not used in grad
        let axis = if self.axis >= x.ndim() {
            x.ndim() - 1
        } else {
            self.axis
        };
        let (xp, perm_opt) = permute_to_last(&x, axis);
        let shape = xp.shape().to_vec();
        let ndim = xp.ndim();
        let nrows = shape.iter().take(ndim - 1).product::<usize>();
        let features = shape[ndim - 1];
        // reshape output_grad as well
        let og_perm = output_grad
            .to_shape((nrows, features))
            .expect("Reshape og to 2D failed")
            .to_owned();

        // fetch cache
        let lock = match self.cache.lock() {
            Ok(l) => l,
            Err(poisoned) => {
                log::error!("Failed to acquire LayerNorm cache lock: {:?}", poisoned);
                // Return zero gradients if we cannot access cached values; this avoids panicking.
                let grad_x = ArrayD::zeros(IxDyn(&shape));
                let grad_gamma = ArrayD::zeros(IxDyn(&[features]));
                let grad_beta = ArrayD::zeros(IxDyn(&[features]));
                return vec![grad_x, grad_gamma, grad_beta];
            }
        };
        let (normalized, inv_std) = if let Some((ref n, ref i)) = *lock {
            (n.clone(), i.clone())
        } else {
            log::error!("LayerNorm backward called without forward cache — forward cache missing. Returning zero grads.");
            let grad_x = ArrayD::zeros(IxDyn(&shape));
            let grad_gamma = ArrayD::zeros(IxDyn(&[features]));
            let grad_beta = ArrayD::zeros(IxDyn(&[features]));
            return vec![grad_x, grad_gamma, grad_beta];
        };
        let normalized2 = normalized
            .to_shape((nrows, features))
            .expect("Reshape normalized 2D failed");
        let inv2 = inv_std
            .to_shape((nrows, 1))
            .expect("Reshape inv std 2D failed");

        // grad w.r.t gamma and beta
        let mut grad_gamma = ArrayD::zeros(IxDyn(&[features]));
        let mut grad_beta = ArrayD::zeros(IxDyn(&[features]));
        for j in 0..features {
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            for irow in 0..nrows {
                let dop = og_perm[[irow, j]];
                let norm = normalized2[[irow, j]];
                sum_g += dop * norm;
                sum_b += dop;
            }
            grad_gamma[[j]] = sum_g;
            grad_beta[[j]] = sum_b;
        }

        // grad w.r.t input
        let mut grad_x2 = ArrayD::zeros(IxDyn(&[nrows, features]));
        for irow in 0..nrows {
            // compute per-row mean1 and mean2
            let mut mean1 = 0.0f32;
            let mut mean2 = 0.0f32;
            for j in 0..features {
                let g = og_perm[[irow, j]];
                let gam = if gamma.ndim() == 1 {
                    if let Some(slice) = gamma.as_slice() {
                        slice[j]
                    } else {
                        gamma[[j]]
                    }
                } else {
                    gamma[[j]]
                };
                let dnormalized = g * gam;
                mean1 += dnormalized;
                mean2 += dnormalized * normalized2[[irow, j]];
            }
            mean1 /= features as f32;
            mean2 /= features as f32;
            let inv = inv2[[irow, 0]];
            for j in 0..features {
                let dnormalized = og_perm[[irow, j]]
                    * if gamma.ndim() == 1 {
                        if let Some(slice) = gamma.as_slice() {
                            slice[j]
                        } else {
                            gamma[[j]]
                        }
                    } else {
                        gamma[[j]]
                    };
                let norm = normalized2[[irow, j]];
                let val = inv * (dnormalized - mean1 - norm * mean2);
                grad_x2[[irow, j]] = val;
            }
        }

        // reshape back and permute back
        let grad_x_perm = grad_x2
            .into_dyn()
            .to_shape(IxDyn(&shape))
            .expect("Reshape grad back failed")
            .to_owned();
        let grad_x = if let Some(ref perm) = perm_opt {
            permute_back(grad_x_perm, perm)
        } else {
            grad_x_perm
        };
        vec![grad_x, grad_gamma, grad_beta]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl CrossEntropyLogits {
    pub fn new(axis: usize) -> Self {
        CrossEntropyLogits { axis }
    }
}

impl Operation for CrossEntropyLogits {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let logits = inputs[0].to_f32_array();
        let targets = inputs[1].to_f32_array();
        let axis = if self.axis >= logits.ndim() {
            logits.ndim() - 1
        } else {
            self.axis
        };
        // Permute logits so the class axis becomes the last axis, and reshape to (nrows, classes)
        let (permuted_logits, perm_opt) = permute_to_last(&logits, axis);
        // `perm_opt` is used directly below; no need to clone into an unused variable
        let shape = permuted_logits.shape().to_vec();
        let ndim = permuted_logits.ndim();
        let nrows = shape.iter().take(ndim - 1).product::<usize>();
        let classes = shape[ndim - 1];
        let logits_2d = match permuted_logits.to_shape((nrows, classes)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("CrossEntropyLogits forward: Reshape to 2D logits failed: {}", e);
                return;
            }
        };

        // Determine target format: index vector 1D with len nrows, or one-hot with same shape as logits
        let mut per_sample = Vec::new();
        if targets.ndim() == 1 && targets.shape()[0] == nrows {
            // integer class indices in float representation
            for i in 0..nrows {
                // compute log-softmax for row i: logp = logits[i,j] - logsumexp(row)
                let row = logits_2d.row(i);
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter() {
                    sum += (v - max).exp();
                }
                let logsum = sum.ln();
                let j = targets[[i]] as usize;
                let logprob = logits_2d[[i, j]] - max - logsum;
                per_sample.push(-logprob);
            }
        } else if targets.ndim() == logits.ndim() {
            // assume one-hot of same shape as logits; permute targets similarly if needed
            let perm_targets = if let Some(ref permv) = perm_opt {
                targets.view().permuted_axes(permv.clone()).to_owned()
            } else {
                targets.clone()
            };
            let t_2d = perm_targets
                .to_shape((nrows, classes))
                .expect("Reshape targets one-hot failed")
                .to_owned();
            for i in 0..nrows {
                let mut acc = 0.0f32;
                for j in 0..classes {
                    acc += t_2d[[i, j]] * logits_2d[[i, j]];
                }
                // subtract logsum via logsumexp
                let max = logits_2d.row(i).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;
                for j in 0..classes {
                    sum += (logits_2d[[i, j]] - max).exp();
                }
                let logsum = sum.ln();
                per_sample.push(-(acc - logsum));
            }
        } else {
            log::error!("CrossEntropyLogits: target shape incompatible with logits and axis; logits shape: {:?}, targets shape: {:?}, axis: {}",
                logits.shape(), targets.shape(), axis);
            // Set output to NaN to indicate invalid computation
            *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
            return;
        }
        // average
        let mean = per_sample.iter().sum::<f32>() / (per_sample.len() as f32);
        *output = ArrayD::from_elem(IxDyn(&[]), mean);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let logits = inputs[0].to_f32_array();
        let targets = inputs[1].to_f32_array();
        let axis = if self.axis >= logits.ndim() {
            logits.ndim() - 1
        } else {
            self.axis
        };
        // permute and reshape logits into (nrows, classes)
        let (permuted_logits, perm_opt) = permute_to_last(&logits, axis);
        // `perm_opt` is used directly below; no need to clone into an unused variable
        let shape = permuted_logits.shape().to_vec();
        let ndim = permuted_logits.ndim();
        let nrows = shape.iter().take(ndim - 1).product::<usize>();
        let classes = shape[ndim - 1];
        let logits_2d = match permuted_logits.to_shape((nrows, classes)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("SoftmaxCrossEntropy forward: Reshape permuted logits failed: {}", e);
                let grad_logits = ArrayD::zeros(logits.dim());
                let grad_targets = ArrayD::zeros(targets.dim());
                return vec![grad_logits, grad_targets];
            }
        };
        // compute soft
        let mut soft = logits_2d.clone();
        for mut row in soft.rows_mut() {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v = *v / sum;
            }
        }
        // compute grad in 2D then reshape back and permute back
        let og = output_grad.iter().next().copied().unwrap_or_else(||{
            log::error!("SoftmaxCrossEntropy backward: expected scalar output_grad, defaulting to 1.0");
            1.0f32
        });
        let grad_logits_2d = ArrayD::zeros(IxDyn(&[nrows, classes]));
        let mut grad_view = grad_logits_2d
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        if targets.ndim() == 1 && targets.shape()[0] == nrows {
            // Use safe indexing into targets; if non-contiguous, accessing via index still works.
            for i in 0..nrows {
                for j in 0..classes {
                    grad_view[[i, j]] = soft[[i, j]];
                }
                let idx = targets[[i]] as usize;
                grad_view[[i, idx]] -= 1.0;
                for j in 0..classes {
                    grad_view[[i, j]] *= og / (nrows as f32);
                }
            }
        } else if targets.ndim() == logits.ndim() {
            let perm_targets = if let Some(ref permv) = perm_opt {
                targets.view().permuted_axes(permv.clone()).to_owned()
            } else {
                targets.clone()
            };
            let t_2d = match perm_targets.to_shape((nrows, classes)) {
                Ok(v) => v,
                Err(e) => {
                    log::error!("CrossEntropyLogits backward: Reshape targets one-hot failed: {}", e);
                    let grad_logits = ArrayD::zeros(logits.dim());
                    let grad_targets = ArrayD::zeros(targets.dim());
                    return vec![grad_logits, grad_targets];
                }
            };
            for i in 0..nrows {
                for j in 0..classes {
                    grad_view[[i, j]] = (soft[[i, j]] - t_2d[[i, j]]) * og / (nrows as f32);
                }
            }
        } else {
            log::error!("CrossEntropyLogits backward: target shape incompatible; logits shape: {:?}, targets shape: {:?}, axis: {}",
                logits.shape(), targets.shape(), axis);
            let grad_logits = ArrayD::zeros(logits.dim());
            let grad_targets = ArrayD::zeros(targets.dim());
            return vec![grad_logits, grad_targets];
        }
        let grad_permuted = match grad_view.into_dyn().to_shape(IxDyn(&shape)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("CrossEntropyLogits backward: Reshape back failed: {}", e);
                let grad_logits = ArrayD::zeros(logits.dim());
                let grad_targets = ArrayD::zeros(targets.dim());
                return vec![grad_logits, grad_targets];
            }
        };
        let grad_logits = if let Some(ref permv) = perm_opt {
            permute_back(grad_permuted, permv)
        } else {
            grad_permuted
        };
        let grad_targets = ArrayD::zeros(targets.dim());
        vec![grad_logits, grad_targets]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Combined Softmax + CrossEntropy op for logits - avoids extra allocation and is numerically stable.
/// Inputs: logits (N, C), targets: 1D labels or 2D one-hot.
pub struct SoftmaxCrossEntropyLogits {
    pub axis: usize,
}

/// NLLLoss (Negative Log Likelihood Loss) for logits in log-space (expects log_probs).
/// Targets are 1D integer labels stored as floats (one label per row) or 2D one-hot vectors.
pub struct NLLLoss;

impl NLLLoss {
    pub fn new() -> Self {
        NLLLoss
    }
}

impl Operation for NLLLoss {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let log_probs = inputs[0].to_f32_array();
        let targets = inputs[1].to_f32_array();
        if log_probs.ndim() < 1 {
            log::error!("NLLLoss: log_probs must be at least 1D; got shape: {:?}", log_probs.shape());
            *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
            return;
        }
        // Permute log_probs to bring class axis to last
        let axis = log_probs.ndim() - 1;
        let (permuted, perm_opt) = permute_to_last(&log_probs, axis);
        let shape = permuted.shape().to_vec();
        let ndim = permuted.ndim();
        let nrows = shape.iter().take(ndim - 1).product::<usize>();
        let classes = shape[ndim - 1];
        let lp_2d = match permuted.to_shape((nrows, classes)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("NLLLoss forward: Reshape log_probs to 2D failed: {}", e);
                *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
                return;
            }
        };
        let mut total = 0.0f32;
        if targets.ndim() == 1 && targets.shape()[0] == nrows {
            for i in 0..nrows {
                let idx = targets[[i]] as usize;
                total += -lp_2d[[i, idx]];
            }
        } else if targets.ndim() == log_probs.ndim() {
            let perm_targets = if let Some(ref permv) = perm_opt {
                targets.view().permuted_axes(permv.clone()).to_owned()
            } else {
                targets.clone()
            };
            let t_2d = match perm_targets.to_shape((nrows, classes)) {
                Ok(v) => v,
                Err(e) => {
                    log::error!("NLLLoss forward: Reshape targets one-hot failed: {}", e);
                    *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
                    return;
                }
            };
            for i in 0..nrows {
                for j in 0..classes {
                    total += -lp_2d[[i, j]] * t_2d[[i, j]];
                }
            }
        } else {
            log::error!("NLLLoss: targets shape incompatible; log_probs shape: {:?}, targets shape: {:?}", log_probs.shape(), targets.shape());
            *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
            return;
        }
        *output = ArrayD::from_elem(IxDyn(&[]), total / (nrows as f32));
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let log_probs = inputs[0].to_f32_array();
        let targets = inputs[1].to_f32_array();
        let axis = log_probs.ndim() - 1;
        let (permuted, perm_opt) = permute_to_last(&log_probs, axis);
        let shape = permuted.shape().to_vec();
        let ndim = permuted.ndim();
        let nrows = shape.iter().take(ndim - 1).product::<usize>();
        let classes = shape[ndim - 1];
        let og = output_grad.iter().next().copied().unwrap_or_else(|| {
            log::error!("NLLLoss backward: expected scalar output_grad, defaulting to 1.0");
            1.0f32
        });
        let grad_2d = ArrayD::zeros(IxDyn(&[nrows, classes]));
        let mut grad_view = match grad_2d.into_dimensionality::<ndarray::Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("NLLLoss backward: failed to convert grad to 2D: {}", e);
                let shape0 = inputs[0].lock().storage.shape().to_vec();
                let shape1 = inputs[1].lock().storage.shape().to_vec();
                let grad_logits = ArrayD::zeros(IxDyn(&shape0));
                let grad_targets = ArrayD::zeros(IxDyn(&shape1));
                return vec![grad_logits, grad_targets];
            }
        };
        if targets.ndim() == 1 && targets.shape()[0] == nrows {
            for i in 0..nrows {
                let idx = targets[[i]] as usize;
                grad_view[[i, idx]] = -og / (nrows as f32);
            }
        } else {
            let perm_targets = if let Some(ref permv) = perm_opt {
                targets.view().permuted_axes(permv.clone()).to_owned()
            } else {
                targets.clone()
            };
            let t_2d = match perm_targets.to_shape((nrows, classes)) {
                Ok(v) => v,
                Err(e) => {
                    log::error!("NLLLoss backward: Reshape targets one-hot failed: {}", e);
                    let shape0 = inputs[0].lock().storage.shape().to_vec();
                    let shape1 = inputs[1].lock().storage.shape().to_vec();
                    let grad_logits = ArrayD::zeros(IxDyn(&shape0));
                    let grad_targets = ArrayD::zeros(IxDyn(&shape1));
                    return vec![grad_logits, grad_targets];
                }
            };
            for i in 0..nrows {
                for j in 0..classes {
                    grad_view[[i, j]] = -t_2d[[i, j]] * og / (nrows as f32);
                }
            }
        }
        // targets are non-differentiable
        let grad_permuted = match grad_view.into_dyn().to_shape(IxDyn(&shape)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("NLLLoss backward: Reshape grad failed: {}", e);
                let shape0 = inputs[0].lock().storage.shape().to_vec();
                let grad_logits = ArrayD::zeros(IxDyn(&shape0));
                let shape1 = inputs[1].lock().storage.shape().to_vec();
                let grad_targets = ArrayD::zeros(IxDyn(&shape1));
                return vec![grad_logits, grad_targets];
            }
        };
        let grad_logprobs = if let Some(ref permv) = perm_opt {
            permute_back(grad_permuted, permv)
        } else {
            grad_permuted
        };
        let targets_grad = ArrayD::zeros(targets.dim());
        vec![grad_logprobs, targets_grad]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl SoftmaxCrossEntropyLogits {
    pub fn new(axis: usize) -> Self {
        SoftmaxCrossEntropyLogits { axis }
    }
}

impl Operation for SoftmaxCrossEntropyLogits {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let logits = inputs[0].to_f32_array();
        let targets = inputs[1].to_f32_array();
        let axis = if self.axis >= logits.ndim() {
            logits.ndim() - 1
        } else {
            self.axis
        };
        // Permute logits to move class axis to last and reshape to (nrows, classes)
        let (permuted_logits, perm_opt) = permute_to_last(&logits, axis);
        // `perm_opt` is used directly below; no need to clone into an unused variable
        let shape = permuted_logits.shape().to_vec();
        let ndim = permuted_logits.ndim();
        let nrows = shape.iter().take(ndim - 1).product::<usize>();
        let classes = shape[ndim - 1];
        let logits_2d = match permuted_logits.to_shape((nrows, classes)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!(
                    "SoftmaxCrossEntropyLogits forward: Reshape logits to 2D failed: {}",
                    e
                );
                // set output to NaN to indicate invalid computation and return
                *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
                return;
            }
        };
        let mut loss_sum = 0.0f32;
        if targets.ndim() == 1 && targets.shape()[0] == nrows {
            for i in 0..nrows {
                let max = logits_2d.row(i).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;
                for j in 0..classes {
                    sum += (logits_2d[[i, j]] - max).exp();
                }
                let logsum = sum.ln();
                let j = targets[[i]] as usize;
                let logprob = logits_2d[[i, j]] - max - logsum;
                loss_sum += -logprob;
            }
        } else if targets.ndim() == logits.ndim() {
            let perm_targets = if let Some(ref permv) = perm_opt {
                targets.view().permuted_axes(permv.clone()).to_owned()
            } else {
                targets.clone()
            };
            let t_2d = perm_targets
                .to_shape((nrows, classes))
                .expect("Reshape targets one-hot failed")
                .to_owned();
            for i in 0..nrows {
                let max = logits_2d.row(i).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;
                for j in 0..classes {
                    sum += (logits_2d[[i, j]] - max).exp();
                }
                let logsum = sum.ln();
                let mut acc = 0.0f32;
                for j in 0..classes {
                    acc += t_2d[[i, j]] * (logits_2d[[i, j]] - max - logsum);
                }
                loss_sum += -acc;
            }
        } else {
            log::error!(
                "SoftmaxCrossEntropyLogits: target shape incompatible with logits and axis; logits shape: {:?}, targets shape: {:?}, axis: {}",
                logits.shape(), targets.shape(), axis
            );
            *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
            return;
        }
        *output = ArrayD::from_elem(IxDyn(&[]), loss_sum / (nrows as f32));
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let logits = inputs[0].to_f32_array();
        let targets = inputs[1].to_f32_array();
        let axis = if self.axis >= logits.ndim() {
            logits.ndim() - 1
        } else {
            self.axis
        };
        let (permuted_logits, perm_opt) = permute_to_last(&logits, axis);
        // `perm_opt` is used directly below; no need to clone into an unused variable
        let shape = permuted_logits.shape().to_vec();
        let ndim = permuted_logits.ndim();
        let nrows = shape.iter().take(ndim - 1).product::<usize>();
        let classes = shape[ndim - 1];
        let logits_2d = match permuted_logits.to_shape((nrows, classes)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!(
                    "SoftmaxCrossEntropyLogits backward: Reshape logits to 2D failed: {}",
                    e
                );
                // return zero grads
                let shape0 = inputs[0].lock().storage.shape().to_vec();
                let shape1 = inputs[1].lock().storage.shape().to_vec();
                let grad_logits = ArrayD::zeros(IxDyn(&shape0));
                let grad_targets = ArrayD::zeros(IxDyn(&shape1));
                return vec![grad_logits, grad_targets];
            }
        };
        let mut soft = logits_2d.clone();
        for mut row in soft.rows_mut() {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v = *v / sum;
            }
        }
        let og = output_grad.iter().next().copied().unwrap_or_else(|| {
            log::error!("Softmax backward: expected scalar output_grad, defaulting to 1.0");
            1.0f32
        });
        let grad_logits_2d = ArrayD::zeros(IxDyn(&[nrows, classes]));
        let mut grad_view = match grad_logits_2d.into_dimensionality::<ndarray::Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Softmax backward: failed to convert grad to 2D: {}", e);
                let shape0 = inputs[0].lock().storage.shape().to_vec();
                let grad_logits = ArrayD::zeros(IxDyn(&shape0));
                let shape1 = inputs[1].lock().storage.shape().to_vec();
                let grad_targets = ArrayD::zeros(IxDyn(&shape1));
                return vec![grad_logits, grad_targets];
            }
        };
        if targets.ndim() == 1 && targets.shape()[0] == nrows {
            for i in 0..nrows {
                for j in 0..classes {
                    grad_view[[i, j]] = soft[[i, j]];
                }
                let j = targets[[i]] as usize;
                grad_view[[i, j]] -= 1.0;
                for k in 0..classes {
                    grad_view[[i, k]] *= og / (nrows as f32);
                }
            }
        } else if targets.ndim() == logits.ndim() {
            let perm_targets = if let Some(ref permv) = perm_opt {
                targets.view().permuted_axes(permv.clone()).to_owned()
            } else {
                targets.clone()
            };
            let t_2d = match perm_targets.to_shape((nrows, classes)) {
                Ok(v) => v.to_owned(),
                Err(e) => {
                    log::error!(
                        "SoftmaxCrossEntropyLogits backward: Reshape targets one-hot failed: {}",
                        e
                    );
                    let shape0 = inputs[0].lock().storage.shape().to_vec();
                    let grad_logits = ArrayD::zeros(IxDyn(&shape0));
                    let shape1 = inputs[1].lock().storage.shape().to_vec();
                    let grad_targets = ArrayD::zeros(IxDyn(&shape1));
                    return vec![grad_logits, grad_targets];
                }
            };
            for i in 0..nrows {
                for j in 0..classes {
                    grad_view[[i, j]] = (soft[[i, j]] - t_2d[[i, j]]) * og / (nrows as f32);
                }
            }
        } else {
            log::error!(
                "SoftmaxCrossEntropyLogits backward: target shape incompatible; logits shape: {:?}, targets shape: {:?}, axis: {}",
                logits.shape(), targets.shape(), axis
            );
            let shape0 = inputs[0].lock().storage.shape().to_vec();
            let grad_logits = ArrayD::zeros(IxDyn(&shape0));
            let shape1 = inputs[1].lock().storage.shape().to_vec();
            let grad_targets = ArrayD::zeros(IxDyn(&shape1));
            return vec![grad_logits, grad_targets];
        }
        let grad_permuted = match grad_view.into_dyn().to_shape(IxDyn(&shape)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("SoftmaxCrossEntropyLogits backward: Reshape grad failed: {}", e);
                let shape0 = inputs[0].lock().storage.shape().to_vec();
                let grad_logits = ArrayD::zeros(IxDyn(&shape0));
                let shape1 = inputs[1].lock().storage.shape().to_vec();
                let grad_targets = ArrayD::zeros(IxDyn(&shape1));
                return vec![grad_logits, grad_targets];
            }
        };
        let grad_logits = if let Some(ref permv) = perm_opt {
            permute_back(grad_permuted, permv)
        } else {
            grad_permuted
        };
        // grad for targets not supported
        let grad_targets = ArrayD::zeros(targets.dim());
        vec![grad_logits, grad_targets]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The concatenate operation.
pub struct Concat(pub usize);

impl Operation for Concat {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let axis = self.0;
        let mut arrays = Vec::new();
        for input in inputs {
            arrays.push(input.lock().storage.to_f32_array());
        }
        *output = match ndarray::concatenate(
            Axis(axis),
            &arrays.iter().map(|x| x.view()).collect::<Vec<_>>(),
        ) {
            Ok(v) => v,
            Err(e) => {
                log::error!("Concat forward failed: {}", e);
                ArrayD::<f32>::zeros(IxDyn(&[0]))
            }
        };
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let axis = self.0;
        let mut grads = Vec::new();
        let mut current_index = 0;
        for input in inputs {
            let input_lock = input.lock();
            let input_shape = input_lock.storage.shape();
            let mut slice_info_elems: Vec<SliceInfoElem> = Vec::new();
            for i in 0..input_shape.len() {
                if i == axis {
                    slice_info_elems
                        .push((current_index..current_index + input_shape[axis]).into());
                } else {
                    slice_info_elems.push((..).into());
                }
            }
            let slice_info: SliceInfo<_, IxDyn, IxDyn> =
                unsafe { match SliceInfo::new(slice_info_elems) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("Concat backward: invalid slice info: {}", e);
                        // push zeros for this input to preserve shape
                        grads.push(ArrayD::<f32>::zeros(IxDyn(&input_shape)));
                        current_index += input_shape[axis];
                        continue;
                    }
                }};
            grads.push(output_grad.slice(slice_info).to_owned().into_dyn());
            current_index += input_shape[axis];
        }
        grads
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The stack operation.
pub struct Stack(pub usize);

impl Operation for Stack {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let axis = self.0;
        let mut arrays = Vec::new();
        for input in inputs {
            arrays.push(input.lock().storage.to_f32_array());
        }
        *output = match ndarray::stack(
            Axis(axis),
            &arrays.iter().map(|x| x.view()).collect::<Vec<_>>(),
        ) {
            Ok(v) => v,
            Err(e) => {
                log::error!("Stack forward failed: {}", e);
                ArrayD::<f32>::zeros(IxDyn(&[0]))
            }
        };
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let axis = self.0;
        let mut grads = Vec::new();
        for (i, _input) in inputs.iter().enumerate() {
            let mut slice_info_elems: Vec<SliceInfoElem> = Vec::new();
            for j in 0..output_grad.ndim() {
                if j == axis {
                    slice_info_elems.push((i..i + 1).into());
                } else {
                    slice_info_elems.push((..).into());
                }
            }
            let slice_info: SliceInfo<_, IxDyn, IxDyn> =
                unsafe { match SliceInfo::new(slice_info_elems) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("Stack backward: invalid slice info: {}", e);
                        // push zeros default
                        grads.push(ArrayD::<f32>::zeros(IxDyn(&[0])));
                        continue;
                    }
                }};
            grads.push(
                output_grad
                    .slice(slice_info)
                    .to_owned()
                    .into_dyn()
                    .remove_axis(Axis(axis)),
            );
        }
        grads
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Slice operation slicing contiguous columns/axes in a 2D tensor; returns the slice along axis.
pub struct Slice {
    pub axis: usize,
    pub start: usize,
    pub len: usize,
}

impl Slice {
    pub fn new(axis: usize, start: usize, len: usize) -> Self {
        Slice { axis, start, len }
    }
}

impl Operation for Slice {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        // For now we only support 2D
        let a2 = a
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Slice: expected 2D input");
        let out = a2
            .slice(s![.., self.start..self.start + self.len])
            .to_owned();
        *output = out.into_dyn();
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // place the output_grad back into the correct slice positions for the input shape
        let a_shape = inputs[0].lock().storage.shape().to_vec();
        let res = ArrayD::<f32>::zeros(IxDyn(&a_shape));
        let mut res2 = match res.clone().into_dimensionality::<ndarray::Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Slice backward: expected 2D input, failed to convert: {}", e);
                return vec![ArrayD::zeros(IxDyn(&a_shape))];
            }
        };
        let og2 = match output_grad.clone().into_dimensionality::<ndarray::Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Slice backward: output_grad not 2D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&a_shape))];
            }
        };
        for i in 0..res2.dim().0 {
            for j in 0..og2.dim().1 {
                res2[[i, self.start + j]] = og2[[i, j]];
            }
        }
        vec![res2.into_dyn()]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The Conv2D operation (NCHW layout) with optional bias
pub struct Conv2D {
    pub stride: usize,
    pub padding: usize,
}

impl Conv2D {
    pub fn new(stride: usize, padding: usize) -> Self {
        Conv2D { stride, padding }
    }
}

impl Operation for Conv2D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // inputs: [input (N,Cin,H,W), weight (Cout,Cin,kH,kW), bias (Cout) optional]
        let input = inputs[0].to_f32_array();
        let weights = inputs[1].to_f32_array();
        let bias_opt = if inputs.len() > 2 {
            Some(inputs[2].lock().storage.to_f32_array())
        } else {
            None
        };

        let input = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv2D forward: input is not 4D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv2D forward: weights are not 4D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
        let (n, cin, hin, win) = input.dim();
        let (cout, cin2, kh, kw) = w.dim();
        assert_eq!(cin, cin2, "Conv2D: input channel mismatch with weight");

        let stride = self.stride as isize;
        let pad = self.padding as isize;
        let hout = ((hin as isize - kh as isize + 2 * pad) / stride + 1) as usize;
        let wout = ((win as isize - kw as isize + 2 * pad) / stride + 1) as usize;

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, hout, wout]));
        let mut out4 = match out.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv2D forward: output buffer reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };

        for batch in 0..n {
            for oc in 0..cout {
                for oh in 0..hout {
                    for ow in 0..wout {
                        let mut sum = 0.0f32;
                        for ic in 0..cin {
                            for kh_i in 0..kh {
                                for kw_i in 0..kw {
                                    let ih = oh as isize * stride + kh_i as isize - pad;
                                    let iw = ow as isize * stride + kw_i as isize - pad;
                                    if ih >= 0 && ih < hin as isize && iw >= 0 && iw < win as isize
                                    {
                                        let iv = input[[batch, ic, ih as usize, iw as usize]];
                                        let wv = w[[oc, ic, kh_i, kw_i]];
                                        sum += iv * wv;
                                    }
                                }
                            }
                        }
                        if let Some(ref b) = bias_opt {
                            sum += b[[oc]];
                        }
                        out4[[batch, oc, oh, ow]] = sum;
                    }
                }
            }
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let input = inputs[0].to_f32_array();
        let weights = inputs[1].to_f32_array();
        let input = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv2D backward: input is not 4D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv2D backward: weights are not 4D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let (n, cin, hin, win) = input.dim();
        let (cout, _, kh, kw) = w.dim();
        let outg_data = output_grad.clone();
        let outg = match outg_data.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv2D backward: output_grad is not 4D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
                return vec![grad_in, grad_w, grad_b];
            }
        };

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hin, win]));
        let mut grad_w = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, kh, kw]));
        let mut grad_b = None;
        if inputs.len() > 2 {
            grad_b = Some(ArrayD::<f32>::zeros(IxDyn(&[cout])));
        }

        let mut grad_in4 = match grad_in.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv2D backward: failed to reshape grad_in to 4D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let mut grad_w4 = grad_w
            .view_mut()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap();
        let mut grad_b_view = grad_b
            .as_mut()
            .map(|x| x.view_mut().into_dimensionality::<ndarray::Ix1>().unwrap());

        let stride = self.stride as isize;
        let pad = self.padding as isize;

        let hout = outg.dim().2;
        let wout = outg.dim().3;

        // grad_input: accumulate contributions from weights * output_grad
        for batch in 0..n {
            for oc in 0..cout {
                for oh in 0..hout {
                    for ow in 0..wout {
                        let ogv = outg[[batch, oc, oh, ow]];
                        for ic in 0..cin {
                            for kh_i in 0..kh {
                                for kw_i in 0..kw {
                                    let ih = oh as isize * stride + kh_i as isize - pad;
                                    let iw = ow as isize * stride + kw_i as isize - pad;
                                    if ih >= 0 && ih < hin as isize && iw >= 0 && iw < win as isize
                                    {
                                        grad_in4[[batch, ic, ih as usize, iw as usize]] +=
                                            ogv * w[[oc, ic, kh_i, kw_i]];
                                    }
                                }
                            }
                        }
                        if let Some(ref mut gb) = grad_b_view {
                            gb[oc] += ogv;
                        }
                    }
                }
            }
        }

        // grad_w: correlate input with output_grad
        for oc in 0..cout {
            for ic in 0..cin {
                for kh_i in 0..kh {
                    for kw_i in 0..kw {
                        let mut sum = 0f32;
                        for batch in 0..n {
                            for oh in 0..hout {
                                for ow in 0..wout {
                                    let ih = oh as isize * stride + kh_i as isize - pad;
                                    let iw = ow as isize * stride + kw_i as isize - pad;
                                    if ih >= 0 && ih < hin as isize && iw >= 0 && iw < win as isize
                                    {
                                        sum += outg[[batch, oc, oh, ow]]
                                            * input[[batch, ic, ih as usize, iw as usize]];
                                    }
                                }
                            }
                        }
                        grad_w4[[oc, ic, kh_i, kw_i]] = sum;
                    }
                }
            }
        }

        let mut ret: Vec<ArrayD<f32>> = vec![grad_in, grad_w];
        if let Some(gb) = grad_b {
            ret.push(gb);
        }
        ret
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Dropout operation. `p` is dropout probability (0.0 .. 1.0). Training mode applies dropout.
pub struct Dropout {
    pub p: f32,
    pub training: bool,
    mask: std::sync::Mutex<Option<ArrayD<f32>>>,
}

impl Dropout {
    pub fn new(p: f32, training: bool) -> Self {
        Dropout {
            p,
            training,
            mask: std::sync::Mutex::new(None),
        }
    }
}

impl Operation for Dropout {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        if !self.training || (self.p - 0.0).abs() < std::f32::EPSILON {
            *output = x.clone();
            return;
        }
        let keep = 1.0 - self.p;
        let mut rng = rand::thread_rng();
        let mut mask = ArrayD::<f32>::zeros(x.dim());
        // `_xv` is intentionally unused; we only use the mask values during construction
        for (m, _xv) in mask.iter_mut().zip(x.iter()) {
            let r: f32 = rng.gen();
            if r < keep {
                *m = 1.0 / keep;
            } else {
                *m = 0.0;
            }
        }
        *output = x * &mask;
        let mut lock = self
            .mask
            .lock()
            .expect("Failed to acquire Dropout mask lock — Mutex is poisoned or unavailable.");
        *lock = Some(mask);
    }

    fn backward(&self, _inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        if !self.training || (self.p - 0.0).abs() < std::f32::EPSILON {
            return vec![output_grad.clone()];
        }
        let mask = self
            .mask
            .lock()
            .expect("Failed to acquire Dropout mask lock — Mutex is poisoned or unavailable.");
        if let Some(m) = &*mask {
            vec![output_grad * m]
        } else {
            // no mask, just return zeros
            vec![ArrayD::zeros(output_grad.dim())]
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The MaxPool2D operation.
pub struct MaxPool2D {
    pub kernel_size: usize,
    pub stride: usize,
}

impl Operation for MaxPool2D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let input = inputs[0].to_f32_array();
        let input_view = input.view().into_dimensionality::<ndarray::Ix4>().unwrap();
        let (batch, c, h, w) = input_view.dim();
        let oh = (h - self.kernel_size) / self.stride + 1;
        let ow = (w - self.kernel_size) / self.stride + 1;
        let mut out = ArrayD::zeros(IxDyn(&[batch, c, oh, ow]));
        for b in 0..batch {
            for ch in 0..c {
                for i in 0..oh {
                    for j in 0..ow {
                        let window = input_view.slice(s![
                            b,
                            ch,
                            i * self.stride..i * self.stride + self.kernel_size,
                            j * self.stride..j * self.stride + self.kernel_size
                        ]);
                        out[[b, ch, i, j]] =
                            window.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    }
                }
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], _output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let input = inputs[0].to_f32_array();
        let input_view = input.view().into_dimensionality::<ndarray::Ix4>().unwrap();
        let (batch, c, h, w) = input_view.dim();
        let oh = (h - self.kernel_size) / self.stride + 1;
        let ow = (w - self.kernel_size) / self.stride + 1;
        let mut grad_in = ArrayD::zeros(IxDyn(&[batch, c, h, w]));
        let mut grad_view = grad_in
            .view_mut()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap();
        let out_grad_view = _output_grad
            .view()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap();
        for b in 0..batch {
            for ch in 0..c {
                for i in 0..oh {
                    for j in 0..ow {
                        let window = input_view.slice(s![
                            b,
                            ch,
                            i * self.stride..i * self.stride + self.kernel_size,
                            j * self.stride..j * self.stride + self.kernel_size
                        ]);
                        let max_val = window.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        // count number of times max occurs
                        let mut count = 0usize;
                        for v in window.iter() {
                            if (*v - max_val).abs() < 1e-6 {
                                count += 1;
                            }
                        }
                        if count == 0 {
                            continue;
                        }
                        let grad_share = out_grad_view[[b, ch, i, j]] / (count as f32);
                        for (wi, wv) in window.indexed_iter() {
                            if (wv - max_val).abs() < 1e-6 {
                                // Calculate global coordinates
                                let gi = i * self.stride + wi.0;
                                let gj = j * self.stride + wi.1;
                                grad_view[[b, ch, gi, gj]] += grad_share;
                            }
                        }
                    }
                }
            }
        }
        vec![grad_in]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
