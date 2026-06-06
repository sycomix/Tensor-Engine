use crate::backend::traits::{Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use ndarray::{ArrayD, Axis, IxDyn};
use rayon::prelude::*;

pub struct CpuBackend {
    thread_pool: rayon::ThreadPool,
}

impl Default for CpuBackend {
    fn default() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4); // Fallback to 4 threads

        log::info!("CPU Backend initialized with {} threads", num_threads);

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap_or_else(|e| {
                log::warn!(
                    "Failed to build rayon thread pool with {} threads: {}; using rayon default",
                    num_threads,
                    e
                );
                rayon::ThreadPoolBuilder::new()
                    .build()
                    .expect("Failed to build default rayon thread pool")
            });

        CpuBackend { thread_pool }
    }
}

impl CpuBackend {
    pub fn new(num_threads: usize) -> Self {
        let actual_threads = num_threads.max(1).min(64); // Clamp between 1 and 64

        log::info!("CPU Backend initialized with {} threads", actual_threads);

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(actual_threads)
            .build()
            .unwrap_or_else(|e| {
                log::warn!(
                    "Failed to build rayon thread pool with {} threads: {}; using rayon default",
                    actual_threads,
                    e
                );
                rayon::ThreadPoolBuilder::new()
                    .build()
                    .expect("Failed to build default rayon thread pool")
            });

        CpuBackend { thread_pool }
    }

    pub fn num_threads(&self) -> usize {
        self.thread_pool.current_num_threads()
    }

    fn matmul_2d_optimized(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];

        #[cfg(feature = "openblas")]
        {
            use matrixmultiply;

            let a_owned: Option<ArrayD<f32>> = if !a.is_standard_layout() {
                Some(a.as_standard_layout().into_owned())
            } else {
                None
            };
            let b_owned: Option<ArrayD<f32>> = if !b.is_standard_layout() {
                Some(b.as_standard_layout().into_owned())
            } else {
                None
            };

            let a_ref = a_owned.as_ref().unwrap_or(a);
            let b_ref = b_owned.as_ref().unwrap_or(b);

            let a_slice = a_ref
                .as_slice()
                .expect("A must be contiguous after standard layout conversion");
            let b_slice = b_ref
                .as_slice()
                .expect("B must be contiguous after standard layout conversion");

            let mut result = ArrayD::<f32>::zeros(IxDyn(&[m, n]));

            // Safety: a_slice/b_slice are contiguous row-major buffers of the correct dimensions.
            // Strides: A[m×k] → rsa=k, csa=1; B[k×n] → rsb=n, csb=1; C[m×n] → rsc=n, csc=1.
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    n,
                    1.0_f32,
                    a_slice.as_ptr(),
                    k as isize,
                    1,
                    b_slice.as_ptr(),
                    n as isize,
                    1,
                    0.0_f32,
                    result.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }

            Some(result)
        }

        #[cfg(not(feature = "openblas"))]
        {
            let rows: Vec<Vec<f32>> = self.thread_pool.install(|| {
                (0..m)
                    .into_par_iter()
                    .map(|i| {
                        (0..n)
                            .map(|j| (0..k).map(|l| a[[i, l]] * b[[l, j]]).sum::<f32>())
                            .collect()
                    })
                    .collect()
            });

            let mut result = ArrayD::<f32>::zeros(IxDyn(&[m, n]));
            for (i, row) in rows.into_iter().enumerate() {
                for (j, val) in row.into_iter().enumerate() {
                    result[[i, j]] = val;
                }
            }

            Some(result)
        }
    }

    fn matmul_3d_optimized(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        let batch = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        let n = b.shape()[2];

        if b.shape()[1] != k {
            log::warn!("matmul_3d: Shape mismatch");
            return None;
        }

        let results: Vec<ArrayD<f32>> = self.thread_pool.install(|| {
            (0..batch)
                .into_par_iter()
                .map(|i| {
                    let a_i = a.index_axis(Axis(0), i);
                    let b_i = b.index_axis(Axis(0), i);

                    if let (Ok(a_2d), Ok(b_2d)) = (
                        a_i.into_dimensionality::<ndarray::Ix2>(),
                        b_i.into_dimensionality::<ndarray::Ix2>(),
                    ) {
                        a_2d.dot(&b_2d).into_dyn()
                    } else {
                        ArrayD::zeros(IxDyn(&[m, n]))
                    }
                })
                .collect()
        });

        let mut c = ArrayD::<f32>::zeros(IxDyn(&[batch, m, n]));

        for (i, result) in results.into_iter().enumerate() {
            if i < batch {
                c.index_axis_mut(Axis(0), i).assign(&result);
            }
        }

        Some(c)
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "CPU"
    }

    fn create_from_data(&self, data: ArrayD<f32>, dtype: DType) -> Box<dyn Storage> {
        if dtype == DType::F32 {
            Box::new(TensorStorage::F32(data))
        } else {
            Box::new(TensorStorage::from_f32_array(&data, dtype))
        }
    }

    fn create_zeros(&self, shape: &[usize]) -> Box<dyn Storage> {
        let shape_ix = IxDyn(shape);
        let data = ArrayD::zeros(shape_ix);
        Box::new(TensorStorage::F32(data))
    }

    fn create_ones(&self, shape: &[usize]) -> Box<dyn Storage> {
        let shape_ix = IxDyn(shape);
        let data = ArrayD::from_elem(shape_ix, 1.0);
        Box::new(TensorStorage::F32(data))
    }

    fn matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        match (a.ndim(), b.ndim()) {
            (2, 2) => self.matmul_2d_optimized(a, b),
            (3, 3) => self.matmul_3d_optimized(a, b),
            _ => None,
        }
    }

    fn softmax(&self, input: &ArrayD<f32>, axis: isize) -> Option<ArrayD<f32>> {
        let mut output = input.clone();
        let ndim = input.ndim();

        if axis < 0 || (axis as usize) >= ndim {
            log::error!(
                "Softmax: Invalid axis {} for tensor with {} dimensions",
                axis,
                ndim
            );
            return None;
        }

        let norm_axis = axis as usize;

        // Compute max for numerical stability along the normalization axis
        let max_val: f32 = output
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

        // Subtract max and compute exp (numerically stable)
        output.mapv_inplace(|x| (x - max_val).exp());

        // Sum along axis for normalization
        let sum_array = output.sum_axis(Axis(norm_axis));

        // Check if any value is NaN or <= 0
        let has_invalid = sum_array.iter().any(|&x| x.is_nan() || x <= 0.0);

        if has_invalid {
            log::warn!("Softmax: Invalid sum detected, returning zeros");
            return Some(ArrayD::zeros(input.shape().to_vec()));
        }

        // Convert to scalar f32 and normalize
        let sum: f32 = match sum_array.len() {
            1 => *sum_array.get(0).unwrap_or(&1.0),
            _ => return None, // Should not happen for valid input
        };

        if sum > 1e-8 {
            output.mapv_inplace(|x| x / sum);
        } else {
            log::warn!("Softmax: Near-zero denominator detected");
        }

        Some(output)
    }

    fn rms_norm(
        &self,
        input: &ArrayD<f32>,
        weight: &ArrayD<f32>,
        eps: f32,
        axis: isize,
    ) -> Option<ArrayD<f32>> {
        let ndim = input.ndim();
        let norm_axis = if axis < 0 {
            let positive = ndim as isize + axis;
            if positive < 0 {
                log::error!(
                    "RMSNorm: Invalid axis {} for tensor with {} dimensions",
                    axis,
                    ndim
                );
                return None;
            }
            positive as usize
        } else {
            axis as usize
        };

        if norm_axis >= ndim {
            log::error!(
                "RMSNorm: Invalid axis {} for tensor with {} dimensions",
                axis,
                ndim
            );
            return None;
        }

        let lane_len = input.shape()[norm_axis];
        if weight.len() != lane_len {
            log::error!(
                "RMSNorm: Weight length {} does not match axis {} dimension {}",
                weight.len(),
                norm_axis,
                lane_len
            );
            return None;
        }

        let n = lane_len as f32;
        let mut output = input.clone();

        for mut lane in output.lanes_mut(Axis(norm_axis)) {
            let sum_sq: f32 = lane.iter().map(|&x| x * x).sum();
            let rms = (sum_sq / n + eps).sqrt();
            if rms < 1e-8 {
                log::warn!(
                    "RMSNorm: Near-zero RMS value detected; skipping normalization for lane"
                );
                continue;
            }
            for (val, &w) in lane.iter_mut().zip(weight.iter()) {
                *val = (*val / rms) * w;
            }
        }

        Some(output)
    }

    fn rope(
        &self,
        x: &ArrayD<f32>,
        freqs: &ArrayD<f32>,
        seq_len: usize,
        head_dim: usize,
    ) -> Option<ArrayD<f32>> {
        if x.ndim() < 3 || x.shape()[1] != seq_len || x.shape()[2] != head_dim {
            log::error!(
                "RoPE: Invalid input shape {:?} for seq_len={} and head_dim={}",
                x.shape(),
                seq_len,
                head_dim
            );
            return None;
        }

        let mut output = x.clone();

        // Apply rotary embeddings - use sequential iteration to avoid borrow checker issues
        for pos in 0..seq_len {
            for batch in 0..x.shape()[0] {
                for i in (0..head_dim).step_by(2) {
                    if i + 1 >= head_dim {
                        break;
                    }

                    let freq_idx = i / 2;
                    if freq_idx >= freqs.len() {
                        log::warn!(
                            "RoPE: Frequency index {} out of bounds for head_dim={}",
                            freq_idx,
                            head_dim
                        );
                        continue;
                    }

                    let theta = freqs[freq_idx];

                    // Get the two values to rotate
                    let x_i = output[[batch, pos, i]];
                    let x_i1 = output[[batch, pos, i + 1]];

                    // Apply rotation: [cos θ -sin θ; sin θ cos θ]
                    let cos_theta = (pos as f32 * theta).cos();
                    let sin_theta = (pos as f32 * theta).sin();

                    output[[batch, pos, i]] = x_i * cos_theta - x_i1 * sin_theta;
                    output[[batch, pos, i + 1]] = x_i * sin_theta + x_i1 * cos_theta;
                }
            }
        }

        Some(output)
    }

    fn synchronize(&self) {
        // CPU is synchronous by nature - no-op
    }

    fn memory_info(&self) -> (usize, usize) {
        let total = 16 * 1024 * 1024 * 1024; // Assume 16GB
        let used = 2 * 1024 * 1024 * 1024; // Estimate 2GB used
        (used, total)
    }
}
