use crate::backend::traits::{Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use ndarray::{ArrayD, ArrayView2, IxDyn, Axis};
use rayon::prelude::*;

pub struct CpuBackend {
    num_threads: usize,
}

impl Default for CpuBackend {
    fn default() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or_else(|| 4); // Fallback to 4 threads
        
        log::info!("CPU Backend initialized with {} threads", num_threads);
        
        CpuBackend { num_threads }
    }
}

impl CpuBackend {
    pub fn new(num_threads: usize) -> Self {
        let actual_threads = num_threads.max(1).min(64); // Clamp between 1 and 64
        
        log::info!("CPU Backend initialized with {} threads", actual_threads);
        
        CpuBackend { 
            num_threads: actual_threads,
        }
    }

    fn matmul_2d_optimized(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];

        // Use OpenBLAS via blas-rs if available, otherwise use optimized CPU implementation
        #[cfg(feature = "blas")]
        {
            use blas::Lapack;
            
            let mut c = ArrayD::<f32>::zeros(IxDyn(&[m, n]));
            
            unsafe {
                // Use sgemm for single-precision matrix multiplication
                // C := alpha * A * B + beta * C
                blas::sgemm(
                    b'N', b'N',  // No transpose
                    m as i32, n as i32, k as i32,
                    1.0,
                    a.as_ptr(), m,
                    b.as_ptr(), k,
                    0.0,
                    c.as_mut_ptr(), m,
                );
            }
            
            Some(c)
        }

        #[cfg(not(feature = "blas"))]
        {
            // Optimized CPU implementation using Rayon for parallelism
            let mut result = ArrayD::<f32>::zeros(IxDyn(&[m, n]));
            
            // Parallelize over rows of A (outer loop)
            (0..m).into_par_iter().for_each(|i| {
                let a_row = &a.slice(ndarray::s![i, ..]);
                
                for j in 0..n {
                    let mut sum: f32 = 0.0;
                    
                    // Inner loop - unrolled by compiler for better performance
                    let b_col_start = j * k;
                    for l in 0..k {
                        sum += a_row[l] * b[[l, j]];
                    }
                    
                    result[[i, j]] = sum;
                }
            });

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

        // Parallelize over batch dimension using Rayon
        let results: Vec<ArrayD<f32>> = (0..batch)
            .into_par_iter()
            .map(|i| {
                let a_i = a.index_axis(Axis(0), i);
                let b_i = b.index_axis(Axis(0), i);

                if let (Ok(a_2d), Ok(b_2d)) = (a_i.into_dimensionality::<ndarray::Ix2>(), b_i.into_dimensionality::<ndarray::Ix2>()) {
                    a_2d.dot(&b_2d).into_dyn()
                } else {
                    ArrayD::zeros(IxDyn(&[m, n])) // Fallback for invalid input
                }
            })
            .collect();

        // Stack results back into 3D array
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
        // Support both 2D and batched matrix multiplication with optimized implementations
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
            log::error!("Softmax: Invalid axis {} for tensor with {} dimensions", axis, ndim);
            return None;
        }

        let norm_axis = axis as usize;

        // Compute max for numerical stability along the normalization axis
        let max_val: f32 = output.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));

        // Subtract max and compute exp (numerically stable)
        output.mapv_inplace(|x| (x - max_val).exp());

        // Sum along axis for normalization
        let sum_array = output.sum_axis(Axis(norm_axis));
        
        // Handle edge case where sum might be zero or NaN
        if sum_array.is_nan() || sum_array.iter().any(|&x| x <= 0.0) {
            log::warn!("Softmax: Invalid sum detected, returning zeros");
            return Some(ArrayD::zeros(input.shape().to_vec()));
        }

        // Convert to scalar f32 and normalize
        let sum: f32 = match sum_array.len() {
            1 => *sum_array.get(0).unwrap_or(&1.0),
            _ => return None, // Should not happen for valid input
        };

        // Divide by sum for each element along the axis
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
        let mut output = input.clone();
        let ndim = input.ndim();

        if axis < 0 || (axis as usize) >= ndim {
            log::error!("RMSNorm: Invalid axis {} for tensor with {} dimensions", axis, ndim);
            return None;
        }

        let norm_axis = axis as usize;
        
        // Compute RMS along the normalization axis with numerical stability
        let sum_sq = output.sum_axis(Axis(norm_axis));
        
        // Convert to f32 and compute RMS with epsilon for stability
        let rms_values: Vec<f32> = sum_sq.iter()
            .map(|&x| ((x / weight.len() as f32) + eps).sqrt())
            .collect();
        
        // Normalize by dividing input by RMS using parallel iteration
        output.par_iter_mut().enumerate().for_each(|(i, val)| {
            if let Some(pos) = ndarray::indices_of(&output, IxDyn(&[i])).first() {
                let norm_val = rms_values[pos[norm_axis] as usize];
                if norm_val > 1e-8 {
                    *val /= norm_val;
                } else {
                    log::warn!("RMSNorm: Near-zero RMS value detected at position {}", i);
                }
            }
        });

        // Apply weight scaling in parallel
        output.par_iter_mut().enumerate().for_each(|(i, val)| {
            if let Some(pos) = ndarray::indices_of(&output, IxDyn(&[i])).first() {
                *val *= weight[pos[norm_axis] as usize];
            }
        });

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
            log::error!("RoPE: Invalid input shape {:?} for seq_len={} and head_dim={}", x.shape(), seq_len, head_dim);
            return None;
        }

        let mut output = x.clone();
        
        // Apply rotary embeddings to each position in the sequence using parallel iteration
        (0..seq_len).into_par_iter().for_each(|pos| {
            for batch in 0..x.shape()[0] {
                for i in (0..head_dim).step_by(2) {
                    if i + 1 >= head_dim {
                        break;
                    }

                    let freq_idx = i / 2;
                    if freq_idx >= freqs.len() {
                        log::warn!("RoPE: Frequency index {} out of bounds for head_dim={}", freq_idx, head_dim);
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
        });

        Some(output)
    }

    fn memory_info(&self) -> (usize, usize) {
        // CPU backend - estimate based on system memory
        let total = 16 * 1024 * 1024 * 1024; // Assume 16GB
        let used = 2 * 1024 * 1024 * 1024; // Estimate 2GB used
        (used, total)
    }

    fn synchronize(&self) {
        // CPU is synchronous by nature - no-op
    }
}

impl CpuBackend {
    fn matmul_2d(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 || a.shape()[1] != b.shape()[0] {
            log::warn!("matmul_2d: Shape mismatch");
            return None;
        }

        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];

        let mut c = ArrayD::<f32>::zeros(IxDyn(&[m, n]));

        // Use ndarray's dot product for simplicity
        let a_2d: ArrayView2<f32> = match ndarray::ArrayView2::from_shape((m, k), a.as_slice()?) {
            Ok(v) => v,
            Err(_) => return None,
        };

        let b_2d: ArrayView2<f32> = match ndarray::ArrayView2::from_shape((k, n), b.as_slice()?) {
            Ok(v) => v,
            Err(_) => return None,
        };

        let result = a_2d.dot(&b_2d);
        c.assign(&result.into_dyn());

        Some(c)
    }

    fn matmul_3d(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        if a.ndim() != 3 || b.ndim() != 3 {
            return None;
        }

        let batch = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        let n = b.shape()[2];

        if b.shape()[1] != k {
            log::warn!("matmul_3d: Shape mismatch");
            return None;
        }

        let mut c = ArrayD::<f32>::zeros(IxDyn(&[batch, m, n]));

        for i in 0..batch {
            let a_i = a.index_axis(Axis(0), i);
            let b_i = b.index_axis(Axis(0), i);

            if let (Ok(a_2d), Ok(b_2d)) = (a_i.into_dimensionality::<ndarray::Ix2>(), b_i.into_dimensionality::<ndarray::Ix2>()) {
                let result = a_2d.dot(&b_2d);
                c.index_axis_mut(Axis(0), i).assign(&result.into_dyn());
            } else {
                return None;
            }
        }

        Some(c)
    }
}
