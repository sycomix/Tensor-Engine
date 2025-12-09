use crate::tensor::Tensor;
use rand::Rng;
use ndarray::{IxDyn, Array2, Axis};
// Matrix multiply acceleration when openblas feature is enabled
fn matmul_row_major(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    a.dot(&b.view())
}
// No openblas-specific matmul: fallback to ndarray::dot is used.

/// Residual Vector Quantizer (RVQ) stub implementation.
/// This is a minimal first pass: it holds a single codebook and supports a quantize() method that
/// returns indices for a given input tensor. Full RVQ logic (multiple levels, residual updates)
/// will be implemented later; for now this provides an API surface for the roadmap.

pub struct RVQ {
    pub codebooks: Vec<Tensor>, // shape per level: [num_codes, dim]
    pub levels: usize,
    pub num_codes: usize,
    pub dim: usize,
    // EMA counts to allow unbiased updates per code: ema_counts[level][code]
    pub ema_counts: Vec<Vec<f32>>,
    // Allow scheduling: only update EMA every N calls (default 1 -> every call)
    pub ema_update_every: usize,
    // Internal training step counter used for scheduling
    pub train_step: usize,
    // If true, reinitialize empty codes from random residuals to avoid dead codes
    pub reinit_empty_codes: bool,
}

impl RVQ {
    pub fn new(num_codes: usize, dim: usize, levels: usize) -> Self {
        let mut codebooks = Vec::new();
        for _ in 0..levels {
            let cb = ndarray::Array::zeros(IxDyn(&[num_codes, dim]));
            codebooks.push(Tensor::new(cb, true));
        }
        let mut ema_counts = Vec::new();
        for _ in 0..levels {
            ema_counts.push(vec![0.0f32; num_codes]);
        }
        RVQ {
            codebooks,
            levels,
            num_codes,
            dim,
            ema_counts,
            ema_update_every: 1,
            train_step: 0,
            reinit_empty_codes: false,
        }
    }

    /// Return a flattened vector of all codebook tensors for optimizer.
    pub fn parameters(&self) -> Vec<crate::tensor::Tensor> {
        self.codebooks.clone()
    }

    /// Set how often EMA updates should occur (every `n` calls to update_ema).
    pub fn set_ema_update_every(&mut self, n: usize) {
        if n == 0 { return; }
        self.ema_update_every = n;
    }

    pub fn set_reinit_empty_codes(&mut self, v: bool) {
        self.reinit_empty_codes = v;
    }

    /// Quantize a tensor into code indices.
    /// For initial skeleton, we just return zeros with shape equal to first two dims of input.
    /// Quantize a tensor into code indices for each level.
    /// Returns a Vec per level: indices[level] is a Vec<usize> length N (flattened positions).
    pub fn quantize(&self, input: &Tensor) -> Vec<Vec<usize>> {
        // Vectorized nearest-neighbor quantization across the last dimension.
        // input: [..., dim] ; codebook: [num_codes, dim]
        let inp_arr = input.lock().storage.to_f32_array();
        let first_cb_arr = self.codebooks[0].lock().storage.to_f32_array();
        // reshape cb to 2D: [num_codes, dim]
        let cb2 = match first_cb_arr.into_dimensionality::<ndarray::Ix2>() {
            Ok(v) => v,
            Err(_) => return vec![],
        };
        let dim = cb2.dim().1;
        let inp_shape = inp_arr.shape().to_vec();
        if inp_shape.len() == 0 || inp_shape.last().unwrap() != &dim {
            // incompatible shapes
            return vec![];
        }
        // Flatten leading dims to [N, dim]
        // Input must be able to be reshaped into 2D (N, dim)
        let n: usize = if inp_shape.len() >= 2 {
            inp_shape.iter().cloned().take(inp_shape.len() - 1).product()
        } else {
            0
        };
        let inp2 = match inp_arr.into_dimensionality::<ndarray::Ix2>() {
            Ok(v) => v,
            Err(_) => return vec![],
        };
        // Prepare outputs: per-level indices
        let mut indices_per_level: Vec<Vec<usize>> = Vec::new();
        indices_per_level.resize_with(self.levels, || Vec::with_capacity(n));
        // residual array: start as inp2
        let mut residual = inp2.clone();
        for level in 0..self.levels {
            let cb_arr = self.codebooks[level].lock().storage.to_f32_array();
            let cb2 = match cb_arr.into_dimensionality::<ndarray::Ix2>() {
                Ok(v) => v,
                Err(_) => return vec![],
            };
            // Vectorized distance computation:
            // dist^2 = ||x||^2 + ||c||^2 - 2 x c^T
            let x2: Array2<f32> = residual.clone().into_dimensionality().unwrap();
            let c2: Array2<f32> = cb2.clone().into_dimensionality().unwrap();
            // Compute squared norms
            let x_norm: Array2<f32> = x2.mapv(|v| v * v).sum_axis(Axis(1)).insert_axis(Axis(1)); // (N,1)
            let c_norm: Array2<f32> = c2.mapv(|v| v * v).sum_axis(Axis(1)).insert_axis(Axis(0)); // (1, num_codes)
            // Compute dot product X (N x dim) dot C^T (dim x num_codes) = (N x num_codes)
            // make contiguous copy of transposed matrix to ensure stable memory layout
            let c2_t = c2.t().to_owned();
            let xc = matmul_row_major(&x2, &c2_t);
            // dist = x_norm + c_norm - 2*xc
            let mut dist = xc.clone();
            for ((i, j), val) in dist.indexed_iter_mut() {
                *val = x_norm[[i, 0]] + c_norm[[0, j]] - 2.0 * *val;
            }
            // Argmin per row
            for i in 0..dist.dim().0 {
                let row = dist.index_axis(Axis(0), i);
                let mut min_idx = 0usize;
                let mut min_val = f32::INFINITY;
                for j in 0..row.len() {
                    let v = row[j];
                    if v < min_val {
                        min_val = v;
                        min_idx = j;
                    }
                }
                indices_per_level[level].push(min_idx);
            }
            // Update residual: residual = residual - codebook[level][index]
            let mut it = 0usize;
            for mut row in residual.outer_iter_mut() {
                let idx = indices_per_level[level][it];
                for k in 0..dim {
                    row[k] -= cb2[[idx, k]];
                }
                it += 1;
            }
        }
        indices_per_level
    }

    /// Update codebooks using exponential moving average (EMA) based on assignments.
    /// `inputs` should be [N, dim] (or flattenable leading dims) and indices must match quantize output.
    pub fn update_ema(&mut self, inputs: &Tensor, indices: &Vec<Vec<usize>>, decay: f32) -> Result<(), String> {
        if indices.len() != self.levels {
            return Err(format!("indices len {} != levels {}", indices.len(), self.levels));
        }
        // Scheduling: only update once every `ema_update_every` calls
        self.train_step = self.train_step.wrapping_add(1);
        if self.ema_update_every > 1 && (self.train_step % self.ema_update_every != 0) {
            return Ok(());
        }
        let inp_arr = inputs.lock().storage.to_f32_array();
        let x2 = match inp_arr.into_dimensionality::<ndarray::Ix2>() {
            Ok(v) => v,
            Err(_) => return Err("inputs must be flattenable to 2D (N, dim)".to_string()),
        };
        let n = x2.dim().0;
        for level in 0..self.levels {
            if indices[level].len() != n {
                return Err(format!("indices[{}] length {} != N {}", level, indices[level].len(), n));
            }
            // Use the residual at this level (i.e., inputs minus all previous levels' codebook contributions)
            // To compute this, iterate through levels until this one and subtract codebook contributions as in quantize.
            // We'll reconstruct residuals per sample in a small loop to compute level-wise sums and counts.
            let mut residual = x2.clone();
            for l in 0..level {
                let cb_arr = self.codebooks[l].lock().storage.to_f32_array();
                let cb2 = cb_arr.into_dimensionality::<ndarray::Ix2>().unwrap();
                for i in 0..n {
                    let idx = indices[l][i];
                    let mut row = residual.index_axis_mut(Axis(0), i);
                    for k in 0..self.dim {
                        row[k] -= cb2[[idx, k]];
                    }
                }
            }
            // Initialize sums and counts for this level based on residuals
            let mut sums = ndarray::Array2::<f32>::zeros((self.num_codes, self.dim));
            let mut counts = vec![0usize; self.num_codes];
            for i in 0..n {
                let idx = indices[level][i];
                counts[idx] += 1;
                let row = residual.index_axis(Axis(0), i);
                for k in 0..self.dim {
                    sums[[idx, k]] += row[k];
                }
            }
            // Compute means and update codebook
            let mut cb_arr = self.codebooks[level].lock();
            let mut cb_data = cb_arr.storage.to_f32_array();
            let ema_counts_level = &mut self.ema_counts[level];
            for c in 0..self.num_codes {
                let cnt = counts[c] as f32;
                // Unbiased EMA: maintain a per-code EMA count, and compute a weighted update to avoid bias
                let prev_count = ema_counts_level[c];
                let new_count = decay * prev_count + (1.0 - decay) * cnt;
                if new_count <= 0.0 {
                    // No observed counts; allow reinitialization of dead codes if configured
                    if counts[c] == 0 && self.reinit_empty_codes {
                        let mut rng = rand::thread_rng();
                        let rand_idx = rng.gen_range(0..n);
                        let row = residual.index_axis(Axis(0), rand_idx);
                        for k in 0..self.dim {
                            cb_data[[c, k]] = row[k];
                        }
                        ema_counts_level[c] = 1.0;
                    } else {
                        // still no information; skip
                        ema_counts_level[c] = new_count;
                    }
                    continue;
                }
                // numerator: decay * prev_count * old_code + (1-decay) * sums[c]
                for k in 0..self.dim {
                    let old = cb_data[[c, k]];
                    let sum_k = sums[[c, k]];
                    let numerator = decay * prev_count * old + (1.0 - decay) * sum_k;
                    cb_data[[c, k]] = numerator / new_count;
                }
                ema_counts_level[c] = new_count;
                // if no actual counts (i.e., counts[c] == 0) and we are configured to reinitialize empty codes,
                // the reinit is handled above in the new_count<=0.0 block.
            }
            // write back
            cb_arr.storage = crate::dtype::TensorStorage::from_f32_array(&cb_data, cb_arr.dtype);
        }
        Ok(())
    }

    /// Dequantize a list of indices into a Tensor with provided lead shape.
    /// `shape`: full shape including last dim equal to codebook dim.
    pub fn dequantize(&self, indices: &Vec<Vec<usize>>, shape: &[usize]) -> Option<Tensor> {
        if indices.len() != self.levels {
            return None;
        }
        let cb_arr = self.codebooks[0].lock().storage.to_f32_array();
        let cb2 = match cb_arr.into_dimensionality::<ndarray::Ix2>() {
            Ok(v) => v,
            Err(_) => return None,
        };
        let dim = cb2.dim().1;
        if shape.last().unwrap() != &dim {
            return None;
        }
        let n = indices[0].len();
        // Build array shape:
        let total: usize = shape.iter().cloned().product();
        if total / dim != n {
            return None;
        }
        // Sum contributions across levels
        let mut flat = Vec::with_capacity(n * dim);
        for i in 0..n {
            for k in 0..dim {
                let mut val = 0.0f32;
                for level in 0..self.levels {
                    let idx = indices[level][i];
                    let cb_arr = self.codebooks[level].lock().storage.to_f32_array();
                    let cb2 = match cb_arr.into_dimensionality::<ndarray::Ix2>() {
                        Ok(v) => v,
                        Err(_) => return None,
                    };
                    val += cb2[[idx, k]];
                }
                flat.push(val);
            }
        }
        let arr = ndarray::Array::from_shape_vec(IxDyn(shape), flat).ok()?;
        Some(Tensor::new(arr.into_dyn(), false))
    }
}
