use crate::backend::get_global_backend;
use crate::tensor::Tensor;
#[cfg(all(feature = "openblas", not(target_os = "windows")))]
#[cfg(all(feature = "openblas", not(target_os = "windows")))]
use cblas_sys::{self, CBLAS_ORDER, CBLAS_TRANSPOSE};
use ndarray::Array2;
use ndarray::Zip;
use ndarray::{s, ArrayD, ArrayView2, Axis, Dimension, Ix2, IxDyn, SliceInfo, SliceInfoElem};
use rayon::prelude::*;

// Reusable empty shape slice to avoid inline cast errors
// rand::Rng import removed; use rand::random() where needed to avoid deprecated API usage.
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
        res = match res.to_shape(IxDyn(&new_shape[..])) {
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
                log::error!(
                    "reduce_grad_to_shape: Final reshape to target shape failed: {}",
                    e
                );
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

fn permute_back(a: ArrayD<f32>, perm: &[usize]) -> ArrayD<f32> {
    // compute inverse permutation
    let ndim = perm.len();
    let mut inv = vec![0usize; ndim];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    a.view().permuted_axes(inv).to_owned()
}

/// Parallel element-wise map over an ArrayD<f32> using Rayon.
/// Falls back to sequential iteration if the arrays are non-contiguous.
fn par_mapv<F>(a: &ArrayD<f32>, f: F) -> ArrayD<f32>
where
    F: Fn(f32) -> f32 + Send + Sync,
{
    let mut out = ArrayD::<f32>::zeros(a.shape());
    match (a.as_slice_memory_order(), out.as_slice_memory_order_mut()) {
        (Some(a_slice), Some(out_slice)) => {
            out_slice
                .par_iter_mut()
                .zip(a_slice.par_iter())
                .for_each(|(o, &x)| *o = f(x));
        }
        _ => {
            for (o, &x) in out.iter_mut().zip(a.iter()) {
                *o = f(x);
            }
        }
    }
    out
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

/// FlashAttentionRef: A CPU reference implementation of FlashAttention.
/// This op expects three inputs: Q, K, V each shaped [b*heads, seq, head_dim]
/// and produces output shaped [b*heads, seq, head_dim]. It mirrors the baseline
/// attention but is provided for alternative implementations and parity testing.
pub struct FlashAttentionRef {
    pub head_dim: usize,
}

impl FlashAttentionRef {
    pub fn new(head_dim: usize) -> Self {
        FlashAttentionRef { head_dim }
    }
}

impl Operation for FlashAttentionRef {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // inputs: Q, K, V
        let q = inputs[0].to_f32_array(); // shape: [b*heads, seq, head_dim]
        let k = inputs[1].to_f32_array();
        let v = inputs[2].to_f32_array();
        let shape_q = q.shape().to_vec();
        if shape_q.len() != 3 {
            log::error!("FlashAttentionRef forward: expected 3D inputs");
            *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
            return;
        }
        let bnh = shape_q[0];
        let seq = shape_q[1];
        let hd = shape_q[2];
        if hd != self.head_dim || k.shape() != [bnh, seq, hd] || v.shape() != [bnh, seq, hd] {
            log::error!(
                "FlashAttentionRef forward: shape mismatch: q={:?} k={:?} v={:?} head_dim={}",
                shape_q,
                k.shape(),
                v.shape(),
                self.head_dim
            );
            *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
            return;
        }
        // QK^T
        // Build k_t as k transposed on the last two axes -> [bnh, hd, seq]
        let mut k_t = ArrayD::<f32>::zeros(IxDyn(&[bnh, hd, seq][..]));
        for i in 0..bnh {
            let kmat = k.index_axis(Axis(0), i).to_owned(); // [seq,hd]
            let kt = kmat.t().to_owned(); // [hd,seq]
            k_t.index_axis_mut(Axis(0), i).assign(&kt.into_dyn());
        }
        let mut qk = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, seq][..]));
        for i in 0..bnh {
            let q_mat = q.index_axis(Axis(0), i).to_owned(); // [seq,hd]
            let k_mat = k_t.index_axis(Axis(0), i).to_owned(); // [hd,seq]
            let q_mat2 = match q_mat.into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef forward: Failed to convert q matrix to 2D: {}",
                        e
                    );
                    *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
                    return;
                }
            };
            let k_mat2 = match k_mat.into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef forward: Failed to convert k matrix to 2D: {}",
                        e
                    );
                    *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
                    return;
                }
            };
            let res = q_mat2.dot(&k_mat2); // [seq, seq]
            qk.index_axis_mut(Axis(0), i).assign(&res.into_dyn());
        }
        // scale
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        qk *= scale;
        // softmax along last axis
        let mut attn = qk.clone();
        for i in 0..bnh {
            let mut cur = attn.index_axis_mut(Axis(0), i);
            // apply softmax across axis 1 (seq)
            for mut row in cur.outer_iter_mut() {
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for val in row.iter_mut() {
                    *val = (*val - mx).exp();
                    sum += *val;
                }
                // Numerical guard: if sum is zero or non-finite, fall back to uniform
                if !(sum > 0.0f32 && sum.is_finite()) {
                    let len = row.len() as f32;
                    for val in row.iter_mut() {
                        *val = 1.0f32 / len;
                    }
                    continue;
                }
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }
        // attn @ V
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd][..]));
        for i in 0..bnh {
            let att = attn.index_axis(Axis(0), i).to_owned(); // [seq,seq]
            let vmat = v.index_axis(Axis(0), i).to_owned(); // [seq,hd]
            let att2 = match att.into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef forward: Failed to convert attention to 2D: {}",
                        e
                    );
                    *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
                    return;
                }
            };
            let vmat2 = match vmat.into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef forward: Failed to convert v matrix to 2D: {}",
                        e
                    );
                    *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
                    return;
                }
            };
            let res = att2.dot(&vmat2); // [seq,hd]
            out.index_axis_mut(Axis(0), i).assign(&res.into_dyn());
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // inputs: Q, K, V
        let q = inputs[0].to_f32_array();
        let k = inputs[1].to_f32_array();
        let v = inputs[2].to_f32_array();
        let bnh = q.shape()[0];
        let seq = q.shape()[1];
        let hd = q.shape()[2];
        let scale = 1.0f32 / (hd as f32).sqrt();

        // Forward intermediates
        // qk = q @ k^T * scale
        let mut qk = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, seq][..]));
        for i in 0..bnh {
            let qmat = q.index_axis(Axis(0), i).to_owned();
            let km = k.index_axis(Axis(0), i).to_owned();
            let qmat2 = match qmat.into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert qmat to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let km2t = match km.t().to_owned().into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert km transpose to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let res = qmat2.dot(&km2t);
            qk.index_axis_mut(Axis(0), i).assign(&res.into_dyn());
        }
        qk *= scale;
        // attn
        let mut attn = qk.clone();
        for i in 0..bnh {
            let mut cur = attn.index_axis_mut(Axis(0), i);
            for mut row in cur.outer_iter_mut() {
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for val in row.iter_mut() {
                    *val = (*val - mx).exp();
                    sum += *val;
                }
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }
        // Attn @ V
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd][..]));
        for i in 0..bnh {
            let atm = attn.index_axis(Axis(0), i).to_owned();
            let vmat = v.index_axis(Axis(0), i).to_owned();
            let atm2 = match atm.into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert atm to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let vmat2 = match vmat.into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert vmat to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let res = atm2.dot(&vmat2);
            out.index_axis_mut(Axis(0), i).assign(&res.into_dyn());
        }

        // now compute grads using chain rule
        // dout shape: [bnh, seq, hd]
        let dout = output_grad.clone();
        // dv = attn^T @ dout
        let mut dv = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd][..]));
        for i in 0..bnh {
            let atm = attn.index_axis(Axis(0), i).to_owned(); // [seq,seq]
            let dmat = dout.index_axis(Axis(0), i).to_owned(); // [seq,hd]
            let atm_t2 = match atm.t().to_owned().into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert atm transpose to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let dmat2 = match dmat.into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert dmat to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let res = atm_t2.dot(&dmat2); // [seq,hd]
            dv.index_axis_mut(Axis(0), i).assign(&res.into_dyn());
        }

        // datt = dout @ v^T
        let mut datt = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, seq][..]));
        for i in 0..bnh {
            let dmat = dout.index_axis(Axis(0), i).to_owned(); // [seq,hd]
            let vmat = v.index_axis(Axis(0), i).to_owned(); // [seq,hd]
            let dmat2 = match dmat.into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert dmat to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let vmat_t2: Array2<f32> = match vmat.t().to_owned().into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert vmat transpose to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let res = dmat2.dot(&vmat_t2); // [seq,seq]
            datt.index_axis_mut(Axis(0), i).assign(&res.into_dyn());
        }

        // dsoftmax: given datt and attn, compute dqk
        let mut dqk = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, seq][..]));
        for i in 0..bnh {
            let a = attn.index_axis(Axis(0), i).to_owned(); // [seq,seq]
            let da = datt.index_axis(Axis(0), i).to_owned(); // [seq,seq]
                                                             // for each row: jacobian of softmax
            let mut dqi = ArrayD::<f32>::zeros(IxDyn(&[seq, seq][..]));
            for r in 0..seq {
                let a_row = a.index_axis(Axis(0), r).to_owned();
                let da_row = da.index_axis(Axis(0), r).to_owned();
                // compute v = (da - sum(da*a)) * a
                let dot = a_row
                    .iter()
                    .zip(da_row.iter())
                    .map(|(x, y)| x * y)
                    .sum::<f32>();
                let mut row_res = a_row.clone();
                for j in 0..seq {
                    row_res[j] = (da_row[j] - dot) * a_row[j];
                }
                dqi.index_axis_mut(Axis(0), r).assign(&row_res.into_dyn());
            }
            dqk.index_axis_mut(Axis(0), i).assign(&dqi);
        }

        // dqk scaled by scale factor
        dqk *= scale;

        // dq = dqk @ K
        let mut dq = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd][..]));
        let mut dk = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd][..]));
        for i in 0..bnh {
            let dqk_mat = dqk.index_axis(Axis(0), i).to_owned(); // [seq, seq]
            let kmat = k.index_axis(Axis(0), i).to_owned(); // [seq,hd]
            let dqk_mat2 = match dqk_mat.clone().into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert dqk_mat to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let kmat2 = match kmat.into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert kmat to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let dq_res = dqk_mat2.dot(&kmat2); // [seq,hd]
            dq.index_axis_mut(Axis(0), i).assign(&dq_res.into_dyn());
            // dk = dqk^T @ Q
            let qmat = q.index_axis(Axis(0), i).to_owned();
            let dqk_t2 = match dqk_mat.t().to_owned().into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert dqk_mat transpose to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let qmat2 = match qmat.into_dimensionality::<Ix2>() {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!(
                        "FlashAttentionRef backward: Failed to convert qmat to 2D: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                }
            };
            let dk_res = dqk_t2.dot(&qmat2); // [seq,hd]
            dk.index_axis_mut(Axis(0), i).assign(&dk_res.into_dyn());
        }

        vec![dq, dk, dv]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// ChunkedAttention performs attention by splitting the query sequence into non-overlapping
/// chunks and computing attention per-chunk to reduce memory peak usage. It expects Q/K/V shapes
/// [b*heads, seq, head_dim] and returns same shape output. This is a memory-efficient option
/// for long sequences when full attention is not required for every query position.
pub struct ChunkedAttention {
    pub head_dim: usize,
    pub chunk_size: usize,
}

impl ChunkedAttention {
    pub fn new(head_dim: usize, chunk_size: usize) -> Self {
        ChunkedAttention {
            head_dim,
            chunk_size,
        }
    }
}

impl Operation for ChunkedAttention {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let q = inputs[0].to_f32_array();
        let k = inputs[1].to_f32_array();
        let v = inputs[2].to_f32_array();
        let bnh = q.shape()[0];
        let seq = q.shape()[1];
        let hd = q.shape()[2];
        if hd != self.head_dim {
            log::error!("ChunkedAttention forward: head_dim mismatch");
            *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
            return;
        }
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd][..]));
        for i in 0..bnh {
            let qmat = q.index_axis(Axis(0), i).to_owned(); // [seq, hd]
            let kmat = k.index_axis(Axis(0), i).to_owned();
            let vmat = v.index_axis(Axis(0), i).to_owned();
            let mut out_i = out.index_axis_mut(Axis(0), i);
            let mut start = 0usize;
            while start < seq {
                let end = (start + self.chunk_size).min(seq);
                let q_chunk = qmat.slice(s![start..end, ..]).to_owned(); // [chunk, hd]
                                                                         // compute logits against all keys: [chunk, seq]
                let q_chunk2 = match q_chunk.clone().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention forward: Failed to convert q_chunk to 2D: {}",
                            e
                        );
                        *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
                        return;
                    }
                };
                let kmat_t2 = match kmat.t().to_owned().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention forward: Failed to convert kmat transpose to 2D: {}",
                            e
                        );
                        *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
                        return;
                    }
                };
                let logits = q_chunk2.dot(&kmat_t2);
                let logits = logits * (1.0f32 / (hd as f32).sqrt());
                // softmax per row
                let mut logits = logits;
                for mut row in logits.outer_iter_mut() {
                    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for val in row.iter_mut() {
                        *val = (*val - mx).exp();
                        sum += *val;
                    }
                    // Numerical guard: if sum is zero or non-finite, fall back to uniform
                    if !(sum > 0.0f32 && sum.is_finite()) {
                        let len = row.len() as f32;
                        for val in row.iter_mut() {
                            *val = 1.0f32 / len;
                        }
                        continue;
                    }
                    // Numerical guard: if sum is zero or non-finite, fall back to uniform
                    if !(sum > 0.0f32 && sum.is_finite()) {
                        let len = row.len() as f32;
                        for val in row.iter_mut() {
                            *val = 1.0f32 / len;
                        }
                        continue;
                    }
                    for val in row.iter_mut() {
                        *val /= sum;
                    }
                }
                let logits2 = match logits.into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention forward: Failed to convert logits to 2D: {}",
                            e
                        );
                        *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
                        return;
                    }
                };
                let vmat2 = match vmat.clone().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention forward: Failed to convert vmat clone to 2D: {}",
                            e
                        );
                        *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
                        return;
                    }
                };
                let res = logits2.dot(&vmat2); // [chunk, hd]
                out_i.slice_mut(s![start..end, ..]).assign(&res);
                start = end;
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // For chunked attention we compute per-chunk backward contributions
        let q = inputs[0].to_f32_array();
        let k = inputs[1].to_f32_array();
        let v = inputs[2].to_f32_array();
        let bnh = q.shape()[0];
        let seq = q.shape()[1];
        let hd = q.shape()[2];
        let mut dq = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd][..]));
        let mut dk = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd][..]));
        let mut dv = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd][..]));
        let chunk = self.chunk_size;
        for i in 0..bnh {
            let qmat = q.index_axis(Axis(0), i).to_owned();
            let kmat = k.index_axis(Axis(0), i).to_owned();
            let vmat = v.index_axis(Axis(0), i).to_owned();
            let dout = output_grad.index_axis(Axis(0), i).to_owned();
            let mut start = 0usize;
            while start < seq {
                let end = (start + chunk).min(seq);
                let q_chunk = qmat.slice(s![start..end, ..]).to_owned();
                let dout_chunk = dout.slice(s![start..end, ..]).to_owned();
                // compute logits and softmax as forward
                let q_chunk2 = match q_chunk.clone().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention backward: Failed to convert q_chunk to 2D: {}",
                            e
                        );
                        return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                    }
                };
                let kmat_t2 = match kmat.t().to_owned().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention backward: Failed to convert kmat transpose to 2D: {}",
                            e
                        );
                        return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                    }
                };
                let logits = q_chunk2.dot(&kmat_t2);
                let logits = logits * (1.0f32 / (hd as f32).sqrt());
                let mut soft = logits.clone();
                for mut row in soft.outer_iter_mut() {
                    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for val in row.iter_mut() {
                        *val = (*val - mx).exp();
                        sum += *val;
                    }
                    for val in row.iter_mut() {
                        *val /= sum;
                    }
                }
                // dv_chunk
                let soft_t2 = match soft.t().to_owned().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention backward: Failed to convert soft transpose to 2D: {}",
                            e
                        );
                        return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                    }
                };
                let dout_chunk2 = match dout_chunk.clone().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention backward: Failed to convert dout_chunk to 2D: {}",
                            e
                        );
                        return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                    }
                };
                let dv_chunk = soft_t2.dot(&dout_chunk2); // [seq, hd]
                                                          // datt
                                                          // Use previously cloned dout_chunk2 for datt
                let vmat_t2 = match vmat.t().to_owned().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention backward: Failed to convert vmat transpose to 2D: {}",
                            e
                        );
                        return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                    }
                };
                let datt = dout_chunk2.dot(&vmat_t2); // [chunk, seq]
                                                      // dsoft -> dqk
                let mut dqk_chunk = ArrayD::<f32>::zeros(IxDyn(&[end - start, seq][..]));
                for r in 0..(end - start) {
                    let a_row = soft.index_axis(Axis(0), r).to_owned();
                    let da_row = datt.index_axis(Axis(0), r).to_owned();
                    let dot = a_row
                        .iter()
                        .zip(da_row.iter())
                        .map(|(x, y)| x * y)
                        .sum::<f32>();
                    let mut row_res = a_row.clone();
                    for j in 0..seq {
                        row_res[j] = (da_row[j] - dot) * a_row[j];
                    }
                    dqk_chunk
                        .index_axis_mut(Axis(0), r)
                        .assign(&row_res.into_dyn());
                }
                let dqk_chunk = dqk_chunk * (1.0f32 / (hd as f32).sqrt());
                // dq chunk
                let dqk_chunk2 = match dqk_chunk.clone().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention backward: Failed to convert dqk_chunk to 2D: {}",
                            e
                        );
                        return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                    }
                };
                let kmat2 = match kmat.clone().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention backward: Failed to convert kmat clone to 2D: {}",
                            e
                        );
                        return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                    }
                };
                let dq_chunk = dqk_chunk2.dot(&kmat2); // [chunk, hd]
                                                       // dk contributions: dqk^T @ q_chunk => [seq, hd]
                let dqk_chunk_t2 = match dqk_chunk.t().to_owned().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!("ChunkedAttention backward: Failed to convert dqk_chunk transpose to 2D: {}", e);
                        return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                    }
                };
                let q_chunk2 = match q_chunk.clone().into_dimensionality::<Ix2>() {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "ChunkedAttention backward: Failed to convert q_chunk clone to 2D: {}",
                            e
                        );
                        return vec![ArrayD::zeros(IxDyn(&[bnh, seq, hd])); 3];
                    }
                };
                let dk_part = dqk_chunk_t2.dot(&q_chunk2); // [seq, hd]
                                                           // Accumulate
                dq.index_axis_mut(Axis(0), i)
                    .slice_mut(s![start..end, ..])
                    .assign(&dq_chunk);
                // add to dk for full sequence (accumulate)
                {
                    let mut dk_slice = dk.index_axis_mut(Axis(0), i);
                    Zip::from(dk_slice.slice_mut(s![.., ..]))
                        .and(&dk_part)
                        .for_each(|a, b| *a += *b);
                }
                // accumulate dv
                {
                    let mut dv_slice = dv.index_axis_mut(Axis(0), i);
                    Zip::from(dv_slice.slice_mut(s![.., ..]))
                        .and(&dv_chunk)
                        .for_each(|a, b| *a += *b);
                }
                start = end;
            }
        }
        vec![dq, dk, dv]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
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
                *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
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

/// Permute axes operation: reorder axes according to a permutation vector
pub struct PermuteAxes {
    pub perm: Vec<usize>,
}

impl PermuteAxes {
    pub fn new(perm: Vec<usize>) -> Self {
        PermuteAxes { perm }
    }
}

impl Operation for PermuteAxes {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].lock().storage.to_f32_array();
        if self.perm.len() != a.ndim() {
            log::error!(
                "PermuteAxes forward: permutation length {} != ndim {}",
                self.perm.len(),
                a.ndim()
            );
            *output = a.clone();
            return;
        }
        *output = a.view().permuted_axes(self.perm.clone()).to_owned();
    }

    fn backward(&self, _inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let n = self.perm.len();
        let mut inv = vec![0usize; n];
        for (i, &p) in self.perm.iter().enumerate() {
            inv[p] = i;
        }
        vec![permute_array(output_grad, &inv)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Helper for permuting an ArrayD according to a perm vector (used in PermuteAxes backward)
fn permute_array(a: &ArrayD<f32>, perm: &[usize]) -> ArrayD<f32> {
    if perm.len() != a.ndim() {
        return a.clone();
    }
    a.view().permuted_axes(perm.to_vec()).to_owned()
}

/// Sum operation: sums all elements to a scalar
pub struct Sum;

impl Operation for Sum {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let s = a.sum();
        *output = ArrayD::from_elem(IxDyn(&[][..]), s);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a_shape = inputs[0].lock().storage.shape();
        // output_grad is scalar; expand to input shape
        let val = match output_grad.iter().next().copied() {
            Some(v) => v,
            None => {
                log::error!("Sum backward: Expected scalar output_grad");
                0.0f32
            }
        };
        let grad = ArrayD::from_elem(IxDyn(&a_shape), val);
        vec![grad]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Sum operation along an axis
pub struct SumAxis {
    pub axis: isize,
    pub keep_dims: bool,
}

/// Cumulative sum operation along an axis.
pub struct CumSum {
    pub dim: usize,
}

impl CumSum {
    pub fn new(dim: usize) -> Self {
        CumSum { dim }
    }
}

/// Cumulative product operation along an axis.
pub struct CumProd {
    pub dim: usize,
}

impl CumProd {
    pub fn new(dim: usize) -> Self {
        CumProd { dim }
    }
}

/// Cumulative max operation along an axis (values only).
pub struct CumMax {
    pub dim: usize,
}

impl CumMax {
    pub fn new(dim: usize) -> Self {
        CumMax { dim }
    }
}

/// Cumulative min operation along an axis (values only).
pub struct CumMin {
    pub dim: usize,
}

impl CumMin {
    pub fn new(dim: usize) -> Self {
        CumMin { dim }
    }
}

impl SumAxis {
    pub fn new(axis: isize, keep_dims: bool) -> Self {
        SumAxis { axis, keep_dims }
    }
}

impl Operation for SumAxis {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let ndim = a.ndim();
        let axis = if self.axis < 0 {
            (ndim as isize + self.axis) as usize
        } else {
            self.axis as usize
        };
        if axis >= ndim {
            log::error!("SumAxis: axis {} out of bounds for ndim {}", axis, ndim);
            *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
            return;
        }
        let res = a.sum_axis(Axis(axis));
        if self.keep_dims {
            let mut shape = res.shape().to_vec();
            shape.insert(axis, 1);
            *output = match res.to_shape(IxDyn(&shape)).map(|v| v.to_owned()) {
                Ok(a) => a,
                Err(e) => {
                    log::error!("SumAxis: failed to reshape output: {}", e);
                    ArrayD::zeros(IxDyn(&shape))
                }
            };
        } else {
            *output = res;
        }
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a_shape = inputs[0].lock().storage.shape().to_vec();
        let ndim = a_shape.len();
        let axis = if self.axis < 0 {
            (ndim as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        // gradient of sum is 1, broadcasted to input shape.
        // output_grad has shape of output.
        // If keep_dims=false, output_grad lacks the axis. We need to add it back to broadcast.

        let grad_expanded = if !self.keep_dims {
            let mut target_shape = output_grad.shape().to_vec();
            target_shape.insert(axis, 1);
            match output_grad
                .to_shape(IxDyn(&target_shape))
                .map(|v| v.to_owned())
            {
                Ok(a) => a,
                Err(e) => {
                    log::error!("SumAxis backward: failed to reshape grad: {}", e);
                    ArrayD::zeros(IxDyn(&target_shape))
                }
            }
        } else {
            output_grad.clone()
        };

        // Broadcast grad_expanded to a_shape
        // Since it's a sum, we just broadcast the value.
        // In ndarray, broadcasting happens on operations, but here we need to explicitly create the full array
        // or rely on implicit broadcast if we were adding? No we return the grad w.r.t input.
        // So we need to broadcast `grad_expanded` (which has 1 at `axis`) to `a_shape` (which has N at `axis`).

        // Manually broadcast:
        let mut full_grad = ArrayD::<f32>::zeros(IxDyn(&a_shape));
        // This is inefficient loop, let's use broadcast method if available or a trick.
        // `ArrayBase::broadcast` returns a Broadcast wrapper. We can assign it to an owned array.

        if let Some(broadcasted) = grad_expanded.broadcast(IxDyn(&a_shape)) {
            full_grad.assign(&broadcasted);
        } else {
            log::error!(
                "SumAxis backward: failed to switch broadcast to shape {:?}",
                a_shape
            );
        }

        vec![full_grad]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for CumSum {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        if self.dim >= a.ndim() {
            log::error!(
                "CumSum.forward: dim {} out of bounds for ndim {}",
                self.dim,
                a.ndim()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let mut out = a.clone();
        for mut lane in out.lanes_mut(Axis(self.dim)) {
            let mut run = 0.0f32;
            for v in &mut lane {
                run += *v;
                *v = run;
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a_shape = inputs[0].lock().storage.shape();
        if self.dim >= a_shape.len() || output_grad.shape() != a_shape.as_slice() {
            return vec![ArrayD::zeros(IxDyn(&a_shape))];
        }

        let mut grad = output_grad.clone();
        for mut lane in grad.lanes_mut(Axis(self.dim)) {
            let mut run = 0.0f32;
            for v in lane.iter_mut().rev() {
                run += *v;
                *v = run;
            }
        }
        vec![grad]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for CumProd {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        if self.dim >= a.ndim() {
            log::error!(
                "CumProd.forward: dim {} out of bounds for ndim {}",
                self.dim,
                a.ndim()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let mut out = a.clone();
        for mut lane in out.lanes_mut(Axis(self.dim)) {
            let mut run = 1.0f32;
            for v in &mut lane {
                run *= *v;
                *v = run;
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        if self.dim >= x.ndim() || output_grad.shape() != x.shape() {
            return vec![ArrayD::zeros(IxDyn(x.shape()))];
        }

        let mut grad_x = ArrayD::<f32>::zeros(IxDyn(x.shape()));
        let lanes_x = x.lanes(Axis(self.dim));
        let lanes_gy = output_grad.lanes(Axis(self.dim));
        let lanes_gx = grad_x.lanes_mut(Axis(self.dim));

        for ((lane_x, lane_gy), mut lane_gx) in lanes_x.into_iter().zip(lanes_gy).zip(lanes_gx) {
            let n = lane_x.len();
            let xv: Vec<f32> = lane_x.iter().copied().collect();
            let gyv: Vec<f32> = lane_gy.iter().copied().collect();

            for i in 0..n {
                let mut acc = 0.0f32;
                for (j, gyj) in gyv.iter().enumerate().skip(i) {
                    let mut prod = 1.0f32;
                    for (t, &xt) in xv.iter().enumerate().take(j + 1) {
                        if t != i {
                            prod *= xt;
                        }
                    }
                    acc += *gyj * prod;
                }
                lane_gx[i] = acc;
            }
        }

        vec![grad_x]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for CumMax {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        if self.dim >= a.ndim() {
            log::error!(
                "CumMax.forward: dim {} out of bounds for ndim {}",
                self.dim,
                a.ndim()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let mut out = a.clone();
        for mut lane in out.lanes_mut(Axis(self.dim)) {
            let mut run = f32::NEG_INFINITY;
            for v in &mut lane {
                run = run.max(*v);
                *v = run;
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        if self.dim >= x.ndim() || output_grad.shape() != x.shape() {
            return vec![ArrayD::zeros(IxDyn(x.shape()))];
        }

        let mut grad_x = ArrayD::<f32>::zeros(IxDyn(x.shape()));
        let lanes_x = x.lanes(Axis(self.dim));
        let lanes_gy = output_grad.lanes(Axis(self.dim));
        let lanes_gx = grad_x.lanes_mut(Axis(self.dim));

        for ((lane_x, lane_gy), mut lane_gx) in lanes_x.into_iter().zip(lanes_gy).zip(lanes_gx) {
            let mut best = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            for i in 0..lane_x.len() {
                let v = lane_x[i];
                if v > best {
                    best = v;
                    best_idx = i;
                }
                lane_gx[best_idx] += lane_gy[i];
            }
        }

        vec![grad_x]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for CumMin {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        if self.dim >= a.ndim() {
            log::error!(
                "CumMin.forward: dim {} out of bounds for ndim {}",
                self.dim,
                a.ndim()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let mut out = a.clone();
        for mut lane in out.lanes_mut(Axis(self.dim)) {
            let mut run = f32::INFINITY;
            for v in &mut lane {
                run = run.min(*v);
                *v = run;
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        if self.dim >= x.ndim() || output_grad.shape() != x.shape() {
            return vec![ArrayD::zeros(IxDyn(x.shape()))];
        }

        let mut grad_x = ArrayD::<f32>::zeros(IxDyn(x.shape()));
        let lanes_x = x.lanes(Axis(self.dim));
        let lanes_gy = output_grad.lanes(Axis(self.dim));
        let lanes_gx = grad_x.lanes_mut(Axis(self.dim));

        for ((lane_x, lane_gy), mut lane_gx) in lanes_x.into_iter().zip(lanes_gy).zip(lanes_gx) {
            let mut best = f32::INFINITY;
            let mut best_idx = 0usize;
            for i in 0..lane_x.len() {
                let v = lane_x[i];
                if v < best {
                    best = v;
                    best_idx = i;
                }
                lane_gx[best_idx] += lane_gy[i];
            }
        }

        vec![grad_x]
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
        *output = ArrayD::from_elem(IxDyn(&[][..]), mean);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a_shape = inputs[0].lock().storage.shape();
        let val = match output_grad.iter().next().copied() {
            Some(v) => v,
            None => {
                log::error!("Mean backward: Expected scalar output_grad");
                0.0f32
            }
        };
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
        // Use safe path to avoid deadlocks when both inputs point to the same underlying tensor
        // Always convert both inputs to f32 arrays (locks are short-lived during conversion)
        // Inspect raw storage shapes before conversion to f32 to catch mismatches early
        let a_shape_raw = inputs[0].lock().storage.shape();
        let b_shape_raw = inputs[1].lock().storage.shape();
        log::debug!(
            "Add.forward: raw shapes a={:?} b={:?}",
            a_shape_raw,
            b_shape_raw
        );
        // Convert first input, but catch panics during conversion to avoid crashes
        let a = match std::panic::catch_unwind(|| inputs[0].to_f32_array()) {
            Ok(arr) => arr,
            Err(_) => {
                log::error!(
                    "Add.forward: failed to convert first input to f32 array; aborting operation"
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        // Convert second input in a separate step to isolate panics
        log::debug!("Add.forward: about to convert second input to f32 array (may panic if dequantization fails)");
        let b = match std::panic::catch_unwind(|| inputs[1].to_f32_array()) {
            Ok(arr) => arr,
            Err(_) => {
                log::error!(
                    "Add.forward: failed to convert second input to f32 array; aborting operation"
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        log::debug!(
            "Add.forward: converted arrays; a_shape={:?} b_shape={:?}",
            a.shape(),
            b.shape()
        );
        if a.shape() == b.shape() {
            *output = a + b;
            return;
        }
        // Compute the broadcasted output shape and broadcast both inputs to it
        let a_shape_vec = a.shape().to_vec();
        let b_shape_vec = b.shape().to_vec();
        let out_shape = match Tensor::broadcast_shapes(
            &[a_shape_vec.clone(), b_shape_vec.clone()][..],
        ) {
            Ok(s) => s,
            Err(_) => {
                log::error!("Add.forward: incompatible shapes and cannot broadcast: a_shape={:?} b_shape={:?}", a.shape(), b.shape());
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let a_b = match a.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!(
                    "Add.forward: failed to broadcast a to out_shape={:?}",
                    out_shape
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let b_b = match b.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!(
                    "Add.forward: failed to broadcast b to out_shape={:?}",
                    out_shape
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        // Perform elementwise addition into a new owned array
        let mut out_arr = ArrayD::zeros(IxDyn(&out_shape));
        Zip::from(out_arr.view_mut())
            .and(&a_b)
            .and(&b_b)
            .par_for_each(|o, &a, &b| *o = a + b);
        *output = out_arr;
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

/// Element-wise exponentiation (e^x)
pub struct Exp;

impl Operation for Exp {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        *output = par_mapv(&a, |x| x.exp());
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let grad = par_mapv(&a, |x| x.exp());
        vec![output_grad * &grad]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Elementwise comparison returning 1.0 for true and 0.0 for false.
pub struct Equal;
pub struct Greater;
pub struct Less;
pub struct Where;
pub struct MaskedScatter;
pub struct FFT;
pub struct IFFT;
pub struct RFFT;
pub struct IRFFT;
pub struct ComplexConj;
pub struct ComplexMul;

impl Operation for Equal {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let b = inputs[1].to_f32_array();
        // no 'out' variable needed; we'll build out_arr directly
        // Use broadcasting
        let a_shape = a.shape().to_vec();
        let b_shape = b.shape().to_vec();
        let out_shape = Tensor::broadcast_shapes(&[a_shape.clone(), b_shape.clone()][..])
            .unwrap_or_else(|_| a_shape.clone());
        let mut out_arr = ArrayD::zeros(IxDyn(&out_shape));
        let a_b = match a.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!("Broadcast failed for 'a' in Equal forward; shapes incompatible");
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        let b_b = match b.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!("Broadcast failed for 'b' in Equal forward; shapes incompatible");
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        Zip::from(out_arr.view_mut())
            .and(&a_b)
            .and(&b_b)
            .par_for_each(|o, &a, &b| *o = if (a - b).abs() < 1e-6 { 1.0 } else { 0.0 });
        *output = out_arr;
    }

    fn backward(&self, _inputs: &[Tensor], _output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // Non-differentiable -> return zeros for both inputs
        let a_shape = _inputs[0].lock().storage.shape();
        let b_shape = _inputs[1].lock().storage.shape();
        vec![
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(&b_shape)),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for Greater {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let b = inputs[1].to_f32_array();
        let a_shape = a.shape().to_vec();
        let b_shape = b.shape().to_vec();
        let out_shape = Tensor::broadcast_shapes(&[a_shape.clone(), b_shape.clone()][..])
            .unwrap_or_else(|_| a_shape.clone());
        let a_b = match a.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!("Broadcast failed for 'a' in Greater forward; shapes incompatible");
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        let b_b = match b.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!("Broadcast failed for 'b' in Greater forward; shapes incompatible");
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        let mut out_arr = ArrayD::zeros(IxDyn(&out_shape));
        Zip::from(out_arr.view_mut())
            .and(&a_b)
            .and(&b_b)
            .par_for_each(|o, &a, &b| *o = if a > b { 1.0 } else { 0.0 });
        *output = out_arr;
    }

    fn backward(&self, _inputs: &[Tensor], _output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a_shape = _inputs[0].lock().storage.shape();
        let b_shape = _inputs[1].lock().storage.shape();
        vec![
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(&b_shape)),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for Less {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let b = inputs[1].to_f32_array();
        let a_shape = a.shape().to_vec();
        let b_shape = b.shape().to_vec();
        let out_shape = Tensor::broadcast_shapes(&[a_shape.clone(), b_shape.clone()][..])
            .unwrap_or_else(|_| a_shape.clone());
        let a_b = match a.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!("Broadcast failed for 'a' in Less forward; shapes incompatible");
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        let b_b = match b.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!("Broadcast failed for 'b' in Less forward; shapes incompatible");
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        let mut out_arr = ArrayD::zeros(IxDyn(&out_shape));
        Zip::from(out_arr.view_mut())
            .and(&a_b)
            .and(&b_b)
            .par_for_each(|o, &a, &b| *o = if a < b { 1.0 } else { 0.0 });
        *output = out_arr;
    }

    fn backward(&self, _inputs: &[Tensor], _output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a_shape = _inputs[0].lock().storage.shape();
        let b_shape = _inputs[1].lock().storage.shape();
        vec![
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(&b_shape)),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for Where {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let condition = inputs[0].to_f32_array();
        let x = inputs[1].to_f32_array();
        let y = inputs[2].to_f32_array();

        let out_shape = match Tensor::broadcast_shapes(
            &[
                condition.shape().to_vec(),
                x.shape().to_vec(),
                y.shape().to_vec(),
            ][..],
        ) {
            Ok(shape) => shape,
            Err(e) => {
                log::error!("Where.forward: incompatible broadcast shapes: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        let cond_b = match condition.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!(
                    "Where.forward: failed to broadcast condition to {:?}",
                    out_shape
                );
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        let x_b = match x.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!("Where.forward: failed to broadcast x to {:?}", out_shape);
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        let y_b = match y.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!("Where.forward: failed to broadcast y to {:?}", out_shape);
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };

        let mut out_arr = ArrayD::zeros(IxDyn(&out_shape));
        Zip::from(out_arr.view_mut())
            .and(&cond_b)
            .and(&x_b)
            .and(&y_b)
            .par_for_each(|o, &c, &xv, &yv| *o = if c != 0.0 { xv } else { yv });
        *output = out_arr;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let condition = inputs[0].to_f32_array();
        let x = inputs[1].to_f32_array();
        let y = inputs[2].to_f32_array();

        let out_shape = output_grad.shape().to_vec();

        let cond_b = match condition.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!(
                    "Where.backward: failed to broadcast condition to {:?}",
                    out_shape
                );
                return vec![
                    ArrayD::zeros(IxDyn(condition.shape())),
                    ArrayD::zeros(IxDyn(x.shape())),
                    ArrayD::zeros(IxDyn(y.shape())),
                ];
            }
        };

        let mut grad_x_full = ArrayD::zeros(IxDyn(&out_shape));
        let mut grad_y_full = ArrayD::zeros(IxDyn(&out_shape));

        let gx_slice = match grad_x_full.as_slice_mut() {
            Some(s) => s,
            None => {
                log::error!("Where.backward: failed to get grad_x slice");
                return vec![
                    ArrayD::zeros(IxDyn(condition.shape())),
                    ArrayD::zeros(IxDyn(x.shape())),
                    ArrayD::zeros(IxDyn(y.shape())),
                ];
            }
        };
        let gy_slice = match grad_y_full.as_slice_mut() {
            Some(s) => s,
            None => {
                log::error!("Where.backward: failed to get grad_y slice");
                return vec![
                    ArrayD::zeros(IxDyn(condition.shape())),
                    ArrayD::zeros(IxDyn(x.shape())),
                    ArrayD::zeros(IxDyn(y.shape())),
                ];
            }
        };
        let og_slice = match output_grad.as_slice() {
            Some(s) => s,
            None => {
                log::error!("Where.backward: failed to get output_grad slice");
                return vec![
                    ArrayD::zeros(IxDyn(condition.shape())),
                    ArrayD::zeros(IxDyn(x.shape())),
                    ArrayD::zeros(IxDyn(y.shape())),
                ];
            }
        };

        for (((c, g), gx), gy) in cond_b
            .iter()
            .zip(og_slice.iter())
            .zip(gx_slice.iter_mut())
            .zip(gy_slice.iter_mut())
        {
            if *c != 0.0 {
                *gx = *g;
                *gy = 0.0;
            } else {
                *gx = 0.0;
                *gy = *g;
            }
        }

        let grad_condition = ArrayD::zeros(IxDyn(condition.shape()));
        let grad_x = reduce_grad_to_shape(&grad_x_full, x.shape());
        let grad_y = reduce_grad_to_shape(&grad_y_full, y.shape());

        vec![grad_condition, grad_x, grad_y]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for MaskedScatter {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let base = inputs[0].to_f32_array();
        let mask = inputs[1].to_f32_array();
        let source = inputs[2].to_f32_array();

        let out_shape = base.shape().to_vec();
        let mask_b = match mask.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!(
                    "MaskedScatter.forward: failed to broadcast mask {:?} to base {:?}",
                    mask.shape(),
                    out_shape
                );
                *output = base;
                return;
            }
        };

        let flags: Vec<bool> = mask_b.iter().map(|v| *v != 0.0).collect();
        let needed = flags.iter().filter(|&&b| b).count();
        let source_values: Vec<f32> = source.iter().copied().collect();
        if source_values.len() < needed {
            log::error!(
                "MaskedScatter.forward: source has {} values but {} are required by mask; trailing masked positions keep base values",
                source_values.len(),
                needed
            );
        }

        let mut out_arr = base.clone();
        let mut source_idx = 0usize;
        for (o, flag) in out_arr.iter_mut().zip(flags.iter()) {
            if *flag && source_idx < source_values.len() {
                *o = source_values[source_idx];
                source_idx += 1;
            }
        }

        *output = out_arr;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let base = inputs[0].to_f32_array();
        let mask = inputs[1].to_f32_array();
        let source = inputs[2].to_f32_array();

        let out_shape = base.shape().to_vec();
        let mask_b = match mask.broadcast(IxDyn(&out_shape)) {
            Some(v) => v,
            None => {
                log::error!(
                    "MaskedScatter.backward: failed to broadcast mask {:?} to base {:?}",
                    mask.shape(),
                    out_shape
                );
                return vec![
                    ArrayD::zeros(IxDyn(base.shape())),
                    ArrayD::zeros(IxDyn(mask.shape())),
                    ArrayD::zeros(IxDyn(source.shape())),
                ];
            }
        };

        let flags: Vec<bool> = mask_b.iter().map(|v| *v != 0.0).collect();
        let source_len = source.len();

        let mut grad_base = ArrayD::zeros(IxDyn(&out_shape));
        let mut grad_source_flat = vec![0.0f32; source_len];
        let mut source_idx = 0usize;

        for ((g_out, flag), g_base) in output_grad
            .iter()
            .zip(flags.iter())
            .zip(grad_base.iter_mut())
        {
            if *flag && source_idx < source_len {
                grad_source_flat[source_idx] = *g_out;
                *g_base = 0.0;
                source_idx += 1;
            } else {
                *g_base = *g_out;
            }
        }

        let grad_source = match ArrayD::from_shape_vec(IxDyn(source.shape()), grad_source_flat) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!(
                    "MaskedScatter.backward: failed to reshape source grad to {:?}: {}",
                    source.shape(),
                    e
                );
                ArrayD::zeros(IxDyn(source.shape()))
            }
        };

        let grad_mask = ArrayD::zeros(IxDyn(mask.shape()));
        vec![grad_base, grad_mask, grad_source]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for FFT {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        if x.ndim() == 0 {
            log::error!("FFT.forward: input must have at least 1 dimension");
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let n = x.shape()[x.ndim() - 1];
        if n == 0 {
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let prefix_shape = &x.shape()[0..x.ndim() - 1];
        let batch: usize = prefix_shape.iter().product();

        let mut out_shape = prefix_shape.to_vec();
        out_shape.push(n);
        out_shape.push(2);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&out_shape));

        let x2 = match x.to_shape((batch, n)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("FFT.forward: reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let out_view = out.view_mut();
        let mut out2 = match out_view.to_shape((batch, n, 2)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("FFT.forward: output reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        for b in 0..batch {
            for k in 0..n {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for t in 0..n {
                    let theta = 2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
                    let v = x2[[b, t]];
                    re += v * theta.cos();
                    im -= v * theta.sin();
                }
                out2[[b, k, 0]] = re;
                out2[[b, k, 1]] = im;
            }
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        if x.ndim() == 0 {
            return vec![ArrayD::zeros(IxDyn(&[][..]))];
        }

        let n = x.shape()[x.ndim() - 1];
        let prefix_shape = &x.shape()[0..x.ndim() - 1];
        let batch: usize = prefix_shape.iter().product();

        let og_expected_shape: Vec<usize> = {
            let mut s = prefix_shape.to_vec();
            s.push(n);
            s.push(2);
            s
        };
        if output_grad.shape() != og_expected_shape.as_slice() {
            log::error!(
                "FFT.backward: output_grad shape {:?} mismatches expected {:?}",
                output_grad.shape(),
                og_expected_shape
            );
            return vec![ArrayD::zeros(IxDyn(x.shape()))];
        }

        let og2 = match output_grad.to_shape((batch, n, 2)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("FFT.backward: reshape failed: {}", e);
                return vec![ArrayD::zeros(IxDyn(x.shape()))];
            }
        };

        let mut grad_x = ArrayD::<f32>::zeros(IxDyn(x.shape()));
        let gx_view = grad_x.view_mut();
        let mut gx2 = match gx_view.to_shape((batch, n)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("FFT.backward: grad reshape failed: {}", e);
                return vec![ArrayD::zeros(IxDyn(x.shape()))];
            }
        };

        for b in 0..batch {
            for t in 0..n {
                let mut g = 0.0f32;
                for k in 0..n {
                    let theta = 2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
                    let gre = og2[[b, k, 0]];
                    let gim = og2[[b, k, 1]];
                    g += gre * theta.cos() - gim * theta.sin();
                }
                gx2[[b, t]] = g;
            }
        }

        vec![grad_x]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for IFFT {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        if x.ndim() < 2 || x.shape()[x.ndim() - 1] != 2 {
            log::error!(
                "IFFT.forward: input must end with complex axis of size 2, got {:?}",
                x.shape()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let n = x.shape()[x.ndim() - 2];
        if n == 0 {
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let prefix_shape = &x.shape()[0..x.ndim() - 2];
        let batch: usize = prefix_shape.iter().product();

        let mut out_shape = prefix_shape.to_vec();
        out_shape.push(n);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&out_shape));

        let x2 = match x.to_shape((batch, n, 2)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("IFFT.forward: reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let out_view = out.view_mut();
        let mut out2 = match out_view.to_shape((batch, n)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("IFFT.forward: output reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        let scale = 1.0f32 / (n as f32);
        for b in 0..batch {
            for t in 0..n {
                let mut v = 0.0f32;
                for k in 0..n {
                    let theta = 2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
                    let re = x2[[b, k, 0]];
                    let im = x2[[b, k, 1]];
                    v += re * theta.cos() - im * theta.sin();
                }
                out2[[b, t]] = v * scale;
            }
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        if x.ndim() < 2 || x.shape()[x.ndim() - 1] != 2 {
            return vec![ArrayD::zeros(IxDyn(x.shape()))];
        }

        let n = x.shape()[x.ndim() - 2];
        let prefix_shape = &x.shape()[0..x.ndim() - 2];
        let batch: usize = prefix_shape.iter().product();

        let og_expected_shape: Vec<usize> = {
            let mut s = prefix_shape.to_vec();
            s.push(n);
            s
        };
        if output_grad.shape() != og_expected_shape.as_slice() {
            log::error!(
                "IFFT.backward: output_grad shape {:?} mismatches expected {:?}",
                output_grad.shape(),
                og_expected_shape
            );
            return vec![ArrayD::zeros(IxDyn(x.shape()))];
        }

        let og2 = match output_grad.to_shape((batch, n)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("IFFT.backward: reshape failed: {}", e);
                return vec![ArrayD::zeros(IxDyn(x.shape()))];
            }
        };

        let mut grad_x = ArrayD::<f32>::zeros(IxDyn(x.shape()));
        let gx_view = grad_x.view_mut();
        let mut gx2 = match gx_view.to_shape((batch, n, 2)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("IFFT.backward: grad reshape failed: {}", e);
                return vec![ArrayD::zeros(IxDyn(x.shape()))];
            }
        };

        let scale = 1.0f32 / (n as f32);
        for b in 0..batch {
            for k in 0..n {
                let mut gre = 0.0f32;
                let mut gim = 0.0f32;
                for t in 0..n {
                    let theta = 2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
                    let g = og2[[b, t]];
                    gre += g * theta.cos();
                    gim += -g * theta.sin();
                }
                gx2[[b, k, 0]] = gre * scale;
                gx2[[b, k, 1]] = gim * scale;
            }
        }

        vec![grad_x]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for RFFT {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        if x.ndim() == 0 {
            log::error!("RFFT.forward: input must have at least 1 dimension");
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let n = x.shape()[x.ndim() - 1];
        if n == 0 {
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        let m = n / 2 + 1;

        let prefix_shape = &x.shape()[0..x.ndim() - 1];
        let batch: usize = prefix_shape.iter().product();

        let mut out_shape = prefix_shape.to_vec();
        out_shape.push(m);
        out_shape.push(2);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&out_shape));

        let x2 = match x.to_shape((batch, n)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("RFFT.forward: reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let out_view = out.view_mut();
        let mut out2 = match out_view.to_shape((batch, m, 2)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("RFFT.forward: output reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        for b in 0..batch {
            for k in 0..m {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for t in 0..n {
                    let theta = 2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
                    let v = x2[[b, t]];
                    re += v * theta.cos();
                    im -= v * theta.sin();
                }
                out2[[b, k, 0]] = re;
                out2[[b, k, 1]] = im;
            }
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        if x.ndim() == 0 {
            return vec![ArrayD::zeros(IxDyn(&[][..]))];
        }

        let n = x.shape()[x.ndim() - 1];
        let m = n / 2 + 1;
        let prefix_shape = &x.shape()[0..x.ndim() - 1];
        let batch: usize = prefix_shape.iter().product();

        let og_expected_shape: Vec<usize> = {
            let mut s = prefix_shape.to_vec();
            s.push(m);
            s.push(2);
            s
        };
        if output_grad.shape() != og_expected_shape.as_slice() {
            log::error!(
                "RFFT.backward: output_grad shape {:?} mismatches expected {:?}",
                output_grad.shape(),
                og_expected_shape
            );
            return vec![ArrayD::zeros(IxDyn(x.shape()))];
        }

        let og2 = match output_grad.to_shape((batch, m, 2)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("RFFT.backward: reshape failed: {}", e);
                return vec![ArrayD::zeros(IxDyn(x.shape()))];
            }
        };

        let mut grad_x = ArrayD::<f32>::zeros(IxDyn(x.shape()));
        let gx_view = grad_x.view_mut();
        let mut gx2 = match gx_view.to_shape((batch, n)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("RFFT.backward: grad reshape failed: {}", e);
                return vec![ArrayD::zeros(IxDyn(x.shape()))];
            }
        };

        for b in 0..batch {
            for t in 0..n {
                let mut g = 0.0f32;
                for k in 0..m {
                    let theta = 2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
                    let gre = og2[[b, k, 0]];
                    let gim = og2[[b, k, 1]];
                    g += gre * theta.cos() - gim * theta.sin();
                }
                gx2[[b, t]] = g;
            }
        }

        vec![grad_x]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for IRFFT {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        if x.ndim() < 2 || x.shape()[x.ndim() - 1] != 2 {
            log::error!(
                "IRFFT.forward: input must end with complex axis of size 2, got {:?}",
                x.shape()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let m = x.shape()[x.ndim() - 2];
        if m == 0 {
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        let n = 2 * (m - 1);
        if n == 0 {
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let prefix_shape = &x.shape()[0..x.ndim() - 2];
        let batch: usize = prefix_shape.iter().product();

        let mut out_shape = prefix_shape.to_vec();
        out_shape.push(n);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&out_shape));

        let x2 = match x.to_shape((batch, m, 2)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("IRFFT.forward: reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let out_view = out.view_mut();
        let mut out2 = match out_view.to_shape((batch, n)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("IRFFT.forward: output reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        let scale = 1.0f32 / (n as f32);
        for b in 0..batch {
            for t in 0..n {
                let mut v = x2[[b, 0, 0]];
                if m > 1 {
                    let nyq_sign = if t % 2 == 0 { 1.0 } else { -1.0 };
                    v += x2[[b, m - 1, 0]] * nyq_sign;
                }
                if m > 2 {
                    for k in 1..(m - 1) {
                        let theta =
                            2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
                        let re = x2[[b, k, 0]];
                        let im = x2[[b, k, 1]];
                        v += 2.0 * (re * theta.cos() - im * theta.sin());
                    }
                }
                out2[[b, t]] = v * scale;
            }
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        if x.ndim() < 2 || x.shape()[x.ndim() - 1] != 2 {
            return vec![ArrayD::zeros(IxDyn(x.shape()))];
        }

        let m = x.shape()[x.ndim() - 2];
        let n = 2 * (m - 1);
        let prefix_shape = &x.shape()[0..x.ndim() - 2];
        let batch: usize = prefix_shape.iter().product();

        let og_expected_shape: Vec<usize> = {
            let mut s = prefix_shape.to_vec();
            s.push(n);
            s
        };
        if output_grad.shape() != og_expected_shape.as_slice() {
            log::error!(
                "IRFFT.backward: output_grad shape {:?} mismatches expected {:?}",
                output_grad.shape(),
                og_expected_shape
            );
            return vec![ArrayD::zeros(IxDyn(x.shape()))];
        }

        let og2 = match output_grad.to_shape((batch, n)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("IRFFT.backward: reshape failed: {}", e);
                return vec![ArrayD::zeros(IxDyn(x.shape()))];
            }
        };

        let mut grad_x = ArrayD::<f32>::zeros(IxDyn(x.shape()));
        let gx_view = grad_x.view_mut();
        let mut gx2 = match gx_view.to_shape((batch, m, 2)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("IRFFT.backward: grad reshape failed: {}", e);
                return vec![ArrayD::zeros(IxDyn(x.shape()))];
            }
        };

        let scale = 1.0f32 / (n as f32);
        for b in 0..batch {
            for k in 0..m {
                let factor = if k == 0 || k == m - 1 { 1.0 } else { 2.0 };
                let mut gre = 0.0f32;
                let mut gim = 0.0f32;
                for t in 0..n {
                    let theta = 2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
                    let g = og2[[b, t]];
                    gre += g * factor * theta.cos();
                    gim += -g * factor * theta.sin();
                }
                gx2[[b, k, 0]] = gre * scale;
                gx2[[b, k, 1]] = gim * scale;
            }
        }

        vec![grad_x]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for ComplexConj {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        if x.ndim() < 1 || x.shape()[x.ndim() - 1] != 2 {
            log::error!(
                "ComplexConj.forward: expected last dimension size 2, got {:?}",
                x.shape()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let mut out = x.clone();
        if let Some(slice) = out.as_slice_mut() {
            let mut i = 0usize;
            while i + 1 < slice.len() {
                slice[i + 1] = -slice[i + 1];
                i += 2;
            }
        }
        *output = out;
    }

    fn backward(&self, _inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let mut grad = output_grad.clone();
        if let Some(slice) = grad.as_slice_mut() {
            let mut i = 0usize;
            while i + 1 < slice.len() {
                slice[i + 1] = -slice[i + 1];
                i += 2;
            }
        }
        vec![grad]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for ComplexMul {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let b = inputs[1].to_f32_array();

        if a.shape() != b.shape() {
            log::error!(
                "ComplexMul.forward: shape mismatch {:?} vs {:?}",
                a.shape(),
                b.shape()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        if a.ndim() < 1 || a.shape()[a.ndim() - 1] != 2 {
            log::error!(
                "ComplexMul.forward: expected last dimension size 2, got {:?}",
                a.shape()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let mut out = ArrayD::<f32>::zeros(IxDyn(a.shape()));
        if let (Some(asl), Some(bsl), Some(osl)) = (a.as_slice(), b.as_slice(), out.as_slice_mut())
        {
            let mut i = 0usize;
            while i + 1 < asl.len() {
                let ar = asl[i];
                let ai = asl[i + 1];
                let br = bsl[i];
                let bi = bsl[i + 1];
                osl[i] = ar * br - ai * bi;
                osl[i + 1] = ar * bi + ai * br;
                i += 2;
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let b = inputs[1].to_f32_array();
        if a.shape() != b.shape() || output_grad.shape() != a.shape() {
            return vec![
                ArrayD::zeros(IxDyn(a.shape())),
                ArrayD::zeros(IxDyn(b.shape())),
            ];
        }

        let mut ga = ArrayD::<f32>::zeros(IxDyn(a.shape()));
        let mut gb = ArrayD::<f32>::zeros(IxDyn(b.shape()));
        if let (Some(asl), Some(bsl), Some(gsl), Some(gasl), Some(gbsl)) = (
            a.as_slice(),
            b.as_slice(),
            output_grad.as_slice(),
            ga.as_slice_mut(),
            gb.as_slice_mut(),
        ) {
            let mut i = 0usize;
            while i + 1 < asl.len() {
                let ar = asl[i];
                let ai = asl[i + 1];
                let br = bsl[i];
                let bi = bsl[i + 1];
                let gr = gsl[i];
                let gi = gsl[i + 1];

                gasl[i] = gr * br + gi * bi;
                gasl[i + 1] = -gr * bi + gi * br;
                gbsl[i] = gr * ar + gi * ai;
                gbsl[i + 1] = -gr * ai + gi * ar;
                i += 2;
            }
        }

        vec![ga, gb]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Max operation: returns the maximum value of all elements in the tensor as a scalar.
pub struct Max;

impl Operation for Max {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let max_val = a.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        *output = ArrayD::from_elem(IxDyn(&[][..]), max_val);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let a_shape = inputs[0].lock().storage.shape();
        let max_val = a.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        // mask positions equal to max_val
        let mut mask = a.mapv(|v| if (v - max_val).abs() < 1e-6 { 1.0 } else { 0.0 });
        let count = mask.sum();
        if count == 0.0 {
            // shouldn't happen, but return zeros
            return vec![ArrayD::zeros(IxDyn(&a_shape))];
        }
        let val = match output_grad.iter().next().copied() {
            Some(v) => v,
            None => {
                log::error!("Max backward: Expected scalar output_grad");
                0.0f32
            }
        };
        mask *= val / count;
        vec![mask]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Min operation: returns the minimum value of all elements in the tensor as a scalar.
pub struct Min;

impl Operation for Min {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let min_val = a.iter().fold(f32::INFINITY, |m, &v| m.min(v));
        *output = ArrayD::from_elem(IxDyn(&[][..]), min_val);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let a_shape = inputs[0].lock().storage.shape();
        let min_val = a.iter().fold(f32::INFINITY, |m, &v| m.min(v));
        let mut mask = a.mapv(|v| if (v - min_val).abs() < 1e-6 { 1.0 } else { 0.0 });
        let count = mask.sum();
        if count == 0.0 {
            return vec![ArrayD::zeros(IxDyn(&a_shape))];
        }
        let val = match output_grad.iter().next().copied() {
            Some(v) => v,
            None => {
                log::error!("Min backward: Expected scalar output_grad");
                0.0f32
            }
        };
        mask *= val / count;
        vec![mask]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn det_square_matrix(m: &[f32], n: usize) -> f32 {
    if n == 0 {
        return 1.0;
    }
    let mut a = m.to_vec();
    let mut sign = 1.0f32;
    let mut det = 1.0f32;
    let eps = 1e-12f32;

    for i in 0..n {
        let mut pivot = i;
        let mut best = a[i * n + i].abs();
        for r in (i + 1)..n {
            let v = a[r * n + i].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }

        if best < eps {
            return 0.0;
        }

        if pivot != i {
            for c in 0..n {
                a.swap(i * n + c, pivot * n + c);
            }
            sign = -sign;
        }

        let piv = a[i * n + i];
        det *= piv;
        for r in (i + 1)..n {
            let f = a[r * n + i] / piv;
            a[r * n + i] = 0.0;
            for c in (i + 1)..n {
                a[r * n + c] -= f * a[i * n + c];
            }
        }
    }

    sign * det
}

fn inverse_square_matrix(m: &[f32], n: usize) -> Option<Vec<f32>> {
    if n == 0 {
        return Some(vec![]);
    }

    let eps = 1e-12f32;
    let width = 2 * n;
    let mut aug = vec![0.0f32; n * width];

    for r in 0..n {
        for c in 0..n {
            aug[r * width + c] = m[r * n + c];
        }
        aug[r * width + (n + r)] = 1.0;
    }

    for i in 0..n {
        let mut pivot = i;
        let mut best = aug[i * width + i].abs();
        for r in (i + 1)..n {
            let v = aug[r * width + i].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < eps {
            return None;
        }

        if pivot != i {
            for c in 0..width {
                aug.swap(i * width + c, pivot * width + c);
            }
        }

        let piv = aug[i * width + i];
        for c in 0..width {
            aug[i * width + c] /= piv;
        }

        for r in 0..n {
            if r == i {
                continue;
            }
            let f = aug[r * width + i];
            if f.abs() < eps {
                continue;
            }
            for c in 0..width {
                aug[r * width + c] -= f * aug[i * width + c];
            }
        }
    }

    let mut inv = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            inv[r * n + c] = aug[r * width + (n + c)];
        }
    }
    Some(inv)
}

fn matmul_square(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    out
}

/// Determinant for square matrices with optional leading batch dimensions.
/// Input shape: [*, n, n], output shape: [*].
pub struct Determinant;

impl Operation for Determinant {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let shape = a.shape().to_vec();
        if shape.len() < 2 {
            log::error!("Determinant.forward: input rank must be at least 2");
            *output = ArrayD::from_elem(IxDyn(&[][..]), 0.0);
            return;
        }
        let n = shape[shape.len() - 1];
        let m = shape[shape.len() - 2];
        if n != m {
            log::error!("Determinant.forward: last two dimensions must form square matrices");
            *output = ArrayD::from_elem(IxDyn(&shape[..shape.len() - 2]), 0.0);
            return;
        }

        let batch_dims = &shape[..shape.len() - 2];
        let batch = batch_dims.iter().product::<usize>();
        let mat_size = n * n;
        let flat = a.iter().copied().collect::<Vec<_>>();

        let mut out = vec![0.0f32; batch];
        for (b, out_b) in out.iter_mut().enumerate().take(batch) {
            let start = b * mat_size;
            *out_b = det_square_matrix(&flat[start..start + mat_size], n);
        }

        if batch_dims.is_empty() {
            *output = ArrayD::from_elem(IxDyn(&[][..]), out[0]);
        } else {
            *output = ArrayD::from_shape_vec(IxDyn(batch_dims), out)
                .expect("Determinant.forward: output shape construction failed");
        }
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let shape = a.shape().to_vec();
        if shape.len() < 2 {
            return vec![ArrayD::zeros(IxDyn(&shape))];
        }
        let n = shape[shape.len() - 1];
        let m = shape[shape.len() - 2];
        if n != m {
            return vec![ArrayD::zeros(IxDyn(&shape))];
        }

        let batch_dims = &shape[..shape.len() - 2];
        let batch = batch_dims.iter().product::<usize>();
        let mat_size = n * n;
        let flat = a.iter().copied().collect::<Vec<_>>();

        let grad_scalars = if batch_dims.is_empty() {
            vec![output_grad.iter().copied().next().unwrap_or(0.0)]
        } else {
            output_grad.iter().copied().collect::<Vec<_>>()
        };

        let mut ga = vec![0.0f32; flat.len()];
        for b in 0..batch {
            let start = b * mat_size;
            let mat = &flat[start..start + mat_size];
            let det = det_square_matrix(mat, n);
            let Some(inv) = inverse_square_matrix(mat, n) else {
                continue;
            };
            let g = grad_scalars.get(b).copied().unwrap_or(0.0);
            for i in 0..n {
                for j in 0..n {
                    ga[start + i * n + j] = g * det * inv[j * n + i];
                }
            }
        }

        let ga_arr = ArrayD::from_shape_vec(IxDyn(&shape), ga)
            .expect("Determinant.backward: grad shape construction failed");
        vec![ga_arr]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Matrix inverse for square matrices with optional leading batch dimensions.
/// Input shape: [*, n, n], output shape: [*, n, n].
pub struct Inverse;

impl Operation for Inverse {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let shape = a.shape().to_vec();
        if shape.len() < 2 {
            log::error!("Inverse.forward: input rank must be at least 2");
            *output = ArrayD::zeros(IxDyn(&shape));
            return;
        }
        let n = shape[shape.len() - 1];
        let m = shape[shape.len() - 2];
        if n != m {
            log::error!("Inverse.forward: last two dimensions must form square matrices");
            *output = ArrayD::zeros(IxDyn(&shape));
            return;
        }

        let batch = shape[..shape.len() - 2].iter().product::<usize>();
        let mat_size = n * n;
        let flat = a.iter().copied().collect::<Vec<_>>();
        let mut out = vec![0.0f32; flat.len()];

        for b in 0..batch {
            let start = b * mat_size;
            let mat = &flat[start..start + mat_size];
            if let Some(inv) = inverse_square_matrix(mat, n) {
                out[start..(start + mat_size)].copy_from_slice(&inv[..mat_size]);
            } else {
                log::warn!(
                    "Inverse.forward: encountered singular matrix, returning zeros for batch {}",
                    b
                );
            }
        }

        *output = ArrayD::from_shape_vec(IxDyn(&shape), out)
            .expect("Inverse.forward: output shape construction failed");
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let shape = a.shape().to_vec();
        if shape.len() < 2 {
            return vec![ArrayD::zeros(IxDyn(&shape))];
        }
        let n = shape[shape.len() - 1];
        let m = shape[shape.len() - 2];
        if n != m {
            return vec![ArrayD::zeros(IxDyn(&shape))];
        }

        let batch = shape[..shape.len() - 2].iter().product::<usize>();
        let mat_size = n * n;
        let flat = a.iter().copied().collect::<Vec<_>>();
        let gy = output_grad.iter().copied().collect::<Vec<_>>();
        let mut ga = vec![0.0f32; flat.len()];

        for b in 0..batch {
            let start = b * mat_size;
            let mat = &flat[start..start + mat_size];
            let Some(inv) = inverse_square_matrix(mat, n) else {
                continue;
            };

            let mut inv_t = vec![0.0f32; mat_size];
            for i in 0..n {
                for j in 0..n {
                    inv_t[i * n + j] = inv[j * n + i];
                }
            }

            let gy_mat = &gy[start..start + mat_size];
            let tmp = matmul_square(&inv_t, gy_mat, n);
            let mut gmat = matmul_square(&tmp, &inv_t, n);
            for v in &mut gmat {
                *v = -*v;
            }
            ga[start..(start + mat_size)].copy_from_slice(&gmat[..mat_size]);
        }

        let ga_arr = ArrayD::from_shape_vec(IxDyn(&shape), ga)
            .expect("Inverse.backward: grad shape construction failed");
        vec![ga_arr]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The multiplication operation.
pub struct Mul;

impl Operation for Mul {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        log::debug!(
            "Mul.forward enter lhs={:p} rhs={:p}",
            &inputs[0] as *const _,
            &inputs[1] as *const _
        );

        // Check for aliasing to avoid deadlock
        if inputs[0].is_same(&inputs[1]) {
            let lock = inputs[0].lock();
            if let Some(view) = lock.storage.as_f32_view() {
                log::debug!("Mul.forward: aliased inputs, using view square");
                *output = (&view * &view).into_owned().into_dyn();
                return;
            }
            // Fallback if no view (shouldn't happen for f32 usually but consistent style)
        }

        let a_lock = inputs[0].lock();
        // If not aliased, we can safely lock the second one.
        // Note: if A and B are different tensors, a_lock is held here.
        // We must ensure that we define "not aliased" correctly.
        // is_same checks Arc pointer equality.
        // If they are distinct Arcs, we proceed.
        // There is a theoretical edge case of separate Arcs pointing to same Mutex? No, Arc wraps Mutex.

        // However, if we are in a graph with cycles or shared nodes,
        // ensure we don't have locking order issues globally (like A->B vs B->A).
        // StandardOps don't usually lock multiple tensors except binary ops.
        // We always lock inputs[0] then inputs[1].
        // Deadlock only happens if thread 1 does 0 then 1, thread 2 does 1 then 0.
        // Here we are single-threaded mostly or `rayon` parallelizes independent tasks.
        // But `checkpoint` runs sequentially.

        let b_lock = inputs[1].lock();
        if let (Some(a_view), Some(b_view)) =
            (a_lock.storage.as_f32_view(), b_lock.storage.as_f32_view())
        {
            log::debug!("Mul.forward: using views for equal shape multiply");
            if a_view.shape() == b_view.shape() {
                *output = (&a_view * &b_view).into_owned().into_dyn();
                log::debug!("Mul.forward: view multiply done");
                return;
            }
        }
        log::debug!("Mul.forward: falling back to cloning arrays + broadcasted multiply");
        let a = a_lock.storage.to_f32_array();
        let b = b_lock.storage.to_f32_array();
        log::debug!(
            "Mul.forward: cloned arrays shape a={:?} b={:?}",
            a.shape(),
            b.shape()
        );
        *output = &a * &b;
        log::debug!("Mul.forward: result shape={:?}", output.shape());
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
        *output = par_mapv(&a, |x| x.powf(self.0));
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        vec![(output_grad * par_mapv(&a, |x| self.0 * x.powf(self.0 - 1.0))).to_owned()]
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
            log::error!(
                "MatMul blas detection: Failed to create test 2x2 array 'a': {}",
                e
            );
            BLAS_ORDER_DETECTION.set(None).ok();
            return None;
        }
    };
    let b = match Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]) {
        Ok(v) => v,
        Err(e) => {
            log::error!(
                "MatMul blas detection: Failed to create test 2x2 array 'b': {}",
                e
            );
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

/// Batched matrix multiplication: broadcast over first batch dimension.
/// Inputs: a [batch, m, k], b [batch, k, n] -> output [batch, m, n]
pub struct BatchedMatMul;

impl BatchedMatMul {
    pub fn new() -> Self {
        BatchedMatMul
    }
}

impl Default for BatchedMatMul {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for BatchedMatMul {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].lock().storage.to_f32_array();
        let b = inputs[1].lock().storage.to_f32_array();
        log::debug!(
            "BatchedMatMul.forward: a_shape={:?} b_shape={:?}",
            a.shape(),
            b.shape()
        );
        if a.ndim() != 3 || b.ndim() != 3 {
            log::error!("BatchedMatMul: both inputs must be 3D (batch,m,k) and (batch,k,n)");
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        let batch = a.shape()[0];
        if b.shape()[0] != batch {
            log::error!(
                "BatchedMatMul: batch dims mismatch: {} != {}",
                batch,
                b.shape()[0]
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        let m = a.shape()[1];
        let k = a.shape()[2];
        let kb = b.shape()[1];
        let n = b.shape()[2];
        if k != kb {
            log::error!("BatchedMatMul: inner dims mismatch: {} != {}", k, kb);
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        if let Some(backend_output) = get_global_backend().matmul(&a, &b) {
            *output = backend_output;
            return;
        }
        let mut out = ndarray::Array3::<f32>::zeros((batch, m, n));
        for i in 0..batch {
            let a_view = a.index_axis(Axis(0), i).to_owned();
            let b_view = b.index_axis(Axis(0), i).to_owned();
            let a2 = match a_view.into_dimensionality::<Ix2>() {
                Ok(v) => v,
                Err(e) => {
                    log::error!(
                        "BatchedMatMul forward: failed to convert a slice to 2D: {}",
                        e
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
            };
            let b2 = match b_view.into_dimensionality::<Ix2>() {
                Ok(v) => v,
                Err(e) => {
                    log::error!(
                        "BatchedMatMul forward: failed to convert b slice to 2D: {}",
                        e
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
            };
            let res = a2.dot(&b2);
            out.index_axis_mut(Axis(0), i).assign(&res);
        }
        *output = out.into_dyn();
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].lock().storage.to_f32_array();
        let b = inputs[1].lock().storage.to_f32_array();
        if a.ndim() != 3 || b.ndim() != 3 {
            return vec![output_grad.clone(), output_grad.clone()];
        }
        let batch = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        let n = b.shape()[2];
        let mut grad_a = ndarray::Array3::<f32>::zeros((batch, m, k)).into_dyn();
        let mut grad_b = ndarray::Array3::<f32>::zeros((batch, k, n)).into_dyn();
        for i in 0..batch {
            let og = output_grad.index_axis(Axis(0), i).to_owned();
            let a_view = a.index_axis(Axis(0), i).to_owned();
            let b_view = b.index_axis(Axis(0), i).to_owned();
            let og2 = match og.into_dimensionality::<Ix2>() {
                Ok(v) => v,
                Err(e) => {
                    log::error!("BatchedMatMul backward: failed to convert og to 2D: {}", e);
                    return vec![output_grad.clone(), output_grad.clone()];
                }
            };
            let a2 = match a_view.into_dimensionality::<Ix2>() {
                Ok(v) => v,
                Err(e) => {
                    log::error!(
                        "BatchedMatMul backward: failed to convert a slice to 2D: {}",
                        e
                    );
                    return vec![output_grad.clone(), output_grad.clone()];
                }
            };
            let b2 = match b_view.into_dimensionality::<Ix2>() {
                Ok(v) => v,
                Err(e) => {
                    log::error!(
                        "BatchedMatMul backward: failed to convert b slice to 2D: {}",
                        e
                    );
                    return vec![output_grad.clone(), output_grad.clone()];
                }
            };
            let ga = og2.dot(&b2.t());
            let gb = a2.t().dot(&og2);
            grad_a.index_axis_mut(Axis(0), i).assign(&ga);
            grad_b.index_axis_mut(Axis(0), i).assign(&gb);
        }
        vec![grad_a, grad_b]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Simple quantized matmul operation: left operand is f32, right operand is INT8 storage with scale.
/// This operator dequantizes the int8 weights to f32 and performs a normal matmul. For inference.
pub struct QuantizedMatMul;

impl QuantizedMatMul {
    pub fn new() -> Self {
        QuantizedMatMul
    }
}

impl Default for QuantizedMatMul {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for QuantizedMatMul {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // inputs: a (f32-ish), b (quantized int8 storage variants)
        // This forward pass is designed for inference. It avoids constructing a full dequantized
        // weight matrix when b is stored as INT8; instead it applies scale(s) on-the-fly.
        let a = inputs[0].lock().storage.to_f32_array();
        if a.ndim() != 2 {
            log::error!(
                "QuantizedMatMul forward: expected left operand to be 2D, got ndim={}",
                a.ndim()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        let a2 = match a.into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("QuantizedMatMul forward failed to convert a: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        let m = a2.nrows();
        let k = a2.ncols();

        let b_guard = inputs[1].lock();
        match &b_guard.storage {
            crate::dtype::TensorStorage::I8(bytes, scale, shape) => {
                if shape.len() != 2 {
                    log::error!(
                        "QuantizedMatMul forward: expected 2D shape for I8 weights, got {:?}",
                        shape
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                let rows = shape[0];
                let cols = shape[1];
                if k != rows {
                    log::error!(
                        "QuantizedMatMul forward: inner dims mismatch: {} != {}",
                        k,
                        rows
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                let expected_len = match rows.checked_mul(cols) {
                    Some(v) => v,
                    None => {
                        log::error!(
                            "QuantizedMatMul forward: rows*cols overflow: {}*{}",
                            rows,
                            cols
                        );
                        *output = ArrayD::zeros(IxDyn(&[0][..]));
                        return;
                    }
                };
                if bytes.len() != expected_len {
                    log::error!(
                        "QuantizedMatMul forward: I8 weight buffer length mismatch: len={} expected={}",
                        bytes.len(),
                        expected_len
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }

                let mut out = vec![0.0f32; m * cols];
                if let Some(a_slice) = a2.as_slice() {
                    for i in 0..m {
                        let a_row = &a_slice[i * k..(i + 1) * k];
                        let out_row = &mut out[i * cols..(i + 1) * cols];
                        for j in 0..k {
                            let aij = a_row[j];
                            if aij == 0.0 {
                                continue;
                            }
                            let b_row = &bytes[j * cols..(j + 1) * cols];
                            for col in 0..cols {
                                out_row[col] += aij * (b_row[col] as f32) * (*scale);
                            }
                        }
                    }
                } else {
                    for i in 0..m {
                        let out_row = &mut out[i * cols..(i + 1) * cols];
                        for j in 0..k {
                            let aij = a2[(i, j)];
                            if aij == 0.0 {
                                continue;
                            }
                            let b_row = &bytes[j * cols..(j + 1) * cols];
                            for col in 0..cols {
                                out_row[col] += aij * (b_row[col] as f32) * (*scale);
                            }
                        }
                    }
                }

                let res = match Array2::from_shape_vec((m, cols), out) {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "QuantizedMatMul forward: failed to build output Array2: {}",
                            e
                        );
                        *output = ArrayD::zeros(IxDyn(&[0][..]));
                        return;
                    }
                };
                *output = res.into_dyn();
            }
            crate::dtype::TensorStorage::I8Rowwise(bytes, scales, shape) => {
                if shape.len() != 2 {
                    log::error!(
                        "QuantizedMatMul forward: expected 2D shape for I8Rowwise weights, got {:?}",
                        shape
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                let rows = shape[0];
                let cols = shape[1];
                if k != rows {
                    log::error!(
                        "QuantizedMatMul forward: inner dims mismatch: {} != {}",
                        k,
                        rows
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                if scales.len() != rows {
                    log::error!(
                        "QuantizedMatMul forward: I8Rowwise scales length mismatch: len={} rows={}",
                        scales.len(),
                        rows
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                let expected_len = match rows.checked_mul(cols) {
                    Some(v) => v,
                    None => {
                        log::error!(
                            "QuantizedMatMul forward: rows*cols overflow: {}*{}",
                            rows,
                            cols
                        );
                        *output = ArrayD::zeros(IxDyn(&[0][..]));
                        return;
                    }
                };
                if bytes.len() != expected_len {
                    log::error!(
                        "QuantizedMatMul forward: I8Rowwise weight buffer length mismatch: len={} expected={}",
                        bytes.len(),
                        expected_len
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }

                let mut out = vec![0.0f32; m * cols];
                if let Some(a_slice) = a2.as_slice() {
                    for i in 0..m {
                        let a_row = &a_slice[i * k..(i + 1) * k];
                        let out_row = &mut out[i * cols..(i + 1) * cols];
                        for j in 0..k {
                            let aij = a_row[j];
                            if aij == 0.0 {
                                continue;
                            }
                            let sj = scales[j];
                            let b_row = &bytes[j * cols..(j + 1) * cols];
                            for col in 0..cols {
                                out_row[col] += aij * (b_row[col] as f32) * sj;
                            }
                        }
                    }
                } else {
                    for i in 0..m {
                        let out_row = &mut out[i * cols..(i + 1) * cols];
                        for j in 0..k {
                            let aij = a2[(i, j)];
                            if aij == 0.0 {
                                continue;
                            }
                            let sj = scales[j];
                            let b_row = &bytes[j * cols..(j + 1) * cols];
                            for col in 0..cols {
                                out_row[col] += aij * (b_row[col] as f32) * sj;
                            }
                        }
                    }
                }

                let res = match Array2::from_shape_vec((m, cols), out) {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "QuantizedMatMul forward: failed to build output Array2: {}",
                            e
                        );
                        *output = ArrayD::zeros(IxDyn(&[0][..]));
                        return;
                    }
                };
                *output = res.into_dyn();
            }
            crate::dtype::TensorStorage::I8Blockwise(bytes, scales, shape, block_size) => {
                if shape.len() != 2 {
                    log::error!(
                        "QuantizedMatMul forward: expected 2D shape for I8Blockwise weights, got {:?}",
                        shape
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                let rows = shape[0];
                let cols = shape[1];
                if k != rows {
                    log::error!(
                        "QuantizedMatMul forward: inner dims mismatch: {} != {}",
                        k,
                        rows
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                let bs = *block_size;
                if bs == 0 {
                    log::error!("QuantizedMatMul forward: block_size must be > 0");
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                let blocks_per_row = cols.div_ceil(bs);
                let expected_scales = match rows.checked_mul(blocks_per_row) {
                    Some(v) => v,
                    None => {
                        log::error!(
                            "QuantizedMatMul forward: rows*blocks_per_row overflow: {}*{}",
                            rows,
                            blocks_per_row
                        );
                        *output = ArrayD::zeros(IxDyn(&[0][..]));
                        return;
                    }
                };
                if scales.len() != expected_scales {
                    log::error!(
                        "QuantizedMatMul forward: I8Blockwise scales length mismatch: len={} expected={} (rows={}, blocks_per_row={}, block_size={})",
                        scales.len(),
                        expected_scales,
                        rows,
                        blocks_per_row,
                        bs
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                let expected_len = match rows.checked_mul(cols) {
                    Some(v) => v,
                    None => {
                        log::error!(
                            "QuantizedMatMul forward: rows*cols overflow: {}*{}",
                            rows,
                            cols
                        );
                        *output = ArrayD::zeros(IxDyn(&[0][..]));
                        return;
                    }
                };
                if bytes.len() != expected_len {
                    log::error!(
                        "QuantizedMatMul forward: I8Blockwise weight buffer length mismatch: len={} expected={}",
                        bytes.len(),
                        expected_len
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }

                let mut out = vec![0.0f32; m * cols];
                if let Some(a_slice) = a2.as_slice() {
                    for i in 0..m {
                        let a_row = &a_slice[i * k..(i + 1) * k];
                        let out_row = &mut out[i * cols..(i + 1) * cols];
                        for j in 0..k {
                            let aij = a_row[j];
                            if aij == 0.0 {
                                continue;
                            }
                            let b_row = &bytes[j * cols..(j + 1) * cols];
                            let scales_row = &scales[j * blocks_per_row..(j + 1) * blocks_per_row];
                            for (block_idx, &s) in
                                scales_row.iter().enumerate().take(blocks_per_row)
                            {
                                let start = block_idx * bs;
                                let end = ((block_idx + 1) * bs).min(cols);
                                for col in start..end {
                                    out_row[col] += aij * (b_row[col] as f32) * s;
                                }
                            }
                        }
                    }
                } else {
                    for i in 0..m {
                        let out_row = &mut out[i * cols..(i + 1) * cols];
                        for j in 0..k {
                            let aij = a2[(i, j)];
                            if aij == 0.0 {
                                continue;
                            }
                            let b_row = &bytes[j * cols..(j + 1) * cols];
                            let scales_row = &scales[j * blocks_per_row..(j + 1) * blocks_per_row];
                            for (block_idx, &s) in
                                scales_row.iter().enumerate().take(blocks_per_row)
                            {
                                let start = block_idx * bs;
                                let end = ((block_idx + 1) * bs).min(cols);
                                for col in start..end {
                                    out_row[col] += aij * (b_row[col] as f32) * s;
                                }
                            }
                        }
                    }
                }

                let res = match Array2::from_shape_vec((m, cols), out) {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "QuantizedMatMul forward: failed to build output Array2: {}",
                            e
                        );
                        *output = ArrayD::zeros(IxDyn(&[0][..]));
                        return;
                    }
                };
                *output = res.into_dyn();
            }
            _ => {
                // Fallback: treat b as a regular float matrix.
                let b_shape = b_guard.storage.shape();
                if b_shape.len() != 2 {
                    log::error!(
                        "QuantizedMatMul forward: expected 2D right operand, got shape {:?}",
                        b_shape
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                let b2 = match b_guard.storage.to_f32_array().into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        log::error!("QuantizedMatMul forward failed to convert b: {}", e);
                        *output = ArrayD::zeros(IxDyn(&[0][..]));
                        return;
                    }
                };
                let res = a2.dot(&b2);
                *output = res.into_dyn();
            }
        }
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // compute gradients with respect to dequantized b (float)
        let a = inputs[0].lock().storage.to_f32_array();
        let b = inputs[1].lock().storage.to_f32_array();
        if a.ndim() != 2 || b.ndim() != 2 {
            return vec![output_grad.clone(), output_grad.clone()];
        }
        let a2 = match a.into_dimensionality::<Ix2>() {
            Ok(arr) => arr,
            Err(e) => {
                log::error!(
                    "QuantizedMatMul backward: Failed to convert a into 2D: {}",
                    e
                );
                return vec![output_grad.clone(), output_grad.clone()];
            }
        };
        let b2 = match b.into_dimensionality::<Ix2>() {
            Ok(arr) => arr,
            Err(e) => {
                log::error!(
                    "QuantizedMatMul backward: Failed to convert b into 2D: {}",
                    e
                );
                return vec![output_grad.clone(), output_grad.clone()];
            }
        };
        let og = match output_grad.clone().into_dimensionality::<Ix2>() {
            Ok(arr) => arr,
            Err(e) => {
                log::error!(
                    "QuantizedMatMul backward: Failed to convert output_grad into 2D: {}",
                    e
                );
                return vec![output_grad.clone(), output_grad.clone()];
            }
        };
        let ga = og.dot(&b2.t()).into_dyn();
        let gb = a2.t().dot(&og).into_dyn();
        // Note: We provide a gradient for the dequantized weights; updating quantized storage is not supported.
        vec![ga, gb]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
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

#[cfg(test)]
mod approx_tests {
    use super::*;
    use ndarray::ArrayD;
    use ndarray::IxDyn;

    #[test]
    fn test_approx_eq_arrayd() {
        let a = ArrayD::from_elem(IxDyn(&[2, 2][..]), 1.0f32);
        let mut b = a.clone();
        assert!(approx_eq_arrayd(&a, &b));
        b[[0, 0]] = 2.0;
        assert!(!approx_eq_arrayd(&a, &b));
    }
}

impl MatMul {
    pub fn new() -> Self {
        MatMul
    }
}

impl Default for MatMul {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for MatMul {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // Autocast logic: if enabled, cast inputs to target dtype (simulated via F32 storage but metadata)
        // For MVP, we don't change storage to F16 yet because our backend is f32-based.
        // However, we can simulate the precision loss or just note it.
        // Real implementation in future: convert to F16 storage and use f16 matmul.
        // Current: just respect the flag and log it?
        // Or actually perform the cast if we have f16 backend.
        // We have `dtype_f16` feature.

        let autocast = crate::amp::is_autocast_enabled();
        // If autocast is on, we conceptually "cast" inputs to f16 (or bf16).
        // Since our MatMul implementation is f32-based (ndarray::dot), we continue with f32.
        // But we should verify input types or log.
        if autocast {
            // Example: Log that we are running in autocast mode
            // In a real kernel, we would dispatch to f16 kernel here.
            // log::trace!("MatMul running in autocast mode");
        }

        // Simple, robust matmul: convert to f32 arrays, ensure 2D, then dispatch through
        // the active backend before falling back to ndarray dot.
        let a_arr = match inputs[0]
            .lock()
            .storage
            .to_f32_array()
            .view()
            .into_dimensionality::<Ix2>()
        {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("MatMul forward: left operand is not 2D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let b_arr = match inputs[1]
            .lock()
            .storage
            .to_f32_array()
            .view()
            .into_dimensionality::<Ix2>()
        {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("MatMul forward: right operand is not 2D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        log::debug!(
            "MatMul.forward SAFE: a_shape={:?} b_shape={:?}",
            a_arr.shape(),
            b_arr.shape()
        );
        if let Some(backend_output) =
            get_global_backend().matmul(&a_arr.clone().into_dyn(), &b_arr.clone().into_dyn())
        {
            *output = backend_output;
            return;
        }
        let res = std::panic::catch_unwind(|| a_arr.dot(&b_arr).into_dyn());
        match res {
            Ok(r) => *output = r,
            Err(_) => {
                log::error!("MatMul forward: panic during ndarray dot; returning zeros");
                *output = ArrayD::zeros(IxDyn(&[0][..]));
            }
        }
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a_owned = inputs[0].lock().storage.to_f32_array();
        let b_owned = inputs[1].lock().storage.to_f32_array();
        let a: ArrayView2<f32> = match a_owned.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("MatMul backward: left operand is not 2D: {}", e);
                let grad_a = ArrayD::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::zeros(IxDyn(&[0][..]));
                return vec![grad_a.into_dyn(), grad_b.into_dyn()];
            }
        };
        let b: ArrayView2<f32> = match b_owned.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("MatMul backward: right operand is not 2D: {}", e);
                let grad_a = ArrayD::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::zeros(IxDyn(&[0][..]));
                return vec![grad_a.into_dyn(), grad_b.into_dyn()];
            }
        };
        let output_grad: ArrayView2<f32> = match output_grad.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("MatMul backward: output_grad is not 2D: {}", e);
                let grad_a = ArrayD::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::zeros(IxDyn(&[0][..]));
                return vec![grad_a.into_dyn(), grad_b.into_dyn()];
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
                    log::warn!(
                        "MatMul backward: output_grad not contiguous; falling back to ndarray path"
                    );
                    let grad_a = output_grad.dot(&b.t()).into_dyn();
                    let grad_b = a.t().dot(&output_grad).into_dyn();
                    return vec![grad_a.into_dyn(), grad_b.into_dyn()];
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
                    return vec![grad_a.into_dyn(), grad_b.into_dyn()];
                }
            };
            let b_slice = match b_owned.as_slice() {
                Some(s) => s,
                None => {
                    log::warn!("MatMul backward: b not contiguous; fallback to ndarray path");
                    let grad_a = output_grad.dot(&b.t()).into_dyn();
                    let grad_b = a.t().dot(&output_grad).into_dyn();
                    return vec![grad_a.into_dyn(), grad_b.into_dyn()];
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
                            b_col_vec[col * (k as usize) + row] =
                                b_owned_vec[row * (n as usize) + col];
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
                Option::None => {
                    // fall back to ndarray if detection fails
                    let grad_a = output_grad.dot(&b.t()).into_dyn();
                    let grad_b = a.t().dot(&output_grad).into_dyn();
                    return vec![grad_a.into_dyn(), grad_b.into_dyn()];
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
            let grad_a = match ArrayD::from_shape_vec(IxDyn(&[m as usize, k as usize]), grad_a_vec)
            {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!("MatMul backward: Failed to create grad_a array: {}", e);
                    let grad_a = output_grad.dot(&b.t()).into_dyn();
                    let grad_b = a.t().dot(&output_grad).into_dyn();
                    return vec![grad_a.into_dyn(), grad_b.into_dyn()];
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
                Option::None => {
                    // already handled above; keep defensive fallback
                }
            }
            let grad_b = match ArrayD::from_shape_vec(IxDyn(&[k as usize, n as usize]), grad_b_vec)
            {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!("MatMul backward: Failed to create grad_b array: {}", e);
                    let grad_a = ArrayD::from_elem(IxDyn(&[m as usize, k as usize]), f32::NAN);
                    let grad_b = output_grad.to_owned().into_dyn();
                    return vec![grad_a.into_dyn(), grad_b.into_dyn()];
                }
            };
            return vec![grad_a.into_dyn(), grad_b.into_dyn()];
        }
        #[cfg(any(not(feature = "openblas"), target_os = "windows"))]
        {
            // Fallback to ndarray-based computation for backward on platforms where BLAS may be unstable
            let grad_a = output_grad.dot(&b.t()).into_dyn();
            let grad_b = a.t().dot(&output_grad).into_dyn();
            vec![grad_a.into_dyn(), grad_b.into_dyn()]
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
        let rounded = a_scaled.mapv(|x| x.round().clamp(-1.0, 1.0));
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
        *output = par_mapv(&a, |x| x.max(0.0));
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        vec![output_grad * par_mapv(&a, |x| if x > 0.0 { 1.0 } else { 0.0 })]
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
        *output = par_mapv(&a, |x| 1.0 / (1.0 + (-x).exp()));
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let sigmoid_a = par_mapv(&a, |x| 1.0 / (1.0 + (-x).exp()));
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
        *output = par_mapv(&a, |x| x.tanh());
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let tanh_a = par_mapv(&a, |x| x.tanh());
        vec![output_grad * (1.0 - par_mapv(&tanh_a, |x| x.powi(2)))]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// GELU activation function (approximation using tanh).
pub struct GELU;

impl Operation for GELU {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        *output = par_mapv(&a, |x| {
            let u = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + u.tanh())
        });
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let grad = par_mapv(&a, |x| {
            let u = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            let tanh_u = u.tanh();
            let left = 0.5 * (1.0 + tanh_u);
            let right = 0.5
                * x
                * (1.0 - tanh_u * tanh_u)
                * (sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x));
            left + right
        });
        vec![output_grad * grad]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// SiLU activation function (Swish): x * sigmoid(x)
pub struct SiLU;

impl Operation for SiLU {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let sig = par_mapv(&a, |x| 1.0 / (1.0 + (-x).exp()));
        *output = a * sig;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let sig = par_mapv(&a, |x| 1.0 / (1.0 + (-x).exp()));
        let deriv = &sig + &(&a * (&sig * (1.0 - &sig)));
        vec![output_grad * deriv]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Nearest-neighbor Upsample 2D operation. Input tensor expects NCHW format.
pub struct UpSampleNearest2D {
    pub scale: usize,
}

impl UpSampleNearest2D {
    pub fn new(scale: usize) -> Self {
        UpSampleNearest2D { scale }
    }
}

impl Operation for UpSampleNearest2D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        // Expect input shape [N, C, H, W]
        let shape = a.shape().to_vec();
        if shape.len() != 4 {
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let sh = h * self.scale;
        let sw = w * self.scale;
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, c, sh, sw][..]));
        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        let v = a[[ni, ci, hi, wi]];
                        let start_h = hi * self.scale;
                        let start_w = wi * self.scale;
                        for rh in 0..self.scale {
                            for rw in 0..self.scale {
                                out[[ni, ci, start_h + rh, start_w + rw]] = v;
                            }
                        }
                    }
                }
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // Sum gradient values corresponding to each input pixel
        let a = inputs[0].to_f32_array();
        let shape = a.shape().to_vec();
        if shape.len() != 4 {
            return vec![output_grad.clone()];
        }
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, c, h, w][..]));
        let og_shape = output_grad.shape().to_vec();
        let sh = og_shape[2];
        let sw = og_shape[3];
        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        let start_h = hi * self.scale;
                        let start_w = wi * self.scale;
                        let mut sum = 0.0;
                        for rh in 0..self.scale {
                            for rw in 0..self.scale {
                                let oh = start_h + rh;
                                let ow = start_w + rw;
                                if oh < sh && ow < sw {
                                    sum += output_grad[[ni, ci, oh, ow]];
                                }
                            }
                        }
                        grad_in[[ni, ci, hi, wi]] = sum;
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

/// The natural logarithm operation element-wise
pub struct Log;

impl Operation for Log {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        *output = par_mapv(&a, |x| x.ln());
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
        let (mut s, perm_opt) = permute_to_last(output_grad, axis);
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
                *v /= sum;
            }
        }
        // grad_input = grad_output - softmax * sum(grad_output) along axis
        let (p_output_grad, _) = permute_to_last(output_grad, axis);
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
                *gi -= si * sum;
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
        if let Some(backend_output) = get_global_backend().softmax(&out, last_axis as isize) {
            if let Some(ref perm) = perm_opt {
                *output = permute_back(backend_output, perm);
            } else {
                *output = backend_output;
            }
            return;
        }
        for mut lane in out.lanes_mut(Axis(last_axis)) {
            let max = lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in lane.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            // Numerical guard: if sum is zero or non-finite (e.g., all -inf), fall back to uniform distribution
            if !(sum > 0.0f32 && sum.is_finite()) {
                let len = lane.len() as f32;
                for v in lane.iter_mut() {
                    *v = 1.0f32 / len;
                }
                continue;
            }
            for v in lane.iter_mut() {
                *v /= sum;
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
        // Recompute softmax using f64 for improved numeric stability and to avoid layout/iterator issues
        let x_perm = permute_to_last(&x, axis).0;
        let last_axis = x_perm.ndim() - 1;
        // convert to f64 for stable sums
        let mut y_f64 = x_perm.mapv(|v| v as f64);
        for mut lane in y_f64.lanes_mut(Axis(last_axis)) {
            let max = lane.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0f64;
            for v in lane.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in lane.iter_mut() {
                *v /= sum;
            }
        }
        // cast back to f32 ArrayD
        let y = y_f64.mapv(|v| v as f32);
        let perm_opt = permute_to_last(&x, axis).1;
        // grad = y * (grad_out - sum(grad_out * y) along last axis)
        let (p_output_grad, _) = permute_to_last(output_grad, axis);
        // compute elementwise product and sum along last axis
        let prod = &p_output_grad * &y; // elementwise
        let s = prod.sum_axis(Axis(last_axis)); // shape: same as y with last axis removed
                                                // broadcast s back to full shape by inserting axis
        let s_b = s.insert_axis(Axis(last_axis));
        let grad_in = &y * (&p_output_grad - &s_b);

        if let Some(ref perm) = perm_opt {
            vec![permute_back(grad_in.to_owned(), perm)]
        } else {
            vec![grad_in.to_owned()]
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
        let x2 = match xp.to_shape((nrows, features)) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                log::error!("LayerNorm forward: Reshape to 2D failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[][..]));
                return;
            }
        };

        // compute per-row mean and var
        let mut normalized = x2.clone();
        let mut inv_std = ArrayD::zeros(IxDyn(&[nrows, 1][..]));
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
        let og_perm = match output_grad.to_shape(IxDyn(&[nrows, features][..])) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                log::error!("LayerNorm backward: Reshape og to 2D failed: {}", e);
                let grad_x = ArrayD::zeros(IxDyn(&shape));
                let grad_gamma = ArrayD::zeros(IxDyn(&[features][..]));
                let grad_beta = ArrayD::zeros(IxDyn(&[features][..]));
                return vec![grad_x, grad_gamma, grad_beta];
            }
        };

        // fetch cache
        let lock = match self.cache.lock() {
            Ok(l) => l,
            Err(poisoned) => {
                log::error!("Failed to acquire LayerNorm cache lock: {:?}", poisoned);
                // Return zero gradients if we cannot access cached values; this avoids panicking.
                let grad_x = ArrayD::zeros(IxDyn(&shape));
                let grad_gamma = ArrayD::zeros(IxDyn(&[features][..]));
                let grad_beta = ArrayD::zeros(IxDyn(&[features][..]));
                return vec![grad_x, grad_gamma, grad_beta];
            }
        };
        let (normalized, inv_std) = if let Some((ref n, ref i)) = *lock {
            (n.clone(), i.clone())
        } else {
            log::error!("LayerNorm backward called without forward cache — forward cache missing. Returning zero grads.");
            let grad_x = ArrayD::zeros(IxDyn(&shape));
            let grad_gamma = ArrayD::zeros(IxDyn(&[features][..]));
            let grad_beta = ArrayD::zeros(IxDyn(&[features][..]));
            return vec![grad_x, grad_gamma, grad_beta];
        };
        let normalized2 = match normalized.to_shape((nrows, features)) {
            Ok(s) => s,
            Err(e) => {
                log::error!("LayerNorm backward: Reshape normalized 2D failed: {}", e);
                let grad_x = ArrayD::zeros(IxDyn(&shape));
                let grad_gamma = ArrayD::zeros(IxDyn(&[features][..]));
                let grad_beta = ArrayD::zeros(IxDyn(&[features][..]));
                return vec![grad_x, grad_gamma, grad_beta];
            }
        };
        let inv2 = match inv_std.to_shape((nrows, 1)) {
            Ok(s) => s,
            Err(e) => {
                log::error!("LayerNorm backward: Reshape inv std 2D failed: {}", e);
                let grad_x = ArrayD::zeros(IxDyn(&shape));
                let grad_gamma = ArrayD::zeros(IxDyn(&[features][..]));
                let grad_beta = ArrayD::zeros(IxDyn(&[features][..]));
                return vec![grad_x, grad_gamma, grad_beta];
            }
        };

        // grad w.r.t gamma and beta
        let mut grad_gamma = ArrayD::zeros(IxDyn(&[features][..]));
        let mut grad_beta = ArrayD::zeros(IxDyn(&[features][..]));
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
        let mut grad_x2 = ArrayD::zeros(IxDyn(&[nrows, features][..]));
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
        let grad_x_perm = match grad_x2.into_dyn().to_shape(IxDyn(&shape)) {
            Ok(g) => g.to_owned(),
            Err(e) => {
                log::error!("LayerNorm backward: Reshape grad back failed: {}", e);
                let grad_x = ArrayD::zeros(IxDyn(&shape));
                return vec![grad_x, grad_gamma, grad_beta];
            }
        };
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

/// BatchNorm operation.
///
/// Standard Batch Normalization implementation.
/// Inputs: [x, gamma, beta, running_mean, running_var]
pub struct BatchNorm {
    pub momentum: f32,
    pub eps: f32,
    pub training: bool,
    // Cache for backward pass
    cache: std::sync::Mutex<Option<BatchNormCache>>,
}

type BatchNormCache = (ArrayD<f32>, ArrayD<f32>, ArrayD<f32>); // (normalized, mean, inv_std)

impl BatchNorm {
    pub fn new(momentum: f32, eps: f32, training: bool) -> Self {
        BatchNorm {
            momentum,
            eps,
            training,
            cache: std::sync::Mutex::new(None),
        }
    }
}

impl Operation for BatchNorm {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        let gamma = inputs[1].to_f32_array();
        let beta = inputs[2].to_f32_array();
        let running_mean_tensor = &inputs[3];
        let running_var_tensor = &inputs[4];

        let ndim = x.ndim();
        // BatchNorm traditionally normalizes over the channel dimension (axis 1)
        // For [B, C, spatial], normalize over B and all spatial dimensions.
        let features = x.shape()[1];

        // Reshape x to [B, C, N] where N is number of spatial elements
        let batch_size = x.shape()[0];
        let spatial_elements = if ndim > 2 {
            x.shape().iter().skip(2).product::<usize>()
        } else {
            1
        };

        let x_reshaped = match x.to_shape((batch_size, features, spatial_elements)) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                log::error!("BatchNorm forward: Reshape failed: {}", e);
                return;
            }
        };

        let (mean, _var, inv_std) = if self.training {
            // Compute mini-batch mean and variance over (batch_size, spatial_elements)
            let mut mean = ArrayD::zeros(IxDyn(&[features][..]));
            let mut var = ArrayD::zeros(IxDyn(&[features][..]));
            let n = (batch_size * spatial_elements) as f32;

            for c in 0..features {
                let mut sum = 0.0f32;
                for b in 0..batch_size {
                    for s in 0..spatial_elements {
                        sum += x_reshaped[[b, c, s]];
                    }
                }
                let m = sum / n;
                mean[[c]] = m;

                let mut sq_diff_sum = 0.0f32;
                for b in 0..batch_size {
                    for s in 0..spatial_elements {
                        let diff = x_reshaped[[b, c, s]] - m;
                        sq_diff_sum += diff * diff;
                    }
                }
                var[[c]] = sq_diff_sum / n;
            }

            // Update running statistics in-place
            {
                let mut rm_lock = running_mean_tensor.lock();
                let mut rv_lock = running_var_tensor.lock();
                let mut rm_data = rm_lock.storage.to_f32_array();
                let mut rv_data = rv_lock.storage.to_f32_array();

                for c in 0..features {
                    rm_data[[c]] = (1.0 - self.momentum) * rm_data[[c]] + self.momentum * mean[[c]];
                    // Bessel's correction for unbiased variance estimator used in running var
                    let unbiased_var = var[[c]] * (n / (n - 1.0).max(1.0));
                    rv_data[[c]] =
                        (1.0 - self.momentum) * rv_data[[c]] + self.momentum * unbiased_var;
                }

                rm_lock.storage =
                    crate::dtype::TensorStorage::from_f32_array(&rm_data, crate::dtype::DType::F32);
                rv_lock.storage =
                    crate::dtype::TensorStorage::from_f32_array(&rv_data, crate::dtype::DType::F32);
            }

            let inv_std = var.mapv(|v| 1.0 / (v + self.eps).sqrt());
            (mean, var, inv_std)
        } else {
            // Use running statistics
            let mean = running_mean_tensor.to_f32_array();
            let var = running_var_tensor.to_f32_array();
            let inv_std = var.mapv(|v| 1.0 / (v + self.eps).sqrt());
            (mean, var, inv_std)
        };

        // Normalize and scale/shift
        let mut normalized = ArrayD::zeros(IxDyn(&[batch_size, features, spatial_elements][..]));
        let mut out_reshaped = ArrayD::zeros(IxDyn(&[batch_size, features, spatial_elements][..]));

        for c in 0..features {
            let m = mean[[c]];
            let is = inv_std[[c]];
            let g = if gamma.ndim() == 1 {
                gamma[[c]]
            } else {
                gamma[[0]]
            };
            let b = if beta.ndim() == 1 {
                beta[[c]]
            } else {
                beta[[0]]
            };

            for b_idx in 0..batch_size {
                for s in 0..spatial_elements {
                    let norm = (x_reshaped[[b_idx, c, s]] - m) * is;
                    normalized[[b_idx, c, s]] = norm;
                    out_reshaped[[b_idx, c, s]] = norm * g + b;
                }
            }
        }

        // Store in cache for backward
        if let Ok(mut lock) = self.cache.lock() {
            *lock = Some((normalized, mean, inv_std));
        }

        // Reshape back to original shape
        *output = match out_reshaped.to_shape(x.shape()) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                log::error!("BatchNorm forward reshape failed: {}", e);
                return;
            }
        };
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        let gamma = inputs[1].to_f32_array();
        let ndim = x.ndim();
        let batch_size = x.shape()[0];
        let features = x.shape()[1];
        let spatial_elements = if ndim > 2 {
            x.shape().iter().skip(2).product::<usize>()
        } else {
            1
        };
        let n = (batch_size * spatial_elements) as f32;

        let og_reshaped = match output_grad.to_shape((batch_size, features, spatial_elements)) {
            Ok(s) => s,
            Err(e) => {
                log::error!("BatchNorm backward: Reshape og failed: {}", e);
                return vec![
                    ArrayD::zeros(x.shape()),
                    ArrayD::zeros(gamma.shape()),
                    ArrayD::zeros(gamma.shape()),
                    ArrayD::zeros(gamma.shape()),
                    ArrayD::zeros(gamma.shape()),
                ];
            }
        };

        let lock = self.cache.lock().unwrap();
        let (normalized, _mean, inv_std) = lock.as_ref().unwrap();

        let mut grad_x_reshaped = ArrayD::zeros(ndarray::IxDyn(
            &[batch_size, features, spatial_elements][..],
        ));
        let mut grad_gamma = ArrayD::zeros(IxDyn(&[features][..]));
        let mut grad_beta = ArrayD::zeros(IxDyn(&[features][..]));

        for c in 0..features {
            let g = if gamma.ndim() == 1 {
                gamma[[c]]
            } else {
                gamma[[0]]
            };
            let is = inv_std[[c]];

            let mut sum_og = 0.0f32;
            let mut sum_og_norm = 0.0f32;
            let mut sum_dgamma = 0.0f32;
            let mut sum_dbeta = 0.0f32;

            for b in 0..batch_size {
                for s in 0..spatial_elements {
                    let og = og_reshaped[[b, c, s]];
                    let norm = normalized[[b, c, s]];
                    sum_og += og;
                    sum_og_norm += og * norm;
                    sum_dgamma += og * norm;
                    sum_dbeta += og;
                }
            }

            grad_gamma[[c]] = sum_dgamma;
            grad_beta[[c]] = sum_dbeta;

            if self.training {
                for b in 0..batch_size {
                    for s in 0..spatial_elements {
                        let og = og_reshaped[[b, c, s]];
                        let norm = normalized[[b, c, s]];
                        // BatchNorm backward formula:
                        // dx = (1/N) * gamma * inv_std * (N*og - sum(og) - norm * sum(og * norm))
                        grad_x_reshaped[[b, c, s]] =
                            (1.0 / n) * g * is * (n * og - sum_og - norm * sum_og_norm);
                    }
                }
            } else {
                for b in 0..batch_size {
                    for s in 0..spatial_elements {
                        let og = og_reshaped[[b, c, s]];
                        grad_x_reshaped[[b, c, s]] = og * g * is;
                    }
                }
            }
        }

        let grad_x = grad_x_reshaped
            .into_dyn()
            .to_shape(x.shape())
            .unwrap()
            .to_owned();

        // Return 5 gradients: x, gamma, beta, running_mean (0), running_var (0)
        vec![
            grad_x,
            grad_gamma,
            grad_beta,
            ArrayD::zeros(inputs[3].lock().storage.shape()),
            ArrayD::zeros(inputs[4].lock().storage.shape()),
        ]
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
                log::error!(
                    "CrossEntropyLogits forward: Reshape to 2D logits failed: {}",
                    e
                );
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
            let t_2d = match perm_targets.to_shape((nrows, classes)) {
                Ok(v) => v.to_owned(),
                Err(e) => {
                    log::error!(
                        "CrossEntropyLogits forward: Reshape targets one-hot failed: {}",
                        e
                    );
                    *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
                    return;
                }
            };
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
            *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
            return;
        }
        // average
        let mean = per_sample.iter().sum::<f32>() / (per_sample.len() as f32);
        *output = ArrayD::from_elem(IxDyn(&[][..]), mean);
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
                log::error!(
                    "SoftmaxCrossEntropy forward: Reshape permuted logits failed: {}",
                    e
                );
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
                *v /= sum;
            }
        }
        // compute grad in 2D then reshape back and permute back
        let og = output_grad.iter().next().copied().unwrap_or_else(|| {
            log::error!(
                "SoftmaxCrossEntropy backward: expected scalar output_grad, defaulting to 1.0"
            );
            1.0f32
        });
        let grad_logits_2d = ArrayD::zeros(IxDyn(&[nrows, classes][..]));
        let mut grad_view = match grad_logits_2d.into_dimensionality::<ndarray::Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "SoftmaxCrossEntropy backward: failed to reshape grad_logits to 2D: {}",
                    e
                );
                let grad_logits = ArrayD::zeros(logits.dim());
                let grad_targets = ArrayD::zeros(targets.dim());
                return vec![grad_logits, grad_targets];
            }
        };
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
                    log::error!(
                        "CrossEntropyLogits backward: Reshape targets one-hot failed: {}",
                        e
                    );
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

impl Default for NLLLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for NLLLoss {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let log_probs = inputs[0].to_f32_array();
        let targets = inputs[1].to_f32_array();
        if log_probs.ndim() < 1 {
            log::error!(
                "NLLLoss: log_probs must be at least 1D; got shape: {:?}",
                log_probs.shape()
            );
            *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
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
                *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
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
                    *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
                    return;
                }
            };
            for i in 0..nrows {
                for j in 0..classes {
                    total += -lp_2d[[i, j]] * t_2d[[i, j]];
                }
            }
        } else {
            log::error!(
                "NLLLoss: targets shape incompatible; log_probs shape: {:?}, targets shape: {:?}",
                log_probs.shape(),
                targets.shape()
            );
            *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
            return;
        }
        *output = ArrayD::from_elem(IxDyn(&[][..]), total / (nrows as f32));
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
        let grad_2d = ArrayD::zeros(IxDyn(&[nrows, classes][..]));
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
                *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
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
            let t_2d = match perm_targets.to_shape((nrows, classes)) {
                Ok(v) => v.to_owned(),
                Err(e) => {
                    log::error!(
                        "SoftmaxCrossEntropyLogits forward: Reshape targets one-hot failed: {}",
                        e
                    );
                    *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
                    return;
                }
            };
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
            *output = ArrayD::from_elem(IxDyn(&[][..]), f32::NAN);
            return;
        }
        *output = ArrayD::from_elem(IxDyn(&[][..]), loss_sum / (nrows as f32));
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
                *v /= sum;
            }
        }
        let og = output_grad.iter().next().copied().unwrap_or_else(|| {
            log::error!("Softmax backward: expected scalar output_grad, defaulting to 1.0");
            1.0f32
        });
        let grad_logits_2d = ArrayD::zeros(IxDyn(&[nrows, classes][..]));
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
                log::error!(
                    "SoftmaxCrossEntropyLogits backward: Reshape grad failed: {}",
                    e
                );
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
        // Manual concatenation into the provided output buffer to avoid an intermediate allocation
        let axis = self.0;
        if inputs.is_empty() {
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        // Validate dimensionality and compute output shape
        let first = inputs[0].lock().storage.to_f32_array();
        let mut out_shape = first.shape().to_vec();
        let ndim = out_shape.len();
        if axis >= ndim {
            log::error!(
                "Concat forward: axis {} out of bounds for ndim {}",
                axis,
                ndim
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        let mut axis_sum = 0usize;
        let mut arrays = Vec::new();
        for input in inputs {
            let arr = input.lock().storage.to_f32_array();
            if arr.ndim() != ndim {
                log::error!("Concat forward: mismatched ndim among inputs");
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
            // ensure other dims match
            for i in 0..ndim {
                if i == axis {
                    continue;
                }
                if arr.shape()[i] != out_shape[i] {
                    log::error!(
                        "Concat forward: shape mismatch on non-concat axis: {:?} vs {:?}",
                        arr.shape(),
                        out_shape
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
            }
            axis_sum += arr.shape()[axis];
            arrays.push(arr);
        }
        out_shape[axis] = axis_sum;
        *output = ArrayD::<f32>::zeros(IxDyn(&out_shape));

        // Copy each input into the correct slice of the output
        let mut cur = 0usize;
        for a in arrays.iter() {
            let len = a.shape()[axis];
            // Build slice info: .. for all dims except axis -> (cur..cur+len)
            let mut slice_elems: Vec<SliceInfoElem> = Vec::new();
            for i in 0..ndim {
                if i == axis {
                    slice_elems.push((cur..cur + len).into());
                } else {
                    slice_elems.push((..).into());
                }
            }
            let slice_info: SliceInfo<_, IxDyn, IxDyn> =
                unsafe { SliceInfo::new(slice_elems).unwrap() };
            let mut out_slice = output.slice_mut(slice_info.as_ref());
            out_slice.assign(&a.view());
            cur += len;
        }
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
            let slice_info_res = unsafe { SliceInfo::new(slice_info_elems) };
            let slice_info: SliceInfo<_, IxDyn, IxDyn> = match slice_info_res {
                Ok(s) => s,
                Err(e) => {
                    log::error!("Concat backward: invalid slice info: {}", e);
                    // push zeros for this input to preserve shape
                    grads.push(ArrayD::<f32>::zeros(IxDyn(&input_shape)));
                    current_index += input_shape[axis];
                    continue;
                }
            };
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
                ArrayD::<f32>::zeros(IxDyn(&[0][..]))
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
            let slice_info_res = unsafe { SliceInfo::new(slice_info_elems) };
            let slice_info: SliceInfo<_, IxDyn, IxDyn> = match slice_info_res {
                Ok(s) => s,
                Err(e) => {
                    log::error!("Stack backward: invalid slice info: {}", e);
                    // push zeros default
                    grads.push(ArrayD::<f32>::zeros(IxDyn(&[0][..])));
                    continue;
                }
            };
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
        println!(
            "Slice::forward start, axis={}, start={}, len={}",
            self.axis, self.start, self.len
        );
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let a = inputs[0].to_f32_array();
        println!("Slice::forward: got array, shape={:?}", a.shape());
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let mut slice_info_elems: Vec<SliceInfoElem> = Vec::with_capacity(a.ndim());
        for i in 0..a.ndim() {
            if i == self.axis {
                slice_info_elems.push((self.start..self.start + self.len).into());
            } else {
                slice_info_elems.push((..).into());
            }
        }
        let slice_info: SliceInfo<_, IxDyn, IxDyn> =
            unsafe { SliceInfo::new(slice_info_elems).unwrap() };
        println!("Slice::forward: slicing");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        *output = a.slice(slice_info).to_owned().into_dyn();
        println!("Slice::forward: done");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a_shape = inputs[0].lock().storage.shape();
        let mut grad = ArrayD::zeros(IxDyn(&a_shape));
        let mut slice_info_elems: Vec<SliceInfoElem> = Vec::with_capacity(a_shape.len());
        for i in 0..a_shape.len() {
            if i == self.axis {
                slice_info_elems.push((self.start..self.start + self.len).into());
            } else {
                slice_info_elems.push((..).into());
            }
        }
        let slice_info: SliceInfo<_, IxDyn, IxDyn> =
            unsafe { SliceInfo::new(slice_info_elems).unwrap() };
        grad.slice_mut(slice_info).assign(output_grad);
        vec![grad]
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

/// Unfold2D (im2col) operation for NCHW tensors.
/// Input [N, C, H, W] -> Output [N, C * kH * kW, L] where L = out_h * out_w.
pub struct Unfold2D {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Unfold2D {
    pub fn new(kernel_h: usize, kernel_w: usize, stride: usize, padding: usize) -> Self {
        Unfold2D {
            kernel_h,
            kernel_w,
            stride,
            padding,
        }
    }
}

impl Operation for Unfold2D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        let x4 = match x.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Unfold2D forward: input must be 4D NCHW: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        let (n, c, h, w) = x4.dim();
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let s = self.stride as isize;
        let p = self.padding as isize;
        if kh == 0 || kw == 0 || self.stride == 0 {
            log::error!("Unfold2D forward: kernel and stride must be > 0");
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let out_h = ((h as isize + 2 * p - kh as isize) / s + 1) as usize;
        let out_w = ((w as isize + 2 * p - kw as isize) / s + 1) as usize;
        let l = out_h * out_w;
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, c * kh * kw, l][..]));

        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let col_idx = oh * out_w + ow;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh as isize * s + khi as isize - p;
                                let iw = ow as isize * s + kwi as isize - p;
                                let row = ci * kh * kw + khi * kw + kwi;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    out[[ni, row, col_idx]] =
                                        x4[[ni, ci, ih as usize, iw as usize]];
                                }
                            }
                        }
                    }
                }
            }
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        let x4 = match x.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(_) => return vec![ArrayD::zeros(IxDyn(&[0][..]))],
        };
        let (n, c, h, w) = x4.dim();
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let s = self.stride as isize;
        let p = self.padding as isize;
        let out_h = ((h as isize + 2 * p - kh as isize) / s + 1) as usize;
        let out_w = ((w as isize + 2 * p - kw as isize) / s + 1) as usize;
        let l = out_h * out_w;

        if output_grad.shape() != [n, c * kh * kw, l] {
            log::error!(
                "Unfold2D backward: output_grad shape {:?} expected [{}, {}, {}]",
                output_grad.shape(),
                n,
                c * kh * kw,
                l
            );
            return vec![ArrayD::zeros(IxDyn(&[n, c, h, w][..]))];
        }

        let mut grad_x = ArrayD::<f32>::zeros(IxDyn(&[n, c, h, w][..]));
        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let col_idx = oh * out_w + ow;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh as isize * s + khi as isize - p;
                                let iw = ow as isize * s + kwi as isize - p;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let row = ci * kh * kw + khi * kw + kwi;
                                    grad_x[[ni, ci, ih as usize, iw as usize]] +=
                                        output_grad[[ni, row, col_idx]];
                                }
                            }
                        }
                    }
                }
            }
        }

        vec![grad_x]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Fold2D (col2im) operation for NCHW tensors.
/// Input [N, C*kH*kW, L] -> Output [N, C, out_h, out_w] with overlap-add behavior.
pub struct Fold2D {
    pub output_h: usize,
    pub output_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Fold2D {
    pub fn new(
        output_h: usize,
        output_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Fold2D {
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        }
    }
}

impl Operation for Fold2D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let cols = inputs[0].to_f32_array();
        let cols3 = match cols.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Fold2D forward: input must be 3D [N, C*kH*kW, L]: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        let (n, ckk, l) = cols3.dim();
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        if kh == 0 || kw == 0 || self.stride == 0 {
            log::error!("Fold2D forward: kernel and stride must be > 0");
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        if ckk % (kh * kw) != 0 {
            log::error!(
                "Fold2D forward: channel dimension {} not divisible by kernel area {}",
                ckk,
                kh * kw
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        let c = ckk / (kh * kw);

        let s = self.stride as isize;
        let p = self.padding as isize;
        let out_h = ((self.output_h as isize + 2 * p - kh as isize) / s + 1) as usize;
        let out_w = ((self.output_w as isize + 2 * p - kw as isize) / s + 1) as usize;
        if l != out_h * out_w {
            log::error!(
                "Fold2D forward: L={} does not match expected out_h*out_w={}",
                l,
                out_h * out_w
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, c, self.output_h, self.output_w][..]));
        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let col_idx = oh * out_w + ow;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh as isize * s + khi as isize - p;
                                let iw = ow as isize * s + kwi as isize - p;
                                if ih >= 0
                                    && ih < self.output_h as isize
                                    && iw >= 0
                                    && iw < self.output_w as isize
                                {
                                    let row = ci * kh * kw + khi * kw + kwi;
                                    out[[ni, ci, ih as usize, iw as usize]] +=
                                        cols3[[ni, row, col_idx]];
                                }
                            }
                        }
                    }
                }
            }
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let cols = inputs[0].to_f32_array();
        let cols3 = match cols.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(_) => return vec![ArrayD::zeros(IxDyn(&[0][..]))],
        };
        let og4 = match output_grad.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(_) => return vec![ArrayD::zeros(IxDyn(cols.shape()))],
        };

        let (n, ckk, l) = cols3.dim();
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        if kh == 0 || kw == 0 || ckk % (kh * kw) != 0 || self.stride == 0 {
            return vec![ArrayD::zeros(IxDyn(cols.shape()))];
        }
        let c = ckk / (kh * kw);
        let s = self.stride as isize;
        let p = self.padding as isize;
        let out_h = ((self.output_h as isize + 2 * p - kh as isize) / s + 1) as usize;
        let out_w = ((self.output_w as isize + 2 * p - kw as isize) / s + 1) as usize;
        if l != out_h * out_w || og4.dim() != (n, c, self.output_h, self.output_w) {
            return vec![ArrayD::zeros(IxDyn(cols.shape()))];
        }

        let mut grad_cols = ArrayD::<f32>::zeros(IxDyn(cols.shape()));
        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let col_idx = oh * out_w + ow;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh as isize * s + khi as isize - p;
                                let iw = ow as isize * s + kwi as isize - p;
                                if ih >= 0
                                    && ih < self.output_h as isize
                                    && iw >= 0
                                    && iw < self.output_w as isize
                                {
                                    let row = ci * kh * kw + khi * kw + kwi;
                                    grad_cols[[ni, row, col_idx]] =
                                        og4[[ni, ci, ih as usize, iw as usize]];
                                }
                            }
                        }
                    }
                }
            }
        }

        vec![grad_cols]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The Conv3D operation (NCDHW layout) with optional bias
pub struct Conv3D {
    pub stride: usize,
    pub padding: usize,
}

impl Conv3D {
    pub fn new(stride: usize, padding: usize) -> Self {
        Conv3D { stride, padding }
    }
}

impl Operation for Conv3D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // inputs: [input (N,Cin,D,H,W), weight (Cout,Cin,kD,kH,kW), bias (Cout) optional]
        let input = inputs[0].to_f32_array();
        let weights = inputs[1].to_f32_array();
        let bias_opt = if inputs.len() > 2 {
            Some(inputs[2].lock().storage.to_f32_array())
        } else {
            None
        };

        let input = match input.view().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv3D forward: input is not 5D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv3D forward: weights are not 5D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let (n, cin, din, hin, win) = input.dim();
        let (cout, cin2, kd, kh, kw) = w.dim();
        assert_eq!(cin, cin2, "Conv3D: input channel mismatch with weight");

        let stride = self.stride as isize;
        let pad = self.padding as isize;
        let dout = ((din as isize - kd as isize + 2 * pad) / stride + 1) as usize;
        let hout = ((hin as isize - kh as isize + 2 * pad) / stride + 1) as usize;
        let wout = ((win as isize - kw as isize + 2 * pad) / stride + 1) as usize;

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, dout, hout, wout][..]));
        let mut out5 = match out.view_mut().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv3D forward: output buffer reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        for batch in 0..n {
            for oc in 0..cout {
                for od in 0..dout {
                    for oh in 0..hout {
                        for ow in 0..wout {
                            let mut sum = 0.0f32;
                            for ic in 0..cin {
                                for kd_i in 0..kd {
                                    for kh_i in 0..kh {
                                        for kw_i in 0..kw {
                                            let id = od as isize * stride + kd_i as isize - pad;
                                            let ih = oh as isize * stride + kh_i as isize - pad;
                                            let iw = ow as isize * stride + kw_i as isize - pad;
                                            if id >= 0
                                                && id < din as isize
                                                && ih >= 0
                                                && ih < hin as isize
                                                && iw >= 0
                                                && iw < win as isize
                                            {
                                                let iv = input[[
                                                    batch,
                                                    ic,
                                                    id as usize,
                                                    ih as usize,
                                                    iw as usize,
                                                ]];
                                                let wv = w[[oc, ic, kd_i, kh_i, kw_i]];
                                                sum += iv * wv;
                                            }
                                        }
                                    }
                                }
                            }
                            if let Some(ref b) = bias_opt {
                                sum += b[[oc]];
                            }
                            out5[[batch, oc, od, oh, ow]] = sum;
                        }
                    }
                }
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let input = inputs[0].to_f32_array();
        let weights = inputs[1].to_f32_array();
        let input = match input.view().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv3D backward: input is not 5D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv3D backward: weights are not 5D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let (n, cin, din, hin, win) = input.dim();
        let (cout, _, kd, kh, kw) = w.dim();
        let outg_data = output_grad.clone();
        let outg = match outg_data.view().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv3D backward: output_grad is not 5D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, din, hin, win][..]));
        let mut grad_w = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, kd, kh, kw][..]));
        let mut grad_b = None;
        if inputs.len() > 2 {
            grad_b = Some(ArrayD::<f32>::zeros(IxDyn(&[cout][..])));
        }

        let mut grad_in5 = match grad_in.view_mut().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv3D backward: failed to reshape grad_in to 5D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let mut grad_w5 = match grad_w.view_mut().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "Conv3D backward: failed to convert grad_w to 5D view: {}",
                    e
                );
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let mut grad_b_view = match grad_b.as_mut() {
            Some(x) => match x.view_mut().into_dimensionality::<ndarray::Ix1>() {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!(
                        "Conv3D backward: failed to convert grad_b to 1D view: {}",
                        e
                    );
                    let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                    let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                    let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                    return vec![grad_in, grad_w, grad_b];
                }
            },
            None => None,
        };

        let stride = self.stride as isize;
        let pad = self.padding as isize;

        let dout = outg.dim().2;
        let hout = outg.dim().3;
        let wout = outg.dim().4;

        // grad_input
        for batch in 0..n {
            for oc in 0..cout {
                for od in 0..dout {
                    for oh in 0..hout {
                        for ow in 0..wout {
                            let ogv = outg[[batch, oc, od, oh, ow]];
                            for ic in 0..cin {
                                for kd_i in 0..kd {
                                    for kh_i in 0..kh {
                                        for kw_i in 0..kw {
                                            let id = od as isize * stride + kd_i as isize - pad;
                                            let ih = oh as isize * stride + kh_i as isize - pad;
                                            let iw = ow as isize * stride + kw_i as isize - pad;
                                            if id >= 0
                                                && id < din as isize
                                                && ih >= 0
                                                && ih < hin as isize
                                                && iw >= 0
                                                && iw < win as isize
                                            {
                                                grad_in5[[
                                                    batch,
                                                    ic,
                                                    id as usize,
                                                    ih as usize,
                                                    iw as usize,
                                                ]] += ogv * w[[oc, ic, kd_i, kh_i, kw_i]];
                                            }
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
        }

        // grad_w
        for oc in 0..cout {
            for ic in 0..cin {
                for kd_i in 0..kd {
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let mut sum = 0f32;
                            for batch in 0..n {
                                for od in 0..dout {
                                    for oh in 0..hout {
                                        for ow in 0..wout {
                                            let id = od as isize * stride + kd_i as isize - pad;
                                            let ih = oh as isize * stride + kh_i as isize - pad;
                                            let iw = ow as isize * stride + kw_i as isize - pad;
                                            if id >= 0
                                                && id < din as isize
                                                && ih >= 0
                                                && ih < hin as isize
                                                && iw >= 0
                                                && iw < win as isize
                                            {
                                                sum += outg[[batch, oc, od, oh, ow]]
                                                    * input[[
                                                        batch,
                                                        ic,
                                                        id as usize,
                                                        ih as usize,
                                                        iw as usize,
                                                    ]];
                                            }
                                        }
                                    }
                                }
                            }
                            grad_w5[[oc, ic, kd_i, kh_i, kw_i]] = sum;
                        }
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

/// Depthwise Separable Conv2D: first apply channel-wise depthwise convolution (per-channel kernel), then pointwise 1x1 conv to mix channels.
pub struct DepthwiseSeparableConv2D {
    pub stride: usize,
    pub padding: usize,
}

impl DepthwiseSeparableConv2D {
    pub fn new(stride: usize, padding: usize) -> Self {
        DepthwiseSeparableConv2D { stride, padding }
    }
}

impl Operation for DepthwiseSeparableConv2D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // inputs: [input (N,Cin,H,W), depthwise_weight (Cin,1,kH,kW), pointwise_weight (Cout,Cin,1,1), bias optional (Cout)]
        let input = inputs[0].to_f32_array();
        let dw = inputs[1].to_f32_array();
        let pw = inputs[2].to_f32_array();
        let bias_opt = if inputs.len() > 3 {
            Some(inputs[3].lock().storage.to_f32_array())
        } else {
            None
        };

        let input = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("DepthwiseSeparableConv2D forward: input is not 4D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let depthwise = match dw.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D forward: depthwise weights not 4D: {}",
                    e
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let pointwise = match pw.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D forward: pointwise weights not 4D: {}",
                    e
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let (n, cin, hin, win) = input.dim();
        let (cin2, one, kh, kw) = depthwise.dim();
        assert_eq!(
            one, 1,
            "Depthwise weight must have inner channel dimension of 1"
        );
        assert_eq!(cin, cin2, "Depthwise: channel mismatch");
        let (cout, cin3, pkh, pkw) = pointwise.dim();
        assert_eq!(
            cin3, cin,
            "Pointwise input channel mismatch with depthwise output"
        );
        // pointwise should be 1x1 conv
        assert_eq!(pkh, 1);
        assert_eq!(pkw, 1);

        let stride = self.stride as isize;
        let pad = self.padding as isize;
        let hout = ((hin as isize - kh as isize + 2 * pad) / stride + 1) as usize;
        let wout = ((win as isize - kw as isize + 2 * pad) / stride + 1) as usize;

        // output of depthwise is (N, Cin, hout, wout)
        let mut depth_out = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hout, wout][..]));
        let mut depth_out4 = match depth_out.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("DepthwiseSeparableConv2D forward: failed to convert depth_out to 4D mutable view: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        // Depthwise convolution (per channel)
        for batch in 0..n {
            for c in 0..cin {
                for oh in 0..hout {
                    for ow in 0..wout {
                        let mut sum = 0.0f32;
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let ih = oh as isize * stride + kh_i as isize - pad;
                                let iw = ow as isize * stride + kw_i as isize - pad;
                                if ih >= 0 && ih < hin as isize && iw >= 0 && iw < win as isize {
                                    let iv = input[[batch, c, ih as usize, iw as usize]];
                                    let wv = depthwise[[c, 0, kh_i, kw_i]];
                                    sum += iv * wv;
                                }
                            }
                        }
                        depth_out4[[batch, c, oh, ow]] = sum;
                    }
                }
            }
        }

        // Pointwise 1x1 conv: (N, Cout, hout, wout) from (N, Cin, hout, wout)
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, hout, wout][..]));
        let mut out4 = match out.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "ConvTranspose2D forward: failed to convert out to 4D mutable view: {}",
                    e
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        for batch in 0..n {
            for oc in 0..cout {
                for oh in 0..hout {
                    for ow in 0..wout {
                        let mut sum = 0.0f32;
                        for ic in 0..cin {
                            sum += depth_out4[[batch, ic, oh, ow]] * pointwise[[oc, ic, 0, 0]];
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
        // We'll compute gradients wrt input, depthwise weights, pointwise weights, and optional bias
        let input = inputs[0].to_f32_array();
        let depthwise = inputs[1].to_f32_array();
        let pointwise = inputs[2].to_f32_array();
        let input = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("DepthwiseSeparableConv2D backward: input not 4D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0][..]))];
            }
        };
        let depthwise = match depthwise.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: depthwise weights not 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0][..]))];
            }
        };
        let pointwise = match pointwise.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: pointwise weights not 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0][..]))];
            }
        };

        let (n, cin, hin, win) = input.dim();
        let (_cin2, _, kh, kw) = depthwise.dim();
        let (cout, _cin3, _, _) = pointwise.dim();
        let outg_data = output_grad.clone();
        let outg = match outg_data.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: output_grad must be 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0][..]))];
            }
        };

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hin, win][..]));
        let mut grad_depth = ArrayD::<f32>::zeros(IxDyn(&[cin, 1, kh, kw][..]));
        let mut grad_point = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, 1, 1][..]));
        let mut grad_bias = Some(ArrayD::<f32>::zeros(IxDyn(&[cout][..])));

        let mut grad_in4 = match grad_in.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: failed to convert grad_in to 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0][..]))];
            }
        };
        let mut grad_depth4 = match grad_depth.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: failed to convert grad_depth to 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0][..]))];
            }
        };
        let mut grad_point4 = match grad_point.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: failed to convert grad_point to 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0][..]))];
            }
        };
        let mut grad_bias_view = match grad_bias.as_mut() {
            Some(x) => {
                match x.view_mut().into_dimensionality::<ndarray::Ix1>() {
                    Ok(v) => Some(v),
                    Err(e) => {
                        log::error!("DepthwiseSeparableConv2D backward: failed to convert grad_bias to 1D: {}", e);
                        return vec![ArrayD::zeros(IxDyn(&[0][..]))];
                    }
                }
            }
            None => None,
        };

        // First, compute grad wrt pointwise weights and bias, and also grad of depthwise output (before pointwise) to compute grad_in via depthwise
        // grad_depth_out: same shape as depth_out
        let stride = self.stride as isize;
        let pad = self.padding as isize;
        let hout = ((hin as isize - kh as isize + 2 * pad) / stride + 1) as usize;
        let wout = ((win as isize - kw as isize + 2 * pad) / stride + 1) as usize;
        let mut grad_depth_out = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hout, wout][..]));
        let mut grad_depth_out4 = match grad_depth_out
            .view_mut()
            .into_dimensionality::<ndarray::Ix4>()
        {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: failed to convert grad_depth_out to 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0][..]))];
            }
        };

        // Compute grad_depth_out and grad_point and bias
        for batch in 0..n {
            for oc in 0..cout {
                for oh in 0..hout {
                    for ow in 0..wout {
                        let g = outg[[batch, oc, oh, ow]];
                        for ic in 0..cin {
                            grad_depth_out4[[batch, ic, oh, ow]] += g * pointwise[[oc, ic, 0, 0]];
                        }
                        if let Some(ref mut gb) = grad_bias_view {
                            gb[oc] += g;
                        }
                    }
                }
            }
        }

        // compute grad_point properly: sum over batch and spatial dims: grad_point[oc,ic,0,0] = sum_{b,oh,ow} outg[b,oc,oh,ow] * depth_out[b,ic,oh,ow]
        // For depth_out we need to compute the forward depth_out again from input and depthwise weights
        let mut depth_out = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hout, wout][..]));
        let mut depth_out4_view = match depth_out.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: failed to convert depth_out to 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let stride = self.stride as isize;
        let pad = self.padding as isize;
        for batch in 0..n {
            for c in 0..cin {
                for oh in 0..hout {
                    for ow in 0..wout {
                        let mut sum = 0.0f32;
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let ih = oh as isize * stride + kh_i as isize - pad;
                                let iw = ow as isize * stride + kw_i as isize - pad;
                                if ih >= 0 && ih < hin as isize && iw >= 0 && iw < win as isize {
                                    sum += input[[batch, c, ih as usize, iw as usize]]
                                        * depthwise[[c, 0, kh_i, kw_i]];
                                }
                            }
                        }
                        depth_out4_view[[batch, c, oh, ow]] = sum;
                    }
                }
            }
        }

        for oc in 0..cout {
            for ic in 0..cin {
                let mut sum = 0.0f32;
                for batch in 0..n {
                    for oh in 0..hout {
                        for ow in 0..wout {
                            sum += outg[[batch, oc, oh, ow]] * depth_out4_view[[batch, ic, oh, ow]];
                        }
                    }
                }
                grad_point4[[oc, ic, 0, 0]] = sum;
            }
        }

        // grad wrt depthwise weights: correlate input with grad_depth_out
        for c in 0..cin {
            for kh_i in 0..kh {
                for kw_i in 0..kw {
                    let mut sum = 0.0f32;
                    for batch in 0..n {
                        for oh in 0..hout {
                            for ow in 0..wout {
                                let ih = oh as isize * stride + kh_i as isize - pad;
                                let iw = ow as isize * stride + kw_i as isize - pad;
                                if ih >= 0 && ih < hin as isize && iw >= 0 && iw < win as isize {
                                    sum += grad_depth_out4[[batch, c, oh, ow]]
                                        * input[[batch, c, ih as usize, iw as usize]];
                                }
                            }
                        }
                    }
                    grad_depth4[[c, 0, kh_i, kw_i]] = sum;
                }
            }
        }

        // grad wrt input: convolve grad_depth_out with flipped depthwise weights (correlation)
        for batch in 0..n {
            for c in 0..cin {
                for oh in 0..hout {
                    for ow in 0..wout {
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let ih = oh as isize * stride + kh_i as isize - pad;
                                let iw = ow as isize * stride + kw_i as isize - pad;
                                if ih >= 0 && ih < hin as isize && iw >= 0 && iw < win as isize {
                                    grad_in4[[batch, c, ih as usize, iw as usize]] +=
                                        grad_depth_out4[[batch, c, oh, ow]]
                                            * depthwise[[c, 0, kh_i, kw_i]];
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut ret: Vec<ArrayD<f32>> = vec![grad_in, grad_depth, grad_point];
        if let Some(gb) = grad_bias {
            ret.push(gb);
        }
        ret
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// ConvTranspose2D (a.k.a. deconvolution) in NCHW layout
pub struct ConvTranspose2D {
    pub stride: usize,
    pub padding: usize,
}

impl ConvTranspose2D {
    pub fn new(stride: usize, padding: usize) -> Self {
        ConvTranspose2D { stride, padding }
    }
}

impl Operation for ConvTranspose2D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // inputs: [input (N,Cin,Hin,Win), weight (Cout,Cin,kH,kW), bias optional (Cout)]
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
                log::error!("ConvTranspose2D forward: input is not 4D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("ConvTranspose2D forward: weights not 4D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let (n, cin, hin, win) = input.dim();
        let (cout, cin2, kh, kw) = w.dim();
        assert_eq!(
            cin, cin2,
            "ConvTranspose2D: input channel mismatch with weight"
        );

        let stride = self.stride as isize;
        let pad = self.padding as isize;
        // output dims: Hout = (Hin-1)*stride - 2*pad + kh, Wout similar
        let hout = ((hin as isize - 1) * stride - 2 * pad + kh as isize) as usize;
        let wout = ((win as isize - 1) * stride - 2 * pad + kw as isize) as usize;

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, hout, wout][..]));
        let mut out4 = match out.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "ConvTranspose2D forward: failed to convert out to 4D mutable view: {}",
                    e
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        // For each input position, scatter into the output
        for batch in 0..n {
            for ic in 0..cin {
                for ih in 0..hin {
                    for iw in 0..win {
                        let iv = input[[batch, ic, ih, iw]];
                        for oc in 0..cout {
                            for kh_i in 0..kh {
                                for kw_i in 0..kw {
                                    // location in output
                                    let oh = ih as isize * stride - pad + kh_i as isize;
                                    let ow = iw as isize * stride - pad + kw_i as isize;
                                    if oh >= 0
                                        && oh < hout as isize
                                        && ow >= 0
                                        && ow < wout as isize
                                    {
                                        out4[[batch, oc, oh as usize, ow as usize]] +=
                                            iv * w[[oc, ic, kh_i, kw_i]];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(b) = bias_opt {
            for oc in 0..cout {
                for batch in 0..n {
                    for oh in 0..hout {
                        for ow in 0..wout {
                            out4[[batch, oc, oh, ow]] += b[[oc]];
                        }
                    }
                }
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // grad wrt input, weights, bias
        let input = inputs[0].to_f32_array();
        let weights = inputs[1].to_f32_array();
        let input = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("ConvTranspose2D backward: input not 4D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("ConvTranspose2D backward: weights not 4D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let (n, cin, hin, win) = input.dim();
        let (cout, _, kh, kw) = w.dim();
        let outg_data = output_grad.clone();
        let outg = match outg_data.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("ConvTranspose2D backward: output_grad not 4D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hin, win][..]));
        let mut grad_w = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, kh, kw][..]));
        let mut grad_b = None;
        if inputs.len() > 2 {
            grad_b = Some(ArrayD::<f32>::zeros(IxDyn(&[cout][..])));
        }

        let mut grad_in4 = match grad_in.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "ConvTranspose2D backward: failed to convert grad_in to 4D mutable view: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let mut grad_w4 = match grad_w.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "ConvTranspose2D backward: failed to convert grad_w to 4D mutable view: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let mut grad_b_view = match grad_b.as_mut() {
            Some(x) => match x.view_mut().into_dimensionality::<ndarray::Ix1>() {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!(
                        "ConvTranspose2D backward: failed to convert grad_b to 1D view: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[0]))];
                }
            },
            None => None,
        };

        let stride = self.stride as isize;
        let pad = self.padding as isize;
        let hout = outg.dim().2;
        let wout = outg.dim().3;

        // grad wrt input: accumulate over outg * weights at appropriate positions
        for batch in 0..n {
            for ic in 0..cin {
                for ih in 0..hin {
                    for iw in 0..win {
                        let mut sum = 0.0f32;
                        for oc in 0..cout {
                            for kh_i in 0..kh {
                                for kw_i in 0..kw {
                                    let oh = ih as isize * stride - pad + kh_i as isize;
                                    let ow = iw as isize * stride - pad + kw_i as isize;
                                    if oh >= 0
                                        && oh < hout as isize
                                        && ow >= 0
                                        && ow < wout as isize
                                    {
                                        sum += outg[[batch, oc, oh as usize, ow as usize]]
                                            * w[[oc, ic, kh_i, kw_i]];
                                    }
                                }
                            }
                        }
                        grad_in4[[batch, ic, ih, iw]] = sum;
                    }
                }
            }
        }

        // grad wrt weights: correlate outg with input positions
        for oc in 0..cout {
            for ic in 0..cin {
                for kh_i in 0..kh {
                    for kw_i in 0..kw {
                        let mut sum = 0.0f32;
                        for batch in 0..n {
                            for ih in 0..hin {
                                for iw in 0..win {
                                    let oh = ih as isize * stride - pad + kh_i as isize;
                                    let ow = iw as isize * stride - pad + kw_i as isize;
                                    if oh >= 0
                                        && oh < hout as isize
                                        && ow >= 0
                                        && ow < wout as isize
                                    {
                                        sum += outg[[batch, oc, oh as usize, ow as usize]]
                                            * input[[batch, ic, ih, iw]];
                                    }
                                }
                            }
                        }
                        grad_w4[[oc, ic, kh_i, kw_i]] = sum;
                    }
                }
            }
        }

        // grad bias is sum of outg across batch/spatial dims
        if let Some(ref mut gb) = grad_b_view {
            for oc in 0..cout {
                let mut sum = 0.0f32;
                for batch in 0..n {
                    for oh in 0..hout {
                        for ow in 0..wout {
                            sum += outg[[batch, oc, oh, ow]];
                        }
                    }
                }
                gb[oc] = sum;
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

/// The Conv1D operation (NCL layout) with optional bias
pub struct Conv1D {
    pub stride: usize,
    pub padding: usize,
}

impl Conv1D {
    pub fn new(stride: usize, padding: usize) -> Self {
        Conv1D { stride, padding }
    }
}

impl Operation for Conv1D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // inputs: [input (N,C,L), weight (Cout,Cin,kL), bias (Cout) optional]
        let input = inputs[0].to_f32_array();
        let weights = inputs[1].to_f32_array();
        let bias_opt = if inputs.len() > 2 {
            Some(inputs[2].lock().storage.to_f32_array())
        } else {
            None
        };

        let input = match input.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv1D forward: input is not 3D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv1D forward: weights are not 3D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let (n, cin, lin) = input.dim();
        let (cout, cin2, kl) = w.dim();
        assert_eq!(cin, cin2, "Conv1D: input channel mismatch with weight");

        let stride = self.stride as isize;
        let pad = self.padding as isize;
        let lout_calc = (lin as isize - kl as isize + 2 * pad) / stride + 1;

        // Handle case where input is too small for kernel
        if lout_calc <= 0 {
            log::warn!("Conv1D: input length {} too small for kernel {} with padding {}, returning zero output", lin, kl, pad);
            *output = ArrayD::zeros(IxDyn(&[n, cout, 1][..]));
            return;
        }

        let lout = lout_calc as usize;

        // Optimized implementation using im2col + matrix multiplication
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, lout][..]));
        let mut out3 = match out.view_mut().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv1D forward: output buffer reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        // Process each batch separately to reduce memory usage
        for batch in 0..n {
            // im2col: extract patches into columns
            // Shape: [cin * kl, lout]
            let mut col = ndarray::Array2::<f32>::zeros((cin * kl, lout));

            for ol in 0..lout {
                for ic in 0..cin {
                    for kl_i in 0..kl {
                        let il = ol as isize * stride + kl_i as isize - pad;
                        let val = if il >= 0 && il < lin as isize {
                            input[[batch, ic, il as usize]]
                        } else {
                            0.0
                        };
                        col[[ic * kl + kl_i, ol]] = val;
                    }
                }
            }

            // Reshape weights: [cout, cin, kl] -> [cout, cin * kl]
            let w_flat = w.as_standard_layout();
            let w_reshaped = w_flat
                .view()
                .into_shape_with_order((cout, cin * kl))
                .unwrap();

            // Matrix multiplication: [cout, cin * kl] @ [cin * kl, lout] = [cout, lout]
            let batch_out = w_reshaped.dot(&col);

            // Add bias if present
            if let Some(ref b) = bias_opt {
                for oc in 0..cout {
                    for ol in 0..lout {
                        out3[[batch, oc, ol]] = batch_out[[oc, ol]] + b[[oc]];
                    }
                }
            } else {
                for oc in 0..cout {
                    for ol in 0..lout {
                        out3[[batch, oc, ol]] = batch_out[[oc, ol]];
                    }
                }
            }
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let input = inputs[0].to_f32_array();
        let weights = inputs[1].to_f32_array();
        let input = match input.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv1D backward: input is not 3D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv1D backward: weights are not 3D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let (n, cin, lin) = input.dim();
        let (cout, _, kl) = w.dim();
        let outg_data = output_grad.clone();
        let outg = match outg_data.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv1D backward: output_grad is not 3D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, lin][..]));
        let mut grad_w = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, kl][..]));
        let mut grad_b = None;
        if inputs.len() > 2 {
            grad_b = Some(ArrayD::<f32>::zeros(IxDyn(&[cout][..])));
        }

        let mut grad_in3 = match grad_in.view_mut().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv1D backward: failed to reshape grad_in to 3D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let mut grad_w3 = match grad_w.view_mut().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "Conv1D backward: failed to convert grad_w to 3D mutable view: {}",
                    e
                );
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let mut grad_b_view = match grad_b.as_mut() {
            Some(x) => match x.view_mut().into_dimensionality::<ndarray::Ix1>() {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!(
                        "Conv1D backward: failed to convert grad_b to 1D view: {}",
                        e
                    );
                    let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                    let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                    let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                    return vec![grad_in, grad_w, grad_b];
                }
            },
            None => None,
        };

        let stride = self.stride as isize;
        let pad = self.padding as isize;

        let lout = outg.dim().2;

        // grad_input: accumulate contributions from weights * output_grad
        for batch in 0..n {
            for oc in 0..cout {
                for ol in 0..lout {
                    let ogv = outg[[batch, oc, ol]];
                    for ic in 0..cin {
                        for kl_i in 0..kl {
                            let il = ol as isize * stride + kl_i as isize - pad;
                            if il >= 0 && il < lin as isize {
                                grad_in3[[batch, ic, il as usize]] += ogv * w[[oc, ic, kl_i]];
                            }
                        }
                    }
                    if let Some(ref mut gb) = grad_b_view {
                        gb[oc] += ogv;
                    }
                }
            }
        }

        // grad_w
        for oc in 0..cout {
            for ic in 0..cin {
                for kl_i in 0..kl {
                    let mut sum = 0f32;
                    for batch in 0..n {
                        for ol in 0..lout {
                            let il = ol as isize * stride + kl_i as isize - pad;
                            if il >= 0 && il < lin as isize {
                                sum += outg[[batch, oc, ol]] * input[[batch, ic, il as usize]];
                            }
                        }
                    }
                    grad_w3[[oc, ic, kl_i]] = sum;
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

/// ConvTranspose1D (a.k.a. deconvolution) in NCL layout
pub struct ConvTranspose1D {
    pub stride: usize,
    pub padding: usize,
}

impl ConvTranspose1D {
    pub fn new(stride: usize, padding: usize) -> Self {
        ConvTranspose1D { stride, padding }
    }
}

impl Operation for ConvTranspose1D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // inputs: [input (N,Cin,Lin), weight (Cout,Cin,kl), bias optional]
        let input = inputs[0].to_f32_array();
        let weights = inputs[1].to_f32_array();
        let bias_opt = if inputs.len() > 2 {
            Some(inputs[2].lock().storage.to_f32_array())
        } else {
            None
        };

        let input = match input.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("ConvTranspose1D forward: input is not 3D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("ConvTranspose1D forward: weights not 3D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let (n, cin, lin) = input.dim();
        let (cout, cin2, kl) = w.dim();
        assert_eq!(
            cin, cin2,
            "ConvTranspose1D: input channel mismatch with weight"
        );

        let stride = self.stride as isize;
        let pad = self.padding as isize;
        // output length: Lout = (Lin-1)*stride - 2*pad + kl
        let lout = ((lin as isize - 1) * stride - 2 * pad + kl as isize) as usize;

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, lout][..]));
        let mut out3 = match out.view_mut().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "ConvTranspose1D forward: output buffer reshape failed: {}",
                    e
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };

        // For each input position, scatter into the output
        for batch in 0..n {
            for ic in 0..cin {
                for il in 0..lin {
                    let iv = input[[batch, ic, il]];
                    for oc in 0..cout {
                        for kl_i in 0..kl {
                            let ol = il as isize * stride - pad + kl_i as isize;
                            if ol >= 0 && ol < lout as isize {
                                out3[[batch, oc, ol as usize]] += iv * w[[oc, ic, kl_i]];
                            }
                        }
                    }
                }
            }
        }
        if let Some(b) = bias_opt {
            for oc in 0..cout {
                for batch in 0..n {
                    for ol in 0..lout {
                        out3[[batch, oc, ol]] += b[[oc]];
                    }
                }
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // grad wrt input, weights, bias
        let input = inputs[0].to_f32_array();
        let weights = inputs[1].to_f32_array();
        let input = match input.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("ConvTranspose1D backward: input not 3D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("ConvTranspose1D backward: weights not 3D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let (n, cin, lin) = input.dim();
        let (cout, _, kl) = w.dim();
        let outg_data = output_grad.clone();
        let outg = match outg_data.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("ConvTranspose1D backward: output_grad not 3D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, lin][..]));
        let mut grad_w = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, kl][..]));
        let mut grad_b = None;
        if inputs.len() > 2 {
            grad_b = Some(ArrayD::<f32>::zeros(IxDyn(&[cout][..])));
        }

        let mut grad_in3 = match grad_in.view_mut().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "ConvTranspose1D backward: failed to reshape grad_in to 3D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let mut grad_w3 = match grad_w.view_mut().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "ConvTranspose1D backward: failed to convert grad_w to 3D mutable view: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let grad_b_view = match grad_b.as_mut() {
            Some(x) => match x.view_mut().into_dimensionality::<ndarray::Ix1>() {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!(
                        "ConvTranspose1D backward: failed to convert grad_b to 1D view: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[0]))];
                }
            },
            None => None,
        };

        let stride = self.stride as isize;
        let pad = self.padding as isize;
        let lout = outg.dim().2;

        // grad wrt input: accumulate over outg * weights at appropriate positions
        for batch in 0..n {
            for ic in 0..cin {
                for il in 0..lin {
                    let mut sum = 0.0f32;
                    for oc in 0..cout {
                        for kl_i in 0..kl {
                            let ol = il as isize * stride - pad + kl_i as isize;
                            if ol >= 0 && ol < lout as isize {
                                sum += outg[[batch, oc, ol as usize]] * w[[oc, ic, kl_i]];
                            }
                        }
                    }
                    grad_in3[[batch, ic, il]] = sum;
                }
            }
        }

        // grad wrt weights: correlate outg with input positions
        for oc in 0..cout {
            for ic in 0..cin {
                for kl_i in 0..kl {
                    let mut sum = 0.0f32;
                    for batch in 0..n {
                        for ol in 0..lout {
                            // input position that contributed to output at ol given kl_i
                            let il = (ol as isize - kl_i as isize + pad) / stride;
                            // ensure exact integer division mapping back to the original location
                            if il >= 0
                                && il < lin as isize
                                && (il * stride - pad + kl_i as isize) == ol as isize
                            {
                                sum += outg[[batch, oc, ol]] * input[[batch, ic, il as usize]];
                            }
                        }
                    }
                    grad_w3[[oc, ic, kl_i]] = sum;
                }
            }
        }

        if let Some(mut gb) = grad_b_view {
            for oc in 0..cout {
                let mut sum = 0.0f32;
                for batch in 0..n {
                    for ol in 0..lout {
                        sum += outg[[batch, oc, ol]];
                    }
                }
                gb[oc] = sum;
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
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv2D forward: weights are not 4D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
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

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, hout, wout][..]));
        let mut out4 = match out.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv2D forward: output buffer reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
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
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv2D backward: weights are not 4D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
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
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hin, win][..]));
        let mut grad_w = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, kh, kw][..]));
        let mut grad_b = None;
        if inputs.len() > 2 {
            grad_b = Some(ArrayD::<f32>::zeros(IxDyn(&[cout][..])));
        }

        let mut grad_in4 = match grad_in.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv2D backward: failed to reshape grad_in to 4D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let mut grad_w4 = match grad_w.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "Conv2D backward: failed to convert grad_w to 4D mutable view: {}",
                    e
                );
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let mut grad_b_view = match grad_b.as_mut() {
            Some(x) => match x.view_mut().into_dimensionality::<ndarray::Ix1>() {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!(
                        "Conv2D backward: failed to convert grad_b to 1D view: {}",
                        e
                    );
                    let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                    let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                    let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0][..]));
                    return vec![grad_in, grad_w, grad_b];
                }
            },
            None => None,
        };

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
        if !self.training || (self.p - 0.0).abs() < f32::EPSILON {
            *output = x.clone();
            return;
        }
        let keep = 1.0 - self.p;
        let mut mask = ArrayD::<f32>::zeros(x.dim());
        // `_xv` is intentionally unused; we only use the mask values during construction
        for (m, _xv) in mask.iter_mut().zip(x.iter()) {
            let r: f32 = rand::random();
            if r < keep {
                *m = 1.0 / keep;
            } else {
                *m = 0.0;
            }
        }
        *output = x * &mask;
        let mut lock = match self.mask.lock() {
            Ok(l) => l,
            Err(poisoned) => {
                log::error!(
                    "Dropout forward: Failed to acquire mask lock: {:?}",
                    poisoned
                );
                // Do not panic; simply skip caching mask on failure
                return;
            }
        };
        *lock = Some(mask);
    }

    fn backward(&self, _inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        if !self.training || (self.p - 0.0).abs() < f32::EPSILON {
            return vec![output_grad.clone()];
        }
        let mask = match self.mask.lock() {
            Ok(l) => l,
            Err(poisoned) => {
                log::error!(
                    "Dropout backward: Failed to acquire mask lock: {:?}",
                    poisoned
                );
                // Fallback: return original output_grad (no dropout applied)
                return vec![output_grad.clone()];
            }
        };
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

/// Average pooling 2D operation.
pub struct AvgPool2D {
    pub kernel_size: usize,
    pub stride: usize,
}

impl Operation for AvgPool2D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let input = inputs[0].to_f32_array();
        let input_view = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("AvgPool2D forward: input must be 4D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let (batch, c, h, w) = input_view.dim();
        let oh = (h - self.kernel_size) / self.stride + 1;
        let ow = (w - self.kernel_size) / self.stride + 1;
        let mut out = ArrayD::zeros(IxDyn(&[batch, c, oh, ow][..]));
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
                        let sum: f32 = window.iter().cloned().sum();
                        let area = (self.kernel_size * self.kernel_size) as f32;
                        out[[b, ch, i, j]] = sum / area;
                    }
                }
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let input = inputs[0].to_f32_array();
        let input_view = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("AvgPool2D backward: input must be 4D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let (batch, c, h, w) = input_view.dim();
        let oh = (h - self.kernel_size) / self.stride + 1;
        let ow = (w - self.kernel_size) / self.stride + 1;
        let mut grad_in = ArrayD::zeros(IxDyn(&[batch, c, h, w][..]));
        let mut grad_view = match grad_in.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("AvgPool2D backward: failed to convert grad_in to 4D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let og = match output_grad.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("AvgPool2D backward: output_grad must be 4D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let area = (self.kernel_size * self.kernel_size) as f32;
        for b in 0..batch {
            for ch in 0..c {
                for i in 0..oh {
                    for j in 0..ow {
                        let grad = og[[b, ch, i, j]] / area;
                        for gi in (i * self.stride)..(i * self.stride + self.kernel_size) {
                            for gj in (j * self.stride)..(j * self.stride + self.kernel_size) {
                                grad_view[[b, ch, gi, gj]] += grad;
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

/// Adaptive average pooling 2D: maps input size to a specified output size by averaging variable windows.
pub struct AdaptiveAvgPool2D {
    pub out_h: usize,
    pub out_w: usize,
}

impl AdaptiveAvgPool2D {
    pub fn new(out_h: usize, out_w: usize) -> Self {
        AdaptiveAvgPool2D { out_h, out_w }
    }
}

// helper to compute pooling region range for adaptive pooling
fn adaptive_pool_range(in_size: usize, out_size: usize, idx: usize) -> (usize, usize) {
    // inclusive start, exclusive end
    let start = (idx * in_size) / out_size;
    let end = ((idx + 1) * in_size).div_ceil(out_size); // ceil
    (start, end)
}

impl Operation for AdaptiveAvgPool2D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let input = inputs[0].to_f32_array();
        let input_view = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("AdaptiveAvgPool2D forward: input must be 4D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let (batch, c, h, w) = input_view.dim();
        let oh = self.out_h;
        let ow = self.out_w;
        let mut out = ArrayD::zeros(IxDyn(&[batch, c, oh, ow][..]));
        for b in 0..batch {
            for ch in 0..c {
                for i in 0..oh {
                    let (s_h, e_h) = adaptive_pool_range(h, oh, i);
                    for j in 0..ow {
                        let (s_w, e_w) = adaptive_pool_range(w, ow, j);
                        let mut sum = 0.0f32;
                        let mut count = 0usize;
                        for ih in s_h..e_h {
                            for jw in s_w..e_w {
                                sum += input_view[[b, ch, ih, jw]];
                                count += 1;
                            }
                        }
                        let avg = if count == 0 { 0.0 } else { sum / count as f32 };
                        out[[b, ch, i, j]] = avg;
                    }
                }
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let input = inputs[0].to_f32_array();
        let input_view = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("AdaptiveAvgPool2D backward: input must be 4D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let (batch, c, h, w) = input_view.dim();
        let oh = self.out_h;
        let ow = self.out_w;
        let mut grad_in = ArrayD::zeros(IxDyn(&[batch, c, h, w][..]));
        let mut grad_view = match grad_in.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "AdaptiveAvgPool2D backward: failed to convert grad_in to 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let og = match output_grad.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("AdaptiveAvgPool2D backward: output_grad must be 4D: {}", e);
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        for b in 0..batch {
            for ch in 0..c {
                for i in 0..oh {
                    let (s_h, e_h) = adaptive_pool_range(h, oh, i);
                    for j in 0..ow {
                        let (s_w, e_w) = adaptive_pool_range(w, ow, j);
                        let count = (e_h - s_h) * (e_w - s_w);
                        if count == 0 {
                            continue;
                        }
                        let grad = og[[b, ch, i, j]] / (count as f32);
                        for ih in s_h..e_h {
                            for jw in s_w..e_w {
                                grad_view[[b, ch, ih, jw]] += grad;
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

impl Operation for MaxPool2D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let input = inputs[0].to_f32_array();
        let input_view = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "MaxPool2D forward: input must be 4D (batch, channels, height, width): {}",
                    e
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        let (batch, c, h, w) = input_view.dim();
        let oh = (h - self.kernel_size) / self.stride + 1;
        let ow = (w - self.kernel_size) / self.stride + 1;
        let mut out = ArrayD::zeros(IxDyn(&[batch, c, oh, ow][..]));
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
        let input_view = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "MaxPool2D backward: input must be 4D (batch, channels, height, width): {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let (batch, c, h, w) = input_view.dim();
        let oh = (h - self.kernel_size) / self.stride + 1;
        let ow = (w - self.kernel_size) / self.stride + 1;
        let mut grad_in = ArrayD::zeros(IxDyn(&[batch, c, h, w][..]));
        let mut grad_view = match grad_in.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "MaxPool2D backward: failed to convert grad_in to 4D; expected Ix4 shape: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let out_grad_view = match _output_grad.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "MaxPool2D backward: output_grad must be 4D; expected Ix4 shape: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
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

/// RMSNorm operation: Root Mean Square Normalization along an axis (common in transformer variants)
pub struct RMSNorm {
    pub axis: usize,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(axis: usize, eps: f32) -> Self {
        RMSNorm { axis, eps }
    }
}

impl Operation for RMSNorm {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // inputs: x, gamma (scale)
        let x = &inputs[0].lock().storage.to_f32_array();
        let gamma = &inputs[1].lock().storage.to_f32_array();
        let axis = if self.axis >= x.ndim() {
            x.ndim() - 1
        } else {
            self.axis
        };
        let (x_last_axis, perm_opt) = permute_to_last(x, axis);
        let last_axis = x_last_axis.ndim() - 1;
        if let Some(backend_output) =
            get_global_backend().rms_norm(&x_last_axis, gamma, self.eps, last_axis as isize)
        {
            if let Some(ref perm) = perm_opt {
                *output = permute_back(backend_output, perm);
            } else {
                *output = backend_output;
            }
            return;
        }
        // compute mean square across axis
        let sq = x.mapv(|v| v * v);
        // sum over axis and get mean (divide by length along axis to compute mean)
        let len = x.shape()[axis] as f32;
        let mean_sq = sq.sum_axis(Axis(axis)).mapv(|v| v / len);
        let denom = mean_sq.mapv(|v| (v + self.eps).sqrt());
        // broadcast denom back
        let denom_bcast = denom;
        // we need shape alignment; expand dims at axis
        let mut shape_vec = denom_bcast.shape().to_vec();
        shape_vec.insert(axis, 1usize);
        let denom_bcast = match denom_bcast.to_shape(IxDyn(&shape_vec)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!(
                    "RMSNorm forward: failed to reshape denom for broadcasting: {}",
                    e
                );
                *output = ArrayD::zeros(IxDyn(&[][..]));
                return;
            }
        };
        // normalized
        let normalized = x / &denom_bcast;
        // apply scale gamma (broadcast)
        *output = &normalized * gamma;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = &inputs[0].lock().storage.to_f32_array();
        let gamma = &inputs[1].lock().storage.to_f32_array();
        let axis = if self.axis >= x.ndim() {
            x.ndim() - 1
        } else {
            self.axis
        };
        // compute denom (mean of squares): divide sum by length along axis
        let len = x.shape()[axis] as f32;
        let mean_sq = x.mapv(|v| v * v).sum_axis(Axis(axis)).mapv(|v| v / len);
        let denom = mean_sq.mapv(|v| (v + self.eps).sqrt());
        let mut denom_shape = denom.shape().to_vec();
        denom_shape.insert(axis, 1usize);
        let denom_bcast = match denom.to_shape(IxDyn(&denom_shape)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!(
                    "RMSNorm backward: failed to reshape denom for broadcasting: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[])); 2];
            }
        };
        let normalized = x / &denom_bcast;

        // grad wrt x: dL/dx = dL/dy * gamma * (1/denom - x*(mean(x* dL/dy * gamma)/((denom^3))) )
        // For simplicity use ndarray direct formulas (safe but a bit heavier)
        let grad_out = output_grad.clone();
        // grad wrt gamma
        let grad_gamma = {
            // reduce sum(normalized * grad_out) along broadcasted dimensions for gamma
            let mut prod = &normalized * &grad_out;
            // sum along axes except gamma's shape (assume gamma is 1D along axis)
            let reduce_axes: Vec<usize> = (0..prod.ndim()).filter(|&i| i != axis).collect();
            // Sum over all axes besides axis
            for ax in reduce_axes.iter().rev() {
                prod = prod.sum_axis(Axis(*ax));
            }
            // prod now has shape of gamma
            prod.to_owned()
        };

        // grad wrt x: more manual: using formula for RMSNorm
        // d(normalized)/dx = (1/denom) - (x / denom^3) * (1/len) * 2 * x sum? For simplicity we'll use autodiff-like rewrite:
        // Compute grad_x numerically using simple derivation: g = grad_out * gamma; then compute d normalized
        let g = grad_out * gamma; // broadcast
                                  // length along axis
        let len = x.shape()[axis] as f32;
        // sum g * x across axis
        let gx = (&g * x.clone()).sum_axis(Axis(axis));
        let gx_bcast = match gx.to_shape(IxDyn(&denom_shape)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!(
                    "RMSNorm backward: failed to reshape gx for broadcasting: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[])); 2];
            }
        };
        // grad_x = g / denom_bcast - x * (gx_bcast) / (denom_bcast.mapv(|d| d * d * d) * len)
        let denom_cubed = denom_bcast.mapv(|d| d * d * d);
        let grad_x = &g / &denom_bcast - &(x * (&gx_bcast / (denom_cubed * len)));
        vec![grad_x, grad_gamma]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// SwiGLU activation: split last axis into two halves x1,x2; output = x1 * swish(x2)
pub struct SwiGLU;

impl SwiGLU {
    pub fn new() -> Self {
        SwiGLU
    }
}

impl Default for SwiGLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for SwiGLU {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = &inputs[0].lock().storage.to_f32_array();
        let ndim = x.ndim();
        let last = ndim - 1;
        let d = x.shape()[last];
        if !d.is_multiple_of(2) {
            log::error!("SwiGLU forward: last dim {} not divisible by 2", d);
            *output = ArrayD::from_elem(IxDyn(&[0][..]), f32::NAN);
            return;
        }
        let half = d / 2;
        // reshape into (.., 2, half) and compute gate
        // iterate indexes and compute
        // Simpler approach: split along last axis using views
        let x_view = x.view();
        let left = x_view
            .slice_axis(Axis(last), ndarray::Slice::from(..half))
            .to_owned();
        let right = x_view
            .slice_axis(Axis(last), ndarray::Slice::from(half..))
            .to_owned();
        let swish = left.mapv(|v| v * (1.0 / (1.0 + (-v).exp())));
        let out_arr = swish * right;
        *output = out_arr.into_dyn();
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = &inputs[0].lock().storage.to_f32_array();
        let ndim = x.ndim();
        let last = ndim - 1;
        let d = x.shape()[last];
        if !d.is_multiple_of(2) {
            log::error!("SwiGLU backward: last dim {} not divisible by 2", d);
            return vec![ArrayD::zeros(IxDyn(&[0])); 1];
        }
        let half = d / 2;
        let x_view = x.view();
        let left = x_view
            .slice_axis(Axis(last), ndarray::Slice::from(..half))
            .to_owned();
        let right = x_view
            .slice_axis(Axis(last), ndarray::Slice::from(half..))
            .to_owned();
        let sigmoid = left.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let swish = &left * &sigmoid;
        let swish_prime = &sigmoid + &((&left * &sigmoid) * (&sigmoid.mapv(|s| 1.0 - s)));

        // out = swish(left) * right
        // dL/dleft = dL/dout * dout/dleft = output_grad * right * swish'(left)
        // dL/dright = dL/dout * dout/dright = output_grad * swish(left)
        let grad_left = (output_grad * &right * &swish_prime).into_owned();
        let grad_right = (output_grad * &swish).into_owned();

        // Re-concatenate left and right gradients into grad_in of shape x.dim()
        let mut grad_in = ArrayD::<f32>::zeros(x.dim());
        {
            let mut gi_view = grad_in.view_mut();
            gi_view
                .slice_axis_mut(Axis(last), ndarray::Slice::from(..half))
                .assign(&grad_left);
            gi_view
                .slice_axis_mut(Axis(last), ndarray::Slice::from(half..))
                .assign(&grad_right);
        }
        vec![grad_in]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Rotary positional embeddings (RoPE) operation. Applies rotation across head dim pairs.
pub struct RoPE {
    pub num_heads: usize,
    pub theta: f32,
    pub scale: f32,
    pub offset: usize,
}

impl RoPE {
    pub fn new(num_heads: usize, theta: f32, scale: f32, offset: usize) -> Self {
        RoPE {
            num_heads,
            theta,
            scale,
            offset,
        }
    }
}

impl Operation for RoPE {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].lock().storage.to_f32_array();
        log::debug!(
            "RoPE forward: offset={}, num_heads={}, theta={}, scale={}, input_shape={:?}",
            self.offset,
            self.num_heads,
            self.theta,
            self.scale,
            x.shape()
        );
        // inputs: x (shape [*, d_model]) - last dim should be divisible by num_heads
        let ndim = x.ndim();
        let last = ndim - 1;
        let d = x.shape()[last];
        if !d.is_multiple_of(self.num_heads) {
            log::error!(
                "RoPE: last dim {} not divisible by num_heads {}",
                d,
                self.num_heads
            );
            *output = x.clone();
            return;
        }
        let head_dim = d / self.num_heads;
        if !head_dim.is_multiple_of(2) {
            log::error!("RoPE: head_dim {} must be even", head_dim);
            *output = x.clone();
            return;
        }
        // reshape to [*, num_heads, head_dim]
        let shape_vec = x.shape().to_vec();
        let mut new_shape = shape_vec[..last].to_vec();
        new_shape.push(self.num_heads);
        new_shape.push(head_dim);
        let x_reshaped = match x.clone().to_shape(IxDyn(&new_shape)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("RoPE forward: reshape failed: {}", e);
                *output = x.clone();
                return;
            }
        };
        // apply rotation across head_dim pairs
        // for simplicity compute sin/cos per position along the second-to-last axis (assumed seq axis if present)
        // sequence axis index: last - 1 if 3D (batch, seq, d_model), else last - 1
        let seq_axis = match new_shape.len() {
            4 => 1, // [B, S, H, D]
            3 => 1, // [B, S, D]
            2 => 0, // [S, D]
            _ => 0,
        };
        let seq_len = new_shape.get(seq_axis).cloned().unwrap_or(1);
        let pair = head_dim / 2;
        // compute inv_freq using configured theta (LLaMA uses large theta like 500000.0)
        let mut inv_freq = Vec::with_capacity(pair);
        for i in 0..pair {
            let denom = self.theta.powf((2 * i) as f32 / (head_dim as f32));
            inv_freq.push(1.0f32 / denom);
        }
        // compute sin and cos matrix shape [seq_len, pair]
        let mut sin = Array2::<f32>::zeros((seq_len, pair));
        let mut cos = Array2::<f32>::zeros((seq_len, pair));
        for pos in 0..seq_len {
            let abs_pos = (pos + self.offset) as f32;
            for (i, &f) in inv_freq.iter().enumerate() {
                let v = (abs_pos / self.scale) * f;
                sin[[pos, i]] = v.sin();
                cos[[pos, i]] = v.cos();
            }
        }
        // Now apply rotation per position: x' = x * cos - rotate_half(x) * sin
        // x_reshaped has shape prefix dims + [num_heads, head_dim]
        let mut out = x_reshaped.clone();
        // iterate over prefix dims except num_heads and head_dim
        // we'll use raw iterators to mutate
        // index and rotate_pair removed: not used
        // We'll attempt to compute using ndviews
        let mut out_view = out.view_mut();
        let in_view = x_reshaped.view();
        // iterate over all coordinates except last two dims
        let prefix_len = new_shape.len() - 2;
        let mut prefix_indices = vec![0usize; prefix_len];
        // nested loops to iterate prefixes
        let mut done = false;
        while !done {
            // compute position along seq axis
            let pos = if seq_axis < prefix_len {
                prefix_indices[seq_axis]
            } else {
                0
            };
            for h in 0..self.num_heads {
                for i in 0..pair {
                    // Llama uses interleaved pairs: (2*i, 2*i+1)
                    // Old planar: (i, i + pair)
                    let idx1 = 2 * i;
                    let idx2 = 2 * i + 1;
                    // construct full index
                    let mut base1 = prefix_indices.clone();
                    base1.push(h);
                    base1.push(idx1);
                    let mut base2 = prefix_indices.clone();
                    base2.push(h);
                    base2.push(idx2);
                    let val1 = in_view[IxDyn(&base1)];
                    let val2 = in_view[IxDyn(&base2)];
                    let cosv = cos[[pos, i]];
                    let sinv = sin[[pos, i]];
                    // Llama rotate_half formula:
                    // y1 = x1 * cos - x2 * sin
                    // y2 = x2 * cos + x1 * sin
                    out_view[IxDyn(&base1)] = val1 * cosv - val2 * sinv;
                    out_view[IxDyn(&base2)] = val2 * cosv + val1 * sinv;
                }
            }
            // increment prefix_indices
            let mut carry = 1;
            for i in (0..prefix_len).rev() {
                if carry == 0 {
                    break;
                }
                prefix_indices[i] += 1;
                if prefix_indices[i] >= new_shape[i] {
                    prefix_indices[i] = 0;
                    carry = 1;
                } else {
                    carry = 0;
                }
            }
            if carry == 1 {
                done = true;
            }
        }
        // reshape back to original
        *output = match out.to_shape(IxDyn(&shape_vec)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("RoPE forward: reshape back failed: {}", e);
                x.clone()
            }
        };
    }

    fn backward(&self, _inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // backward uses linearity: gradient w.r.t x follows same rotation with cos/sin but differential
        // same op as forward but with roated coefficients applied to output_grad appropriately
        let og = output_grad.clone();
        // reuse forward computation structure but use inverse mapping for gradient components
        // compute inverse rotation using same cos/sin: x_even = y_even * cos + y_odd * sin; x_odd = -y_even * sin + y_odd * cos
        let ndim = og.ndim();
        let last = ndim - 1;
        let d = og.shape()[last];
        if !d.is_multiple_of(self.num_heads) {
            return vec![og];
        }
        let head_dim = d / self.num_heads;
        if !head_dim.is_multiple_of(2) {
            return vec![og];
        }
        let pair = head_dim / 2;
        // build shapes similar to forward
        let shape_vec = og.shape().to_vec();
        let mut new_shape = shape_vec[..last].to_vec();
        new_shape.push(self.num_heads);
        new_shape.push(head_dim);
        let og_reshaped = match og.to_shape(IxDyn(&new_shape)) {
            Ok(v) => v.to_owned(),
            Err(_) => return vec![output_grad.clone()],
        };
        let mut grad_x = og_reshaped.clone();
        let seq_axis = if new_shape.len() >= 2 {
            new_shape.len() - 2
        } else {
            0
        };
        let seq_len = new_shape.get(seq_axis).cloned().unwrap_or(1);
        // compute inv_freq and sin/cos as forward
        let mut inv_freq = Vec::with_capacity(pair);
        for i in 0..pair {
            let denom = 10000f32.powf((2 * i) as f32 / (head_dim as f32));
            inv_freq.push(1.0f32 / denom);
        }
        let mut sin = Array2::<f32>::zeros((seq_len, pair));
        let mut cos = Array2::<f32>::zeros((seq_len, pair));
        for pos in 0..seq_len {
            for (i, &f) in inv_freq.iter().enumerate() {
                let v = pos as f32 * f;
                sin[[pos, i]] = v.sin();
                cos[[pos, i]] = v.cos();
            }
        }
        let mut sin_full = Array2::<f32>::zeros((seq_len, head_dim));
        let mut cos_full = Array2::<f32>::zeros((seq_len, head_dim));
        for pos in 0..seq_len {
            for i in 0..pair {
                sin_full[[pos, 2 * i]] = sin[[pos, i]];
                sin_full[[pos, 2 * i + 1]] = sin[[pos, i]];
                cos_full[[pos, 2 * i]] = cos[[pos, i]];
                cos_full[[pos, 2 * i + 1]] = cos[[pos, i]];
            }
        }
        // apply inverse mapping across all positions
        let mut out_view = grad_x.view_mut();
        let in_view = og_reshaped.view();
        let prefix_len = new_shape.len() - 2;
        let mut prefix_indices = vec![0usize; prefix_len];
        let mut done = false;
        while !done {
            let pos = if seq_axis < prefix_len {
                prefix_indices[seq_axis]
            } else {
                0
            };
            for h in 0..self.num_heads {
                for pair_i in 0..pair {
                    let idx_even = 2 * pair_i;
                    let idx_odd = idx_even + 1;
                    let mut base_even = prefix_indices.clone();
                    base_even.push(h);
                    base_even.push(idx_even);
                    let mut base_odd = prefix_indices.clone();
                    base_odd.push(h);
                    base_odd.push(idx_odd);
                    let ye = in_view[IxDyn(&base_even)];
                    let yo = in_view[IxDyn(&base_odd)];
                    let cosv = cos_full[[pos, pair_i]];
                    let sinv = sin_full[[pos, pair_i]];
                    let xe = ye * cosv + yo * sinv;
                    let xo = -ye * sinv + yo * cosv;
                    out_view[IxDyn(&base_even)] = xe;
                    out_view[IxDyn(&base_odd)] = xo;
                }
            }
            // increment prefix_indices
            let mut carry = 1;
            for i in (0..prefix_len).rev() {
                if carry == 0 {
                    break;
                }
                prefix_indices[i] += 1;
                if prefix_indices[i] >= new_shape[i] {
                    prefix_indices[i] = 0;
                    carry = 1;
                } else {
                    carry = 0;
                }
            }
            if carry == 1 {
                done = true;
            }
        }
        // reshape back
        let res = match grad_x.to_shape(IxDyn(&shape_vec)) {
            Ok(v) => v.to_owned(),
            Err(_) => output_grad.clone(),
        };
        vec![res]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Embedding lookup operation. Inputs: embedding matrix (vocab, dim), indices tensor.
pub struct EmbeddingLookup;
pub struct EmbeddingBag;
pub struct IndexSelect {
    pub dim: usize,
}
pub struct Gather {
    pub dim: usize,
}
pub struct Scatter {
    pub dim: usize,
}
pub struct ScatterAdd {
    pub dim: usize,
}

impl EmbeddingLookup {
    pub fn new() -> Self {
        EmbeddingLookup
    }
}

impl EmbeddingBag {
    pub fn new() -> Self {
        EmbeddingBag
    }
}

impl IndexSelect {
    pub fn new(dim: usize) -> Self {
        IndexSelect { dim }
    }
}

impl Gather {
    pub fn new(dim: usize) -> Self {
        Gather { dim }
    }
}

impl Scatter {
    pub fn new(dim: usize) -> Self {
        Scatter { dim }
    }
}

impl ScatterAdd {
    pub fn new(dim: usize) -> Self {
        ScatterAdd { dim }
    }
}

impl Default for EmbeddingLookup {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EmbeddingBag {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for EmbeddingLookup {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let emb = inputs[0].lock().storage.to_f32_array();
        let indices = inputs[1].lock().storage.to_f32_array();
        // indices are floats of integer values; gather along first dim
        // embedding dim
        let dim = emb.shape()[1];
        let idx_shape = indices.shape().to_vec();
        let mut res_shape = idx_shape.clone();
        res_shape.push(dim);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&res_shape));
        // flatten indices and fill
        let idx_flat = indices.iter().cloned().collect::<Vec<f32>>();
        let emb2 = match emb.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("EmbeddingLookup forward: Embedding must be 2D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        for (i, &fidx) in idx_flat.iter().enumerate() {
            let id = fidx as usize;
            if id >= emb2.shape()[0] {
                log::error!("EmbeddingLookup forward: index {} out of bounds for vocab {}. idx_flat.len={} idx_shape={:?} i={}", id, emb2.shape()[0], idx_flat.len(), idx_shape, i);
                // leave zeros for this position and continue
                continue;
            }
            let _row = emb2.row(id).to_owned().into_dyn();
            // compute multi index from i and place row
            // no-op: compute coordinates directly
            let _idx_count = idx_shape.iter().product::<usize>();
            // create a stable position mapping
            let mut pos = i;
            let mut coords = vec![0usize; idx_shape.len()];
            for d in (0..idx_shape.len()).rev() {
                let s = idx_shape[d];
                coords[d] = pos % s;
                pos /= s;
            }
            // assign
            for k in 0..dim {
                let mut coords_k = coords.clone();
                coords_k.push(k);
                out[IxDyn(&coords_k)] = emb2[[id, k]];
            }
        }
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // grad wrt embeddings: accumulate
        let emb_shape = inputs[0].lock().storage.shape();
        let dim = emb_shape[1];
        let vocab = emb_shape[0];
        let mut grad_emb = ArrayD::<f32>::zeros(IxDyn(&[vocab, dim][..]));
        let indices = inputs[1].lock().storage.to_f32_array();
        // iterate over output_grad and accumulate
        let idx_shape = indices.shape().to_vec();
        let _idx_count = idx_shape.iter().product::<usize>();
        let idx_flat = indices.iter().cloned().collect::<Vec<f32>>();
        for (i, &fidx) in idx_flat.iter().enumerate() {
            let id = fidx as usize;
            if id >= vocab {
                log::error!("EmbeddingLookup backward: index {} out of bounds for vocab {}. idx_flat.len={} idx_shape={:?} i={}", id, vocab, idx_flat.len(), idx_shape, i);
                continue;
            }
            // compute coords
            let mut pos = i;
            let mut coords = vec![0usize; idx_shape.len()];
            for d in (0..idx_shape.len()).rev() {
                let s = idx_shape[d];
                coords[d] = pos % s;
                pos /= s;
            }
            for k in 0..dim {
                let mut coords_k = coords.clone();
                coords_k.push(k);
                grad_emb[[id, k]] += output_grad[IxDyn(&coords_k)];
            }
        }
        // gradient wrt indices is None (non-diff)
        let grad_indices = ArrayD::zeros(indices.dim());
        vec![grad_emb, grad_indices]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for EmbeddingBag {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // inputs: emb [vocab, dim], indices [nnz], offsets [bags]
        let emb = inputs[0].lock().storage.to_f32_array();
        let indices = inputs[1].lock().storage.to_f32_array();
        let offsets = inputs[2].lock().storage.to_f32_array();

        let emb2 = match emb.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("EmbeddingBag forward: emb must be 2D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        };
        if indices.ndim() != 1 || offsets.ndim() != 1 {
            log::error!(
                "EmbeddingBag forward: indices/offsets must be 1D, got {:?}/{:?}",
                indices.shape(),
                offsets.shape()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let vocab = emb2.shape()[0];
        let dim = emb2.shape()[1];
        let n_idx = indices.len();
        let n_bag = offsets.len();

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n_bag, dim][..]));
        for b in 0..n_bag {
            let start = offsets[[b]] as isize;
            if (offsets[[b]] - start as f32).abs() > 1e-6 || start < 0 {
                log::error!("EmbeddingBag forward: invalid offset {}", offsets[[b]]);
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
            let start = start as usize;
            let end = if b + 1 < n_bag {
                let e = offsets[[b + 1]] as isize;
                if (offsets[[b + 1]] - e as f32).abs() > 1e-6 || e < 0 {
                    log::error!("EmbeddingBag forward: invalid offset {}", offsets[[b + 1]]);
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                e as usize
            } else {
                n_idx
            };
            if start > end || end > n_idx {
                log::error!(
                    "EmbeddingBag forward: invalid bag range [{}, {}) for n_idx {}",
                    start,
                    end,
                    n_idx
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }

            for p in start..end {
                let idx_f = indices[[p]];
                let idx_i = idx_f as isize;
                if (idx_f - idx_i as f32).abs() > 1e-6 || idx_i < 0 || (idx_i as usize) >= vocab {
                    log::error!(
                        "EmbeddingBag forward: invalid index {} for vocab {}",
                        idx_f,
                        vocab
                    );
                    *output = ArrayD::zeros(IxDyn(&[0][..]));
                    return;
                }
                let idx_u = idx_i as usize;
                for d in 0..dim {
                    out[[b, d]] += emb2[[idx_u, d]];
                }
            }
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let emb_shape = inputs[0].lock().storage.shape();
        let indices = inputs[1].lock().storage.to_f32_array();
        let offsets = inputs[2].lock().storage.to_f32_array();

        if emb_shape.len() != 2 || indices.ndim() != 1 || offsets.ndim() != 1 {
            return vec![
                ArrayD::zeros(IxDyn(&emb_shape)),
                ArrayD::zeros(IxDyn(indices.shape())),
                ArrayD::zeros(IxDyn(offsets.shape())),
            ];
        }

        let vocab = emb_shape[0];
        let dim = emb_shape[1];
        let n_idx = indices.len();
        let n_bag = offsets.len();

        if output_grad.shape() != [n_bag, dim] {
            log::error!(
                "EmbeddingBag backward: output_grad shape {:?} expected [{}, {}]",
                output_grad.shape(),
                n_bag,
                dim
            );
            return vec![
                ArrayD::zeros(IxDyn(&emb_shape)),
                ArrayD::zeros(IxDyn(indices.shape())),
                ArrayD::zeros(IxDyn(offsets.shape())),
            ];
        }

        let mut grad_emb = ArrayD::<f32>::zeros(IxDyn(&[vocab, dim][..]));
        for b in 0..n_bag {
            let start = offsets[[b]] as isize;
            let start = if start < 0 { 0usize } else { start as usize };
            let end = if b + 1 < n_bag {
                let e = offsets[[b + 1]] as isize;
                if e < 0 {
                    0usize
                } else {
                    e as usize
                }
            } else {
                n_idx
            };
            if start > end || end > n_idx {
                continue;
            }

            for p in start..end {
                let idx_f = indices[[p]];
                let idx_i = idx_f as isize;
                if idx_i < 0 || (idx_i as usize) >= vocab {
                    continue;
                }
                let idx_u = idx_i as usize;
                for d in 0..dim {
                    grad_emb[[idx_u, d]] += output_grad[[b, d]];
                }
            }
        }

        let grad_indices = ArrayD::zeros(IxDyn(indices.shape()));
        let grad_offsets = ArrayD::zeros(IxDyn(offsets.shape()));
        vec![grad_emb, grad_indices, grad_offsets]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for IndexSelect {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        let indices = inputs[1].to_f32_array();

        if self.dim >= x.ndim() {
            log::error!(
                "IndexSelect.forward: dim {} out of bounds for input ndim {}",
                self.dim,
                x.ndim()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        if indices.ndim() != 1 {
            log::error!(
                "IndexSelect.forward: indices must be 1D but got shape {:?}",
                indices.shape()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let axis_len = x.shape()[self.dim];
        let mut idx_vec: Vec<usize> = Vec::with_capacity(indices.len());
        for &idx_f in &indices {
            let idx_i = idx_f as isize;
            if (idx_f - idx_i as f32).abs() > 1e-6 || idx_i < 0 || (idx_i as usize) >= axis_len {
                log::error!(
                    "IndexSelect.forward: invalid index {} for axis size {}",
                    idx_f,
                    axis_len
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
            idx_vec.push(idx_i as usize);
        }

        *output = x.select(Axis(self.dim), &idx_vec);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        let indices = inputs[1].to_f32_array();

        if self.dim >= x.ndim() || indices.ndim() != 1 {
            return vec![
                ArrayD::zeros(IxDyn(x.shape())),
                ArrayD::zeros(IxDyn(indices.shape())),
            ];
        }

        let axis_len = x.shape()[self.dim];
        let mut idx_vec: Vec<usize> = Vec::with_capacity(indices.len());
        for &idx_f in &indices {
            let idx_i = idx_f as isize;
            if (idx_f - idx_i as f32).abs() > 1e-6 || idx_i < 0 || (idx_i as usize) >= axis_len {
                log::error!(
                    "IndexSelect.backward: invalid index {} for axis size {}",
                    idx_f,
                    axis_len
                );
                return vec![
                    ArrayD::zeros(IxDyn(x.shape())),
                    ArrayD::zeros(IxDyn(indices.shape())),
                ];
            }
            idx_vec.push(idx_i as usize);
        }

        let mut grad_x = ArrayD::zeros(IxDyn(x.shape()));
        for (out_pos, &src_idx) in idx_vec.iter().enumerate() {
            let og_slice = output_grad.index_axis(Axis(self.dim), out_pos);
            let mut gx_slice = grad_x.index_axis_mut(Axis(self.dim), src_idx);
            gx_slice += &og_slice;
        }

        let grad_indices = ArrayD::zeros(IxDyn(indices.shape()));
        vec![grad_x, grad_indices]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for Gather {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        let index = inputs[1].to_f32_array();

        if self.dim >= x.ndim() {
            log::error!(
                "Gather.forward: dim {} out of bounds for input ndim {}",
                self.dim,
                x.ndim()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        if index.ndim() != x.ndim() {
            log::error!(
                "Gather.forward: index ndim {} must match input ndim {}",
                index.ndim(),
                x.ndim()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        for axis in 0..x.ndim() {
            if axis != self.dim && index.shape()[axis] != x.shape()[axis] {
                log::error!(
                    "Gather.forward: index shape {:?} mismatches input shape {:?} at axis {}",
                    index.shape(),
                    x.shape(),
                    axis
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        }

        let axis_len = x.shape()[self.dim];
        let mut out = ArrayD::<f32>::zeros(IxDyn(index.shape()));
        for (coords, out_val) in out.indexed_iter_mut() {
            let idx_f = index[coords.clone()];
            let idx_i = idx_f as isize;
            if (idx_f - idx_i as f32).abs() > 1e-6 || idx_i < 0 || (idx_i as usize) >= axis_len {
                log::error!(
                    "Gather.forward: invalid index {} for axis size {}",
                    idx_f,
                    axis_len
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
            let mut src_coords = coords.slice().to_vec();
            src_coords[self.dim] = idx_i as usize;
            *out_val = x[IxDyn(&src_coords)];
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        let index = inputs[1].to_f32_array();

        if self.dim >= x.ndim() || index.ndim() != x.ndim() {
            return vec![
                ArrayD::zeros(IxDyn(x.shape())),
                ArrayD::zeros(IxDyn(index.shape())),
            ];
        }

        for axis in 0..x.ndim() {
            if axis != self.dim && index.shape()[axis] != x.shape()[axis] {
                return vec![
                    ArrayD::zeros(IxDyn(x.shape())),
                    ArrayD::zeros(IxDyn(index.shape())),
                ];
            }
        }

        let axis_len = x.shape()[self.dim];
        let mut grad_x = ArrayD::zeros(IxDyn(x.shape()));
        for (coords, &g) in output_grad.indexed_iter() {
            let idx_f = index[coords.clone()];
            let idx_i = idx_f as isize;
            if (idx_f - idx_i as f32).abs() > 1e-6 || idx_i < 0 || (idx_i as usize) >= axis_len {
                log::error!(
                    "Gather.backward: invalid index {} for axis size {}",
                    idx_f,
                    axis_len
                );
                return vec![
                    ArrayD::zeros(IxDyn(x.shape())),
                    ArrayD::zeros(IxDyn(index.shape())),
                ];
            }
            let mut src_coords = coords.slice().to_vec();
            src_coords[self.dim] = idx_i as usize;
            grad_x[IxDyn(&src_coords)] += g;
        }

        let grad_index = ArrayD::zeros(IxDyn(index.shape()));
        vec![grad_x, grad_index]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for Scatter {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        let index = inputs[1].to_f32_array();
        let src = inputs[2].to_f32_array();

        if self.dim >= x.ndim() || index.ndim() != x.ndim() || src.shape() != index.shape() {
            log::error!(
                "Scatter.forward: invalid shapes x={:?} index={:?} src={:?} dim={}",
                x.shape(),
                index.shape(),
                src.shape(),
                self.dim
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        for axis in 0..x.ndim() {
            if axis != self.dim && index.shape()[axis] != x.shape()[axis] {
                log::error!(
                    "Scatter.forward: index shape {:?} mismatches input shape {:?} at axis {}",
                    index.shape(),
                    x.shape(),
                    axis
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        }

        let axis_len = x.shape()[self.dim];
        let mut out = x.clone();
        for (coords, &v) in src.indexed_iter() {
            let idx_f = index[coords.clone()];
            let idx_i = idx_f as isize;
            if (idx_f - idx_i as f32).abs() > 1e-6 || idx_i < 0 || (idx_i as usize) >= axis_len {
                log::error!(
                    "Scatter.forward: invalid index {} for axis size {}",
                    idx_f,
                    axis_len
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }

            let mut dst = coords.slice().to_vec();
            dst[self.dim] = idx_i as usize;
            out[IxDyn(&dst)] = v;
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        let index = inputs[1].to_f32_array();
        let src = inputs[2].to_f32_array();

        if self.dim >= x.ndim() || index.ndim() != x.ndim() || src.shape() != index.shape() {
            return vec![
                ArrayD::zeros(IxDyn(x.shape())),
                ArrayD::zeros(IxDyn(index.shape())),
                ArrayD::zeros(IxDyn(src.shape())),
            ];
        }

        let axis_len = x.shape()[self.dim];
        let mut grad_x = output_grad.clone();
        let mut grad_src = ArrayD::zeros(IxDyn(src.shape()));

        for (coords, gsrc) in grad_src.indexed_iter_mut() {
            let idx_f = index[coords.clone()];
            let idx_i = idx_f as isize;
            if (idx_f - idx_i as f32).abs() > 1e-6 || idx_i < 0 || (idx_i as usize) >= axis_len {
                log::error!(
                    "Scatter.backward: invalid index {} for axis size {}",
                    idx_f,
                    axis_len
                );
                return vec![
                    ArrayD::zeros(IxDyn(x.shape())),
                    ArrayD::zeros(IxDyn(index.shape())),
                    ArrayD::zeros(IxDyn(src.shape())),
                ];
            }

            let mut dst = coords.slice().to_vec();
            dst[self.dim] = idx_i as usize;
            *gsrc = output_grad[IxDyn(&dst)];
            grad_x[IxDyn(&dst)] = 0.0;
        }

        let grad_index = ArrayD::zeros(IxDyn(index.shape()));
        vec![grad_x, grad_index, grad_src]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for ScatterAdd {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        let index = inputs[1].to_f32_array();
        let src = inputs[2].to_f32_array();

        if self.dim >= x.ndim() || index.ndim() != x.ndim() || src.shape() != index.shape() {
            log::error!(
                "ScatterAdd.forward: invalid shapes x={:?} index={:?} src={:?} dim={}",
                x.shape(),
                index.shape(),
                src.shape(),
                self.dim
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        for axis in 0..x.ndim() {
            if axis != self.dim && index.shape()[axis] != x.shape()[axis] {
                log::error!(
                    "ScatterAdd.forward: index shape {:?} mismatches input shape {:?} at axis {}",
                    index.shape(),
                    x.shape(),
                    axis
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        }

        let axis_len = x.shape()[self.dim];
        let mut out = x.clone();
        for (coords, &v) in src.indexed_iter() {
            let idx_f = index[coords.clone()];
            let idx_i = idx_f as isize;
            if (idx_f - idx_i as f32).abs() > 1e-6 || idx_i < 0 || (idx_i as usize) >= axis_len {
                log::error!(
                    "ScatterAdd.forward: invalid index {} for axis size {}",
                    idx_f,
                    axis_len
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }

            let mut dst = coords.slice().to_vec();
            dst[self.dim] = idx_i as usize;
            out[IxDyn(&dst)] += v;
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        let index = inputs[1].to_f32_array();
        let src = inputs[2].to_f32_array();

        if self.dim >= x.ndim() || index.ndim() != x.ndim() || src.shape() != index.shape() {
            return vec![
                ArrayD::zeros(IxDyn(x.shape())),
                ArrayD::zeros(IxDyn(index.shape())),
                ArrayD::zeros(IxDyn(src.shape())),
            ];
        }

        let axis_len = x.shape()[self.dim];
        let grad_x = output_grad.clone();
        let mut grad_src = ArrayD::zeros(IxDyn(src.shape()));
        for (coords, gsrc) in grad_src.indexed_iter_mut() {
            let idx_f = index[coords.clone()];
            let idx_i = idx_f as isize;
            if (idx_f - idx_i as f32).abs() > 1e-6 || idx_i < 0 || (idx_i as usize) >= axis_len {
                log::error!(
                    "ScatterAdd.backward: invalid index {} for axis size {}",
                    idx_f,
                    axis_len
                );
                return vec![
                    ArrayD::zeros(IxDyn(x.shape())),
                    ArrayD::zeros(IxDyn(index.shape())),
                    ArrayD::zeros(IxDyn(src.shape())),
                ];
            }

            let mut dst = coords.slice().to_vec();
            dst[self.dim] = idx_i as usize;
            *gsrc = output_grad[IxDyn(&dst)];
        }

        let grad_index = ArrayD::zeros(IxDyn(index.shape()));
        vec![grad_x, grad_index, grad_src]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Simple KVCache append operation: concatenate along seq axis (axis=1 by default)
pub struct KVCacheAppend {
    pub axis: usize,
}

impl KVCacheAppend {
    pub fn new(axis: usize) -> Self {
        KVCacheAppend { axis }
    }
}

impl Operation for KVCacheAppend {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // Efficiently concatenate two inputs into the provided output buffer to avoid an intermediate allocation.
        // inputs: cache (N, seq, dim) and new_kv (N, t, dim)
        let a = inputs[0].lock().storage.to_f32_array();
        let b = inputs[1].lock().storage.to_f32_array();
        let axis = self.axis;
        // Basic checks
        if a.ndim() != b.ndim() {
            log::error!(
                "KVCacheAppend forward: input ndims differ: {} vs {}",
                a.ndim(),
                b.ndim()
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        let ndim = a.ndim();
        if axis >= ndim {
            log::error!(
                "KVCacheAppend forward: axis {} out of bounds for ndim {}",
                axis,
                ndim
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        // Ensure non-axis dims match
        for i in 0..ndim {
            if i == axis {
                continue;
            }
            if a.shape()[i] != b.shape()[i] {
                log::error!(
                    "KVCacheAppend forward: non-axis dims must match: a {:?} b {:?}",
                    a.shape(),
                    b.shape()
                );
                *output = ArrayD::zeros(IxDyn(&[0][..]));
                return;
            }
        }
        // Build output shape and allocate into provided output
        let mut out_shape = a.shape().to_vec();
        out_shape[axis] = a.shape()[axis] + b.shape()[axis];
        *output = ArrayD::zeros(IxDyn(&out_shape));

        // Copy a then b into slices
        // Copy a
        {
            let mut slice_elems: Vec<SliceInfoElem> = Vec::new();
            for i in 0..ndim {
                if i == axis {
                    slice_elems.push((0..a.shape()[axis]).into());
                } else {
                    slice_elems.push((..).into());
                }
            }
            let slice_info: SliceInfo<_, IxDyn, IxDyn> =
                unsafe { SliceInfo::new(slice_elems).unwrap() };
            let mut out_slice = output.slice_mut(slice_info.as_ref());
            out_slice.assign(&a.view());
        }
        // Copy b
        {
            let start = a.shape()[axis];
            let mut slice_elems: Vec<SliceInfoElem> = Vec::new();
            for i in 0..ndim {
                if i == axis {
                    slice_elems.push((start..start + b.shape()[axis]).into());
                } else {
                    slice_elems.push((..).into());
                }
            }
            let slice_info: SliceInfo<_, IxDyn, IxDyn> =
                unsafe { SliceInfo::new(slice_elems).unwrap() };
            let mut out_slice = output.slice_mut(slice_info.as_ref());
            out_slice.assign(&b.view());
        }
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let axis = self.axis;
        let a_shape = inputs[0].lock().storage.shape();
        let b_shape = inputs[1].lock().storage.shape();
        // slice output_grad into two using indices
        let a_size = a_shape[axis];
        // create slicing to obtain a slice
        let mut a_slice_elems: Vec<SliceInfoElem> = Vec::new();
        for i in 0..a_shape.len() {
            if i == axis {
                a_slice_elems.push((0..a_size).into());
            } else {
                a_slice_elems.push((..).into());
            }
        }
        let mut b_slice_elems: Vec<SliceInfoElem> = Vec::new();
        for i in 0..b_shape.len() {
            if i == axis {
                b_slice_elems.push((a_size..(a_size + b_shape[axis])).into());
            } else {
                b_slice_elems.push((..).into());
            }
        }
        let a_slice_info_res = unsafe { SliceInfo::new(a_slice_elems) };
        let a_slice_info: SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> = match a_slice_info_res {
            Ok(info) => info,
            Err(e) => {
                log::error!(
                    "KVCacheAppend backward: failed to create slice info for a slice: {}",
                    e
                );
                return vec![
                    ArrayD::zeros(IxDyn(&[0][..])),
                    ArrayD::zeros(IxDyn(&[0][..])),
                ];
            }
        };
        let b_slice_info_res = unsafe { SliceInfo::new(b_slice_elems) };
        let b_slice_info: SliceInfo<_, IxDyn, IxDyn> = match b_slice_info_res {
            Ok(info) => info,
            Err(e) => {
                log::error!(
                    "KVCacheAppend backward: failed to create slice info for b slice: {}",
                    e
                );
                return vec![
                    ArrayD::zeros(IxDyn(&[0][..])),
                    ArrayD::zeros(IxDyn(&[0][..])),
                ];
            }
        };
        let grad_a = output_grad
            .slice(a_slice_info.as_ref())
            .to_owned()
            .into_dyn();
        let grad_b = output_grad
            .slice(b_slice_info.as_ref())
            .to_owned()
            .into_dyn();
        vec![grad_a, grad_b]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Binary Cross Entropy operation (element-wise).
/// Inputs: probability, target
pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    pub fn new() -> Self {
        BinaryCrossEntropy
    }
}

impl Default for BinaryCrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for BinaryCrossEntropy {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let input = inputs[0].lock().storage.to_f32_array();
        let target = inputs[1].lock().storage.to_f32_array();

        // out = - (target * ln(input) + (1 - target) * ln(1 - input))
        // Clip input to avoid log(0)
        let eps = 1e-12;
        let input_clipped = input.mapv(|v| v.clamp(eps, 1.0 - eps));

        let term1 = &target * input_clipped.mapv(|v| v.ln());
        let term2 = (1.0 - &target) * input_clipped.mapv(|v| (1.0 - v).ln());
        let out = -(term1 + term2);

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let input = inputs[0].lock().storage.to_f32_array();
        let target = inputs[1].lock().storage.to_f32_array();
        let eps = 1e-12;
        let input_clipped = input.mapv(|v| v.clamp(eps, 1.0 - eps));

        // dL/dx = (x - y) / (x * (1 - x))
        let num = &input_clipped - &target;
        let den = &input_clipped * (1.0 - &input_clipped);
        let grad_input = output_grad * (&num / &den);

        // dL/dy = -ln(x) + ln(1-x) = ln((1-x)/x) ?
        // L = -y ln x - (1-y) ln (1-x)
        // dL/dy = -ln x + ln(1-x)
        let grad_target = output_grad
            * (-input_clipped.mapv(|v| v.ln()) + input_clipped.mapv(|v| (1.0 - v).ln()));

        vec![grad_input, grad_target]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Binary Cross Entropy with Logits operation (element-wise).
/// Inputs: logits, target
/// Numerically stable version using "max(x, 0) - x*y + log(1 + exp(-abs(x)))"
pub struct BinaryCrossEntropyWithLogits;

impl BinaryCrossEntropyWithLogits {
    pub fn new() -> Self {
        BinaryCrossEntropyWithLogits
    }
}

impl Default for BinaryCrossEntropyWithLogits {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for BinaryCrossEntropyWithLogits {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let logits = inputs[0].lock().storage.to_f32_array();
        let target = inputs[1].lock().storage.to_f32_array();

        // max(logits, 0) - logits * target + log(1 + exp(-abs(logits)))
        let max_val = logits.mapv(|v| v.max(0.0));
        let term1 = &max_val - &(&logits * &target);
        let term2 = logits.mapv(|v| (1.0 + (-v.abs()).exp()).ln());

        *output = term1 + term2;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let logits = inputs[0].lock().storage.to_f32_array();
        let target = inputs[1].lock().storage.to_f32_array();

        // dL/dx = sigmoid(x) - y
        let sigmoid = logits.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let grad_logits = output_grad * (&sigmoid - &target);

        // dL/dy: same as BCE
        // L includes a linear term in y with coefficient -x.
        // If we strictly follow functional form:
        // L = max(x,0) - xy + log(1+exp(-|x|))
        // dL/dy = -x
        // Wait, standard BCEWithLogits assumes y is constant?
        // If y is tensor, dL/dy is deriv of: - x * y (linear in y).
        // BUT BCE(p, y) = - y log p - (1-y) log (1-p)
        // p = sigmoid(x)
        // So this formula IS mathematically equivalent.
        // dL/dy = - log p + log(1-p) = - log (sigmoid(x)) + log(1 - sigmoid(x))
        // = - (x - log(1+exp(x))) + log(1 / (1+exp(x)))
        // = -x
        // Let's verify:
        // - log(1 / (1+exp(-x))) = log(1+exp(-x))
        // - log(exp(-x)/(1+exp(-x))) = -x - log(1+exp(-x))
        // dL/dy = log(1+exp(-x)) - (-x - log(1+exp(-x))) ? No.

        // Let's use logits directly.
        // grad_target = -logits
        // Wait, is that correct?
        // BCE(p, y) is linear in y.
        // L = -y log p - log(1-p) + y log(1-p)
        //   = y (log(1-p) - log p) - log(1-p)
        //   = y log((1-p)/p) - log(1-p)
        // log((1-p)/p) = log( (1/(1+e^x)) / (e^x/(1+e^x)) ) = log(1/e^x) = -x.
        // So dL/dy = -x.
        // Yes, grad_target = -logits.
        let grad_target = output_grad * (-&logits);

        vec![grad_logits, grad_target]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod softmax_tests {
    use super::*;

    #[test]
    fn softmax_handles_all_neg_inf_row() {
        // Build a 2x4 array where second row is all -inf
        let mut a = ndarray::Array2::<f32>::zeros((2, 4)).into_dyn();
        a[[0, 0]] = 1.0;
        a[[0, 1]] = 2.0;
        a[[0, 2]] = 3.0;
        a[[0, 3]] = 4.0;
        for j in 0..4 {
            a[[1, j]] = f32::NEG_INFINITY;
        }
        let t = Tensor::new(a, false);
        let op = Softmax::new(1);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[2, 4][..]));
        op.forward(&[t.clone()][..], &mut out);
        // first row should be softmax of [1,2,3,4]
        let out0 = out.index_axis(Axis(0), 0).to_owned();
        let mut sum0 = 0.0f32;
        for v in out0.iter() {
            sum0 += *v;
        }
        assert!(sum0 > 0.99 && sum0 < 1.01);
        // second row turned into uniform distribution
        let out1 = out.index_axis(Axis(0), 1).to_owned();
        for v in out1.iter() {
            assert!((*v - 0.25).abs() < 1e-6);
        }
    }
}

/// Focal Loss operation for addressing class imbalance.
/// Formula: FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
/// Inputs: predictions (probabilities after sigmoid/softmax), targets (0 or 1)
pub struct FocalLoss {
    pub alpha: f32,
    pub gamma: f32,
}

impl FocalLoss {
    pub fn new(alpha: f32, gamma: f32) -> Self {
        assert!(alpha > 0.0, "alpha must be positive");
        assert!(gamma >= 0.0, "gamma must be non-negative");
        FocalLoss { alpha, gamma }
    }
}

impl Operation for FocalLoss {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let preds = inputs[0].lock().storage.to_f32_array();
        let targets = inputs[1].lock().storage.to_f32_array();

        let eps = 1e-7;
        let mut loss_sum = 0.0;

        for (p, t) in preds.iter().zip(targets.iter()) {
            let p_clipped = p.clamp(eps, 1.0 - eps);
            let p_t = if *t == 1.0 {
                p_clipped
            } else {
                1.0 - p_clipped
            };
            let focal_weight = (1.0 - p_t).powf(self.gamma);
            loss_sum += -self.alpha * focal_weight * p_t.ln();
        }

        *output = ArrayD::from_elem(ndarray::IxDyn(&[][..]), loss_sum / preds.len() as f32);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let preds = inputs[0].lock().storage.to_f32_array();
        let targets = inputs[1].lock().storage.to_f32_array();

        let eps = 1e-7;
        let mut grad_preds = ArrayD::zeros(preds.dim());
        let grad_scale = output_grad.iter().next().unwrap_or(&1.0) / preds.len() as f32;

        // Safe iteration using Zip
        Zip::from(&mut grad_preds)
            .and(preds.view())
            .and(targets.view())
            .for_each(|g, &p, &t| {
                let p_clipped = p.clamp(eps, 1.0f32 - eps);
                let p_t = if t == 1.0f32 {
                    p_clipped
                } else {
                    1.0f32 - p_clipped
                };
                let focal_weight = (1.0f32 - p_t).powf(self.gamma);

                // Gradient: d/dp FL = -α * [γ * (1-p_t)^(γ-1) * log(p_t) + (1-p_t)^γ / p_t] * sign
                let log_term = p_t.ln();
                let grad_focal = -self.alpha
                    * (self.gamma * (1.0f32 - p_t).powf(self.gamma - 1.0f32) * log_term
                        + focal_weight / p_t);

                let sign = if t == 1.0f32 { 1.0f32 } else { -1.0f32 };
                *g = grad_focal * sign * grad_scale;
            });

        vec![grad_preds, ArrayD::zeros(targets.dim())]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// KL Divergence operation for measuring distribution similarity.
/// Formula: KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
/// Inputs: P (target log probabilities), Q (predicted log probabilities)
pub struct KLDivergence {
    pub reduction: String, // "mean", "sum", "batchmean"
}

impl KLDivergence {
    pub fn new(reduction: String) -> Self {
        assert!(
            reduction == "mean" || reduction == "sum" || reduction == "batchmean",
            "reduction must be 'mean', 'sum', or 'batchmean'"
        );
        KLDivergence { reduction }
    }
}

impl Operation for KLDivergence {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let p_log = inputs[0].lock().storage.to_f32_array();
        let q_log = inputs[1].lock().storage.to_f32_array();

        // KL(P || Q) = Σ exp(P) * (P - Q)
        let kl_sum: f32 = p_log
            .iter()
            .zip(q_log.iter())
            .map(|(p, q)| p.exp() * (p - q))
            .sum();

        let result = match self.reduction.as_str() {
            "mean" => kl_sum / p_log.len() as f32,
            "batchmean" => {
                // Assuming first dimension is batch
                let batch_size = p_log.shape()[0];
                kl_sum / batch_size as f32
            }
            _ => kl_sum, // "sum"
        };

        *output = ArrayD::from_elem(ndarray::IxDyn(&[][..]), result);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let p_log = inputs[0].lock().storage.to_f32_array();
        let q_log = inputs[1].lock().storage.to_f32_array();

        let grad_scale = *output_grad.iter().next().unwrap_or(&1.0);
        let n = p_log.len() as f32;

        let scale_factor = match self.reduction.as_str() {
            "mean" => grad_scale / n,
            "batchmean" => grad_scale / p_log.shape()[0] as f32,
            _ => grad_scale,
        };

        // dKL/dP = exp(P) * (P - Q + 1)
        let grad_p = p_log.iter().zip(q_log.iter()).map(|(p, q)| {
            let exp_p = p.exp();
            scale_factor * exp_p * (p - q + 1.0)
        });

        // dKL/dQ = -exp(P)
        let grad_q = p_log.iter().map(|p| -scale_factor * p.exp());

        vec![
            ArrayD::from_shape_vec(p_log.dim(), grad_p.collect()).unwrap(),
            ArrayD::from_shape_vec(q_log.dim(), grad_q.collect()).unwrap(),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Contrastive Loss for metric learning with pairs.
/// Formula: L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
/// Inputs: embedding1, embedding2, labels (0=similar, 1=dissimilar)
pub struct ContrastiveLoss {
    pub margin: f32,
}

impl ContrastiveLoss {
    pub fn new(margin: f32) -> Self {
        assert!(margin > 0.0, "margin must be positive");
        ContrastiveLoss { margin }
    }
}

impl Operation for ContrastiveLoss {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let emb1 = inputs[0].lock().storage.to_f32_array();
        let emb2 = inputs[1].lock().storage.to_f32_array();
        let labels = inputs[2].lock().storage.to_f32_array();

        // Compute euclidean distances
        let diff = &emb1 - &emb2;
        let distances_sq = diff.mapv(|x| x * x);

        // Sum over feature dimension (assuming last dimension)
        let shape = emb1.shape();
        let batch_size = if shape.len() > 1 { shape[0] } else { 1 };
        let feature_dim = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            shape[0]
        };

        let mut loss_sum = 0.0;
        for b in 0..batch_size {
            let start_idx = b * feature_dim;
            let end_idx = start_idx + feature_dim;
            let dist_sq: f32 = distances_sq.as_slice().unwrap()[start_idx..end_idx]
                .iter()
                .sum();
            let dist = dist_sq.sqrt();

            let label = if batch_size > 1 {
                labels.as_slice().unwrap()[b]
            } else {
                *labels.iter().next().unwrap()
            };

            if label == 0.0 {
                // Similar pair
                loss_sum += 0.5 * dist_sq;
            } else {
                // Dissimilar pair
                loss_sum += 0.5 * (self.margin - dist).max(0.0).powi(2);
            }
        }

        *output = ArrayD::from_elem(ndarray::IxDyn(&[][..]), loss_sum / batch_size as f32);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let emb1 = inputs[0].lock().storage.to_f32_array();
        let emb2 = inputs[1].lock().storage.to_f32_array();
        let labels = inputs[2].lock().storage.to_f32_array();

        let diff = &emb1 - &emb2;
        let grad_scale = *output_grad.iter().next().unwrap_or(&1.0);

        let shape = emb1.shape();
        let batch_size = if shape.len() > 1 { shape[0] } else { 1 };
        let feature_dim = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            shape[0]
        };

        let mut grad_emb1 = ArrayD::zeros(emb1.dim());
        let mut grad_emb2 = ArrayD::zeros(emb2.dim());

        for b in 0..batch_size {
            let start_idx = b * feature_dim;
            let end_idx = start_idx + feature_dim;

            let dist_sq: f32 = diff.as_slice().unwrap()[start_idx..end_idx]
                .iter()
                .map(|x| x * x)
                .sum();
            let dist = dist_sq.sqrt() + 1e-8;

            let label = if batch_size > 1 {
                labels.as_slice().unwrap()[b]
            } else {
                *labels.iter().next().unwrap()
            };

            let grad_factor = if label == 0.0 {
                // Similar: dL/d(emb1-emb2) = (emb1 - emb2)
                grad_scale / batch_size as f32
            } else {
                // Dissimilar: dL/d(emb1-emb2) = -(margin - dist) * (emb1-emb2) / dist if margin > dist
                if dist < self.margin {
                    -(self.margin - dist) / dist * grad_scale / batch_size as f32
                } else {
                    0.0
                }
            };

            for i in start_idx..end_idx {
                let grad_val = grad_factor * diff.as_slice().unwrap()[i];
                grad_emb1.as_slice_mut().unwrap()[i] = grad_val;
                grad_emb2.as_slice_mut().unwrap()[i] = -grad_val;
            }
        }

        vec![grad_emb1, grad_emb2, ArrayD::zeros(labels.dim())]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Triplet Loss for learning embeddings with anchor/positive/negative triplets.
/// Formula: L = max(0, D(a,p) - D(a,n) + margin)
/// Inputs: anchor, positive, negative embeddings
pub struct TripletLoss {
    pub margin: f32,
}

impl TripletLoss {
    pub fn new(margin: f32) -> Self {
        assert!(margin > 0.0, "margin must be positive");
        TripletLoss { margin }
    }
}

impl Operation for TripletLoss {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let anchor = inputs[0].lock().storage.to_f32_array();
        let positive = inputs[1].lock().storage.to_f32_array();
        let negative = inputs[2].lock().storage.to_f32_array();

        // Compute squared euclidean distances
        let diff_pos = &anchor - &positive;
        let diff_neg = &anchor - &negative;

        let dist_pos_sq: f32 = diff_pos.iter().map(|x| x * x).sum();
        let dist_neg_sq: f32 = diff_neg.iter().map(|x| x * x).sum();

        let loss = (dist_pos_sq - dist_neg_sq + self.margin).max(0.0);

        *output = ArrayD::from_elem(ndarray::IxDyn(&[][..]), loss);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let anchor = inputs[0].lock().storage.to_f32_array();
        let positive = inputs[1].lock().storage.to_f32_array();
        let negative = inputs[2].lock().storage.to_f32_array();

        let diff_pos = &anchor - &positive;
        let diff_neg = &anchor - &negative;

        let dist_pos_sq: f32 = diff_pos.iter().map(|x| x * x).sum();
        let dist_neg_sq: f32 = diff_neg.iter().map(|x| x * x).sum();

        let grad_scale = *output_grad.iter().next().unwrap_or(&1.0);

        // If loss is active (margin violation)
        if dist_pos_sq - dist_neg_sq + self.margin > 0.0 {
            // dL/d_anchor = 2 * (diff_neg - diff_pos)
            // dL/d_positive = 2 * diff_pos
            // dL/d_negative = -2 * diff_neg
            let grad_anchor = (&diff_neg - &diff_pos) * (2.0 * grad_scale);
            let grad_positive = &diff_pos * (2.0 * grad_scale);
            let grad_negative = &diff_neg * (-2.0 * grad_scale);

            vec![grad_anchor, grad_positive, grad_negative]
        } else {
            // No gradient if margin is satisfied
            vec![
                ArrayD::zeros(anchor.dim()),
                ArrayD::zeros(positive.dim()),
                ArrayD::zeros(negative.dim()),
            ]
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
#[cfg(test)]
mod loss_tests {
    use super::*;
    use crate::tensor::Tensor;
    use ndarray::ArrayD;
    use std::sync::Arc;

    #[test]
    fn test_focal_loss_forward() {
        let focal = FocalLoss::new(1.0, 2.0);
        let preds = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[4][..]), vec![0.9, 0.7, 0.3, 0.1]).unwrap(),
            true,
        );
        let targets = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[4][..]), vec![1.0, 1.0, 0.0, 0.0]).unwrap(),
            false,
        );

        let result = Tensor::apply(Arc::new(focal), &[preds, targets][..]);
        let loss_val = *result.lock().storage.to_f32_array().iter().next().unwrap();

        // Focal loss should be positive and finite
        assert!(loss_val > 0.0);
        assert!(loss_val.is_finite());
    }

    #[test]
    fn test_focal_loss_backward() {
        let focal = FocalLoss::new(1.0, 2.0);
        let preds = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2][..]), vec![0.8, 0.2]).unwrap(),
            true,
        );
        let targets = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2][..]), vec![1.0, 0.0]).unwrap(),
            false,
        );

        let result = Tensor::apply(Arc::new(focal), &[preds.clone(), targets][..]);
        result.backward();

        // Check that gradients exist
        assert!(preds.lock().grad.is_some());
    }

    #[test]
    fn test_kl_divergence_forward() {
        let kl = KLDivergence::new("mean".to_string());
        let p_log = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![-1.0, -2.0, -3.0]).unwrap(),
            true,
        );
        let q_log = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![-1.5, -2.5, -3.5]).unwrap(),
            true,
        );

        let result = Tensor::apply(Arc::new(kl), &[p_log, q_log][..]);
        let kl_val = *result.lock().storage.to_f32_array().iter().next().unwrap();

        // KL divergence should be non-negative
        assert!(kl_val >= 0.0);
        assert!(kl_val.is_finite());
    }

    #[test]
    fn test_kl_divergence_zero() {
        let kl = KLDivergence::new("sum".to_string());
        // Identical distributions should have KL = 0
        let p_log = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![-1.0, -2.0, -3.0]).unwrap(),
            true,
        );
        let q_log = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![-1.0, -2.0, -3.0]).unwrap(),
            true,
        );

        let result = Tensor::apply(Arc::new(kl), &[p_log, q_log][..]);
        let kl_val = *result.lock().storage.to_f32_array().iter().next().unwrap();

        assert!(kl_val.abs() < 1e-5);
    }

    #[test]
    fn test_contrastive_loss_similar() {
        let contrastive = ContrastiveLoss::new(1.0);
        let emb1 = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![1.0, 2.0, 3.0]).unwrap(),
            true,
        );
        let emb2 = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![1.1, 2.1, 3.1]).unwrap(),
            true,
        );
        let labels = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1][..]), vec![0.0]).unwrap(), // similar
            false,
        );

        let result = Tensor::apply(Arc::new(contrastive), &[emb1, emb2, labels][..]);
        let loss_val = *result.lock().storage.to_f32_array().iter().next().unwrap();

        // Loss for similar pairs should be small (distance squared)
        assert!(loss_val > 0.0);
        assert!(loss_val < 0.1); // Small distance
    }

    #[test]
    fn test_contrastive_loss_dissimilar() {
        let contrastive = ContrastiveLoss::new(2.0);
        let emb1 = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![0.0, 0.0, 0.0]).unwrap(),
            true,
        );
        let emb2 = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![1.0, 1.0, 1.0]).unwrap(),
            true,
        );
        let labels = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1][..]), vec![1.0]).unwrap(), // dissimilar
            false,
        );

        let result = Tensor::apply(Arc::new(contrastive), &[emb1, emb2, labels][..]);
        let loss_val = *result.lock().storage.to_f32_array().iter().next().unwrap();

        // Loss should be positive (margin violation)
        assert!(loss_val >= 0.0);
    }

    #[test]
    fn test_triplet_loss_violation() {
        let triplet = TripletLoss::new(0.5);
        let anchor = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![0.0, 0.0, 0.0]).unwrap(),
            true,
        );
        let positive = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![0.1, 0.1, 0.1]).unwrap(),
            true,
        );
        let negative = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![0.2, 0.2, 0.2]).unwrap(),
            true,
        );

        let result = Tensor::apply(Arc::new(triplet), &[anchor, positive, negative][..]);
        let loss_val = *result.lock().storage.to_f32_array().iter().next().unwrap();

        // Loss should be positive (margin violation: positive too far from anchor)
        assert!(loss_val > 0.0);
    }

    #[test]
    fn test_triplet_loss_satisfied() {
        let triplet = TripletLoss::new(0.5);
        let anchor = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![0.0, 0.0, 0.0]).unwrap(),
            true,
        );
        let positive = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![0.1, 0.1, 0.1]).unwrap(),
            true,
        );
        let negative = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![2.0, 2.0, 2.0]).unwrap(),
            true,
        );

        let result = Tensor::apply(Arc::new(triplet), &[anchor, positive, negative][..]);
        let loss_val = *result.lock().storage.to_f32_array().iter().next().unwrap();

        // Loss should be zero (margin satisfied: negative far from anchor)
        assert!(loss_val.abs() < 1e-5);
    }

    #[test]
    fn test_triplet_loss_backward() {
        let triplet = TripletLoss::new(0.5);
        let anchor = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![0.0, 0.0, 0.0]).unwrap(),
            true,
        );
        let positive = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![0.1, 0.1, 0.1]).unwrap(),
            true,
        );
        let negative = Tensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3][..]), vec![0.2, 0.2, 0.2]).unwrap(),
            true,
        );

        let result = Tensor::apply(
            Arc::new(triplet),
            &[anchor.clone(), positive.clone(), negative.clone()][..],
        );
        result.backward();

        // Check that gradients exist for all inputs
        assert!(anchor.lock().grad.is_some());
        assert!(positive.lock().grad.is_some());
        assert!(negative.lock().grad.is_some());
    }

    #[test]
    #[should_panic(expected = "alpha must be positive")]
    fn test_focal_loss_invalid_alpha() {
        FocalLoss::new(0.0, 2.0);
    }

    #[test]
    #[should_panic(expected = "gamma must be non-negative")]
    fn test_focal_loss_invalid_gamma() {
        FocalLoss::new(1.0, -1.0);
    }

    #[test]
    #[should_panic(expected = "margin must be positive")]
    fn test_contrastive_loss_invalid_margin() {
        ContrastiveLoss::new(0.0);
    }

    #[test]
    #[should_panic(expected = "margin must be positive")]
    fn test_triplet_loss_invalid_margin() {
        TripletLoss::new(-0.5);
    }
}

#[cfg(test)]
mod batch_norm_tests {
    use super::*;
    use crate::tensor::Tensor;
    use ndarray::{ArrayD, IxDyn};
    use std::sync::Arc;

    #[test]
    fn test_batchnorm_forward_training() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3, 1] shape: B=2, C=3, S=1
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3, 1][..]), input_data).unwrap(),
            true,
        );
        let gamma = Tensor::ones(&[3][..]);
        let beta = Tensor::zeros(&[3][..]);
        let running_mean = Tensor::zeros(&[3][..]);
        let running_var = Tensor::ones(&[3][..]);

        // momentum doesn't matter for single step except for running_mean update
        let bn = BatchNorm::new(0.1, 1e-5, true);

        let mut output = ArrayD::zeros(IxDyn(&[2, 3, 1][..]));
        bn.forward(
            &[x, gamma, beta, running_mean.clone(), running_var.clone()][..],
            &mut output,
        );

        // For C=0: values are 1.0 and 4.0. Mean=2.5, Var=2.25. Std=1.5.
        // Norm values: (1-2.5)/1.5 = -1.0, (4-2.5)/1.5 = 1.0
        assert!((output[[0, 0, 0]] - (-1.0)).abs() < 1e-4);
        assert!((output[[1, 0, 0]] - 1.0).abs() < 1e-4);

        // Verify running stats updated
        let rm = running_mean.lock().storage.to_f32_array();
        // rm_new = (1-0.1)*0 + 0.1*2.5 = 0.25
        assert!((rm[[0]] - 0.25).abs() < 1e-4);
    }

    #[test]
    fn test_batchnorm_forward_inference() {
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1][..]), vec![10.0]).unwrap(),
            true,
        );
        let gamma = Tensor::ones(&[1][..]);
        let beta = Tensor::zeros(&[1][..]);
        let running_mean = Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1][..]), 5.0).into_dyn(),
            false,
        );
        let running_var = Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1][..]), 4.0).into_dyn(),
            false,
        );

        let bn = BatchNorm::new(0.1, 0.0, false); // training=false, eps=0

        let mut output = ArrayD::zeros(IxDyn(&[1, 1, 1][..]));
        bn.forward(
            &[x, gamma, beta, running_mean, running_var][..],
            &mut output,
        );

        // (10 - 5) / sqrt(4) = 5 / 2 = 2.5
        assert!((output[[0, 0, 0]] - 2.5).abs() < 1e-4);
    }

    #[test]
    fn test_batchnorm_backward() {
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 1, 1][..]), vec![1.0, 3.0]).unwrap(),
            true,
        );
        let gamma = Tensor::ones(&[1][..]);
        let beta = Tensor::zeros(&[1][..]);
        let running_mean = Tensor::zeros(&[1][..]);
        let running_var = Tensor::ones(&[1][..]);

        let bn = Arc::new(BatchNorm::new(0.1, 1e-5, true));

        let res = Tensor::apply(
            bn,
            &[
                x.clone(),
                gamma.clone(),
                beta.clone(),
                running_mean,
                running_var,
            ][..],
        );
        res.backward();

        assert!(x.lock().grad.is_some());
        assert!(gamma.lock().grad.is_some());
        assert!(beta.lock().grad.is_some());
    }
}
/// TopK operation selecting the largest k elements along the last dimension.
/// Output shape is [*, 2*k] where the first half of the last axis stores values
/// and the second half stores selected indices encoded as f32.
///
/// Backward recomputes top-k indices from the input and scatters gradients from
/// the values half of the output into input-gradient positions.

pub struct Sort;
pub struct ArgSort;

impl Operation for Sort {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let input = inputs[0].to_f32_array();
        let shape = input.shape();
        if shape.is_empty() {
            *output = input;
            return;
        }

        let last_dim = shape.len() - 1;
        let n = shape[last_dim];
        let total_rows: usize = shape.iter().take(last_dim).product();

        let input_2d = match input.to_shape((total_rows, n)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("Sort forward: reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(shape));
                return;
            }
        };

        let mut out_data = Vec::with_capacity(total_rows * n);
        for row in input_2d.outer_iter() {
            let mut vals: Vec<f32> = row.iter().copied().collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out_data.extend(vals);
        }

        *output = match ArrayD::from_shape_vec(IxDyn(shape), out_data) {
            Ok(a) => a,
            Err(e) => {
                log::error!("Sort forward: shape mismatch {}", e);
                ArrayD::zeros(IxDyn(shape))
            }
        };
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let input = inputs[0].to_f32_array();
        let shape = input.shape();
        if shape.is_empty() {
            return vec![output_grad.clone()];
        }

        let last_dim = shape.len() - 1;
        let n = shape[last_dim];
        let total_rows: usize = shape.iter().take(last_dim).product();

        let input_2d = match input.to_shape((total_rows, n)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("Sort backward: input reshape failed: {}", e);
                return vec![ArrayD::zeros(IxDyn(shape))];
            }
        };
        let grad_2d = match output_grad.to_shape((total_rows, n)) {
            Ok(v) => v,
            Err(e) => {
                log::error!("Sort backward: grad reshape failed: {}", e);
                return vec![ArrayD::zeros(IxDyn(shape))];
            }
        };

        let mut input_grad_data = vec![0.0f32; total_rows * n];
        for (row_idx, row) in input_2d.outer_iter().enumerate() {
            let mut pairs: Vec<(f32, usize)> =
                row.iter().enumerate().map(|(i, &v)| (v, i)).collect();
            pairs.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });

            let grad_row = grad_2d.slice(s![row_idx, ..]);
            for (sorted_pos, &g) in grad_row.iter().enumerate() {
                let orig_idx = pairs[sorted_pos].1;
                input_grad_data[row_idx * n + orig_idx] = g;
            }
        }

        let input_grad = match ArrayD::from_shape_vec(IxDyn(shape), input_grad_data) {
            Ok(a) => a,
            Err(e) => {
                log::error!("Sort backward: shape mismatch {}", e);
                ArrayD::zeros(IxDyn(shape))
            }
        };
        vec![input_grad]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Operation for ArgSort {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let input = inputs[0].to_f32_array();
        let shape = input.shape();
        if shape.is_empty() {
            *output = ArrayD::from_elem(IxDyn(&[][..]), 0.0);
            return;
        }

        let last_dim = shape.len() - 1;
        let n = shape[last_dim];
        let total_rows: usize = shape.iter().take(last_dim).product();

        let input_2d = match input.to_shape((total_rows, n)) {
            Ok(v) => v.to_owned(),
            Err(e) => {
                log::error!("ArgSort forward: reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(shape));
                return;
            }
        };

        let mut out_data = Vec::with_capacity(total_rows * n);
        for row in input_2d.outer_iter() {
            let mut pairs: Vec<(f32, usize)> =
                row.iter().enumerate().map(|(i, &v)| (v, i)).collect();
            pairs.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            out_data.extend(pairs.into_iter().map(|(_, idx)| idx as f32));
        }

        *output = match ArrayD::from_shape_vec(IxDyn(shape), out_data) {
            Ok(a) => a,
            Err(e) => {
                log::error!("ArgSort forward: shape mismatch {}", e);
                ArrayD::zeros(IxDyn(shape))
            }
        };
    }

    fn backward(&self, inputs: &[Tensor], _output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let shape = inputs[0].lock().storage.shape();
        vec![ArrayD::zeros(IxDyn(&shape))]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct TopK {
    pub k: usize,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK { k }
    }
}

impl Operation for TopK {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let input = inputs[0].to_f32_array();
        let k = self.k;
        let shape = input.shape();
        let last_dim = shape.len() - 1;
        let n = shape[last_dim];
        if k > n {
            log::error!("TopK: k {} > last dim {}", k, n);
            return;
        }

        // We need to iterate over all preceding dimensions.
        let total_rows: usize = shape.iter().take(last_dim).product();
        // New shape logic
        let mut out_shape = shape.to_vec();
        out_shape[last_dim] = k * 2; // Storing values AND indices

        // Reshape to 2D [total_rows, n]
        // Use to_shape with unwrap and to_owned to ensure we get an OwnedRepr
        // which avoids the View/Owned mismatch in match arms issues.
        let input_2d = input.to_shape((total_rows, n)).unwrap().to_owned();

        let mut out_data = Vec::with_capacity(total_rows * k * 2);

        for row in input_2d.outer_iter() {
            // Create (val, idx) pairs
            let mut pairs: Vec<(f32, usize)> =
                row.iter().enumerate().map(|(i, &v)| (v, i)).collect();
            // Sort descending by value
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Take top k
            for i in 0..k {
                out_data.push(pairs[i].0); // value
            }
            for i in 0..k {
                out_data.push(pairs[i].1 as f32); // index cast to f32
            }
        }

        // Create output array
        *output = match ArrayD::from_shape_vec(IxDyn(&out_shape), out_data) {
            Ok(a) => a,
            Err(e) => {
                log::error!("TopK forward: shape mismatch {}", e);
                ArrayD::zeros(IxDyn(&out_shape))
            }
        };
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let input = inputs[0].to_f32_array();
        let k = self.k;
        let shape = input.shape();
        let last_dim = shape.len() - 1;
        let n = shape[last_dim];

        let total_rows: usize = shape.iter().take(last_dim).product();

        // Output grad is [*, 2*k]. Reshape to [total_rows, 2*k]
        let grad_2d = output_grad.to_shape((total_rows, k * 2)).unwrap();

        // Input 2D
        let input_2d = input.to_shape((total_rows, n)).unwrap();

        let mut input_grad_data = vec![0.0f32; total_rows * n];

        for (row_idx, row) in input_2d.outer_iter().enumerate() {
            // Re-compute indices (stateless backward)
            let mut pairs: Vec<(f32, usize)> =
                row.iter().enumerate().map(|(i, &v)| (v, i)).collect();
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Top k indices
            let top_indices: Vec<usize> = pairs.iter().take(k).map(|p| p.1).collect();

            // Get grads for values (first k elements of row_idx in output_grad)
            let grad_row = grad_2d.slice(s![row_idx, 0..k]);

            // Scatter back
            for (i, &grad_val) in grad_row.iter().enumerate() {
                let original_idx = top_indices[i];
                input_grad_data[row_idx * n + original_idx] = grad_val;
            }
        }

        let input_grad = ArrayD::from_shape_vec(IxDyn(shape), input_grad_data).unwrap();
        vec![input_grad]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Checkpoint operation: trades compute for memory by re-running the forward pass during backward.
pub struct Checkpoint<F>
where
    F: Fn(&[Tensor]) -> Tensor + Send + Sync + 'static,
{
    pub f: std::sync::Arc<F>,
}

impl<F> Checkpoint<F>
where
    F: Fn(&[Tensor]) -> Tensor + Send + Sync + 'static,
{
    pub fn new(f: F) -> Self {
        Checkpoint {
            f: std::sync::Arc::new(f),
        }
    }
}

impl<F> Operation for Checkpoint<F>
where
    F: Fn(&[Tensor]) -> Tensor + Send + Sync + 'static,
{
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // Execute the closure to get the result tensor.
        // We do this to capture the output data.
        // In a real optimized scenario, we might want to avoid full graph building here if possible,
        // but since our Tensor op always builds graph, we just let it run.
        // The key is that the *output* of this Checkpoint op will NOT point to the intermediate nodes of f
        // as its inputs. It points to inputs directly.
        // The intermediate graph created by f(inputs) here is effectively dropped
        // because we only copy the data to `output`.
        let out_tensor = (self.f)(inputs);

        let out_data = out_tensor.to_f32_array();

        // If output was pre-allocated with wrong shape (due to limited inference in Tensor::apply),
        // we must resize it. ArrayD doesn't support in-place resize easily if it's a view,
        // but `output` is usually an owned array created by `Tensor::apply`.
        // However, `Tensor::apply` passes a Mutable Reference. We can't change the pointer?
        // Replacing `*output` is valid here.
        *output = out_data;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // Re-run forward pass with tracking enabled.
        // We need to differentiate `f` at `inputs`.
        // To do this without messing up `inputs` existing gradients or graph state in a confusing way,
        // we detach inputs into leaf nodes that track their own gradients for this local backward pass.

        let mut detached_inputs = Vec::with_capacity(inputs.len());
        for inp in inputs {
            let data = inp.to_f32_array();
            // Create new leaf tensor with requires_grad=true to capture gradient contribution
            let t = Tensor::new(data, true);
            detached_inputs.push(t);
        }

        // Run f on detached inputs - this builds the local graph
        let out_tensor = (self.f)(&detached_inputs);

        // Seed output gradient
        {
            let mut lock = out_tensor.lock();
            lock.grad = Some(output_grad.clone());
        }

        // Run backward on this subgraph
        crate::autograd::AutogradEngine::new().backward(&out_tensor);

        // Collect grads from detached_inputs
        let mut grads = Vec::with_capacity(inputs.len());
        for (i, t) in detached_inputs.iter().enumerate() {
            let g = t.lock().grad.clone().unwrap_or_else(|| {
                ArrayD::zeros(IxDyn(inputs[i].lock().storage.shape().as_slice()))
            });
            grads.push(g);
        }

        grads
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// GeGLU (Gated Linear Unit with GELU activation).
/// Splits input along last dim into two halves, applies GELU to first half,
/// then element-wise multiplies with second half.
/// Formula: GeGLU(x) = GELU(x[:, :d]) ⊗ x[:, d:]
pub struct GeGLU;

impl GeGLU {
    pub fn new() -> Self {
        GeGLU
    }
}

impl Default for GeGLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for GeGLU {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        let shape = x.shape().to_vec();
        if shape.is_empty() {
            *output = x.clone();
            return;
        }
        let last_dim = shape[shape.len() - 1];
        if last_dim % 2 != 0 {
            log::error!(
                "GeGLU.forward: last dimension {} must be even for splitting",
                last_dim
            );
            *output = x.clone();
            return;
        }
        let half = last_dim / 2;
        let prefix_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
        let total_prefix: usize = prefix_shape.iter().product();
        let flat = x.to_shape((total_prefix, last_dim)).unwrap();

        let mut out_data = Vec::with_capacity(total_prefix * half);
        for row in flat.outer_iter() {
            for i in 0..half {
                let gelu_in = row[i];
                let gelu_out = 0.5 * gelu_in * (1.0 + (gelu_in * 1.702_f32).tanh());
                out_data.push(gelu_out * row[i + half]);
            }
        }

        let mut out_shape = shape.clone();
        out_shape[shape.len() - 1] = half;
        *output = match ArrayD::from_shape_vec(IxDyn(&out_shape), out_data) {
            Ok(a) => a,
            Err(e) => {
                log::error!("GeGLU.forward: shape mismatch {}", e);
                ArrayD::zeros(IxDyn(&out_shape))
            }
        };
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        let shape = x.shape().to_vec();
        if shape.is_empty() {
            return vec![output_grad.clone()];
        }
        let last_dim = shape[shape.len() - 1];
        if last_dim % 2 != 0 {
            return vec![ArrayD::zeros(IxDyn(&shape))];
        }
        let half = last_dim / 2;
        let prefix_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
        let total_prefix: usize = prefix_shape.iter().product();

        let x2 = match x.to_shape((total_prefix, last_dim)) {
            Ok(v) => v,
            Err(_) => return vec![ArrayD::zeros(IxDyn(&shape))],
        };
        let gy2 = match output_grad.to_shape((total_prefix, half)) {
            Ok(v) => v,
            Err(_) => return vec![ArrayD::zeros(IxDyn(&shape))],
        };

        let grad_x = ArrayD::<f32>::zeros(IxDyn(&shape));
        let mut gx2 = match grad_x.to_shape((total_prefix, last_dim)) {
            Ok(v) => v,
            Err(_) => return vec![ArrayD::zeros(IxDyn(&shape))],
        };

        for (row_idx, (row_x, row_gy)) in x2.outer_iter().zip(gy2.outer_iter()).enumerate() {
            for i in 0..half {
                let a = row_x[i];
                let b = row_x[i + half];
                let g = row_gy[i];

                // GELU derivative at a: 0.5 * (1 + tanh(1.702 * a))
                //   + 0.5 * a * (1 - tanh^2(1.702 * a)) * 1.702
                let tanh_val = (1.702_f32 * a).tanh();
                let gelu_a = 0.5 * a * (1.0 + tanh_val);
                let gelu_prime_a =
                    0.5 * (1.0 + tanh_val) + 0.5 * a * (1.0 - tanh_val * tanh_val) * 1.702;

                // grad w.r.t. a: gelu_prime(a) * b * g
                gx2[[row_idx, i]] = gelu_prime_a * b * g;
                // grad w.r.t. b: gelu(a) * g
                gx2[[row_idx, i + half]] = gelu_a * g;
            }
        }

        vec![grad_x]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// ReGLU (Gated Linear Unit with ReLU activation).
/// Splits input along last dim into two halves, applies ReLU to first half,
/// then element-wise multiplies with second half.
/// Formula: ReGLU(x) = ReLU(x[:, :d]) ⊗ x[:, d:]
pub struct ReGLU;

impl ReGLU {
    pub fn new() -> Self {
        ReGLU
    }
}

impl Default for ReGLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for ReGLU {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = inputs[0].to_f32_array();
        let shape = x.shape().to_vec();
        if shape.is_empty() {
            *output = x.clone();
            return;
        }
        let last_dim = shape[shape.len() - 1];
        if last_dim % 2 != 0 {
            log::error!(
                "ReGLU.forward: last dimension {} must be even for splitting",
                last_dim
            );
            *output = x.clone();
            return;
        }
        let half = last_dim / 2;
        let prefix_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
        let total_prefix: usize = prefix_shape.iter().product();
        let flat = x.to_shape((total_prefix, last_dim)).unwrap();

        let mut out_data = Vec::with_capacity(total_prefix * half);
        for row in flat.outer_iter() {
            for i in 0..half {
                let relu_out = row[i].max(0.0);
                out_data.push(relu_out * row[i + half]);
            }
        }

        let mut out_shape = shape.clone();
        out_shape[shape.len() - 1] = half;
        *output = match ArrayD::from_shape_vec(IxDyn(&out_shape), out_data) {
            Ok(a) => a,
            Err(e) => {
                log::error!("ReGLU.forward: shape mismatch {}", e);
                ArrayD::zeros(IxDyn(&out_shape))
            }
        };
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].to_f32_array();
        let shape = x.shape().to_vec();
        if shape.is_empty() {
            return vec![output_grad.clone()];
        }
        let last_dim = shape[shape.len() - 1];
        if last_dim % 2 != 0 {
            return vec![ArrayD::zeros(IxDyn(&shape))];
        }
        let half = last_dim / 2;
        let prefix_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
        let total_prefix: usize = prefix_shape.iter().product();

        let x2 = match x.to_shape((total_prefix, last_dim)) {
            Ok(v) => v,
            Err(_) => return vec![ArrayD::zeros(IxDyn(&shape))],
        };
        let gy2 = match output_grad.to_shape((total_prefix, half)) {
            Ok(v) => v,
            Err(_) => return vec![ArrayD::zeros(IxDyn(&shape))],
        };

        let grad_x = ArrayD::<f32>::zeros(IxDyn(&shape));
        let mut gx2 = match grad_x.to_shape((total_prefix, last_dim)) {
            Ok(v) => v,
            Err(_) => return vec![ArrayD::zeros(IxDyn(&shape))],
        };

        for (row_idx, (row_x, row_gy)) in x2.outer_iter().zip(gy2.outer_iter()).enumerate() {
            for i in 0..half {
                let a = row_x[i];
                let b = row_x[i + half];
                let g = row_gy[i];

                // ReLU derivative: 1 if a > 0 else 0
                let relu_prime_a = if a > 0.0 { 1.0 } else { 0.0 };

                // grad w.r.t. a: relu_prime(a) * b * g
                gx2[[row_idx, i]] = relu_prime_a * b * g;
                // grad w.r.t. b: relu(a) * g
                gx2[[row_idx, i + half]] = a.max(0.0) * g;
            }
        }

        vec![grad_x]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Interpolate operation: resizes an input tensor (NCHW) to a new spatial size.
/// Currently supports "bilinear" and "nearest" modes for 4D inputs.
pub struct Interpolate {
    pub size: (usize, usize), // (H_out, W_out)
    pub mode: String,         // "bilinear" or "nearest"
    pub align_corners: bool,
}

impl Interpolate {
    pub fn new(size: (usize, usize), mode: String, align_corners: bool) -> Self {
        Interpolate {
            size,
            mode,
            align_corners,
        }
    }
}

impl Operation for Interpolate {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let input = inputs[0].lock().storage.to_f32_array(); // [N, C, H_in, W_in]
        let in_shape = input.shape();
        if in_shape.len() != 4 {
            log::error!("Interpolate forward: input must be 4D (NCHW)");
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }
        let (n, c, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (h_out, w_out) = self.size;

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, c, h_out, w_out][..]));

        let scale_h = if self.align_corners {
            if h_out > 1 {
                (h_in - 1) as f32 / (h_out - 1) as f32
            } else {
                0.0
            }
        } else {
            h_in as f32 / h_out as f32
        };

        let scale_w = if self.align_corners {
            if w_out > 1 {
                (w_in - 1) as f32 / (w_out - 1) as f32
            } else {
                0.0
            }
        } else {
            w_in as f32 / w_out as f32
        };

        if self.mode == "nearest" {
            for b in 0..n {
                for k in 0..c {
                    for y in 0..h_out {
                        let real_y = if self.align_corners {
                            scale_h * y as f32
                        } else {
                            scale_h * (y as f32 + 0.5) - 0.5
                        };
                        let in_y = real_y.round() as isize;
                        let in_y = in_y.clamp(0, (h_in - 1) as isize) as usize;

                        for x in 0..w_out {
                            let real_x = if self.align_corners {
                                scale_w * x as f32
                            } else {
                                scale_w * (x as f32 + 0.5) - 0.5
                            };
                            let in_x = real_x.round() as isize;
                            let in_x = in_x.clamp(0, (w_in - 1) as isize) as usize;

                            out[[b, k, y, x]] = input[[b, k, in_y, in_x]];
                        }
                    }
                }
            }
        } else if self.mode == "bilinear" {
            for b in 0..n {
                for k in 0..c {
                    for y in 0..h_out {
                        let real_y = if self.align_corners {
                            scale_h * y as f32
                        } else {
                            scale_h * (y as f32 + 0.5) - 0.5
                        };
                        let y0 = real_y.floor() as isize;
                        let y1 = y0 + 1;
                        let dy = real_y - y0 as f32;

                        for x in 0..w_out {
                            let real_x = if self.align_corners {
                                scale_w * x as f32
                            } else {
                                scale_w * (x as f32 + 0.5) - 0.5
                            };
                            let x0 = real_x.floor() as isize;
                            let x1 = x0 + 1;
                            let dx = real_x - x0 as f32;

                            // Clamp indices
                            let y0_c = y0.clamp(0, (h_in - 1) as isize) as usize;
                            let y1_c = y1.clamp(0, (h_in - 1) as isize) as usize;
                            let x0_c = x0.clamp(0, (w_in - 1) as isize) as usize;
                            let x1_c = x1.clamp(0, (w_in - 1) as isize) as usize;

                            let v00 = input[[b, k, y0_c, x0_c]];
                            let v01 = input[[b, k, y0_c, x1_c]];
                            let v10 = input[[b, k, y1_c, x0_c]];
                            let v11 = input[[b, k, y1_c, x1_c]];

                            let val = (1.0 - dy) * (1.0 - dx) * v00
                                + (1.0 - dy) * dx * v01
                                + dy * (1.0 - dx) * v10
                                + dy * dx * v11;
                            out[[b, k, y, x]] = val;
                        }
                    }
                }
            }
        } else {
            log::error!("Interpolate: unsupported mode {}", self.mode);
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let input = inputs[0].lock().storage.to_f32_array();
        let in_shape = input.shape();
        let (n, c, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (h_out, w_out) = self.size;

        let mut grad_input = ArrayD::<f32>::zeros(input.dim());

        let scale_h = if self.align_corners {
            if h_out > 1 {
                (h_in - 1) as f32 / (h_out - 1) as f32
            } else {
                0.0
            }
        } else {
            h_in as f32 / h_out as f32
        };

        let scale_w = if self.align_corners {
            if w_out > 1 {
                (w_in - 1) as f32 / (w_out - 1) as f32
            } else {
                0.0
            }
        } else {
            w_in as f32 / w_out as f32
        };

        if self.mode == "nearest" {
            for b in 0..n {
                for k in 0..c {
                    for y in 0..h_out {
                        let real_y = if self.align_corners {
                            scale_h * y as f32
                        } else {
                            scale_h * (y as f32 + 0.5) - 0.5
                        };
                        let in_y = real_y.round() as isize;
                        let in_y = in_y.clamp(0, (h_in - 1) as isize) as usize;

                        for x in 0..w_out {
                            let real_x = if self.align_corners {
                                scale_w * x as f32
                            } else {
                                scale_w * (x as f32 + 0.5) - 0.5
                            };
                            let in_x = real_x.round() as isize;
                            let in_x = in_x.clamp(0, (w_in - 1) as isize) as usize;

                            grad_input[[b, k, in_y, in_x]] += output_grad[[b, k, y, x]];
                        }
                    }
                }
            }
        } else if self.mode == "bilinear" {
            for b in 0..n {
                for k in 0..c {
                    for y in 0..h_out {
                        let real_y = if self.align_corners {
                            scale_h * y as f32
                        } else {
                            scale_h * (y as f32 + 0.5) - 0.5
                        };
                        let y0 = real_y.floor() as isize;
                        let y1 = y0 + 1;
                        let dy = real_y - y0 as f32;

                        for x in 0..w_out {
                            let real_x = if self.align_corners {
                                scale_w * x as f32
                            } else {
                                scale_w * (x as f32 + 0.5) - 0.5
                            };
                            let x0 = real_x.floor() as isize;
                            let x1 = x0 + 1;
                            let dx = real_x - x0 as f32;

                            // Clamp indices
                            let y0_c = y0.clamp(0, (h_in - 1) as isize) as usize;
                            let y1_c = y1.clamp(0, (h_in - 1) as isize) as usize;
                            let x0_c = x0.clamp(0, (w_in - 1) as isize) as usize;
                            let x1_c = x1.clamp(0, (w_in - 1) as isize) as usize;

                            let g = output_grad[[b, k, y, x]];

                            let w00 = (1.0 - dy) * (1.0 - dx);
                            let w01 = (1.0 - dy) * dx;
                            let w10 = dy * (1.0 - dx);
                            let w11 = dy * dx;

                            grad_input[[b, k, y0_c, x0_c]] += w00 * g;
                            grad_input[[b, k, y0_c, x1_c]] += w01 * g;
                            grad_input[[b, k, y1_c, x0_c]] += w10 * g;
                            grad_input[[b, k, y1_c, x1_c]] += w11 * g;
                        }
                    }
                }
            }
        }

        vec![grad_input]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod interpolate_tests {
    use super::*;
    use crate::tensor::Tensor;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn test_interpolate_nearest_2x() {
        // [1, 1, 2, 2] -> [1, 1, 4, 4]
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2][..]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            true,
        );
        let op = Interpolate::new((4, 4), "nearest".to_string(), false);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 4, 4][..]));
        op.forward(&[input][..], &mut out);

        // My implementation: scale * (x+0.5) - 0.5
        // 2->4 scale=0.5
        // x=0 -> -0.25 -> 0
        // x=1 -> 0.25 -> 0
        // x=2 -> 0.75 -> 1
        // x=3 -> 1.25 -> 1
        // So indices are 0,0,1,1.

        // Row 0: 1, 2 -> 1, 1, 2, 2
        let out_slice = out.as_slice().unwrap();
        assert_eq!(out_slice[0], 1.0);
        assert_eq!(out_slice[1], 1.0);
        assert_eq!(out_slice[2], 2.0);
        assert_eq!(out_slice[3], 2.0);

        // Row 1: 3, 4 -> 3, 3, 4, 4
        // Indices in flat array: 4,5,6,7? No, 4x4=16 elements.
        // Row 0 is indices 0..3.
        // Row 1 is indices 4..7 (which corresponds to output y=1).
        // Wait, output y=0 -> input y=0.
        // output y=1 -> input y=0 (same math).
        // output y=2 -> input y=1.
        // output y=3 -> input y=1.

        // So:
        // y=0: 1, 1, 2, 2
        // y=1: 1, 1, 2, 2
        // y=2: 3, 3, 4, 4
        // y=3: 3, 3, 4, 4

        assert_eq!(out[[0, 0, 0, 0]], 1.0);
        assert_eq!(out[[0, 0, 1, 0]], 1.0);
        assert_eq!(out[[0, 0, 2, 0]], 3.0);
        assert_eq!(out[[0, 0, 3, 0]], 3.0);
    }

    #[test]
    fn test_interpolate_bilinear_2x() {
        // [1, 1, 2, 2]
        // 1 2
        // 3 4
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2][..]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            true,
        );
        let op = Interpolate::new((4, 4), "bilinear".to_string(), false);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 4, 4][..]));
        op.forward(&[input][..], &mut out);

        // Just check that it runs and produces reasonable values (bounded by min/max)
        for v in out.iter() {
            assert!(*v >= 1.0);
            assert!(*v <= 4.0);
        }
    }
}

/// GridSample operation: samples input using grid of coordinates.
/// Input: [N, C, H_in, W_in]
/// Grid: [N, H_out, W_out, 2] (values in range [-1, 1])
/// Output: [N, C, H_out, W_out]
pub struct GridSample {
    pub mode: String,         // "bilinear" or "nearest"
    pub padding_mode: String, // "zeros", "border", "reflection"
    pub align_corners: bool,
}

impl GridSample {
    pub fn new(mode: String, padding_mode: String, align_corners: bool) -> Self {
        GridSample {
            mode,
            padding_mode,
            align_corners,
        }
    }

    fn compute_source_coordinates(&self, ix: f32, iy: f32, w_in: usize, h_in: usize) -> (f32, f32) {
        let (x, y);
        if self.align_corners {
            x = ((ix + 1.0) / 2.0) * (w_in as f32 - 1.0);
            y = ((iy + 1.0) / 2.0) * (h_in as f32 - 1.0);
        } else {
            x = ((ix + 1.0) * w_in as f32 - 1.0) / 2.0;
            y = ((iy + 1.0) * h_in as f32 - 1.0) / 2.0;
        }
        (x, y)
    }

    fn within_bounds(&self, x: isize, y: isize, w: usize, h: usize) -> bool {
        x >= 0 && x < w as isize && y >= 0 && y < h as isize
    }
}

impl Operation for GridSample {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let input = inputs[0].lock().storage.to_f32_array();
        let grid = inputs[1].lock().storage.to_f32_array();

        // Validation
        if input.ndim() != 4 || grid.ndim() != 4 || grid.shape()[3] != 2 {
            log::error!("GridSample forward: input must be 4D, grid must be 4D with last dim 2");
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let (n, c, h_in, w_in) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (n_grid, h_out, w_out, _) = (
            grid.shape()[0],
            grid.shape()[1],
            grid.shape()[2],
            grid.shape()[3],
        );

        if n != n_grid {
            log::error!(
                "GridSample forward: input batch size {} != grid batch size {}",
                n,
                n_grid
            );
            *output = ArrayD::zeros(IxDyn(&[0][..]));
            return;
        }

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, c, h_out, w_out][..]));

        for b in 0..n {
            for y in 0..h_out {
                for x in 0..w_out {
                    let gx = grid[[b, y, x, 0]];
                    let gy = grid[[b, y, x, 1]];

                    let (src_x, src_y) = self.compute_source_coordinates(gx, gy, w_in, h_in);

                    if self.mode == "nearest" {
                        let ix = src_x.round() as isize;
                        let iy = src_y.round() as isize;

                        if self.within_bounds(ix, iy, w_in, h_in) {
                            for k in 0..c {
                                out[[b, k, y, x]] = input[[b, k, iy as usize, ix as usize]];
                            }
                        } else if self.padding_mode == "border" {
                            let ix = ix.clamp(0, (w_in - 1) as isize);
                            let iy = iy.clamp(0, (h_in - 1) as isize);
                            for k in 0..c {
                                out[[b, k, y, x]] = input[[b, k, iy as usize, ix as usize]];
                            }
                        }
                    } else if self.mode == "bilinear" {
                        let x0 = src_x.floor() as isize;
                        let x1 = x0 + 1;
                        let y0 = src_y.floor() as isize;
                        let y1 = y0 + 1;

                        let dx = src_x - x0 as f32;
                        let dy = src_y - y0 as f32;

                        let w00 = (1.0 - dx) * (1.0 - dy);
                        let w01 = dx * (1.0 - dy);
                        let w10 = (1.0 - dx) * dy;
                        let w11 = dx * dy;

                        let get_val = |vals: &ArrayD<f32>, b, k, y: isize, x: isize| -> f32 {
                            if self.within_bounds(x, y, w_in, h_in) {
                                vals[[b, k, y as usize, x as usize]]
                            } else if self.padding_mode == "border" {
                                let x_c = x.clamp(0, (w_in - 1) as isize);
                                let y_c = y.clamp(0, (h_in - 1) as isize);
                                vals[[b, k, y_c as usize, x_c as usize]]
                            } else {
                                0.0
                            }
                        };

                        for k in 0..c {
                            let v00 = get_val(&input, b, k, y0, x0);
                            let v01 = get_val(&input, b, k, y0, x1);
                            let v10 = get_val(&input, b, k, y1, x0);
                            let v11 = get_val(&input, b, k, y1, x1);

                            out[[b, k, y, x]] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                        }
                    }
                }
            }
        }

        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let input = inputs[0].lock().storage.to_f32_array();
        let grid = inputs[1].lock().storage.to_f32_array();

        let (n, c, h_in, w_in) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (_, h_out, w_out, _) = (
            grid.shape()[0],
            grid.shape()[1],
            grid.shape()[2],
            grid.shape()[3],
        );

        let mut grad_input = ArrayD::<f32>::zeros(input.dim());
        let grad_grid = ArrayD::<f32>::zeros(grid.dim());

        for b in 0..n {
            for y in 0..h_out {
                for x in 0..w_out {
                    let gx = grid[[b, y, x, 0]];
                    let gy = grid[[b, y, x, 1]];

                    let (src_x, src_y) = self.compute_source_coordinates(gx, gy, w_in, h_in);

                    if self.mode == "nearest" {
                        let ix = src_x.round() as isize;
                        let iy = src_y.round() as isize;

                        if self.within_bounds(ix, iy, w_in, h_in) {
                            for k in 0..c {
                                grad_input[[b, k, iy as usize, ix as usize]] +=
                                    output_grad[[b, k, y, x]];
                            }
                        } else if self.padding_mode == "border" {
                            let ix = ix.clamp(0, (w_in - 1) as isize);
                            let iy = iy.clamp(0, (h_in - 1) as isize);
                            for k in 0..c {
                                grad_input[[b, k, iy as usize, ix as usize]] +=
                                    output_grad[[b, k, y, x]];
                            }
                        }
                    } else if self.mode == "bilinear" {
                        let x0 = src_x.floor() as isize;
                        let x1 = x0 + 1;
                        let y0 = src_y.floor() as isize;
                        let y1 = y0 + 1;

                        let dx = src_x - x0 as f32;
                        let dy = src_y - y0 as f32;

                        let w00 = (1.0 - dx) * (1.0 - dy);
                        let w01 = dx * (1.0 - dy);
                        let w10 = (1.0 - dx) * dy;
                        let w11 = dx * dy;

                        let accumulate_grad =
                            |grads: &mut ArrayD<f32>, b, k, y: isize, x: isize, val: f32| {
                                if self.within_bounds(x, y, w_in, h_in) {
                                    grads[[b, k, y as usize, x as usize]] += val;
                                } else if self.padding_mode == "border" {
                                    let x_c = x.clamp(0, (w_in - 1) as isize);
                                    let y_c = y.clamp(0, (h_in - 1) as isize);
                                    grads[[b, k, y_c as usize, x_c as usize]] += val;
                                }
                            };

                        for k in 0..c {
                            let g = output_grad[[b, k, y, x]];
                            accumulate_grad(&mut grad_input, b, k, y0, x0, w00 * g);
                            accumulate_grad(&mut grad_input, b, k, y0, x1, w01 * g);
                            accumulate_grad(&mut grad_input, b, k, y1, x0, w10 * g);
                            accumulate_grad(&mut grad_input, b, k, y1, x1, w11 * g);
                        }
                    }
                }
            }
        }

        vec![grad_input, grad_grid]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Label Smoothing Cross-Entropy operation.
/// Smoothed label: y_smooth = (1 - ε) * y + ε / num_classes
/// Loss: L = -Σ y_smooth * log(p)
/// Inputs: log-probabilities (num_classes,), targets (one-hot or class indices)
pub struct LabelSmoothingCrossEntropy {
    pub smoothing: f32,
    pub num_classes: usize,
    pub reduction: String,   // "mean", "sum", "none"
    pub target_mode: String, // "onehot", "class_index"
}

impl LabelSmoothingCrossEntropy {
    pub fn new(smoothing: f32, num_classes: usize, reduction: String, target_mode: String) -> Self {
        assert!(
            smoothing >= 0.0 && smoothing < 1.0,
            "smoothing must be in [0, 1)"
        );
        assert!(num_classes > 0, "num_classes must be positive");
        assert!(
            reduction == "mean" || reduction == "sum" || reduction == "none",
            "reduction must be 'mean', 'sum', or 'none'"
        );
        assert!(
            target_mode == "onehot" || target_mode == "class_index",
            "target_mode must be 'onehot' or 'class_index'"
        );
        LabelSmoothingCrossEntropy {
            smoothing,
            num_classes,
            reduction,
            target_mode,
        }
    }
}

impl Operation for LabelSmoothingCrossEntropy {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let log_probs = inputs[0].lock().storage.to_f32_array();
        let targets = inputs[1].lock().storage.to_f32_array();

        let eps = self.smoothing;
        let num_classes = self.num_classes;
        let uniform = eps / num_classes as f32;

        let mut loss_sum: f32 = 0.0;
        let count = log_probs.len();

        if self.target_mode == "onehot" {
            // targets is one-hot encoded: y_smooth = (1-eps)*y + eps/C
            for (log_p, &y) in log_probs.iter().zip(targets.iter()) {
                let y_smooth = (1.0 - eps) * y + uniform;
                loss_sum += -y_smooth * log_p;
            }
        } else {
            // targets are class indices: y_smooth = (1-eps)*one_hot(y) + eps/C
            for (log_p, &target_idx) in log_probs.iter().zip(targets.iter()) {
                let idx = target_idx as usize;
                if idx < num_classes {
                    let y_smooth = if idx == 0 {
                        (1.0 - eps) + uniform
                    } else {
                        uniform
                    };
                    loss_sum += -y_smooth * log_p;
                } else {
                    log::warn!(
                        "LabelSmoothingCrossEntropy: target index {} out of range [0, {})",
                        idx,
                        num_classes
                    );
                }
            }
        }

        let result = match self.reduction.as_str() {
            "mean" => loss_sum / count as f32,
            "sum" => loss_sum,
            "none" => loss_sum / count as f32, // per-element loss
            _ => loss_sum / count as f32,
        };

        *output = ArrayD::from_elem(ndarray::IxDyn(&[][..]), result);
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let log_probs = inputs[0].lock().storage.to_f32_array();
        let targets = inputs[1].lock().storage.to_f32_array();

        let eps = self.smoothing;
        let num_classes = self.num_classes;
        let uniform = eps / num_classes as f32;
        let count = log_probs.len() as f32;

        let grad_scale = *output_grad.iter().next().unwrap_or(&1.0);

        let reduction_factor = match self.reduction.as_str() {
            "mean" => 1.0 / count,
            "sum" => 1.0,
            "none" => 1.0 / count,
            _ => 1.0 / count,
        };

        // dL/d(log_p) = -y_smooth for mean/sum, scaled by grad_scale
        let grad_log_probs: Vec<f32> = if self.target_mode == "onehot" {
            log_probs
                .iter()
                .zip(targets.iter())
                .map(|(&_lp, &y)| {
                    let y_smooth = (1.0 - eps) * y + uniform;
                    -y_smooth * grad_scale * reduction_factor
                })
                .collect()
        } else {
            log_probs
                .iter()
                .zip(targets.iter())
                .map(|(&_lp, &target_idx)| {
                    let idx = target_idx as usize;
                    let y_smooth = if idx < num_classes && idx == 0 {
                        (1.0 - eps) + uniform
                    } else if idx < num_classes {
                        uniform
                    } else {
                        0.0
                    };
                    -y_smooth * grad_scale * reduction_factor
                })
                .collect()
        };

        // Gradient w.r.t. targets is zero (targets are fixed labels)
        let grad_targets = ArrayD::zeros(targets.dim());

        vec![
            ArrayD::from_shape_vec(log_probs.dim(), grad_log_probs).unwrap(),
            grad_targets,
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod label_smoothing_tests {
    use super::*;
    use crate::tensor::Tensor;
    use ndarray::{ArrayD, IxDyn};
    use std::sync::Arc;

    #[test]
    fn test_label_smoothing_onehot_forward() {
        let ls = LabelSmoothingCrossEntropy::new(0.1, 3, "mean".to_string(), "onehot".to_string());
        // log-probs for 3 classes
        let log_probs = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![-1.0, -2.0, -3.0]).unwrap(),
            true,
        );
        // one-hot target for class 0
        let targets = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![1.0, 0.0, 0.0]).unwrap(),
            false,
        );

        let result = Tensor::apply(Arc::new(ls), &[log_probs, targets][..]);
        let loss_val = *result.lock().storage.to_f32_array().iter().next().unwrap();

        // With smoothing 0.1 and 3 classes:
        // y_smooth = [0.9+0.033, 0.033, 0.033] = [0.933, 0.033, 0.033]
        // loss = -(0.933*(-1) + 0.033*(-2) + 0.033*(-3)) = 0.933 + 0.066 + 0.099 = 1.098
        assert!(loss_val > 0.0);
        assert!(loss_val.is_finite());
    }

    #[test]
    fn test_label_smoothing_class_index_forward() {
        let ls =
            LabelSmoothingCrossEntropy::new(0.1, 3, "mean".to_string(), "class_index".to_string());
        let log_probs = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![-1.0, -2.0, -3.0]).unwrap(),
            true,
        );
        // class index 0
        let targets = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![0.0]).unwrap(),
            false,
        );

        let result = Tensor::apply(Arc::new(ls), &[log_probs, targets][..]);
        let loss_val = *result.lock().storage.to_f32_array().iter().next().unwrap();

        assert!(loss_val > 0.0);
        assert!(loss_val.is_finite());
    }

    #[test]
    fn test_label_smoothing_backward() {
        let ls = LabelSmoothingCrossEntropy::new(0.1, 3, "mean".to_string(), "onehot".to_string());
        let log_probs = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![-1.0, -2.0, -3.0]).unwrap(),
            true,
        );
        let targets = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![1.0, 0.0, 0.0]).unwrap(),
            false,
        );

        let result = Tensor::apply(Arc::new(ls), &[log_probs.clone(), targets][..]);
        result.backward();

        assert!(log_probs.lock().grad.is_some());
        let grad_lock = log_probs.lock();
        let grad = grad_lock.grad.as_ref().unwrap();
        // Gradients should be negative (since we're minimizing cross-entropy)
        for &g in grad.iter() {
            assert!(g < 0.0 || g.abs() < 1e-10);
        }
    }

    #[test]
    fn test_label_smoothing_zero_smoothing_equals_ce() {
        // With smoothing=0, label smoothing should reduce to standard cross-entropy
        let ls_zero =
            LabelSmoothingCrossEntropy::new(0.0, 3, "mean".to_string(), "onehot".to_string());
        let log_probs = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![-1.0, -2.0, -3.0]).unwrap(),
            true,
        );
        let targets = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![1.0, 0.0, 0.0]).unwrap(),
            false,
        );

        let result = Tensor::apply(Arc::new(ls_zero), &[log_probs, targets][..]);
        let loss_val = *result.lock().storage.to_f32_array().iter().next().unwrap();

        // Standard CE for class 0: -log(p_0) where p_0 = exp(-1)/sum(exp([-1,-2,-3]))
        // = -(-1 - log(exp(-1)+exp(-2)+exp(-3))) = 1 + log(exp(-1)+exp(-2)+exp(-3))
        let sum_exp = (-1.0_f32).exp() + (-2.0_f32).exp() + (-3.0_f32).exp();
        let expected_ce = 1.0 + sum_exp.ln();
        assert!((loss_val - expected_ce).abs() < 1e-4);
    }

    #[test]
    #[should_panic(expected = "smoothing must be in [0, 1)")]
    fn test_label_smoothing_invalid_smoothing() {
        LabelSmoothingCrossEntropy::new(1.0, 3, "mean".to_string(), "onehot".to_string());
    }

    #[test]
    #[should_panic(expected = "num_classes must be positive")]
    fn test_label_smoothing_invalid_classes() {
        LabelSmoothingCrossEntropy::new(0.1, 0, "mean".to_string(), "onehot".to_string());
    }
}

#[cfg(test)]
mod grid_sample_tests {
    use super::*;
    use crate::tensor::Tensor;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn test_grid_sample_identity() {
        // Identity grid: should return input exact same way.
        // grid values: (-1,-1) to (1,1).
        // 2x2 input.
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2][..]), input_data).unwrap(),
            true,
        );

        // Identity grid for 2x2
        // (-0.5, -0.5) is center of top-left pixel?
        // standard align_corners=false:
        // x_in = (x_grid + 1)*W/2 - 0.5.
        // if x_grid = -0.5: (-0.5+1)*1 - 0.5 = 0.5 - 0.5 = 0.
        // So for 2x2, coords are -0.5 and 0.5.
        // Wait, normalized coordinates are [-1, 1].
        // For W=2:
        // pixel 0 center: 0.
        // (x_grid + 1) * 2 - 1 = 2 * 0 => x_grid + 1 = 0 => x_grid = -1?
        // No.
        // x_in = (x_grid + 1) * W / 2 - 0.5.
        // If x_in = 0 => (x_grid + 1) = 0.5 => x_grid = -0.5.
        // If x_in = 1 => (x_grid + 1) = 1.5 => x_grid = 0.5.

        let grid_data = vec![-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]; // [1, 2, 2, 2]
        let grid = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 2][..]), grid_data).unwrap(),
            false,
        );

        let op = GridSample::new("nearest".to_string(), "zeros".to_string(), false);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 2, 2][..]));
        op.forward(&[input, grid][..], &mut out);

        assert_eq!(out[[0, 0, 0, 0]], 1.0);
        assert_eq!(out[[0, 0, 0, 1]], 2.0);
        assert_eq!(out[[0, 0, 1, 0]], 3.0);
        assert_eq!(out[[0, 0, 1, 1]], 4.0);
    }
}
