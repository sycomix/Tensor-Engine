use crate::tensor::Tensor;
#[cfg(all(feature = "openblas", not(target_os = "windows")))]
use cblas_sys::{self, CBLAS_ORDER, CBLAS_TRANSPOSE};
#[cfg(all(feature = "openblas", not(target_os = "windows")))]
use ndarray::Array2;
use ndarray::Zip;
use ndarray::{s, ArrayD, ArrayView2, Axis, Ix2, IxDyn, SliceInfo, SliceInfoElem};
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
            *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
            return;
        }
        let bnh = shape_q[0];
        let seq = shape_q[1];
        let hd = shape_q[2];
        if hd != self.head_dim || k.shape() != &[bnh, seq, hd] || v.shape() != &[bnh, seq, hd] {
            log::error!(
                "FlashAttentionRef forward: shape mismatch: q={:?} k={:?} v={:?} head_dim={}",
                shape_q,
                k.shape(),
                v.shape(),
                self.head_dim
            );
            *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
            return;
        }
        // QK^T
        // Build k_t as k transposed on the last two axes -> [bnh, hd, seq]
        let mut k_t = ArrayD::<f32>::zeros(IxDyn(&[bnh, hd, seq]));
        for i in 0..bnh {
            let kmat = k.index_axis(Axis(0), i).to_owned(); // [seq,hd]
            let kt = kmat.t().to_owned(); // [hd,seq]
            k_t.index_axis_mut(Axis(0), i).assign(&kt.into_dyn());
        }
        let mut qk = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, seq]));
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
                    *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
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
                    *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
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
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd]));
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
                    *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
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
                    *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
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
        let mut qk = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, seq]));
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
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd]));
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
        let mut dv = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd]));
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
        let mut datt = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, seq]));
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
            let vmat_t2: ndarray::Array2<f32> = match vmat
                .t()
                .to_owned()
                .into_dimensionality::<Ix2>()
            {
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
        let mut dqk = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, seq]));
        for i in 0..bnh {
            let a = attn.index_axis(Axis(0), i).to_owned(); // [seq,seq]
            let da = datt.index_axis(Axis(0), i).to_owned(); // [seq,seq]
                                                             // for each row: jacobian of softmax
            let mut dqi = ArrayD::<f32>::zeros(IxDyn(&[seq, seq]));
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
        let mut dq = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd]));
        let mut dk = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd]));
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
            *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
            return;
        }
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd]));
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
                        *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
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
                        *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
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
                        *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
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
                        *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
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
        let mut dq = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd]));
        let mut dk = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd]));
        let mut dv = ArrayD::<f32>::zeros(IxDyn(&[bnh, seq, hd]));
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
                let mut dqk_chunk = ArrayD::<f32>::zeros(IxDyn(&[end - start, seq]));
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
        *output = ArrayD::from_elem(IxDyn(&[]), s);
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
            a_shape_raw, b_shape_raw
        );
        // Convert first input, but catch panics during conversion to avoid crashes
        let a = match std::panic::catch_unwind(|| inputs[0].to_f32_array()) {
            Ok(arr) => arr,
            Err(_) => {
                log::error!(
                    "Add.forward: failed to convert first input to f32 array; aborting operation"
                );
                *output = ArrayD::zeros(IxDyn(&[0]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
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
        let out_shape = match crate::tensor::Tensor::broadcast_shapes(&[
            a_shape_vec.clone(),
            b_shape_vec.clone(),
        ]) {
            Ok(s) => s,
            Err(_) => {
                log::error!("Add.forward: incompatible shapes and cannot broadcast: a_shape={:?} b_shape={:?}", a.shape(), b.shape());
                *output = ArrayD::zeros(IxDyn(&[0]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
        // Perform elementwise addition into a new owned array
        let mut out_arr = ArrayD::zeros(IxDyn(&out_shape));
        let out_slice = match out_arr.as_slice_mut() {
            Some(s) => s,
            None => {
                log::error!("Add.forward: failed to get mutable slice for output");
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        for ((aa, bb), o) in a_b.iter().zip(b_b.iter()).zip(out_slice.iter_mut()) {
            *o = *aa + *bb;
        }
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
        *output = a.mapv(|x| x.exp());
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let grad = a.mapv(|x| x.exp());
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

impl Operation for Equal {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let b = inputs[1].to_f32_array();
        // no 'out' variable needed; we'll build out_arr directly
        // Use broadcasting
        let a_shape = a.shape().to_vec();
        let b_shape = b.shape().to_vec();
        let out_shape =
            crate::tensor::Tensor::broadcast_shapes(&[a_shape.clone(), b_shape.clone()])
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
        let out_slice = match out_arr.as_slice_mut() {
            Some(s) => s,
            None => {
                log::error!("Failed to get mutable slice for output array in Equal forward");
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        for ((oa, ob), o) in a_b.iter().zip(b_b.iter()).zip(out_slice.iter_mut()) {
            *o = if (oa - ob).abs() < 1e-6 { 1.0 } else { 0.0 };
        }
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
        let out_shape =
            crate::tensor::Tensor::broadcast_shapes(&[a_shape.clone(), b_shape.clone()])
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
        let out_slice = match out_arr.as_slice_mut() {
            Some(s) => s,
            None => {
                log::error!("Failed to get mutable slice for output array in Greater forward");
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        for ((oa, ob), o) in a_b.iter().zip(b_b.iter()).zip(out_slice.iter_mut()) {
            *o = if oa > ob { 1.0 } else { 0.0 };
        }
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
        let out_shape =
            crate::tensor::Tensor::broadcast_shapes(&[a_shape.clone(), b_shape.clone()])
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
        let out_slice = match out_arr.as_slice_mut() {
            Some(s) => s,
            None => {
                log::error!("Failed to get mutable slice for output array in Less forward");
                *output = ArrayD::zeros(IxDyn(&out_shape));
                return;
            }
        };
        for ((oa, ob), o) in a_b.iter().zip(b_b.iter()).zip(out_slice.iter_mut()) {
            *o = if oa < ob { 1.0 } else { 0.0 };
        }
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

/// Max operation: returns the maximum value of all elements in the tensor as a scalar.
pub struct Max;

impl Operation for Max {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let max_val = a.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        *output = ArrayD::from_elem(IxDyn(&[]), max_val);
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
        *output = ArrayD::from_elem(IxDyn(&[]), min_val);
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

/// The multiplication operation.
pub struct Mul;

impl Operation for Mul {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        log::debug!(
            "Mul.forward enter lhs={:p} rhs={:p}",
            &inputs[0] as *const _,
            &inputs[1] as *const _
        );
        let a_lock = inputs[0].lock();
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
            *output = ArrayD::zeros(IxDyn(&[0]));
            return;
        }
        let batch = a.shape()[0];
        if b.shape()[0] != batch {
            log::error!(
                "BatchedMatMul: batch dims mismatch: {} != {}",
                batch,
                b.shape()[0]
            );
            *output = ArrayD::zeros(IxDyn(&[0]));
            return;
        }
        let m = a.shape()[1];
        let k = a.shape()[2];
        let kb = b.shape()[1];
        let n = b.shape()[2];
        if k != kb {
            log::error!("BatchedMatMul: inner dims mismatch: {} != {}", k, kb);
            *output = ArrayD::zeros(IxDyn(&[0]));
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
                    *output = ArrayD::zeros(IxDyn(&[0]));
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
                    *output = ArrayD::zeros(IxDyn(&[0]));
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
            *output = ArrayD::zeros(IxDyn(&[0]));
            return;
        }
        let a2 = match a.into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("QuantizedMatMul forward failed to convert a: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
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
                    *output = ArrayD::zeros(IxDyn(&[0]));
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
                    *output = ArrayD::zeros(IxDyn(&[0]));
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
                        *output = ArrayD::zeros(IxDyn(&[0]));
                        return;
                    }
                };
                if bytes.len() != expected_len {
                    log::error!(
                        "QuantizedMatMul forward: I8 weight buffer length mismatch: len={} expected={}",
                        bytes.len(),
                        expected_len
                    );
                    *output = ArrayD::zeros(IxDyn(&[0]));
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

                let res = match ndarray::Array2::from_shape_vec((m, cols), out) {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "QuantizedMatMul forward: failed to build output Array2: {}",
                            e
                        );
                        *output = ArrayD::zeros(IxDyn(&[0]));
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
                    *output = ArrayD::zeros(IxDyn(&[0]));
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
                    *output = ArrayD::zeros(IxDyn(&[0]));
                    return;
                }
                if scales.len() != rows {
                    log::error!(
                        "QuantizedMatMul forward: I8Rowwise scales length mismatch: len={} rows={}",
                        scales.len(),
                        rows
                    );
                    *output = ArrayD::zeros(IxDyn(&[0]));
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
                        *output = ArrayD::zeros(IxDyn(&[0]));
                        return;
                    }
                };
                if bytes.len() != expected_len {
                    log::error!(
                        "QuantizedMatMul forward: I8Rowwise weight buffer length mismatch: len={} expected={}",
                        bytes.len(),
                        expected_len
                    );
                    *output = ArrayD::zeros(IxDyn(&[0]));
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

                let res = match ndarray::Array2::from_shape_vec((m, cols), out) {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "QuantizedMatMul forward: failed to build output Array2: {}",
                            e
                        );
                        *output = ArrayD::zeros(IxDyn(&[0]));
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
                    *output = ArrayD::zeros(IxDyn(&[0]));
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
                    *output = ArrayD::zeros(IxDyn(&[0]));
                    return;
                }
                let bs = *block_size;
                if bs == 0 {
                    log::error!("QuantizedMatMul forward: block_size must be > 0");
                    *output = ArrayD::zeros(IxDyn(&[0]));
                    return;
                }
                let blocks_per_row = (cols + bs - 1) / bs;
                let expected_scales = match rows.checked_mul(blocks_per_row) {
                    Some(v) => v,
                    None => {
                        log::error!(
                            "QuantizedMatMul forward: rows*blocks_per_row overflow: {}*{}",
                            rows,
                            blocks_per_row
                        );
                        *output = ArrayD::zeros(IxDyn(&[0]));
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
                    *output = ArrayD::zeros(IxDyn(&[0]));
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
                        *output = ArrayD::zeros(IxDyn(&[0]));
                        return;
                    }
                };
                if bytes.len() != expected_len {
                    log::error!(
                        "QuantizedMatMul forward: I8Blockwise weight buffer length mismatch: len={} expected={}",
                        bytes.len(),
                        expected_len
                    );
                    *output = ArrayD::zeros(IxDyn(&[0]));
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
                            for block_idx in 0..blocks_per_row {
                                let s = scales_row[block_idx];
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
                            for block_idx in 0..blocks_per_row {
                                let s = scales_row[block_idx];
                                let start = block_idx * bs;
                                let end = ((block_idx + 1) * bs).min(cols);
                                for col in start..end {
                                    out_row[col] += aij * (b_row[col] as f32) * s;
                                }
                            }
                        }
                    }
                }

                let res = match ndarray::Array2::from_shape_vec((m, cols), out) {
                    Ok(arr) => arr,
                    Err(e) => {
                        log::error!(
                            "QuantizedMatMul forward: failed to build output Array2: {}",
                            e
                        );
                        *output = ArrayD::zeros(IxDyn(&[0]));
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
                    *output = ArrayD::zeros(IxDyn(&[0]));
                    return;
                }
                let b2 = match b_guard.storage.to_f32_array().into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        log::error!("QuantizedMatMul forward failed to convert b: {}", e);
                        *output = ArrayD::zeros(IxDyn(&[0]));
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

impl MatMul {
    pub fn new() -> Self {
        MatMul
    }
    pub fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        <Self as Operation>::forward(self, inputs, output)
    }
}

impl Operation for MatMul {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // Simple, robust matmul: convert to f32 arrays, ensure 2D, then use ndarray dot
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
                *output = ArrayD::zeros(IxDyn(&[0]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
        log::debug!(
            "MatMul.forward SAFE: a_shape={:?} b_shape={:?}",
            a_arr.shape(),
            b_arr.shape()
        );
        let res = std::panic::catch_unwind(|| a_arr.dot(&b_arr).into_dyn());
        match res {
            Ok(r) => *output = r,
            Err(_) => {
                log::error!("MatMul forward: panic during ndarray dot; returning zeros");
                *output = ArrayD::zeros(IxDyn(&[0]));
            }
        }
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a_lock = inputs[0].lock();
        let b_lock = inputs[1].lock();
        let a_owned = a_lock.storage.to_f32_array();
        let b_owned = b_lock.storage.to_f32_array();
        let a: ArrayView2<f32> = match a_owned.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("MatMul backward: left operand is not 2D: {}", e);
                let grad_a = ArrayD::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::zeros(IxDyn(&[0]));
                return vec![grad_a.into_dyn(), grad_b.into_dyn()];
            }
        };
        let b: ArrayView2<f32> = match b_owned.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("MatMul backward: right operand is not 2D: {}", e);
                let grad_a = ArrayD::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::zeros(IxDyn(&[0]));
                return vec![grad_a.into_dyn(), grad_b.into_dyn()];
            }
        };
        let output_grad: ArrayView2<f32> = match output_grad.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("MatMul backward: output_grad is not 2D: {}", e);
                let grad_a = ArrayD::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::zeros(IxDyn(&[0]));
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
                None => {
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
                None => {
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
                    return vec![grad_a.into_dyn(), grad_b];
                }
            };
            return vec![grad_a.into_dyn(), grad_b.into_dyn()];
        }
        #[cfg(any(not(feature = "openblas"), target_os = "windows"))]
        {
            // Fallback to ndarray-based computation for backward on platforms where BLAS may be unstable
            let grad_a = output_grad.dot(&b.t()).into_dyn();
            let grad_b = a.t().dot(&output_grad).into_dyn();
            return vec![grad_a.into_dyn(), grad_b.into_dyn()];
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

/// GELU activation function (approximation using tanh).
pub struct GELU;

impl Operation for GELU {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let a = inputs[0].to_f32_array();
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let out = a.mapv(|x| {
            let u = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + u.tanh())
        });
        *output = out;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let grad = a.mapv(|x| {
            let u = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            let tanh_u = u.tanh();
            let left = 0.5 * (1.0 + tanh_u);
            let right = 0.5
                * x
                * (1.0 - tanh_u * tanh_u)
                * (sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x));
            left + right
        });
        let g = output_grad * &grad;
        vec![g]
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
        let sig = a.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        *output = a * sig;
        // SiLU.forward completed
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = inputs[0].to_f32_array();
        let sig = a.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        // derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
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
            *output = ArrayD::zeros(IxDyn(&[0]));
            return;
        }
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let sh = h * self.scale;
        let sw = w * self.scale;
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, c, sh, sw]));
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
        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, c, h, w]));
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
            // Numerical guard: if sum is zero or non-finite (e.g., all -inf), fall back to uniform distribution
            if !(sum > 0.0f32 && sum.is_finite()) {
                let len = lane.len() as f32;
                for v in lane.iter_mut() {
                    *v = 1.0f32 / len;
                }
                continue;
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
                *v = *v / sum;
            }
        }
        // cast back to f32 ArrayD
        let y = y_f64.mapv(|v| v as f32);
        let perm_opt = permute_to_last(&x, axis).1;
        // grad = y * (grad_out - sum(grad_out * y) along last axis)
        let (p_output_grad, _) = permute_to_last(&output_grad, axis);
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
                *output = ArrayD::zeros(IxDyn(&[]));
                return;
            }
        };

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
        let og_perm = match output_grad.to_shape((nrows, features)) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                log::error!("LayerNorm backward: Reshape og to 2D failed: {}", e);
                let grad_x = ArrayD::zeros(IxDyn(&shape));
                let grad_gamma = ArrayD::zeros(IxDyn(&[features]));
                let grad_beta = ArrayD::zeros(IxDyn(&[features]));
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
        let normalized2 = match normalized.to_shape((nrows, features)) {
            Ok(s) => s,
            Err(e) => {
                log::error!("LayerNorm backward: Reshape normalized 2D failed: {}", e);
                let grad_x = ArrayD::zeros(IxDyn(&shape));
                let grad_gamma = ArrayD::zeros(IxDyn(&[features]));
                let grad_beta = ArrayD::zeros(IxDyn(&[features]));
                return vec![grad_x, grad_gamma, grad_beta];
            }
        };
        let inv2 = match inv_std.to_shape((nrows, 1)) {
            Ok(s) => s,
            Err(e) => {
                log::error!("LayerNorm backward: Reshape inv std 2D failed: {}", e);
                let grad_x = ArrayD::zeros(IxDyn(&shape));
                let grad_gamma = ArrayD::zeros(IxDyn(&[features]));
                let grad_beta = ArrayD::zeros(IxDyn(&[features]));
                return vec![grad_x, grad_gamma, grad_beta];
            }
        };

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
                    *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
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
                *v = *v / sum;
            }
        }
        // compute grad in 2D then reshape back and permute back
        let og = output_grad.iter().next().copied().unwrap_or_else(|| {
            log::error!(
                "SoftmaxCrossEntropy backward: expected scalar output_grad, defaulting to 1.0"
            );
            1.0f32
        });
        let grad_logits_2d = ArrayD::zeros(IxDyn(&[nrows, classes]));
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

impl Operation for NLLLoss {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let log_probs = inputs[0].to_f32_array();
        let targets = inputs[1].to_f32_array();
        if log_probs.ndim() < 1 {
            log::error!(
                "NLLLoss: log_probs must be at least 1D; got shape: {:?}",
                log_probs.shape()
            );
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
            log::error!(
                "NLLLoss: targets shape incompatible; log_probs shape: {:?}, targets shape: {:?}",
                log_probs.shape(),
                targets.shape()
            );
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
            let t_2d = match perm_targets.to_shape((nrows, classes)) {
                Ok(v) => v.to_owned(),
                Err(e) => {
                    log::error!(
                        "SoftmaxCrossEntropyLogits forward: Reshape targets one-hot failed: {}",
                        e
                    );
                    *output = ArrayD::from_elem(IxDyn(&[]), f32::NAN);
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
            *output = ArrayD::zeros(IxDyn(&[0]));
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
            *output = ArrayD::zeros(IxDyn(&[0]));
            return;
        }
        let mut axis_sum = 0usize;
        let mut arrays = Vec::new();
        for input in inputs {
            let arr = input.lock().storage.to_f32_array();
            if arr.ndim() != ndim {
                log::error!("Concat forward: mismatched ndim among inputs");
                *output = ArrayD::zeros(IxDyn(&[0]));
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
                    *output = ArrayD::zeros(IxDyn(&[0]));
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
            let slice_info: SliceInfo<_, IxDyn, IxDyn> = unsafe {
                match SliceInfo::new(slice_elems) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("Concat forward: invalid slice info at position {}: {}", cur, e);
                        panic!("Failed to create slice info for concatenation");
                    }
                }
            };
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
            let slice_info: SliceInfo<_, IxDyn, IxDyn> = unsafe {
                match SliceInfo::new(slice_info_elems) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("Concat backward: invalid slice info: {}", e);
                        // push zeros for this input to preserve shape
                        grads.push(ArrayD::<f32>::zeros(IxDyn(&input_shape)));
                        current_index += input_shape[axis];
                        continue;
                    }
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
            let slice_info: SliceInfo<_, IxDyn, IxDyn> = unsafe {
                match SliceInfo::new(slice_info_elems) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("Stack backward: invalid slice info: {}", e);
                        // push zeros default
                        grads.push(ArrayD::<f32>::zeros(IxDyn(&[0])));
                        continue;
                    }
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
        let a = inputs[0].to_f32_array();
        // For now we only support 2D
        let a2 = match a.clone().into_dimensionality::<ndarray::Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Slice forward: expected 2D input: {}", e);
                *output = ArrayD::zeros(IxDyn(&[]));
                return;
            }
        };
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
                log::error!(
                    "Slice backward: expected 2D input, failed to convert: {}",
                    e
                );
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
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv3D forward: weights are not 5D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
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

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, dout, hout, wout]));
        let mut out5 = match out.view_mut().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv3D forward: output buffer reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
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
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv3D backward: weights are not 5D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
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
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
                return vec![grad_in, grad_w, grad_b];
            }
        };

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, din, hin, win]));
        let mut grad_w = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, kd, kh, kw]));
        let mut grad_b = None;
        if inputs.len() > 2 {
            grad_b = Some(ArrayD::<f32>::zeros(IxDyn(&[cout])));
        }

        let mut grad_in5 = match grad_in.view_mut().into_dimensionality::<ndarray::Ix5>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv3D backward: failed to reshape grad_in to 5D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
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
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
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
                    let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                    let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                    let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
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
        let mut depth_out = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hout, wout]));
        let mut depth_out4 = match depth_out.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("DepthwiseSeparableConv2D forward: failed to convert depth_out to 4D mutable view: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
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
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, hout, wout]));
        let mut out4 = match out.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "ConvTranspose2D forward: failed to convert out to 4D mutable view: {}",
                    e
                );
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
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let depthwise = match depthwise.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: depthwise weights not 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let pointwise = match pointwise.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: pointwise weights not 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
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
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hin, win]));
        let mut grad_depth = ArrayD::<f32>::zeros(IxDyn(&[cin, 1, kh, kw]));
        let mut grad_point = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, 1, 1]));
        let mut grad_bias = Some(ArrayD::<f32>::zeros(IxDyn(&[cout])));

        let mut grad_in4 = match grad_in.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: failed to convert grad_in to 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let mut grad_depth4 = match grad_depth.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: failed to convert grad_depth to 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let mut grad_point4 = match grad_point.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "DepthwiseSeparableConv2D backward: failed to convert grad_point to 4D: {}",
                    e
                );
                return vec![ArrayD::zeros(IxDyn(&[0]))];
            }
        };
        let mut grad_bias_view = match grad_bias.as_mut() {
            Some(x) => {
                match x.view_mut().into_dimensionality::<ndarray::Ix1>() {
                    Ok(v) => Some(v),
                    Err(e) => {
                        log::error!("DepthwiseSeparableConv2D backward: failed to convert grad_bias to 1D: {}", e);
                        return vec![ArrayD::zeros(IxDyn(&[0]))];
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
        let mut grad_depth_out = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hout, wout]));
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
                return vec![ArrayD::zeros(IxDyn(&[0]))];
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
        let mut depth_out = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hout, wout]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("ConvTranspose2D forward: weights not 4D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
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

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, hout, wout]));
        let mut out4 = match out.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "ConvTranspose2D forward: failed to convert out to 4D mutable view: {}",
                    e
                );
                *output = ArrayD::zeros(IxDyn(&[0]));
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

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, hin, win]));
        let mut grad_w = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, kh, kw]));
        let mut grad_b = None;
        if inputs.len() > 2 {
            grad_b = Some(ArrayD::<f32>::zeros(IxDyn(&[cout])));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv1D forward: weights are not 3D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
        let (n, cin, lin) = input.dim();
        let (cout, cin2, kl) = w.dim();
        assert_eq!(cin, cin2, "Conv1D: input channel mismatch with weight");

        let stride = self.stride as isize;
        let pad = self.padding as isize;
        let lout = ((lin as isize - kl as isize + 2 * pad) / stride + 1) as usize;

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, lout]));
        let mut out3 = match out.view_mut().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv1D forward: output buffer reshape failed: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };

        for batch in 0..n {
            for oc in 0..cout {
                for ol in 0..lout {
                    let mut sum = 0.0f32;
                    for ic in 0..cin {
                        for kl_i in 0..kl {
                            let il = ol as isize * stride + kl_i as isize - pad;
                            if il >= 0 && il < lin as isize {
                                let iv = input[[batch, ic, il as usize]];
                                let wv = w[[oc, ic, kl_i]];
                                sum += iv * wv;
                            }
                        }
                    }
                    if let Some(ref b) = bias_opt {
                        sum += b[[oc]];
                    }
                    out3[[batch, oc, ol]] = sum;
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
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
                return vec![grad_in, grad_w, grad_b];
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv1D backward: weights are not 3D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
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
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
                return vec![grad_in, grad_w, grad_b];
            }
        };

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, lin]));
        let mut grad_w = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, kl]));
        let mut grad_b = None;
        if inputs.len() > 2 {
            grad_b = Some(ArrayD::<f32>::zeros(IxDyn(&[cout])));
        }

        let mut grad_in3 = match grad_in.view_mut().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Conv1D backward: failed to reshape grad_in to 3D: {}", e);
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
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
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
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
                    let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                    let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                    let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
        let w = match weights.view().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("ConvTranspose1D forward: weights not 3D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
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

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, cout, lout]));
        let mut out3 = match out.view_mut().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "ConvTranspose1D forward: output buffer reshape failed: {}",
                    e
                );
                *output = ArrayD::zeros(IxDyn(&[0]));
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

        let mut grad_in = ArrayD::<f32>::zeros(IxDyn(&[n, cin, lin]));
        let mut grad_w = ArrayD::<f32>::zeros(IxDyn(&[cout, cin, kl]));
        let mut grad_b = None;
        if inputs.len() > 2 {
            grad_b = Some(ArrayD::<f32>::zeros(IxDyn(&[cout])));
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
        let mut grad_w4 = match grad_w.view_mut().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "Conv2D backward: failed to convert grad_w to 4D mutable view: {}",
                    e
                );
                let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
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
                    let grad_in = ArrayD::<f32>::zeros(IxDyn(&[0]));
                    let grad_w = ArrayD::<f32>::zeros(IxDyn(&[0]));
                    let grad_b = ArrayD::<f32>::zeros(IxDyn(&[0]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
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
        let mut grad_in = ArrayD::zeros(IxDyn(&[batch, c, h, w]));
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
    let end = ((idx + 1) * in_size + out_size - 1) / out_size; // ceil
    (start, end)
}

impl Operation for AdaptiveAvgPool2D {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let input = inputs[0].to_f32_array();
        let input_view = match input.view().into_dimensionality::<ndarray::Ix4>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("AdaptiveAvgPool2D forward: input must be 4D: {}", e);
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
        let (batch, c, h, w) = input_view.dim();
        let oh = self.out_h;
        let ow = self.out_w;
        let mut out = ArrayD::zeros(IxDyn(&[batch, c, oh, ow]));
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
        let mut grad_in = ArrayD::zeros(IxDyn(&[batch, c, h, w]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
                return;
            }
        };
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
        let mut grad_in = ArrayD::zeros(IxDyn(&[batch, c, h, w]));
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
                *output = ArrayD::zeros(IxDyn(&[]));
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

impl Operation for SwiGLU {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        let x = &inputs[0].lock().storage.to_f32_array();
        let ndim = x.ndim();
        let last = ndim - 1;
        let d = x.shape()[last];
        if d % 2 != 0 {
            log::error!("SwiGLU forward: last dim {} not divisible by 2", d);
            *output = ArrayD::from_elem(IxDyn(&[0]), f32::NAN);
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
        let swish = right.mapv(|v| v * (1.0 / (1.0 + (-v).exp())));
        let out_arr = left * swish;
        *output = out_arr.into_dyn();
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let x = &inputs[0].lock().storage.to_f32_array();
        let ndim = x.ndim();
        let last = ndim - 1;
        let d = x.shape()[last];
        if d % 2 != 0 {
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
        let sigmoid = right.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        // swish' = sigmoid + right * sigmoid * (1 - sigmoid)
        let swish_prime = &sigmoid + &((&right * &sigmoid) * (&sigmoid.mapv(|s| 1.0 - s)));
        // grad_left = grad_out * swish
        let go = output_grad.clone();
        let go_view = go.view();
        let go_left = go_view
            .slice_axis(Axis(last), ndarray::Slice::from(..half))
            .to_owned();
        let _go_right = go_view
            .slice_axis(Axis(last), ndarray::Slice::from(half..))
            .to_owned();
        // Actually left -> out = left * swish(right) => dLoss/dleft = grad_out * swish
        let swish = &right * &sigmoid;
        let grad_left = &go_left * &swish;
        // grad_right = grad_out * left * swish'
        let grad_right = &go_left * &left * &swish_prime;
        // Re-concatenate left and right
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
}

impl RoPE {
    pub fn new(num_heads: usize, theta: f32) -> Self {
        RoPE { num_heads, theta }
    }
}

impl Operation for RoPE {
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // inputs: x (shape [*, d_model]) - last dim should be divisible by num_heads
        let x = inputs[0].lock().storage.to_f32_array();
        let ndim = x.ndim();
        let last = ndim - 1;
        let d = x.shape()[last];
        if d % self.num_heads != 0 {
            log::error!(
                "RoPE: last dim {} not divisible by num_heads {}",
                d,
                self.num_heads
            );
            *output = x.clone();
            return;
        }
        let head_dim = d / self.num_heads;
        if head_dim % 2 != 0 {
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
        let seq_axis = if new_shape.len() >= 2 {
            new_shape.len() - 2
        } else {
            0
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
        let mut sin = ndarray::Array2::<f32>::zeros((seq_len, pair));
        let mut cos = ndarray::Array2::<f32>::zeros((seq_len, pair));
        for pos in 0..seq_len {
            for (i, &f) in inv_freq.iter().enumerate() {
                let v = pos as f32 * f;
                sin[[pos, i]] = v.sin();
                cos[[pos, i]] = v.cos();
            }
        }
        // build full sin/cos along head_dim by repeating pairs
        // sin_full shape [seq_len, head_dim]
        let mut sin_full = ndarray::Array2::<f32>::zeros((seq_len, head_dim));
        let mut cos_full = ndarray::Array2::<f32>::zeros((seq_len, head_dim));
        for pos in 0..seq_len {
            for i in 0..pair {
                sin_full[[pos, 2 * i]] = sin[[pos, i]];
                sin_full[[pos, 2 * i + 1]] = sin[[pos, i]];
                cos_full[[pos, 2 * i]] = cos[[pos, i]];
                cos_full[[pos, 2 * i + 1]] = cos[[pos, i]];
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
                for pair_i in 0..pair {
                    let idx_even = 2 * pair_i;
                    let idx_odd = idx_even + 1;
                    // construct full index
                    let mut base_even = prefix_indices.clone();
                    base_even.push(h);
                    base_even.push(idx_even);
                    let mut base_odd = prefix_indices.clone();
                    base_odd.push(h);
                    base_odd.push(idx_odd);
                    let even_val = in_view[IxDyn(&base_even)];
                    let odd_val = in_view[IxDyn(&base_odd)];
                    let cosv = cos_full[[pos, pair_i]];
                    let sinv = sin_full[[pos, pair_i]];
                    // rotated vals
                    let out_even = even_val * cosv - odd_val * sinv;
                    let out_odd = even_val * sinv + odd_val * cosv;
                    out_view[IxDyn(&base_even)] = out_even;
                    out_view[IxDyn(&base_odd)] = out_odd;
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
        if d % self.num_heads != 0 {
            return vec![og];
        }
        let head_dim = d / self.num_heads;
        if head_dim % 2 != 0 {
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
        let mut sin = ndarray::Array2::<f32>::zeros((seq_len, pair));
        let mut cos = ndarray::Array2::<f32>::zeros((seq_len, pair));
        for pos in 0..seq_len {
            for (i, &f) in inv_freq.iter().enumerate() {
                let v = pos as f32 * f;
                sin[[pos, i]] = v.sin();
                cos[[pos, i]] = v.cos();
            }
        }
        let mut sin_full = ndarray::Array2::<f32>::zeros((seq_len, head_dim));
        let mut cos_full = ndarray::Array2::<f32>::zeros((seq_len, head_dim));
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

impl EmbeddingLookup {
    pub fn new() -> Self {
        EmbeddingLookup
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
                *output = ArrayD::zeros(IxDyn(&[0]));
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
            // let row = emb2.row(id).to_owned().into_dyn(); // unused
            // compute multi index from i and place row
            // no-op: compute coordinates directly
            // let idx_count = idx_shape.iter().product::<usize>(); // unused
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
        let mut grad_emb = ArrayD::<f32>::zeros(IxDyn(&[vocab, dim]));
        let indices = inputs[1].lock().storage.to_f32_array();
        // iterate over output_grad and accumulate
        let idx_shape = indices.shape().to_vec();
        // let idx_count = idx_shape.iter().product::<usize>(); // unused
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
            *output = ArrayD::zeros(IxDyn(&[0]));
            return;
        }
        let ndim = a.ndim();
        if axis >= ndim {
            log::error!(
                "KVCacheAppend forward: axis {} out of bounds for ndim {}",
                axis,
                ndim
            );
            *output = ArrayD::zeros(IxDyn(&[0]));
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
                *output = ArrayD::zeros(IxDyn(&[0]));
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
            let slice_info: SliceInfo<_, IxDyn, IxDyn> = unsafe {
                match SliceInfo::new(slice_elems) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("KVCacheAppend: invalid slice info for tensor a: {}", e);
                        panic!("Failed to create slice info for KVCacheAppend first tensor");
                    }
                }
            };
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
            let slice_info: SliceInfo<_, IxDyn, IxDyn> = unsafe {
                match SliceInfo::new(slice_elems) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("KVCacheAppend: invalid slice info for tensor b: {}", e);
                        panic!("Failed to create slice info for KVCacheAppend second tensor");
                    }
                }
            };
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
        let a_slice_info: SliceInfo<_, IxDyn, IxDyn> = unsafe {
            match SliceInfo::new(a_slice_elems) {
                Ok(info) => info,
                Err(e) => {
                    log::error!(
                        "KVCacheAppend backward: failed to create slice info for a slice: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[0])), ArrayD::zeros(IxDyn(&[0]))];
                }
            }
        };
        let b_slice_info: SliceInfo<_, IxDyn, IxDyn> = unsafe {
            match SliceInfo::new(b_slice_elems) {
                Ok(info) => info,
                Err(e) => {
                    log::error!(
                        "KVCacheAppend backward: failed to create slice info for b slice: {}",
                        e
                    );
                    return vec![ArrayD::zeros(IxDyn(&[0])), ArrayD::zeros(IxDyn(&[0]))];
                }
            }
        };
        let grad_a = output_grad.slice(a_slice_info).to_owned().into_dyn();
        let grad_b = output_grad.slice(b_slice_info).to_owned().into_dyn();
        vec![grad_a, grad_b]
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
        a[[0, 0]] = 1.0; a[[0, 1]] = 2.0; a[[0, 2]] = 3.0; a[[0, 3]] = 4.0;
        for j in 0..4 { a[[1, j]] = f32::NEG_INFINITY; }
        let t = Tensor::new(a, false);
        let op = Softmax::new(1);
        let mut out = ndarray::ArrayD::<f32>::zeros(IxDyn(&[2, 4]));
        op.forward(&[t.clone()], &mut out);
        // first row should be softmax of [1,2,3,4]
        let out0 = out.index_axis(ndarray::Axis(0), 0).to_owned();
        let mut sum0 = 0.0f32;
        for v in out0.iter() { sum0 += *v; }
        assert!(sum0 > 0.99 && sum0 < 1.01);
        // second row turned into uniform distribution
        let out1 = out.index_axis(ndarray::Axis(0), 1).to_owned();
        for v in out1.iter() { assert!((*v - 0.25).abs() < 1e-6); }
    }
}
