use crate::nn::linear_dispatch::LinearLayer;
use crate::nn::transformer::{BiasFunction, TransformerBlock, TransformerConfig};
use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};
use std::any::Any;
use std::collections::HashMap;

/// LoopedTransformer: weight-tied application of a single TransformerBlock for T steps
/// Returns per-step hidden states and a learned gate distribution p_phi(t|x).
#[derive(Clone)]
pub struct LoopedTransformer {
    pub block: TransformerBlock,
    pub gate: LinearLayer, // projects pooled hidden -> T logits
    pub t_max: usize,
    pub beta: f32,
}

impl LoopedTransformer {
    /// Create a new LoopedTransformer. If `nl_oob_config` is Some, the inner block
    /// will be created with NL-OOB using the provided max scale.
    pub fn new_with_nl_oob(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        nl_oob_config: Option<BiasFunction>,
        nl_oob_max_scale: Option<f32>,
        t_max: usize,
        beta: f32,
    ) -> Result<Self, String> {
        let block = if let Some(cfg) = nl_oob_config {
            TransformerBlock::new_with_nl_oob(
                d_model,
                d_ff,
                num_heads,
                cfg,
                nl_oob_max_scale.unwrap_or(2.0),
            )?
        } else {
            TransformerBlock::new_with_kv_and_rope(TransformerConfig {
                d_model,
                d_ff,
                num_heads,
                kv_heads: num_heads,
                use_rope: false,
                rope_theta: 10000.0,
                rope_scale: 1.0,
                bias: true,
            })?
        };
        let gate = LinearLayer::new_f32(d_model, t_max, true);
        Ok(LoopedTransformer {
            block,
            gate,
            t_max,
            beta,
        })
    }

    /// Forward the block T times (weight-tied). Returns (per_step_hidden_states, p_phi)
    /// - `x` shape: [B, S, D]
    /// - returns Vec of length `t_max`, each Tensor is [B, S, D]
    /// - p_phi shape: [B, t_max]
    pub fn forward_looped(&self, x: &Tensor, dist: Option<&Tensor>) -> (Vec<Tensor>, Tensor) {
        let mut outs: Vec<Tensor> = Vec::with_capacity(self.t_max);
        let mut cur = x.clone();
        for _ in 0..self.t_max {
            let out = if let Some(d) = dist {
                self.block.forward_block_with_distance(&cur, d)
            } else {
                // use non-mutating variant
                self.block.forward_block_no_cache(&cur)
            };
            // debug: ensure out has expected 3D shape
            log::debug!(
                "LoopedTransformer: pushing out shape={:?}",
                out.lock().storage.shape()
            );
            outs.push(out.clone());
            cur = out;
        }

        // pooled representation: use last-token pooling via permute+matmul selector so autograd is preserved
        let shape = cur.lock().storage.shape().to_vec();
        if shape.len() != 3 {
            // fallback: return uniform p_phi
            let b = 1usize;
            let uniform = Tensor::new(
                ndarray::Array::from_elem(
                    IxDyn(&[b, self.t_max][..]),
                    1.0f32 / (self.t_max as f32),
                ),
                false,
            );
            return (outs, uniform);
        }
        let b = shape[0];
        let seq = shape[1];
        let d = shape[2];

        // selector: shape [seq, 1] with 1.0 at last index
        let mut sel_vec = vec![0.0f32; seq * 1];
        if seq > 0 {
            sel_vec[(seq - 1) * 1] = 1.0f32;
        }
        let sel_arr = Array::from_shape_vec((seq, 1), sel_vec).unwrap().into_dyn();
        let sel = Tensor::new(sel_arr, false);

        // pooled = reshape( permute(cur, [0,2,1]) @ sel ) -> [b, d]
        // If the fast path fails fall back to a reliable per-sequence mean so the gate
        // always receives a 2-D tensor shaped [batch, d_model]. Never return a 3-D
        // tensor as "pooled" (that would make the gate run per-token).
        let pooled = match cur.permute(vec![0, 2, 1]).matmul(&sel).reshape(vec![b, d]) {
            Ok(p) => p,
            Err(_) => {
                // fallback: compute mean across the sequence axis using a normalized ones vector
                let ones_arr =
                    ndarray::Array::from_elem(IxDyn(&[seq, 1][..]), 1.0f32 / (seq as f32));
                let ones = Tensor::new(ones_arr, false);
                match cur.permute(vec![0, 2, 1]).matmul(&ones).reshape(vec![b, d]) {
                    Ok(m) => m,
                    Err(_) => {
                        // final fallback: zeros [b, d] to avoid passing a 3-D tensor into the gate
                        Tensor::new(ndarray::Array::zeros(IxDyn(&[b, d][..])), false)
                    }
                }
            }
        };

        // gate logits -> softmax across t dimension
        let gate_logits = self.gate.forward(&pooled); // [B, t_max]
        let p_phi = gate_logits.softmax(1);
        (outs, p_phi)
    }

    /// Stage‑II gate objective (paper Eq.6 POC):
    /// Computes mean_{batch}[ E_{t~p_phi}[ sum_{i<=t} step_loss_i ] - beta * H(p_phi) ]
    /// - `p_phi`: shape [B, T]
    /// - `step_losses`: shape [B, T] (per-step scalar losses)
    /// Returns a scalar Tensor (mean over batch). Uses a small eps for log stability.
    pub fn stage2_loss(&self, p_phi: &Tensor, step_losses: &Tensor) -> Tensor {
        // quick shape checks; on mismatch return zero scalar to avoid panics in callers/tests
        let sp = p_phi.lock().storage.shape().to_vec();
        let sl = step_losses.lock().storage.shape().to_vec();
        if sp.len() != 2
            || sl.len() != 2
            || sp[1] != self.t_max
            || sl[1] != self.t_max
            || sp[0] != sl[0]
        {
            return Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1usize][..]), 0.0f32),
                false,
            );
        }
        let b = sp[0];
        let t = self.t_max;

        // lower-triangular ones matrix M where M[row=k, col=i]=1 iff k>=i (so
        // (p @ M)[i] = sum_{k>=i} p_k). This yields the tail/survival sums we need.
        let mut tri = ndarray::Array2::<f32>::zeros((t, t));
        for i in 0..t {
            for j in 0..=i {
                tri[[i, j]] = 1.0f32;
            }
        }
        let tri_t = Tensor::new(tri.into_dyn(), false);

        // survival[b,i] = sum_{t>=i} p_phi[b,t]  -> survival: [B, T]
        let survival = p_phi.matmul(&tri_t);

        // expected per-batch: sum_i survival[b,i] * step_losses[b,i]
        let ones_col = Tensor::new(ndarray::Array::from_elem(IxDyn(&[t, 1][..]), 1.0f32), false);
        let expected_vec = survival
            .mul(step_losses)
            .matmul(&ones_col)
            .reshape(vec![b])
            .unwrap(); // [B]

        // entropy per-batch: -sum_t p * log(p + eps)
        let eps = 1e-12f32;
        let eps_t = Tensor::new(ndarray::Array::from_elem(IxDyn(&[1, 1][..]), eps), false);
        let p_safe = p_phi.add(&eps_t);
        let entropy_vec = p_phi
            .mul(&p_safe.log())
            .matmul(&ones_col)
            .reshape(vec![b])
            .unwrap()
            .mul(&Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1][..]), -1.0f32),
                false,
            ));

        // mean over batch and combine with beta (we subtract beta * entropy to encourage higher entropy)
        let mean_expected = expected_vec.mean();
        let mean_entropy = entropy_vec.mean();
        mean_expected.sub(&mean_entropy.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1][..]), self.beta),
            false,
        )))
    }
}

impl Module for LoopedTransformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (outs, _) = self.forward_looped(input, None);
        if outs.is_empty() {
            input.clone()
        } else {
            outs.last().unwrap().clone()
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut out = self.block.parameters();
        out.extend(self.gate.parameters());
        out
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = self.block.named_parameters(&format!("{}.block", prefix));
        out.extend(self.gate.named_parameters(&format!("{}.gate", prefix)));
        out
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.block
            .load_state_dict(state, &format!("{}.block", prefix))?;
        self.gate
            .load_state_dict(state, &format!("{}.gate", prefix))?;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
