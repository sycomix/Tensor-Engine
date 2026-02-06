use crate::nn::linear_dispatch::LinearLayer;
use crate::nn::Module;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Sparse Mixture of Experts (SMoE) Layer.
#[derive(Clone)]
pub struct MoELayer {
    pub gate: LinearLayer,
    pub experts_struct: Vec<Expert>,
    pub num_experts: usize,
    pub k: usize,
    pub d_model: usize,
    pub d_ff: usize,
}

#[derive(Clone)]
pub struct Expert {
    pub w1: LinearLayer, // gate_proj
    pub w2: LinearLayer, // down_proj
    pub w3: LinearLayer, // up_proj
}

impl Expert {
    pub fn new(d_model: usize, d_ff: usize, bias: bool) -> Self {
        Self {
            w1: LinearLayer::new_f32(d_model, d_ff, bias),
            w2: LinearLayer::new_f32(d_ff, d_model, bias),
            w3: LinearLayer::new_f32(d_model, d_ff, bias),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // SwiGLU: (silu(w1(x)) * w3(x)) -> w2
        let g = self.w1.forward(x);
        let u = self.w3.forward(x);
        let hidden = g.silu().mul(&u);
        self.w2.forward(&hidden)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.w1.parameters();
        p.extend(self.w2.parameters());
        p.extend(self.w3.parameters());
        p
    }
}

impl MoELayer {
    pub fn new(d_model: usize, d_ff: usize, num_experts: usize, k: usize) -> Self {
        let gate = LinearLayer::new_f32(d_model, num_experts, false);
        let mut experts_struct = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts_struct.push(Expert::new(d_model, d_ff, false));
        }

        Self {
            gate,
            experts_struct,
            num_experts,
            k,
            d_model,
            d_ff,
        }
    }
}

impl Module for MoELayer {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [batch, seq, d_model]
        let shape = x.lock().storage.shape();
        let batch_seq = shape[0] * shape[1];

        // 1. Gating
        let logits = self.gate.forward(x); // [batch, seq, num_experts]

        // 2. Routing: Top-K
        let topk_out = logits.topk(self.k); // [batch, seq, 2*k]

        let out_lock = topk_out.lock();
        let arr = out_lock.storage.to_f32_array();
        let shape_out = arr.shape();
        let dim_k = shape_out[2];

        // Map: expert_idx -> Vec<(flat_token_idx, weight)>
        let mut expert_map: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.num_experts];

        let data_slice = arr.as_slice().unwrap();
        let stride_row = dim_k;

        for i in 0..batch_seq {
            let row_offset = i * stride_row;
            let mut raw_weights = Vec::with_capacity(self.k);
            let mut indices = Vec::with_capacity(self.k);

            for j in 0..self.k {
                raw_weights.push(data_slice[row_offset + j]);
                indices.push(data_slice[row_offset + self.k + j] as usize);
            }

            // Softmax weights
            let max_w: f32 = raw_weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;
            let mut exps = Vec::with_capacity(self.k);
            for &w in &raw_weights {
                let e = (w - max_w).exp();
                exps.push(e);
                sum_exp += e;
            }

            for j in 0..self.k {
                let normalized_w = exps[j] / sum_exp;
                let expert_idx = indices[j];
                if expert_idx < self.num_experts {
                    expert_map[expert_idx].push((i, normalized_w));
                }
            }
        }

        drop(out_lock); // Release lock

        let flat_x = x.reshape(vec![batch_seq, self.d_model]).unwrap();
        let mut final_out = Tensor::zeros(vec![batch_seq, self.d_model].as_slice());

        for (e_idx, items) in expert_map.into_iter().enumerate() {
            if items.is_empty() {
                continue;
            }

            let (indices_vec, weights_vec): (Vec<usize>, Vec<f32>) = items.into_iter().unzip();
            let subset_size = indices_vec.len();

            // Gather input
            let indices_t = Tensor::new(
                ndarray::ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[subset_size]),
                    indices_vec.iter().map(|&i| i as f32).collect(),
                )
                .unwrap(),
                false,
            );

            let sub_x = Tensor::embedding_lookup(&flat_x, &indices_t);

            // Expert forward
            let expert_out = self.experts_struct[e_idx].forward(&sub_x);

            // Apply routing weights: expert_out * weights
            let weights_t = Tensor::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[subset_size, 1]), weights_vec)
                    .unwrap(),
                true,
            );

            let weighted_out = expert_out.mul(&weights_t);

            // Scatter Add using MatMul strategy
            // OneHot [subset, batch_seq] @ weighted_out [subset, d]?? No.
            // weighted_out is [subset, d]. target is [batch_seq, d].
            // We want [batch_seq, subset] @ [subset, d].
            // OneHot matrix P where P[i, j] = 1 if token j in subset corresponds to global token i.
            // j goes 0..subset. i = indices_vec[j].
            // So P[indices_vec[j], j] = 1.

            let mut p_data = vec![0.0f32; batch_seq * subset_size];
            for (j, &global_i) in indices_vec.iter().enumerate() {
                // Row global_i, Col j
                p_data[global_i * subset_size + j] = 1.0;
            }

            let p_mat = Tensor::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[batch_seq, subset_size]), p_data)
                    .unwrap(),
                false, // Constant structure
            );

            let scattered = p_mat.matmul(&weighted_out);
            final_out = final_out.add(&scattered);
        }

        final_out.reshape(shape.to_vec()).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.gate.parameters();
        for e in &self.experts_struct {
            p.extend(e.parameters());
        }
        p
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = self.gate.named_parameters(&format!("{}.gate", prefix));
        for (i, e) in self.experts_struct.iter().enumerate() {
            let mut p =
                e.w1.named_parameters(&format!("{}.experts.{}.w1", prefix, i));
            p.extend(e.w2.named_parameters(&format!("{}.experts.{}.w2", prefix, i)));
            p.extend(e.w3.named_parameters(&format!("{}.experts.{}.w3", prefix, i)));
            out.extend(p);
        }
        out
    }
    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.gate
            .load_state_dict(state, &format!("{}.gate", prefix))?;
        for (i, e) in self.experts_struct.iter_mut().enumerate() {
            e.w1.load_state_dict(state, &format!("{}.experts.{}.w1", prefix, i))?;
            e.w2.load_state_dict(state, &format!("{}.experts.{}.w2", prefix, i))?;
            e.w3.load_state_dict(state, &format!("{}.experts.{}.w3", prefix, i))?;
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
