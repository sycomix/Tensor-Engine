use crate::nn::linear_dispatch::LinearLayer;
use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::IxDyn;
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

        let data_slice = arr.as_slice().expect("moe: non-contiguous");
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

        let flat_x = x.reshape(vec![batch_seq, self.d_model]).expect("failed");
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
                .expect("failed"),
                false,
            );

            let sub_x = Tensor::embedding_lookup(&flat_x, &indices_t);

            // Expert forward
            let expert_out = self.experts_struct[e_idx].forward(&sub_x);

            // Apply routing weights: expert_out * weights
            let weights_t = Tensor::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[subset_size, 1]), weights_vec)
                    .expect("failed"),
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
                    .expect("failed"),
                false, // Constant structure
            );

            let scattered = p_mat.matmul(&weighted_out);
            final_out = final_out.add(&scattered);
        }

        final_out.reshape(shape.to_vec()).expect("failed")
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

/// DeepSeek-style MoE Layer with grouped experts.
/// Uses grouped-query attention pattern for experts:
/// - Experts are divided into groups
/// - Each group shares a single router
/// - Reduces routing computation for large expert counts
#[derive(Clone)]
pub struct DeepSeekMoELayer {
    pub gate: LinearLayer,
    pub experts_struct: Vec<Expert>,
    pub num_experts: usize,
    pub k: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub num_groups: usize,
    pub experts_per_group: usize,
    pub shared_experts: Option<LinearLayer>,
    pub shared_experts_ff: Option<LinearLayer>,
}

impl DeepSeekMoELayer {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        num_experts: usize,
        k: usize,
        num_groups: usize,
        has_shared_experts: bool,
    ) -> Self {
        let gate = LinearLayer::new_f32(d_model, num_experts, false);
        let mut experts_struct = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts_struct.push(Expert::new(d_model, d_ff, false));
        }

        let shared_experts = if has_shared_experts {
            Some(LinearLayer::new_f32(d_model, d_model, false))
        } else {
            None
        };
        let shared_experts_ff = if has_shared_experts {
            Some(LinearLayer::new_f32(d_model, d_model, false))
        } else {
            None
        };

        Self {
            gate,
            experts_struct,
            num_experts,
            k,
            d_model,
            d_ff,
            num_groups,
            experts_per_group: num_experts / num_groups,
            shared_experts,
            shared_experts_ff,
        }
    }
}

impl Module for DeepSeekMoELayer {
    fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.lock().storage.shape();
        let batch_seq = shape[0] * shape[1];

        // 1. Gating
        let logits = self.gate.forward(x);

        // 2. Grouped routing: select top-k experts per group
        let topk_out = logits.topk(self.k);

        let out_lock = topk_out.lock();
        let arr = out_lock.storage.to_f32_array();
        let shape_out = arr.shape();
        let dim_k = shape_out[2];

        let mut expert_map: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.num_experts];

        let data_slice = arr.as_slice().expect("moe: non-contiguous");
        let stride_row = dim_k;

        for i in 0..batch_seq {
            let row_offset = i * stride_row;
            let mut raw_weights = Vec::with_capacity(self.k);
            let mut indices = Vec::with_capacity(self.k);

            for j in 0..self.k {
                raw_weights.push(data_slice[row_offset + j]);
                indices.push(data_slice[row_offset + self.k + j] as usize);
            }

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

        drop(out_lock);

        let flat_x = x.reshape(vec![batch_seq, self.d_model]).expect("failed");
        let mut final_out = Tensor::zeros(vec![batch_seq, self.d_model].as_slice());

        for (e_idx, items) in expert_map.into_iter().enumerate() {
            if items.is_empty() {
                continue;
            }

            let (indices_vec, weights_vec): (Vec<usize>, Vec<f32>) = items.into_iter().unzip();
            let subset_size = indices_vec.len();

            let indices_t = Tensor::new(
                ndarray::ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[subset_size]),
                    indices_vec.iter().map(|&i| i as f32).collect(),
                )
                .expect("failed"),
                false,
            );

            let sub_x = Tensor::embedding_lookup(&flat_x, &indices_t);
            let expert_out = self.experts_struct[e_idx].forward(&sub_x);

            let weights_t = Tensor::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[subset_size, 1]), weights_vec)
                    .expect("failed"),
                true,
            );

            let weighted_out = expert_out.mul(&weights_t);

            let mut p_data = vec![0.0f32; batch_seq * subset_size];
            for (j, &global_i) in indices_vec.iter().enumerate() {
                p_data[global_i * subset_size + j] = 1.0;
            }

            let p_mat = Tensor::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[batch_seq, subset_size]), p_data)
                    .expect("failed"),
                false,
            );

            let scattered = p_mat.matmul(&weighted_out);
            final_out = final_out.add(&scattered);
        }

        // Add shared expert output if present
        if let (Some(shared_gate), Some(shared_ff)) =
            (&self.shared_experts, &self.shared_experts_ff)
        {
            let shared_out = shared_gate.forward(x);
            let shared_out = shared_ff.forward(&shared_out.sigmoid());
            final_out = final_out.add(&shared_out);
        }

        final_out.reshape(shape.to_vec()).expect("failed")
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.gate.parameters();
        for e in &self.experts_struct {
            p.extend(e.parameters());
        }
        if let Some(s) = &self.shared_experts {
            p.extend(s.parameters());
        }
        if let Some(s) = &self.shared_experts_ff {
            p.extend(s.parameters());
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
        if let Some(s) = &self.shared_experts {
            out.extend(s.named_parameters(&format!("{}.shared_experts", prefix)));
        }
        if let Some(s) = &self.shared_experts_ff {
            out.extend(s.named_parameters(&format!("{}.shared_experts_ff", prefix)));
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
        if let Some(s) = &mut self.shared_experts {
            s.load_state_dict(state, &format!("{}.shared_experts", prefix))?;
        }
        if let Some(s) = &mut self.shared_experts_ff {
            s.load_state_dict(state, &format!("{}.shared_experts_ff", prefix))?;
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

/// DeepSeek-style MoE architecture wrapper.
/// Combines DeepSeekMoELayer blocks with standard transformer components.
#[derive(Clone)]
pub struct DeepSeekMoE {
    pub embed_tokens: Tensor,
    pub layers: Vec<DeepSeekMoELayer>,
    pub norm: Tensor,
    pub lm_head: LinearLayer,
    pub vocab_size: usize,
}

impl DeepSeekMoE {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        d_ff: usize,
        num_experts: usize,
        k: usize,
        num_groups: usize,
        has_shared_experts: bool,
    ) -> Result<Self, String> {
        let embed_tokens = Tensor::new(
            ndarray::Array::zeros(IxDyn(&[vocab_size, d_model][..])),
            true,
        );
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(DeepSeekMoELayer::new(
                d_model,
                d_ff,
                num_experts,
                k,
                num_groups,
                has_shared_experts,
            ));
        }
        let norm = Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[d_model][..]), 1.0f32),
            true,
        );
        let lm_head = LinearLayer::new_f32(d_model, vocab_size, false);
        Ok(DeepSeekMoE {
            embed_tokens,
            layers,
            norm,
            lm_head,
            vocab_size,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = Tensor::embedding_lookup(&self.embed_tokens, input);
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x = x.rmsnorm(&self.norm, 2, 1e-5);
        self.lm_head.forward(&x)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.embed_tokens.clone(), self.norm.clone()];
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.lm_head.parameters());
        p
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = vec![
            (
                format!("{}.model.embed_tokens.weight", prefix),
                self.embed_tokens.clone(),
            ),
            (format!("{}.model.norm.weight", prefix), self.norm.clone()),
        ];
        for (i, layer) in self.layers.iter().enumerate() {
            out.extend(layer.named_parameters(&format!("{}.model.layers.{}", prefix, i)));
        }
        out.extend(
            self.lm_head
                .named_parameters(&format!("{}.lm_head", prefix)),
        );
        out
    }

    pub fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        let embed_key = format!("{}.model.embed_tokens.weight", prefix);
        if let Some(t) = state.get(&embed_key) {
            self.embed_tokens = t.clone();
        }
        let norm_key = format!("{}.model.norm.weight", prefix);
        if let Some(t) = state.get(&norm_key) {
            self.norm = t.clone();
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(state, &format!("{}.model.layers.{}", prefix, i))?;
        }
        let lm_key = format!("{}.lm_head.weight", prefix);
        if let Some(lh) = state.get(&lm_key) {
            if let Some(lh_layer) = self.lm_head.as_f32_mut() {
                lh_layer.weight = lh.clone();
            }
        }
        Ok(())
    }
}

impl Module for DeepSeekMoE {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.named_parameters(prefix)
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.load_state_dict(state, prefix)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
