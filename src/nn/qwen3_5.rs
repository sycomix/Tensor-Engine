use ndarray::{Array, Array1, ArrayD, IxDyn};
use std::collections::HashMap;

use crate::nn::linear_dispatch::LinearLayer;
use crate::nn::Module;
use crate::tensor::Tensor;

// ─── Softplus (element-wise) ────────────────────────────────────────────────
fn softplus(x: &Tensor) -> Tensor {
    // softplus(x) = ln(1 + exp(x)), with safe handling for extreme values
    let x_arr = x.to_f32_array();
    let out = x_arr.mapv(|v| {
        if v > 20.0 {
            v
        } else if v < -20.0 {
            v.exp()
        } else {
            (1.0 + v.exp()).ln()
        }
    });
    Tensor::new(out.into_dyn(), false)
}

// ─── RMSNormGated ───────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct Qwen3_5RMSNormGated {
    pub weight: Tensor,
    pub eps: f32,
}

impl Qwen3_5RMSNormGated {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::new(Array::from_elem(IxDyn(&[dim]), 1.0f32), true),
            eps,
        }
    }

    /// x: [*, dim], gate: [*, dim]  (same last dim)
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Tensor {
        let x_arr = x.to_f32_array();
        let gate_arr = gate.to_f32_array();
        let var = (&x_arr * &x_arr)
            .mean_axis(ndarray::Axis(x_arr.ndim() - 1))
            .unwrap();
        let rstd = var.mapv(|v| 1.0 / (v + self.eps).sqrt());
        let mut shape = vec![1; x_arr.ndim()];
        shape[x_arr.ndim() - 1] = x_arr.shape()[x_arr.ndim() - 1];
        let w_arr = self
            .weight
            .to_f32_array()
            .into_shape_with_order(shape)
            .unwrap();
        let rstd_view = rstd
            .into_shape_with_order({
                let mut s = x_arr.shape().to_vec();
                s[x_arr.ndim() - 1] = 1;
                s
            })
            .unwrap();
        let normed = &x_arr * &rstd_view * &w_arr;
        let silu_gate = gate_arr.mapv(|v| v / (1.0 + (-v).exp()));
        let result = &normed * &silu_gate;
        Tensor::new(result.into_dyn(), false)
    }
}

// ─── Full Attention Layer (Qwen3_5Attention) ───────────────────────────────
#[derive(Clone)]
pub struct AttentionState {
    kv_cache: crate::nn::KVCache,
}

#[derive(Clone)]
pub struct Qwen3_5Attention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,
    pub q_norm: LayerRMSNorm,
    pub k_norm: LayerRMSNorm,
    pub d_model: usize,
    pub num_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub state: Option<AttentionState>,
}

impl Qwen3_5Attention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            q_proj: LinearLayer::new_f32(d_model, num_heads * head_dim * 2, false),
            k_proj: LinearLayer::new_f32(d_model, kv_heads * head_dim, false),
            v_proj: LinearLayer::new_f32(d_model, kv_heads * head_dim, false),
            o_proj: LinearLayer::new_f32(num_heads * head_dim, d_model, false),
            q_norm: LayerRMSNorm::new(head_dim, 1e-6),
            k_norm: LayerRMSNorm::new(head_dim, 1e-6),
            d_model,
            num_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            state: None,
        })
    }

    /// Partial planar RoPE matching transformers Qwen3_5 (rope_type "default").
    /// Only the first `rotary_dim` channels of each head are rotated, using the
    /// planar rotate_half convention; the rest pass through unchanged.
    fn apply_rope(&self, x: &Tensor, num_heads_x: usize, offset: usize) -> Tensor {
        let arr = x.to_f32_array();
        let ndim = arr.ndim();
        let d = self.head_dim;
        let h = num_heads_x;
        let rotary = self.rotary_dim.min(d);
        if ndim != 3 || rotary < 2 || rotary % 2 != 0 {
            return Tensor::new(arr, false);
        }
        let pair = rotary / 2;
        let mut cos = vec![0.0f32; pair];
        let mut sin = vec![0.0f32; pair];
        let pos = offset as f32;
        for i in 0..pair {
            let inv = 1.0 / self.rope_theta.powf((2 * i) as f32 / rotary as f32);
            let v = pos * inv;
            cos[i] = v.cos();
            sin[i] = v.sin();
        }
        let b = arr.shape()[0];
        let a3 = match arr.clone().into_dimensionality::<ndarray::Ix3>() {
            Ok(v) => v,
            Err(_) => return Tensor::new(arr, false),
        };
        let mut out = ndarray::Array3::<f32>::zeros((b, 1, h * d));
        for bb in 0..b {
            for hh in 0..h {
                let base = hh * d;
                for i in 0..pair {
                    let x1 = a3[[bb, 0, base + i]];
                    let x2 = a3[[bb, 0, base + i + pair]];
                    out[[bb, 0, base + i]] = x1 * cos[i] - x2 * sin[i];
                    out[[bb, 0, base + i + pair]] = x2 * cos[i] + x1 * sin[i];
                }
                for i in rotary..d {
                    out[[bb, 0, base + i]] = a3[[bb, 0, base + i]];
                }
            }
        }
        Tensor::new(out.into_dyn(), false)
    }

    pub fn forward_single_token(
        &mut self,
        x: &Tensor,
        causal_offset: Option<usize>,
    ) -> Result<Tensor, String> {
        let shape = x.lock().storage.shape().to_vec();
        let b = shape[0];
        let _s = shape[1];

        // Q projection → split into Q and gate
        let q_out = self.q_proj.forward(x);
        let _q_shape = q_out.lock().storage.shape().to_vec();
        // q_out: [b, 1, num_heads * head_dim * 2] → reshape to [b, 1, num_heads, head_dim*2]
        let q_reshaped = q_out
            .reshape(vec![b, 1, self.num_heads, self.head_dim * 2])
            .map_err(|e| format!("q_reshaped: {}", e))?;
        // Split last dim: q [b,1,h,hd] and gate [b,1,h,hd]
        let q_arr = q_reshaped.to_f32_array();
        let _hd2 = self.head_dim * 2;
        let q_part = q_arr
            .slice(ndarray::s![.., .., .., ..self.head_dim])
            .to_owned()
            .into_dyn();
        let gate_arr = q_arr
            .slice(ndarray::s![.., .., .., self.head_dim..])
            .to_owned()
            .into_dyn();
        let mut q = Tensor::new(q_part, false);
        let gate_t = Tensor::new(gate_arr, false);

        // Reshape q to [b, s, num_heads * head_dim] for RoPE
        q = q
            .reshape(vec![b, 1, self.num_heads * self.head_dim])
            .map_err(|e| format!("q reshape after split: {}", e))?;

        // K and V projections
        let v = self.v_proj.forward(x); // [b, 1, kv_heads * head_dim]

        // Apply QK-norm
        // Reshape q → [b, 1, num_heads, head_dim], norm along head_dim, reshape back
        {
            let q_4d = q
                .reshape(vec![b, 1, self.num_heads, self.head_dim])
                .map_err(|e| format!("q 4d reshape: {}", e))?;
            let qn_4d = self.q_norm.forward(&q_4d);
            q = qn_4d
                .reshape(vec![b, 1, self.num_heads * self.head_dim])
                .map_err(|e| format!("q reshape after q_norm: {}", e))?;
        }
        let k = {
            let k_proj = self.k_proj.forward(x);
            let k_4d = k_proj
                .reshape(vec![b, 1, self.kv_heads, self.head_dim])
                .map_err(|e| format!("k 4d reshape: {}", e))?;
            let kn_4d = self.k_norm.forward(&k_4d);
            kn_4d
                .reshape(vec![b, 1, self.kv_heads * self.head_dim])
                .map_err(|e| format!("k reshape after k_norm: {}", e))?
        };

        // Apply RoPE
        let cache_len = self
            .state
            .as_ref()
            .map(|s| {
                let pk = s.kv_cache.packed_keys();
                pk.as_ref()
                    .map(|t| {
                        let shape = t.lock().storage.shape().to_vec();
                        if shape.len() >= 2 {
                            shape[1]
                        } else {
                            0
                        }
                    })
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        let offset = cache_len + causal_offset.unwrap_or(0);
        q = self.apply_rope(&q, self.num_heads, offset);
        let k_rope = self.apply_rope(&k, self.kv_heads, offset);

        // Append to KV cache
        let state = self
            .state
            .as_mut()
            .ok_or("Attention state not initialized")?;
        state
            .kv_cache
            .append_packed(&k_rope, &v)
            .map_err(|e| format!("kv_cache append: {}", e))?;

        // Read back full K, V from cache
        let (k_total, v_total) = {
            let pk = state.kv_cache.packed_keys().ok_or("no packed keys")?;
            let pv = state.kv_cache.packed_values().ok_or("no packed values")?;
            (pk.clone(), pv.clone())
        };

        let k_shape = k_total.lock().storage.shape().to_vec();
        let total_seq = k_shape[1];

        // Reshape for attention: [b, heads, seq, head_dim]
        let q_attn = q
            .reshape(vec![b, self.num_heads, 1, self.head_dim])
            .map_err(|e| format!("q attn reshape: {}", e))?;
        let k_attn = k_total
            .reshape(vec![b, self.kv_heads, total_seq, self.head_dim])
            .map_err(|e| format!("k attn reshape: {}", e))?;
        let v_attn = v_total
            .reshape(vec![b, self.kv_heads, total_seq, self.head_dim])
            .map_err(|e| format!("v attn reshape: {}", e))?;

        // Repeat KV for GQA
        let rep = self.num_heads / self.kv_heads;
        let k_full = if rep > 1 {
            // Repeat kv_heads num_heads times: [b, kv, seq, hd] → [b, num, seq, hd]
            let k_arr = k_attn.to_f32_array();
            let _k_shape_arr = k_arr.shape().to_vec();
            // Use ndarray to repeat
            let mut k_out = ArrayD::zeros(IxDyn(&[b, self.num_heads, total_seq, self.head_dim]));
            let k_view = k_arr.into_dimensionality::<ndarray::Ix4>().unwrap();
            let mut k_out_view = k_out
                .view_mut()
                .into_dimensionality::<ndarray::Ix4>()
                .unwrap();
            for h in 0..self.num_heads {
                let src_h = h / rep;
                k_out_view
                    .slice_mut(ndarray::s![.., h, .., ..])
                    .assign(&k_view.slice(ndarray::s![.., src_h, .., ..]));
            }
            Tensor::new(k_out.into_dyn(), false)
        } else {
            k_attn.clone()
        };
        let v_full = if rep > 1 {
            let v_arr = v_attn.to_f32_array();
            let mut v_out = ArrayD::zeros(IxDyn(&[b, self.num_heads, total_seq, self.head_dim]));
            let v_view = v_arr.into_dimensionality::<ndarray::Ix4>().unwrap();
            let mut v_out_view = v_out
                .view_mut()
                .into_dimensionality::<ndarray::Ix4>()
                .unwrap();
            for h in 0..self.num_heads {
                let src_h = h / rep;
                v_out_view
                    .slice_mut(ndarray::s![.., h, .., ..])
                    .assign(&v_view.slice(ndarray::s![.., src_h, .., ..]));
            }
            Tensor::new(v_out.into_dyn(), false)
        } else {
            v_attn.clone()
        };

        // Flatten batch and heads for matmul
        let q_flat = q_attn
            .reshape(vec![b * self.num_heads, 1, self.head_dim])
            .map_err(|e| format!("q flat: {}", e))?;
        let k_flat = k_full
            .reshape(vec![b * self.num_heads, total_seq, self.head_dim])
            .map_err(|e| format!("k flat: {}", e))?;
        let v_flat = v_full
            .reshape(vec![b * self.num_heads, total_seq, self.head_dim])
            .map_err(|e| format!("v flat: {}", e))?;

        // Scores: [b*h, 1, total_seq]
        let k_arr = k_flat.to_f32_array();
        let k_t = if k_arr.ndim() == 3 {
            let k_ix3 = k_arr
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| format!("k dim: {}", e))?;
            let k_t_view = k_ix3.permuted_axes([0, 2, 1]); // [b*h, head_dim, total_seq]
            Tensor::new(k_t_view.into_owned().into_dyn(), false)
        } else {
            k_flat.transpose()
        };
        let score_arr = q_flat.matmul(&k_t).to_f32_array();
        let scaled = score_arr.mapv(|v| v * (1.0 / (self.head_dim as f32).sqrt()));
        let score = Tensor::new(scaled.into_dyn(), false);

        // Causal mask
        let score_arr = score.to_f32_array();
        let _score_shape = score_arr.shape().to_vec();
        let mut masked = score_arr;
        let last_idx = total_seq - 1;
        if masked.ndim() == 3 {
            for hi in 0..masked.shape()[0] {
                for pos in 0..masked.shape()[2] {
                    if pos > last_idx {
                        masked[[hi, 0, pos]] = -1e10;
                    }
                }
            }
        }
        let score_masked = Tensor::new(masked.into_dyn(), false);

        // Softmax
        let attn_weights = score_masked.softmax(2);

        // Weighted sum: [b*h, 1, total_seq] @ [b*h, total_seq, hd] → [b*h, 1, hd]
        let attn_out = attn_weights.matmul(&v_flat);

        // Reshape back: [b, num_heads, 1, head_dim] → [b, 1, num_heads * head_dim]
        let attn_3d = attn_out
            .reshape(vec![b, self.num_heads * self.head_dim])
            .map_err(|e| format!("attn 3d: {}", e))?;

        // Gate: sigmoid(gate) and element-wise multiply
        let gate_sigmoid = gate_t.sigmoid();
        let gate_3d = gate_sigmoid
            .reshape(vec![b, self.num_heads * self.head_dim])
            .map_err(|e| format!("gate 3d: {}", e))?;
        let gated = attn_3d.mul(&gate_3d);

        // Output projection
        let out = self.o_proj.forward(&gated);
        Ok(out)
    }

    pub fn init_state(&mut self, seq_len: usize) -> Result<(), String> {
        let batch = 1;
        let head_dim = self.head_dim;
        let dim = self.kv_heads * head_dim;
        let mut cache = crate::nn::KVCache::new();
        cache.set_packed_capacity(batch, seq_len, dim);
        self.state = Some(AttentionState { kv_cache: cache });
        Ok(())
    }

    pub fn reset_state(&mut self) {
        if let Some(ref mut s) = self.state {
            s.kv_cache.truncate(s.kv_cache.seq_len());
        }
    }
}

// ─── Linear Attention Layer (GatedDeltaNet) ────────────────────────────────
#[derive(Clone)]
pub struct LinearAttentionState {
    conv_state: Tensor,      // [conv_dim, conv_kernel_size - 1]
    recurrent_state: Tensor, // [num_heads, key_dim, value_dim] (flat)
    filled: usize,
    _capacity: usize,
}

#[allow(non_snake_case)]
pub struct Qwen3_5GatedDeltaNet {
    pub in_proj_qkv: LinearLayer,
    pub in_proj_z: LinearLayer,
    pub in_proj_a: LinearLayer,
    pub in_proj_b: LinearLayer,
    pub out_proj: LinearLayer,
    pub conv1d_weight: Tensor,
    pub conv1d_bias: Option<Tensor>,
    pub A_log: Tensor,
    pub dt_bias: Tensor,
    pub norm: Qwen3_5RMSNormGated,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_dim: usize,
    pub conv_kernel_size: usize,
    pub state: Option<LinearAttentionState>,
}

impl Clone for Qwen3_5GatedDeltaNet {
    fn clone(&self) -> Self {
        Self {
            in_proj_qkv: self.in_proj_qkv.clone(),
            in_proj_z: self.in_proj_z.clone(),
            in_proj_a: self.in_proj_a.clone(),
            in_proj_b: self.in_proj_b.clone(),
            out_proj: self.out_proj.clone(),
            conv1d_weight: self.conv1d_weight.clone(),
            conv1d_bias: self.conv1d_bias.clone(),
            A_log: self.A_log.clone(),
            dt_bias: self.dt_bias.clone(),
            norm: self.norm.clone(),
            num_k_heads: self.num_k_heads,
            num_v_heads: self.num_v_heads,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            head_k_dim: self.head_k_dim,
            head_v_dim: self.head_v_dim,
            conv_dim: self.conv_dim,
            conv_kernel_size: self.conv_kernel_size,
            state: None,
        }
    }
}

fn causal_conv1d_single(
    input: &Tensor,     // [1, 1, conv_dim]
    state: &mut Tensor, // [conv_dim, kernel_size - 1], modified in-place
    weight: &Tensor,    // [conv_dim, 1, kernel_size]
    bias: &Option<Tensor>,
    activation: Option<&str>,
) -> Result<Tensor, String> {
    let conv_dim = weight.lock().storage.shape()[0];
    let kernel = weight.lock().storage.shape()[2];

    // Concatenate state with new input: [conv_dim, kernel_size]
    let inp_arr = input.to_f32_array();
    let state_arr = state.to_f32_array();

    let inp_2d = if inp_arr.ndim() == 3 {
        // [b, 1, conv_dim] → [conv_dim, 1]
        inp_arr
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| format!("conv input dim: {}", e))?
            .index_axis(ndarray::Axis(1), 0)
            .t()
            .to_owned()
    } else {
        // [1, conv_dim] → [conv_dim, 1]
        inp_arr
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("conv input dim: {}", e))?
            .t()
            .to_owned()
    };

    let state_2d = state_arr
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| format!("conv state dim: {}", e))?; // [conv_dim, k-1]

    let mut cat = ndarray::Array2::zeros((conv_dim, kernel));
    cat.slice_mut(ndarray::s![.., ..kernel - 1])
        .assign(&state_2d);
    cat.slice_mut(ndarray::s![.., kernel - 1..]).assign(&inp_2d);

    // Update state with the last kernel-1 values
    let new_state = cat.slice(ndarray::s![.., 1..]).to_owned();
    *state = Tensor::new(new_state.into_dyn(), false);

    // Conv1d: groups=conv_dim, kernel_size=kernel
    let w_arr = weight.to_f32_array();
    let w_2d = w_arr
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| format!("weight dim: {}", e))?; // [conv_dim, 1, kernel]

    // For each group, compute dot product
    let mut out = ArrayD::zeros(IxDyn(&[1, 1, conv_dim]));
    {
        let mut out_3d = out
            .view_mut()
            .into_dimensionality::<ndarray::Ix3>()
            .unwrap();
        for g in 0..conv_dim {
            let mut sum = 0.0f32;
            for k in 0..kernel {
                sum += cat[[g, k]] * w_2d[[g, 0, k]];
            }
            if let Some(ref b) = bias {
                let b_arr = b.to_f32_array();
                sum += b_arr[g];
            }
            out_3d[[0, 0, g]] = sum;
        }
    }

    // Activation
    if let Some("silu") = activation {
        let o_arr = out.mapv(|v| v / (1.0 + (-v).exp()));
        out = o_arr.into_dyn();
    }

    Ok(Tensor::new(out, false))
}

impl Qwen3_5GatedDeltaNet {
    pub fn new(
        d_model: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_kernel_size: usize,
    ) -> Result<Self, String> {
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;

        let conv_w = ArrayD::zeros(IxDyn(&[conv_dim, 1, conv_kernel_size]));
        Ok(Self {
            in_proj_qkv: LinearLayer::new_f32(d_model, conv_dim, false),
            in_proj_z: LinearLayer::new_f32(d_model, value_dim, false),
            in_proj_a: LinearLayer::new_f32(d_model, num_v_heads, false),
            in_proj_b: LinearLayer::new_f32(d_model, num_v_heads, false),
            out_proj: LinearLayer::new_f32(value_dim, d_model, false),
            conv1d_weight: Tensor::new(conv_w, true),
            conv1d_bias: None,
            A_log: Tensor::new(Array::from_elem(IxDyn(&[num_v_heads]), 1.0f32), true),
            dt_bias: Tensor::new(Array::ones(IxDyn(&[num_v_heads])), true),
            norm: Qwen3_5RMSNormGated::new(head_v_dim, 1e-6),
            num_k_heads,
            num_v_heads,
            key_dim,
            value_dim,
            head_k_dim,
            head_v_dim,
            conv_dim,
            conv_kernel_size,
            state: None,
        })
    }

    pub fn forward_single_token(&mut self, x: &Tensor) -> Result<Tensor, String> {
        let state = self
            .state
            .as_mut()
            .ok_or("Linear attention state not initialized")?;
        let b = x.lock().storage.shape()[0];

        // Projections
        let qkv = self.in_proj_qkv.forward(x); // [b, 1, conv_dim]
        let z = self.in_proj_z.forward(x); // [b, 1, value_dim]
        let a = self.in_proj_a.forward(x); // [b, 1, num_v_heads]
        let b_proj = self.in_proj_b.forward(x); // [b, 1, num_v_heads]

        // Conv1d
        let qkv_conv = causal_conv1d_single(
            &qkv,
            &mut state.conv_state,
            &self.conv1d_weight,
            &self.conv1d_bias,
            Some("silu"),
        )?;

        // Split: q, k, v
        let qkv_arr = qkv_conv.to_f32_array();
        let q_part = qkv_arr
            .slice(ndarray::s![.., .., ..self.key_dim])
            .to_owned()
            .into_dyn();
        let k_part = qkv_arr
            .slice(ndarray::s![.., .., self.key_dim..self.key_dim * 2])
            .to_owned()
            .into_dyn();
        let v_part = qkv_arr
            .slice(ndarray::s![.., .., self.key_dim * 2..])
            .to_owned()
            .into_dyn();

        let mut query = Tensor::new(q_part, false);
        let mut key = Tensor::new(k_part, false);
        let value = Tensor::new(v_part, false);

        // Reshape to [b, 1, num_heads, head_dim]
        query = query
            .reshape(vec![b, 1, self.num_k_heads, self.head_k_dim])
            .map_err(|e| format!("q reshape: {}", e))?;
        key = key
            .reshape(vec![b, 1, self.num_k_heads, self.head_k_dim])
            .map_err(|e| format!("k reshape: {}", e))?;
        let value_4d = value
            .reshape(vec![b, 1, self.num_v_heads, self.head_v_dim])
            .map_err(|e| format!("v reshape: {}", e))?;

        // beta = sigmoid(b_proj)
        let beta = b_proj.sigmoid();
        #[allow(non_snake_case)]
        let A_log_exp = self.A_log.exp();
        let dt_bias_3d = self
            .dt_bias
            .reshape(vec![1, 1, self.num_v_heads])
            .map_err(|e| format!("dt_bias reshape: {}", e))?;
        #[allow(non_snake_case)]
        let A_log_3d = A_log_exp
            .reshape(vec![1, 1, self.num_v_heads])
            .map_err(|e| format!("A_log reshape: {}", e))?;
        let a_bias = a.add(&dt_bias_3d);
        let g_val = A_log_3d.mul(&softplus(&a_bias)).neg();

        // Expand: g = [b, 1, num_v_heads] → for decay, squeeze head dims
        let g_exp = g_val; // [1, 1, 16] — used as scalar per head

        // Repeat q, k if needed
        let rep = self.num_v_heads / self.num_k_heads;
        let (q_rep, k_rep) = if rep > 1 {
            let q_arr = query.to_f32_array();
            let k_arr = key.to_f32_array();
            let mut q_out = ArrayD::zeros(IxDyn(&[b, 1, self.num_v_heads, self.head_k_dim]));
            let mut k_out = ArrayD::zeros(IxDyn(&[b, 1, self.num_v_heads, self.head_k_dim]));
            if let (Ok(q4), Ok(k4)) = (
                q_arr.into_dimensionality::<ndarray::Ix4>(),
                k_arr.into_dimensionality::<ndarray::Ix4>(),
            ) {
                if let (Ok(mut qo4), Ok(mut ko4)) = (
                    q_out.view_mut().into_dimensionality::<ndarray::Ix4>(),
                    k_out.view_mut().into_dimensionality::<ndarray::Ix4>(),
                ) {
                    for h in 0..self.num_v_heads {
                        let src = h / rep;
                        qo4.slice_mut(ndarray::s![.., .., h, ..])
                            .assign(&q4.slice(ndarray::s![.., .., src, ..]));
                        ko4.slice_mut(ndarray::s![.., .., h, ..])
                            .assign(&k4.slice(ndarray::s![.., .., src, ..]));
                    }
                }
            }
            (
                Tensor::new(q_out.into_dyn(), false),
                Tensor::new(k_out.into_dyn(), false),
            )
        } else {
            (query.clone(), key.clone())
        };

        // Flatten batch+heads for tensor operations
        let total_h = self.num_v_heads;
        let q_flat = q_rep
            .reshape(vec![b * total_h, self.head_k_dim])
            .map_err(|e| format!("q flat: {}", e))?;
        let k_flat = k_rep
            .reshape(vec![b * total_h, self.head_k_dim])
            .map_err(|e| format!("k flat: {}", e))?;
        let v_flat = value_4d
            .reshape(vec![b * total_h, self.head_v_dim])
            .map_err(|e| format!("v flat: {}", e))?;
        let beta_flat = beta
            .reshape(vec![b * total_h])
            .map_err(|e| format!("beta flat: {}", e))?;

        // Recurrent state S: [total_h, key_dim, value_dim]
        let state_arr = state.recurrent_state.to_f32_array();
        let _state_shape = state_arr.shape().to_vec();
        let mut state_mat = state_arr
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| format!("S dim: {}", e))?
            .to_owned(); // [total_h, head_k_dim, head_v_dim]

        // g_exp: [b, 1, total_h], but we need per-head scalar
        let g_vals = g_exp.to_f32_array();
        let g_scalar = if g_vals.ndim() == 2 {
            // [1, total_h]
            g_vals
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap()
                .row(0)
                .to_owned()
        } else if g_vals.ndim() == 3 {
            // [1, 1, total_h]
            let g2 = g_vals.into_dimensionality::<ndarray::Ix3>().unwrap();
            g2.slice(ndarray::s![0, 0, ..]).to_owned()
        } else {
            return Err("g_exp unexpected dims".into());
        };

        let k_2d = k_flat
            .to_f32_array()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("k_2d: {}", e))?; // [b*total_h, head_k_dim]
        let q_2d = q_flat
            .to_f32_array()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("q_2d: {}", e))?;
        let v_2d = v_flat
            .to_f32_array()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("v_2d: {}", e))?;
        let beta_1d = beta_flat
            .to_f32_array()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|e| format!("beta_1d: {}", e))?;

        // Qwen3.5 delta rule normalizes q and k by their L2 norm (eps=1e-6)
        // and scales q by 1/sqrt(head_k_dim), matching the FLA/transformers
        // reference with use_qk_l2norm_in_kernel=True.
        let scale_q = 1.0 / (self.head_k_dim as f32).sqrt();
        let mut q_n = q_2d.clone();
        let mut k_n = k_2d.clone();
        for r in 0..q_2d.shape()[0] {
            let mut qs = 0.0f32;
            for c in 0..q_2d.shape()[1] {
                let qv = q_2d[[r, c]];
                qs += qv * qv;
            }
            let q_inv = 1.0 / (qs + 1e-6).sqrt();
            let mut ks = 0.0f32;
            for c in 0..k_2d.shape()[1] {
                let kv = k_2d[[r, c]];
                ks += kv * kv;
            }
            let k_inv = 1.0 / (ks + 1e-6).sqrt();
            for c in 0..q_2d.shape()[1] {
                q_n[[r, c]] = q_2d[[r, c]] * q_inv * scale_q;
                k_n[[r, c]] = k_2d[[r, c]] * k_inv;
            }
        }

        // Per-step gated delta rule
        for h in 0..total_h {
            let g_h = g_scalar[h].exp();
            // Decay: state_mat[h] *= exp(g_h)
            state_mat
                .slice_mut(ndarray::s![h, .., ..])
                .mapv_inplace(|v| v * g_h);

            // S @ k → sum over key_dim
            let mut kv_mem: Array1<f32> = ndarray::Array1::zeros(self.head_v_dim);
            for kd in 0..self.head_k_dim {
                for vd in 0..self.head_v_dim {
                    kv_mem[vd] += state_mat[[h, kd, vd]] * k_n[[h, kd]];
                }
            }

            // delta = (v - kv_mem) * beta
            let mut delta: Array1<f32> = ndarray::Array1::zeros(self.head_v_dim);
            for vd in 0..self.head_v_dim {
                delta[vd] = (v_2d[[h, vd]] - kv_mem[vd]) * beta_1d[h];
            }

            // S += outer(k, delta)
            for kd in 0..self.head_k_dim {
                for vd in 0..self.head_v_dim {
                    state_mat[[h, kd, vd]] += k_n[[h, kd]] * delta[vd];
                }
            }
        }

        // Read output: S^T @ q → sum over key_dim
        let mut out_arr = ndarray::Array2::zeros((b * total_h, self.head_v_dim));
        for h in 0..total_h {
            for vd in 0..self.head_v_dim {
                let mut sum = 0.0;
                for kd in 0..self.head_k_dim {
                    sum += state_mat[[h, kd, vd]] * q_n[[h, kd]];
                }
                out_arr[[h, vd]] = sum;
            }
        }

        // Store updated state
        state.recurrent_state = Tensor::new(state_mat.into_dyn(), false);

        // Reshape output: [b, total_h, head_v_dim] → [b, 1, value_dim]
        let out_t = Tensor::new(out_arr.into_dyn(), false);
        let out_3d = out_t
            .reshape(vec![b, 1, self.value_dim])
            .map_err(|e| format!("out 3d: {}", e))?;

        // RMSNormGated: norm(output) * silu(z)
        let z_3d = z
            .reshape(vec![b, 1, self.num_v_heads, self.head_v_dim])
            .map_err(|e| format!("z 4d: {}", e))?;
        let out_4d = out_3d
            .reshape(vec![b, 1, self.num_v_heads, self.head_v_dim])
            .map_err(|e| format!("out 4d: {}", e))?;
        let normed = self.norm.forward(&out_4d, &z_3d);
        let normed_3d = normed
            .reshape(vec![b, 1, self.value_dim])
            .map_err(|e| format!("normed 3d: {}", e))?;

        // Output projection
        let final_out = self.out_proj.forward(&normed_3d);
        Ok(final_out)
    }

    pub fn init_state(&mut self, seq_len: usize) -> Result<(), String> {
        let conv_state = ArrayD::zeros(IxDyn(&[self.conv_dim, self.conv_kernel_size - 1]));
        let rec_state = ArrayD::zeros(IxDyn(&[self.num_v_heads, self.head_k_dim, self.head_v_dim]));
        self.state = Some(LinearAttentionState {
            conv_state: Tensor::new(conv_state, false),
            recurrent_state: Tensor::new(rec_state, false),
            filled: 0,
            _capacity: seq_len,
        });
        Ok(())
    }

    pub fn reset_state(&mut self) {
        if let Some(ref mut s) = self.state {
            s.filled = 0;
            s.conv_state = Tensor::new(
                ArrayD::zeros(IxDyn(&[self.conv_dim, self.conv_kernel_size - 1])),
                false,
            );
            s.recurrent_state = Tensor::new(
                ArrayD::zeros(IxDyn(&[self.num_v_heads, self.head_k_dim, self.head_v_dim])),
                false,
            );
        }
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────
pub enum Qwen3_5DecoderAttention {
    Full(Qwen3_5Attention),
    Linear(Qwen3_5GatedDeltaNet),
}

impl Clone for Qwen3_5DecoderAttention {
    fn clone(&self) -> Self {
        match self {
            Self::Full(a) => Self::Full(a.clone()),
            Self::Linear(a) => Self::Linear(a.clone()),
        }
    }
}

pub struct Qwen3_5DecoderLayer {
    pub input_layernorm: LayerRMSNorm,
    pub attention: Qwen3_5DecoderAttention,
    pub post_attention_layernorm: LayerRMSNorm,
    pub mlp: MLP,
}

impl Clone for Qwen3_5DecoderLayer {
    fn clone(&self) -> Self {
        Self {
            input_layernorm: self.input_layernorm.clone(),
            attention: self.attention.clone(),
            post_attention_layernorm: self.post_attention_layernorm.clone(),
            mlp: self.mlp.clone(),
        }
    }
}

// ─── RMSNorm (standard, for pre-attn/pre-ffn norms) ─────────────────────────
// Qwen3.5 uses the (1 + weight) convention: output = norm(x) * (1 + weight),
// with weight initialized to 0 (see transformers Qwen3_5RMSNorm). The linear
// attention's final norm is the separate Qwen3_5RMSNormGated below, which uses
// plain `weight` scaling with weight initialized to 1.
#[derive(Clone)]
pub struct LayerRMSNorm {
    weight: Tensor,
    eps: f32,
}

impl LayerRMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::new(Array::from_elem(IxDyn(&[dim]), 0.0f32), true),
            eps,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x_arr = x.to_f32_array();
        let last = x_arr.ndim() - 1;
        let var = (&x_arr * &x_arr).mean_axis(ndarray::Axis(last)).unwrap();
        let rstd = var.mapv(|v| 1.0 / (v + self.eps).sqrt());
        let mut rstd_shape = x_arr.shape().to_vec();
        rstd_shape[last] = 1;
        let rstd_view = rstd.into_shape_with_order(IxDyn(&rstd_shape)).unwrap();
        let mut w_shape = vec![1; x_arr.ndim()];
        w_shape[last] = x_arr.shape()[last];
        let w_arr = self
            .weight
            .to_f32_array()
            .into_shape_with_order(IxDyn(&w_shape))
            .unwrap();
        let normed = &x_arr * &rstd_view;
        let scaled = &normed * &w_arr.mapv(|v| v + 1.0);
        Tensor::new(scaled.into_dyn(), false)
    }
}

// ─── MLP (SwiGLU FFN) ───────────────────────────────────────────────────────
#[derive(Clone)]
pub struct MLP {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
}

impl MLP {
    pub fn new(d_model: usize, d_ff: usize) -> Result<Self, String> {
        Ok(Self {
            gate_proj: LinearLayer::new_f32(d_model, d_ff, false),
            up_proj: LinearLayer::new_f32(d_model, d_ff, false),
            down_proj: LinearLayer::new_f32(d_ff, d_model, false),
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward(x).silu();
        let up = self.up_proj.forward(x);
        self.down_proj.forward(&gate.mul(&up))
    }
}

// ─── Text Model ─────────────────────────────────────────────────────────────
static DBG_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn dbg_weights(model: &Qwen3_5TextModel) {
    let e = model.embed_tokens.lock().storage.to_f32_array();
    let ev: Vec<f32> = e.iter().take(8).cloned().collect();
    eprintln!("[dbg] embed_tokens[0][0..8] = {:?}", ev);
    if let Some(lin) = match &model.layers[0].attention {
        Qwen3_5DecoderAttention::Linear(l) => Some(l),
        _ => None,
    } {
        let a_log = lin.A_log.to_f32_array();
        let dt = lin.dt_bias.to_f32_array();
        let conv = lin.conv1d_weight.to_f32_array();
        let al: Vec<f32> = a_log.iter().take(4).cloned().collect();
        let dtr: Vec<f32> = dt.iter().take(4).cloned().collect();
        let c0 = conv[[0, 0, 0]];
        let c1 = conv[[0, 0, 1]];
        eprintln!("[dbg] L0 A_log[0..4] = {:?}", al);
        eprintln!("[dbg] L0 dt_bias[0..4] = {:?}", dtr);
        eprintln!("[dbg] L0 conv1d_w[0][0][0..2] = [{:.6}, {:.6}]", c0, c1);
        if let Some(l) = lin.in_proj_qkv.as_f32() {
            let w = l.weight.to_f32_array();
            let w0 = w[[0, 0]];
            let w1 = w[[0, 1]];
            eprintln!("[dbg] L0 in_proj_qkv_w[0][0..1] = [{:.6}, {:.6}]", w0, w1);
        }
        eprintln!(
            "[dbg] L0 in_proj_qkv has bias: {}",
            lin.in_proj_qkv
                .as_f32()
                .map(|l| l.bias.is_some())
                .unwrap_or(false)
        );
    }
}

fn dbg_logits(logits: &Tensor) {
    let arr = logits.to_f32_array();
    if arr.ndim() == 1 {
        let mut idx: Vec<usize> = (0..arr.len()).collect();
        idx.sort_by(|&a, &b| {
            arr[b]
                .partial_cmp(&arr[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, arr[i])).collect();
        eprintln!("[dbg] top5 logits = {:?}", top);
    }
}

pub struct Qwen3_5TextModel {
    pub embed_tokens: Tensor,
    pub layers: Vec<Qwen3_5DecoderLayer>,
    pub norm: LayerRMSNorm,
    pub d_model: usize,
    pub vocab_size: usize,
}

impl Clone for Qwen3_5TextModel {
    fn clone(&self) -> Self {
        Self {
            embed_tokens: self.embed_tokens.clone(),
            layers: self.layers.clone(),
            norm: self.norm.clone(),
            d_model: self.d_model,
            vocab_size: self.vocab_size,
        }
    }
}

impl Qwen3_5TextModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        layer_types: &[String],
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_kernel_size: usize,
    ) -> Result<Self, String> {
        let embed_tokens = Tensor::new(Array::zeros(IxDyn(&[vocab_size, d_model][..])), true);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let lt = layer_types.get(i).cloned().unwrap_or_default();
            let attn = if lt == "full_attention" {
                Qwen3_5DecoderAttention::Full(Qwen3_5Attention::new(
                    d_model, num_heads, kv_heads, head_dim, rotary_dim, rope_theta,
                )?)
            } else {
                Qwen3_5DecoderAttention::Linear(Qwen3_5GatedDeltaNet::new(
                    d_model,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    conv_kernel_size,
                )?)
            };
            layers.push(Qwen3_5DecoderLayer {
                input_layernorm: LayerRMSNorm::new(d_model, 1e-6),
                attention: attn,
                post_attention_layernorm: LayerRMSNorm::new(d_model, 1e-6),
                mlp: MLP::new(d_model, d_ff)?,
            });
        }

        let norm = LayerRMSNorm::new(d_model, 1e-6);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            d_model,
            vocab_size,
        })
    }

    /// Build from config JSON and pre-loaded state dict.
    /// Needed because the model uses `model.language_model.` prefix for text weights.
    pub fn from_config(
        config: &serde_json::Value,
        state: &HashMap<String, Tensor>,
    ) -> Result<Self, String> {
        let tc = config
            .get("text_config")
            .or_else(|| config.get("text_config"))
            .or(Some(config));
        // Actually read from the value that parse_registry_config provides
        let tc_val = tc.unwrap();

        let vocab_size = tc_val
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(248320) as usize;
        let hidden_size = tc_val
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024) as usize;
        let num_layers = tc_val
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(24) as usize;
        let inter_size = tc_val
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(3584) as usize;
        let num_heads = tc_val
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        let kv_heads = tc_val
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let head_dim = tc_val
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize;
        let num_k_heads = tc_val
            .get("linear_num_key_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;
        let num_v_heads = tc_val
            .get("linear_num_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;
        let head_k_dim = tc_val
            .get("linear_key_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let head_v_dim = tc_val
            .get("linear_value_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let conv_k = tc_val
            .get("linear_conv_kernel_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;
        let partial_rotary = tc_val
            .get("rope_parameters")
            .and_then(|rp| rp.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25);
        let rotary_dim = (head_dim as f64 * partial_rotary) as usize;
        let rope_theta = tc_val
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                tc_val
                    .get("rope_parameters")
                    .and_then(|rp| rp.get("rope_theta"))
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(10000.0) as f32;

        let layer_types: Vec<String> = tc_val
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|s| s.as_str().unwrap_or("linear_attention").to_string())
                    .collect()
            })
            .unwrap_or_else(|| {
                let interval = tc_val
                    .get("full_attention_interval")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4) as usize;
                (0..num_layers)
                    .map(|i| {
                        if (i + 1) % interval == 0 {
                            "full_attention".to_string()
                        } else {
                            "linear_attention".to_string()
                        }
                    })
                    .collect()
            });

        let mut model = Self::new(
            vocab_size,
            hidden_size,
            num_layers,
            inter_size,
            num_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            &layer_types,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_k,
        )?;

        model.apply_state_dict(state, "model.language_model")?;
        Ok(model)
    }

    pub fn forward_single_token(
        &mut self,
        token_id: &Tensor,
        _causal_offset: Option<usize>,
    ) -> Result<Tensor, String> {
        // Embedding lookup
        if DBG_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            dbg_weights(self);
        }
        let mut x = Tensor::embedding_lookup(&self.embed_tokens, token_id);
        let shape = x.lock().storage.shape().to_vec();
        if shape.len() == 2 {
            x = x
                .reshape(vec![1, shape[0], shape[1]])
                .map_err(|e| format!("embed reshape: {}", e))?;
        } else if shape.len() == 1 {
            x = x
                .reshape(vec![1, 1, shape[0]])
                .map_err(|e| format!("embed reshape: {}", e))?;
        }

        // Process through all layers
        for (li, layer) in self.layers.iter_mut().enumerate() {
            let t0 = std::time::Instant::now();
            // Pre-attention norm
            let residual = x.clone();
            x = layer.input_layernorm.forward(&x);
            // Attention
            x = match &mut layer.attention {
                Qwen3_5DecoderAttention::Full(attn) => {
                    attn.forward_single_token(&x, _causal_offset)?
                }
                Qwen3_5DecoderAttention::Linear(lin_attn) => lin_attn.forward_single_token(&x)?,
            };
            let t_attn = std::time::Instant::now();
            // Residual
            x = x.add(&residual);
            // Post-attention norm + MLP
            let residual = x.clone();
            x = layer.post_attention_layernorm.forward(&x);
            x = layer.mlp.forward(&x);
            x = x.add(&residual);
            let t_mlp = std::time::Instant::now();
            eprintln!(
                "[timing] layer {} attn={:.1}ms mlp={:.1}ms",
                li,
                t_attn.duration_since(t0).as_secs_f32() * 1000.0,
                t_mlp.duration_since(t_attn).as_secs_f32() * 1000.0
            );
        }

        // Final norm
        x = self.norm.forward(&x);

        // LM head (tied embeddings)
        let t_lm = std::time::Instant::now();
        let embed_t = self.embed_tokens.lock().storage.to_f32_array();
        let t_after_embed = std::time::Instant::now();
        let out_shape = x.lock().storage.shape().to_vec();
        let flat = x
            .reshape(vec![out_shape[0] * out_shape[1], self.d_model])
            .map_err(|e| format!("flat for lm head: {}", e))?;
        let embed_t_2d = embed_t
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("embed dim: {}", e))?;
        let t_after_reshape = std::time::Instant::now();
        let logits = flat.matmul(&Tensor::new(embed_t_2d.reversed_axes().into_dyn(), false));
        let t_end = std::time::Instant::now();
        eprintln!(
            "[timing] lm_head_total={:.1}ms embed_clone={:.1}ms reshape={:.1}ms matmul={:.1}ms",
            t_end.duration_since(t_lm).as_secs_f32() * 1000.0,
            t_after_embed.duration_since(t_lm).as_secs_f32() * 1000.0,
            t_after_reshape.duration_since(t_after_embed).as_secs_f32() * 1000.0,
            t_end.duration_since(t_after_reshape).as_secs_f32() * 1000.0
        );
        if out_shape[1] == 1 {
            let out = logits.reshape(vec![self.vocab_size])?;
            dbg_logits(&out);
            Ok(out)
        } else {
            let out = logits.reshape(vec![out_shape[0], out_shape[1], self.vocab_size])?;
            dbg_logits(&out);
            Ok(out)
        }
    }

    pub fn init_kv_caches(&mut self, seq_len: usize) -> Result<(), String> {
        for layer in self.layers.iter_mut() {
            match &mut layer.attention {
                Qwen3_5DecoderAttention::Full(attn) => attn.init_state(seq_len)?,
                Qwen3_5DecoderAttention::Linear(lin_attn) => lin_attn.init_state(seq_len)?,
            }
        }
        Ok(())
    }

    pub fn reset_kv_caches(&mut self) {
        for layer in self.layers.iter_mut() {
            match &mut layer.attention {
                Qwen3_5DecoderAttention::Full(attn) => attn.reset_state(),
                Qwen3_5DecoderAttention::Linear(lin_attn) => lin_attn.reset_state(),
            }
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.embed_tokens.clone()];
        p.push(self.norm.weight.clone());
        for layer in &self.layers {
            p.push(layer.input_layernorm.weight.clone());
            match &layer.attention {
                Qwen3_5DecoderAttention::Full(attn) => {
                    p.extend(attn.q_proj.parameters());
                    p.extend(attn.k_proj.parameters());
                    p.extend(attn.v_proj.parameters());
                    p.extend(attn.o_proj.parameters());
                    p.push(attn.q_norm.weight.clone());
                    p.push(attn.k_norm.weight.clone());
                }
                Qwen3_5DecoderAttention::Linear(lin) => {
                    p.extend(lin.in_proj_qkv.parameters());
                    p.extend(lin.in_proj_z.parameters());
                    p.extend(lin.in_proj_a.parameters());
                    p.extend(lin.in_proj_b.parameters());
                    p.extend(lin.out_proj.parameters());
                    p.push(lin.conv1d_weight.clone());
                    p.push(lin.A_log.clone());
                    p.push(lin.dt_bias.clone());
                    p.push(lin.norm.weight.clone());
                }
            }
            p.push(layer.post_attention_layernorm.weight.clone());
            p.extend(layer.mlp.gate_proj.parameters());
            p.extend(layer.mlp.up_proj.parameters());
            p.extend(layer.mlp.down_proj.parameters());
        }
        p
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = vec![
            (
                format!("{}.embed_tokens.weight", prefix),
                self.embed_tokens.clone(),
            ),
            (format!("{}.norm.weight", prefix), self.norm.weight.clone()),
        ];
        for (i, layer) in self.layers.iter().enumerate() {
            let lp = format!("{}.layers.{}", prefix, i);
            out.push((
                format!("{}.input_layernorm.weight", lp),
                layer.input_layernorm.weight.clone(),
            ));
            match &layer.attention {
                Qwen3_5DecoderAttention::Full(attn) => {
                    out.extend(
                        attn.q_proj
                            .named_parameters(&format!("{}.self_attn.q_proj", lp)),
                    );
                    out.extend(
                        attn.k_proj
                            .named_parameters(&format!("{}.self_attn.k_proj", lp)),
                    );
                    out.extend(
                        attn.v_proj
                            .named_parameters(&format!("{}.self_attn.v_proj", lp)),
                    );
                    out.extend(
                        attn.o_proj
                            .named_parameters(&format!("{}.self_attn.o_proj", lp)),
                    );
                    out.push((
                        format!("{}.self_attn.q_norm.weight", lp),
                        attn.q_norm.weight.clone(),
                    ));
                    out.push((
                        format!("{}.self_attn.k_norm.weight", lp),
                        attn.k_norm.weight.clone(),
                    ));
                }
                Qwen3_5DecoderAttention::Linear(lin) => {
                    out.extend(
                        lin.in_proj_qkv
                            .named_parameters(&format!("{}.linear_attn.in_proj_qkv", lp)),
                    );
                    out.extend(
                        lin.in_proj_z
                            .named_parameters(&format!("{}.linear_attn.in_proj_z", lp)),
                    );
                    out.extend(
                        lin.in_proj_a
                            .named_parameters(&format!("{}.linear_attn.in_proj_a", lp)),
                    );
                    out.extend(
                        lin.in_proj_b
                            .named_parameters(&format!("{}.linear_attn.in_proj_b", lp)),
                    );
                    out.extend(
                        lin.out_proj
                            .named_parameters(&format!("{}.linear_attn.out_proj", lp)),
                    );
                    out.push((
                        format!("{}.linear_attn.conv1d.weight", lp),
                        lin.conv1d_weight.clone(),
                    ));
                    out.push((format!("{}.linear_attn.A_log", lp), lin.A_log.clone()));
                    out.push((format!("{}.linear_attn.dt_bias", lp), lin.dt_bias.clone()));
                    out.push((
                        format!("{}.linear_attn.norm.weight", lp),
                        lin.norm.weight.clone(),
                    ));
                }
            }
            out.push((
                format!("{}.post_attention_layernorm.weight", lp),
                layer.post_attention_layernorm.weight.clone(),
            ));
            out.extend(
                layer
                    .mlp
                    .gate_proj
                    .named_parameters(&format!("{}.mlp.gate_proj", lp)),
            );
            out.extend(
                layer
                    .mlp
                    .up_proj
                    .named_parameters(&format!("{}.mlp.up_proj", lp)),
            );
            out.extend(
                layer
                    .mlp
                    .down_proj
                    .named_parameters(&format!("{}.mlp.down_proj", lp)),
            );
        }
        out
    }

    pub fn apply_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        for (name, param) in self.named_parameters(prefix) {
            if let Some(src) = state.get(&name) {
                let mut param_lock = param.lock();
                let src_lock = src.lock();
                let pshape = param_lock.storage.shape().to_vec();
                let sshape = src_lock.storage.shape().to_vec();
                if pshape == sshape {
                    param_lock.storage = src_lock.storage.clone();
                    param_lock.dtype = src_lock.dtype;
                } else if pshape.len() == 2
                    && sshape.len() == 2
                    && pshape[0] == sshape[1]
                    && pshape[1] == sshape[0]
                {
                    // HF stores weights as [out_features, in_features]; transpose to [in, out]
                    let arr = src_lock.storage.to_f32_array();
                    drop(src_lock);
                    if let Ok(m) = arr.into_dimensionality::<ndarray::Ix2>() {
                        let transposed = Tensor::new(m.reversed_axes().into_dyn(), false);
                        let tlock = transposed.lock();
                        param_lock.storage = tlock.storage.clone();
                        param_lock.dtype = tlock.dtype;
                    } else {
                        return Err(format!(
                            "Qwen3.5: failed to transpose 2D weight for '{}': module {:?} vs state {:?}",
                            name, pshape, sshape
                        ));
                    }
                } else {
                    return Err(format!(
                        "Qwen3.5 shape mismatch for '{}': module {:?} vs state {:?}",
                        name, pshape, sshape
                    ));
                }
            } else {
                log::warn!("Qwen3.5: weight '{}' not found in state dict", name);
            }
        }
        Ok(())
    }
}

impl crate::nn::LlamaStyleModel for Qwen3_5TextModel {
    fn forward_single_token(
        &mut self,
        token_id: &Tensor,
        causal_offset: Option<usize>,
    ) -> Result<Tensor, String> {
        self.forward_single_token(token_id, causal_offset)
    }

    fn init_kv_caches(&mut self, seq_len: usize) -> Result<(), String> {
        self.init_kv_caches(seq_len)
    }

    fn reset_kv_caches(&mut self) {
        self.reset_kv_caches()
    }

    fn clone_model(&self) -> Box<dyn crate::nn::LlamaStyleModel> {
        Box::new(self.clone())
    }

    fn apply_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        root: &str,
    ) -> Result<(), String> {
        self.apply_state_dict(state, root)
    }
}
