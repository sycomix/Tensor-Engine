use crate::nn::Linear;
use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};
// ALiBi slope computation helper
fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
    // simple decreasing slopes: 2^(-i/num_heads)
    let mut slopes = Vec::with_capacity(n_heads);
    for i in 0..n_heads {
        let x = (i as f32) / (n_heads as f32);
        slopes.push(2f32.powf(-x));
    }
    slopes
}
// Arc import no longer used here; keeping module compact.

/// MultiHeadAttention implemented using existing Linear layers and matmul.
/// This is a reference implementation optimized for clarity and correct shapes.
pub struct MultiHeadAttention {
    pub linear_q: Linear,
    pub linear_k: Linear,
    pub linear_v: Linear,
    pub linear_o: Linear,
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub kv_heads: usize,
    pub use_rope: bool,
    pub use_alibi: bool,
    pub alibi_slopes: Option<Vec<f32>>,
    pub relative_bias: Option<crate::tensor::Tensor>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );
        let d_k = d_model / num_heads;
        MultiHeadAttention {
            linear_q: Linear::new(d_model, d_model, true),
            linear_k: Linear::new(d_model, d_model, true),
            linear_v: Linear::new(d_model, d_model, true),
            linear_o: Linear::new(d_model, d_model, true),
            num_heads,
            d_model,
            d_k,
            kv_heads: num_heads,
            use_rope: false,
            use_alibi: false,
            alibi_slopes: None,
            relative_bias: None,
        }
    }

    /// Extended constructor with kv_heads and RoPE support
    pub fn new_with_kv_and_rope(
        d_model: usize,
        num_heads: usize,
        kv_heads: usize,
        use_rope: bool,
    ) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );
        assert!(
            num_heads % kv_heads == 0,
            "num_heads must be divisible by kv_heads"
        );
        let d_k = d_model / num_heads;
        MultiHeadAttention {
            linear_q: Linear::new(d_model, d_model, true),
            linear_k: Linear::new(d_model, d_model, true),
            linear_v: Linear::new(d_model, d_model, true),
            linear_o: Linear::new(d_model, d_model, true),
            num_heads,
            d_model,
            d_k,
            kv_heads,
            use_rope,
            use_alibi: false,
            alibi_slopes: None,
            relative_bias: None,
        }
    }

    /// Enable ALiBi for attention and compute default slopes for each head.
    pub fn with_alibi(mut self) -> Self {
        let slopes = compute_alibi_slopes(self.num_heads);
        self.use_alibi = true;
        self.alibi_slopes = Some(slopes);
        self
    }

    /// Attach a relative position bias tensor (num_heads, 2*max_dist+1)
    pub fn with_relative_bias(mut self, bias: crate::tensor::Tensor) -> Self {
        self.relative_bias = Some(bias);
        self
    }

    /// Forward attention. Input: [batch, seq, d_model]. Output: [batch, seq, d_model]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Create Q, K, V
        let q = self.linear_q.forward(x);
        let k = self.linear_k.forward(x);
        let v = self.linear_v.forward(x);

        // reshape to [batch*seq, d_model] -> then split heads by reshaping
        let shape = q.lock().storage.shape();
        if shape.len() != 3 {
            log::error!("MultiHeadAttention::forward expected input with 3 dims [batch, seq, d_model], got {:?}", shape);
            return x.clone();
        }
        let b = shape[0];
        let seq = shape[1];
        let _d = self.d_model;

        // (Note) earlier flattened reshapes removed in favor of per-head processing below

        // Reshape into heads: q/k/v of shape [b, seq, num_heads, head_dim]
        let head_dim = self.d_model / self.num_heads;
        let q_heads = match q.reshape(vec![b, seq, self.num_heads, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("reshape error in MultiHeadAttention::forward (q): {}", e);
                return x.clone();
            }
        };
        // K and V may have kv_heads
        let mut k_heads = match k.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads])
        {
            Ok(t) => t,
            Err(e) => {
                log::error!(
                    "reshape error in MultiHeadAttention::forward (k pre-expansion): {}",
                    e
                );
                return x.clone();
            }
        };
        let mut v_heads = match v.reshape(vec![b, seq, self.kv_heads, self.d_model / self.kv_heads])
        {
            Ok(t) => t,
            Err(e) => {
                log::error!(
                    "reshape error in MultiHeadAttention::forward (v pre-expansion): {}",
                    e
                );
                return x.clone();
            }
        };
        // Expand k/v to match num_heads by reshaping kv head_dim into (group_size, head_dim)
        let group_size = self.num_heads / self.kv_heads;
        if self.kv_heads != self.num_heads {
            let head_dim_kv = self.d_model / self.kv_heads;
            let new_head_dim = head_dim_kv / group_size; // should equal head_dim
            match k_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                Ok(t) => k_heads = t,
                Err(e) => {
                    log::error!(
                        "reshape error in MultiHeadAttention::forward (k split): {}",
                        e
                    );
                    return x.clone();
                }
            }
            match v_heads.reshape(vec![b, seq, self.kv_heads, group_size, new_head_dim]) {
                Ok(t) => v_heads = t,
                Err(e) => {
                    log::error!(
                        "reshape error in MultiHeadAttention::forward (v split): {}",
                        e
                    );
                    return x.clone();
                }
            }
            // Merge kv_heads and group_size into num_heads
            match k_heads.reshape(vec![b, seq, self.kv_heads * group_size, new_head_dim]) {
                Ok(t) => k_heads = t,
                Err(e) => {
                    log::error!(
                        "reshape error in MultiHeadAttention::forward (k merge): {}",
                        e
                    );
                    return x.clone();
                }
            }
            match v_heads.reshape(vec![b, seq, self.kv_heads * group_size, new_head_dim]) {
                Ok(t) => v_heads = t,
                Err(e) => {
                    log::error!(
                        "reshape error in MultiHeadAttention::forward (v merge): {}",
                        e
                    );
                    return x.clone();
                }
            }
        } else {
            // if kv_heads == num_heads, reshape to match dimensions
            match k_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                Ok(t) => k_heads = t,
                Err(e) => {
                    log::error!(
                        "reshape error in MultiHeadAttention::forward (k reshape): {}",
                        e
                    );
                    return x.clone();
                }
            }
            match v_heads.reshape(vec![b, seq, self.num_heads, head_dim]) {
                Ok(t) => v_heads = t,
                Err(e) => {
                    log::error!(
                        "reshape error in MultiHeadAttention::forward (v reshape): {}",
                        e
                    );
                    return x.clone();
                }
            }
        }

        let mut q_proc = q_heads;
        let mut k_proc = k_heads;
        let v_proc = v_heads;
        // Apply RoPE if enabled
        if self.use_rope {
            q_proc = q_proc.rope(self.num_heads);
            k_proc = k_proc.rope(self.num_heads);
        }

        // Permute to batch-first for heads: [b, num_heads, seq, head_dim]
        let q_perm = q_proc.permute(vec![0, 2, 1, 3]);
        let k_perm = k_proc.permute(vec![0, 2, 1, 3]);
        let v_perm = v_proc.permute(vec![0, 2, 1, 3]);

        // reshape to [b*num_heads, seq, head_dim]
        let q2 = match q_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("reshape error (q_heads to 3D): {}", e);
                return x.clone();
            }
        };
        let k2 = match k_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("reshape error (k_heads to 3D): {}", e);
                return x.clone();
            }
        };
        let v2 = match v_perm.reshape(vec![b * self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("reshape error (v_heads to 3D): {}", e);
                return x.clone();
            }
        };

        // Compute attention per head using batched matmul
        let k2t = k2.permute(vec![0, 2, 1]); // [batch, head_dim, seq]
        let qk = q2.batched_matmul(&k2t);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
        let scaled = qk.mul(&scalar_tensor);
            // Optionally add ALiBi or relative positional bias to scaled logits
            let mut scaled_logits = scaled.clone();
            if self.use_alibi {
                // build alibi bias: shape [b*num_heads, seq, seq]
                let slopes = if let Some(s) = &self.alibi_slopes { s.clone() } else { compute_alibi_slopes(self.num_heads) };
                // create bias array
                let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                for batch in 0..b {
                    for h in 0..self.num_heads {
                        let slope = slopes[h];
                        for i in 0..seq {
                            for j in 0..seq {
                                // relative distance j-i
                                let dist = (j as isize - i as isize) as f32;
                                let val = -slope * dist;
                                bias_arr[[batch * self.num_heads + h, i, j]] = val;
                            }
                        }
                    }
                }
                let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                scaled_logits = scaled_logits.add(&bias_t);
            }
            // Add relative bias if provided (shape expected [num_heads, range])
            if let Some(rb) = &self.relative_bias {
                // rb shape [num_heads, 2*max+1]
                // create bias matrix for seq x seq
                let shape = rb.lock().storage.shape();
                if shape.len() == 2 {
                    let max_range = (shape[1] - 1) / 2;
                    let mut bias_arr = ndarray::ArrayD::<f32>::zeros(IxDyn(&[b * self.num_heads, seq, seq]));
                    let rb_arr = rb.lock().storage.to_f32_array();
                    let rb_view = rb_arr.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                    for batch in 0..b {
                        for h in 0..self.num_heads {
                            for i in 0..seq {
                                for j in 0..seq {
                                    let rel = (j as isize - i as isize)
                                        .max(-(max_range as isize))
                                        .min(max_range as isize) as isize;
                                    let idx = (rel + max_range as isize) as usize;
                                    bias_arr[[batch * self.num_heads + h, i, j]] = rb_view[[h, idx]];
                                }
                            }
                        }
                    }
                    let bias_t = crate::tensor::Tensor::new(bias_arr, false);
                    scaled_logits = scaled_logits.add(&bias_t);
                }
            }
        let attn = scaled_logits.softmax(2);
        let out = attn.batched_matmul(&v2);
        // reshape out [b*num_heads, seq, head_dim] -> [b, num_heads, seq, head_dim]
        let out2 = match out.reshape(vec![b, self.num_heads, seq, head_dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("reshape error during attention result reshape: {}", e);
                return x.clone();
            }
        };
        // permute back to [b, seq, num_heads, head_dim]
        let out3 = out2.permute(vec![0, 2, 1, 3]);
        // reshape to [b, seq, d_model]
        let out4 = match out3.reshape(vec![b, seq, self.d_model]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("reshape error during attention final reshape: {}", e);
                return x.clone();
            }
        };
        self.linear_o.forward(&out4)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.linear_q.parameters();
        p.extend(self.linear_k.parameters());
        p.extend(self.linear_v.parameters());
        p.extend(self.linear_o.parameters());
        p
    }
}

/// TransformerBlock: attention + residual + layernorm + feedforward + residual
pub struct TransformerBlock {
    pub mha: MultiHeadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
}

impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self {
        TransformerBlock {
            mha: MultiHeadAttention::new(d_model, num_heads),
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
        }
    }

    pub fn new_with_kv_and_rope(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
        use_rope: bool,
    ) -> Self {
        TransformerBlock {
            mha: MultiHeadAttention::new_with_kv_and_rope(d_model, num_heads, kv_heads, use_rope),
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
        }
    }

    pub fn forward_block(&self, x: &Tensor) -> Tensor {
        log::info!(
            "TransformerBlock forward: input shape {:?}",
            x.lock().storage.shape()
        );
        let attn_out = self.mha.forward(x);
        let x2 = x.add(&attn_out);
        // LayerNorm not exposed as module; use Tensor::layer_norm with axis=-1
        let dim = x.lock().storage.shape()[2];
        // create gamma and beta as ones/zeros (affine) for now
        let gamma = Tensor::new(ndarray::Array::ones(IxDyn(&[dim])), true);
        let beta = Tensor::new(ndarray::Array::zeros(IxDyn(&[dim])), true);
        let x2norm = x2.layer_norm(2, 1e-5, &gamma, &beta);
        let ff = self.linear1.forward(&x2norm).relu();
        let ff = self.linear2.forward(&ff);
        log::debug!(
            "TransformerBlock forward complete: output shape {:?}",
            x2.lock().storage.shape()
        );
        x2.add(&ff)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.mha.parameters();
        p.extend(self.linear1.parameters());
        p.extend(self.linear2.parameters());
        p
    }
}
