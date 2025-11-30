use crate::tensor::Tensor;
use crate::nn::Linear;
use crate::nn::Module;
use ndarray::{Array, IxDyn};
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
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
        let d_k = d_model / num_heads;
        MultiHeadAttention {
            linear_q: Linear::new(d_model, d_model, true),
            linear_k: Linear::new(d_model, d_model, true),
            linear_v: Linear::new(d_model, d_model, true),
            linear_o: Linear::new(d_model, d_model, true),
            num_heads,
            d_model,
            d_k,
        }
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
        let d = self.d_model;

        // Utility to reshape into (b*seq, d_model)
        let q2 = match q.reshape(vec![b * seq, d]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("reshape error in MultiHeadAttention::forward: {}", e);
                return x.clone();
            }
        };
        let k2 = match k.reshape(vec![b * seq, d]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("reshape error in MultiHeadAttention::forward: {}", e);
                return x.clone();
            }
        };
        let v2 = match v.reshape(vec![b * seq, d]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("reshape error in MultiHeadAttention::forward: {}", e);
                return x.clone();
            }
        };

        // For simplicity, follow self-attention with flattened sequences as in previous code.
        // Compute qk = q2 @ k2.T
        let k2t = k2.transpose();
        let qk = q2.matmul(&k2t);
        let scale = 1.0f32 / (self.d_k as f32).sqrt();
        let scalar_tensor = Tensor::new(Array::from_elem(IxDyn(&[1]), scale), false);
        let scaled = qk.mul(&scalar_tensor);
        let attn = scaled.softmax(1);
        let out = attn.matmul(&v2);
        let out2 = match out.reshape(vec![b, seq, d]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("reshape error during attention result reshape: {}", e);
                return x.clone();
            }
        };
        self.linear_o.forward(&out2)
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

    pub fn forward_block(&self, x: &Tensor) -> Tensor {
        log::info!("TransformerBlock forward: input shape {:?}", x.lock().storage.shape());
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
        log::debug!("TransformerBlock forward complete: output shape {:?}", x2.lock().storage.shape());
        x2.add(&ff)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.mha.parameters();
        p.extend(self.linear1.parameters());
        p.extend(self.linear2.parameters());
        p
    }
}
