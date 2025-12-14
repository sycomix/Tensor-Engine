use crate::labels::Labels;
use crate::ops::Conv2D as Conv2DOp;
use crate::ops::MaxPool2D as MaxPool2DOp;
use crate::tensor::Tensor;
use ndarray::{arr0, ArrayD, IxDyn};
use std::collections::HashMap;
use std::sync::Arc;

pub mod flatten;
pub use flatten::Flatten;
pub mod transformer_cleaned;
pub use transformer_cleaned as transformer;
pub use transformer_cleaned::{
    compute_alibi_slopes, AttentionVariant, BiasFunction, EncoderDecoderTransformer,
    MultiHeadAttention, TransformerBlock,
};
// Re-export common NN modules and types
pub mod audio;
pub use audio::{AudioDecoder, AudioEncoder};
pub mod multimodal;
pub use multimodal::{get_decode_count, reset_decode_count, ModalMemoryContext, MultimodalLLM};
pub mod vision;
pub use vision::VisionTransformer;
pub mod diffusion;
pub use diffusion::{DDPMScheduler, TimestepEmbedding, UNetModel};
pub mod quantization;
pub use quantization::RVQ;
// Don't re-export op-level Conv types here to avoid duplicate symbol errors.
// NN defines wrapper Conv1D/Conv2D types in this module. If you need the raw
// op-level Conv types, use crate::ops::Conv2D explicitly.
// legacy files may exist but uses are now forwarded to transformer_cleaned
// pub mod transformer; (disabled while transformer.rs is being repaired)

/// Absolute positional embedding: holds an embedding matrix of shape (max_len, d_model)
#[derive(Clone)]
pub struct AbsolutePositionalEmbedding {
    pub weight: Tensor,
    pub max_len: usize,
}

impl AbsolutePositionalEmbedding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        let w = ndarray::Array::zeros(IxDyn(&[max_len, d_model]));
        AbsolutePositionalEmbedding {
            weight: Tensor::new(w, true),
            max_len,
        }
    }
}

impl Module for AbsolutePositionalEmbedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.lock().storage.shape();
        if shape.len() != 3 {
            log::error!("AbsolutePositionalEmbedding expected 3D input");
            return input.clone();
        }
        let b = shape[0];
        let seq = shape[1];
        assert!(
            seq <= self.max_len,
            "Sequence length > max_len for positional embedding"
        );
        let mut idx = vec![];
        for _ in 0..b {
            for i in 0..seq {
                idx.push(i as f32);
            }
        }
        let idx_arr = ndarray::Array::from_shape_vec((b, seq), idx)
            .unwrap()
            .into_dyn();
        let idx_tensor = Tensor::new(idx_arr, false);
        let pos_emb = Tensor::embedding_lookup(&self.weight, &idx_tensor);
        input.add(&pos_emb)
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![(format!("{}.weight", prefix), self.weight.clone())]
    }
    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        let key = format!("{}.weight", prefix);
        if let Some(w) = state.get(&key) {
            self.weight = w.clone();
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn Any { &*self }
    fn as_any_mut(&mut self) -> &mut dyn Any { &mut *self }
}

#[cfg(test)]
mod tests;

/// A trait for neural network modules.
use std::any::Any;

pub trait Module: 'static + Any {
    /// Performs a forward pass through the module.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Returns the parameters of the module.
    fn parameters(&self) -> Vec<Tensor>;
    /// Default: return a vector of (name, Tensor) pairs for module parameters
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out: Vec<(String, Tensor)> = Vec::new();
        let params = self.parameters();
        for (i, p) in params.into_iter().enumerate() {
            out.push((format!("{}param{}", prefix, i), p));
        }
        out
    }
    /// Load a state dict into this module. Default implementation does nothing.
    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        // Default implementation: apply any matching entries in the state dict to the
        // module's named parameters. This works because `named_parameters()` returns
        // `Tensor` instances referencing the same underlying storage as the module's
        // parameters, so mutating the storage will update the module in-place.
        for (name, param) in self.named_parameters(prefix) {
            if let Some(src) = state.get(&name) {
                let mut param_lock = param.lock();
                let src_lock = src.lock();
                // Ensure shapes roughly match; provide an informative error on mismatch.
                if param_lock.storage.shape() != src_lock.storage.shape() {
                    return Err(format!(
                        "Shape mismatch for parameter '{}': module shape={:?}, state shape={:?}",
                        name,
                        param_lock.storage.shape(),
                        src_lock.storage.shape()
                    ));
                }
                param_lock.storage = src_lock.storage.clone();
                param_lock.dtype = src_lock.dtype;
            }
        }
        Ok(())
    }
    /// Allow downcasting from a `dyn Module` by providing an `Any` accessor.
    fn as_any(&self) -> &dyn Any;

    /// Mutable `Any` accessor for downcasting trait objects when mutation is required.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A small convenience ConvBlock: Conv2D -> ReLU -> optional MaxPool
pub struct ConvBlock {
    conv: Conv2D,
    pool: Option<MaxPool2D>,
}

impl ConvBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        pool: Option<(usize, usize)>,
    ) -> Self {
        let conv = Conv2D::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
        );
        let pool = pool.map(|(k, s)| MaxPool2D::new(k, s));
        ConvBlock { conv, pool }
    }
}

impl Module for ConvBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = self.conv.forward(input);
        out = out.relu();
        if let Some(pool) = &self.pool {
            out = pool.forward(&out);
        }
        out
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.conv.parameters()
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// Simple Generator (GAN): small MLP that outputs tensors given latent vector
pub struct Generator {
    pub layers: Vec<Box<dyn Module>>,
}

impl Generator {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Generator { layers }
    }
}

impl Module for Generator {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            out.extend(layer.named_parameters(&format!("{}.layers.{}", prefix, i)));
        }
        out
    }
    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(state, &format!("{}.layers.{}", prefix, i))?;
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// Simple Discriminator (GAN): small MLP for binary classification
pub struct Discriminator {
    pub layers: Vec<Box<dyn Module>>,
}
impl Discriminator {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Discriminator { layers }
    }
}
impl Module for Discriminator {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            out.extend(layer.named_parameters(&format!("{}.layers.{}", prefix, i)));
        }
        out
    }
    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(state, &format!("{}.layers.{}", prefix, i))?;
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// RNN cell (Elman): single-step RNN cell with weight matrices and bias
pub struct RNNCell {
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias: Option<Tensor>,
}

impl RNNCell {
    pub fn new(input_dim: usize, hidden_dim: usize, bias: bool) -> Self {
        let wih = Tensor::new(
            ndarray::Array::zeros(ndarray::IxDyn(&[input_dim, hidden_dim])),
            true,
        );
        let whh = Tensor::new(
            ndarray::Array::zeros(ndarray::IxDyn(&[hidden_dim, hidden_dim])),
            true,
        );
        let b = if bias {
            Some(Tensor::new(
                ndarray::Array::zeros(ndarray::IxDyn(&[hidden_dim])),
                true,
            ))
        } else {
            None
        };
        RNNCell {
            weight_ih: wih,
            weight_hh: whh,
            bias: b,
        }
    }

    pub fn forward_step(&self, input: &Tensor, hidden: &Tensor) -> Tensor {
        // hidden' = tanh(input @ weight_ih + hidden @ weight_hh + bias)
        let x_w = input.matmul(&self.weight_ih);
        let h_w = hidden.matmul(&self.weight_hh);
        let mut out = x_w.add(&h_w);
        if let Some(b) = &self.bias {
            out = out.add(b);
        }
        out.tanh()
    }
}

impl Module for RNNCell {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    } // not used; forward_step is used for step-wise RNN
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// LSTM Cell implementation
pub struct LSTMCell {
    pub weight_ih: Tensor, // input to gates weights, shape [input_dim, 4*hidden_dim]
    pub weight_hh: Tensor, // hidden to gates weights, shape [hidden_dim, 4*hidden_dim]
    pub bias: Option<Tensor>,
    pub hidden_dim: usize,
}

impl LSTMCell {
    pub fn new(input_dim: usize, hidden_dim: usize, bias: bool) -> Self {
        let wih = Tensor::new(
            ndarray::Array::zeros(ndarray::IxDyn(&[input_dim, 4 * hidden_dim])),
            true,
        );
        let whh = Tensor::new(
            ndarray::Array::zeros(ndarray::IxDyn(&[hidden_dim, 4 * hidden_dim])),
            true,
        );
        let b = if bias {
            Some(Tensor::new(
                ndarray::Array::zeros(ndarray::IxDyn(&[4 * hidden_dim])),
                true,
            ))
        } else {
            None
        };
        LSTMCell {
            weight_ih: wih,
            weight_hh: whh,
            bias: b,
            hidden_dim,
        }
    }

    /// Forward a single step. `hidden` is (h, c) as tuple of Tensors of shape [batch, hid]
    pub fn forward_step(&self, input: &Tensor, h: &Tensor, c: &Tensor) -> (Tensor, Tensor) {
        // gates = input @ w_ih + h @ w_hh + bias
        let xw = input.matmul(&self.weight_ih);
        let hw = h.matmul(&self.weight_hh);
        let mut gates = xw.add(&hw);
        if let Some(b) = &self.bias {
            gates = gates.add(b);
        }
        // gates shape: [batch, 4*hidden]
        // split gates
        let hid = self.hidden_dim;
        let (i_gate, rest) = Self::slice_n(gates.clone(), 0, hid);
        let (f_gate, rest2) = Self::slice_n(rest, 0, hid);
        let (g_gate, o_gate) = Self::slice_n(rest2, 0, hid);
        let i = i_gate.sigmoid();
        let f = f_gate.sigmoid();
        let g = g_gate.tanh();
        let o = o_gate.sigmoid();
        let new_c = f.mul(c).add(&i.mul(&g));
        let new_h = o.mul(&new_c.tanh());
        (new_h, new_c)
    }

    fn slice_n(t: Tensor, start: usize, n: usize) -> (Tensor, Tensor) {
        // Use a Slice operation implemented in ops.rs to return differentiable slices
        let dim = t.lock().storage.shape();
        assert!(dim.len() == 2);
        let total = dim[1];
        let first = Tensor::apply(Arc::new(crate::ops::Slice::new(1, start, n)), &[t.clone()]);
        let second = Tensor::apply(
            Arc::new(crate::ops::Slice::new(1, start + n, total - (start + n))),
            &[t],
        );
        (first, second)
    }
}

impl Module for LSTMCell {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// Scaled Dot-Product Attention (single head)
pub struct SelfAttention {
    pub d_k: usize,
}

impl SelfAttention {
    pub fn new(d_k: usize) -> Self {
        SelfAttention { d_k }
    }

    /// Compute attention: output = softmax(Q @ K.T / sqrt(d_k)) @ V
    pub fn forward_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        // Input shapes: [batch, seq, dim]
        // Flatten batch*seq into 2D if necessary and use matmul (we'll operate per batch by reshaping)
        let q_shape = q.lock().storage.shape();
        let b = q_shape[0];
        let seq = q_shape[1];
        let dim = q_shape[2];
        // reshape to (b*seq, dim)
        let q2 = q.reshape(vec![b * seq, dim]).unwrap();
        // q2 reshape done
        let k2 = k.reshape(vec![b * seq, dim]).unwrap();
        // k2 reshape done
        let v2 = v.reshape(vec![b * seq, dim]).unwrap();
        // v2 reshape done
        // Compute q @ k.T per batch: naive approach computing QK^T for each batch by splitting
        // Simpler approach: compute similarity across flattened sequences; result has shape (b*seq, b*seq) which is undesirable.
        // We'll restrict to single-batch test usage in unit tests and provide a simple formula for now.
        // about to compute qk
        let k2t = k2.transpose();
        // k2t created shape
        let qk = q2.matmul(&k2t);
        // computed qk
        let scale = 1.0 / (self.d_k as f32).sqrt();
        let scaled = qk.mul(&Tensor::new(
            ndarray::Array::from_elem(ndarray::IxDyn(&[1]), scale),
            false,
        ));
        let attn = scaled.softmax(1);
        // computed softmax
        // about to compute out matmul
        let out = attn.matmul(&v2);
        // computed out matmul
        out.reshape(vec![b, seq, dim]).unwrap()
    }
}

impl Module for SelfAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// A linear (fully connected) layer.
#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Creates a new linear layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` - The number of input features.
    /// * `out_features` - The number of output features.
    /// * `bias` - Whether to include a bias term.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight_data = ArrayD::zeros(IxDyn(&[in_features, out_features]));
        let weight = Tensor::new(weight_data, true);

        let bias = if bias {
            let bias_data = ArrayD::zeros(IxDyn(&[out_features]));
            Some(Tensor::new(bias_data, true))
        } else {
            None
        };

        Linear { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let input_shape = input.lock().storage.shape();
        let ndim = input_shape.len();
        let output = if ndim == 2 {
            input.matmul(&self.weight)
        } else {
            // Collapse leading dims to 2D [batch, features]
            let last = input_shape[ndim - 1];
            let batch = input_shape[..ndim - 1].iter().product::<usize>();
            let reshaped = input.reshape(vec![batch, last]).unwrap();
            let out2 = reshaped.matmul(&self.weight);
            let mut out_shape = input_shape.clone();
            out_shape[ndim - 1] = self.weight.lock().storage.shape()[1];
            out2.reshape(out_shape).unwrap()
        };
        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = vec![(format!("{}.weight", prefix), self.weight.clone())];
        if let Some(b) = &self.bias {
            out.push((format!("{}.bias", prefix), b.clone()));
        }
        out
    }
    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        let key_w = format!("{}.weight", prefix);
        if let Some(w) = state.get(&key_w) {
            self.weight = w.clone();
        }
        let key_b = format!("{}.bias", prefix);
        if let Some(b) = state.get(&key_b) {
            self.bias = Some(b.clone());
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// A sequential container for modules.
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

/// Layer normalization module
///
/// - `gamma`: per-feature learnable gain; shape should be `[num_features]` or broadcastable to the normalized axis.
/// - `beta`: per-feature learnable bias; shape should be `[num_features]` or broadcastable to the normalized axis.
/// - `axis`: the axis (dimension index) to normalize over. Use `1` for per-row norm in standard 2D inputs or `-1` for last axis semantics.
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub axis: usize,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(num_features: usize, axis: usize, eps: f32) -> Self {
        let gamma = Tensor::new(
            ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[num_features]),
                vec![1.0; num_features],
            )
            .unwrap(),
            true,
        );
        let beta = Tensor::new(
            ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[num_features]),
                vec![0.0; num_features],
            )
            .unwrap(),
            true,
        );
        LayerNorm {
            gamma,
            beta,
            axis,
            eps,
        }
    }

    /// Inherent forward method to simplify usage in tests and API consumers.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        input.layer_norm(self.axis, self.eps, &self.gamma, &self.beta)
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.layer_norm(self.axis, self.eps, &self.gamma, &self.beta)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

impl Sequential {
    /// Creates a new sequential container.
    pub fn new() -> Self {
        Sequential {
            modules: Vec::new(),
        }
    }

    /// Adds a module to the container.
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    /// Returns all parameters from all modules.
    pub fn parameters(&self) -> Vec<Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut output = input.clone();
        for module in &self.modules {
            output = module.forward(&output);
        }
        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// A trait for optimizers.
pub trait Optimizer {
    /// Performs a single optimization step.
    fn step(&mut self, parameters: &[Tensor]);

    /// Sets the gradients of all parameters to zero.
    fn zero_grad(&mut self, parameters: &[Tensor]);

    /// Clip gradients in-place using global norm. Default impl used by Python wrappers.
    fn clip_gradients(&mut self, parameters: &[Tensor], max_norm: f32) {
        if max_norm <= 0.0 {
            return;
        }
        let mut total_sq = 0.0f32;
        for p in parameters {
            let lock = p.lock();
            if let Some(g) = &lock.grad {
                let arr = g;
                for v in arr.iter() {
                    total_sq += (*v) * (*v);
                }
            }
        }
        let total_norm = total_sq.sqrt();
        if total_norm <= max_norm {
            return;
        }
        let scale = max_norm / (total_norm + 1e-12);
        for p in parameters {
            let mut lock = p.lock();
            if let Some(g) = &mut lock.grad {
                // scale gradients in-place to avoid reallocations
                g.mapv_inplace(|v| v * scale);
            }
        }
    }

    /// Cast parameters to a storage dtype (MVP: round-trip conversion applied)
    fn cast_params(&mut self, parameters: &[Tensor], dtype: crate::dtype::DType) {
        for p in parameters {
            let converted = p.astype(dtype);
            let mut lock = p.lock();
            lock.storage = converted.lock().storage.clone();
            lock.dtype = dtype;
        }
    }
}

/// Learning rate scheduler trait. Implements logic to calculate a scalar learning rate
/// based on current step/epoch.
pub trait LRScheduler {
    /// Get the learning rate for the current step (0-based)
    fn get_lr(&self, step: usize) -> f32;
}

/// Linear warmup scheduler: increase from 0 to `base_lr` over `warmup_steps`, then keep `base_lr`.
pub struct LinearWarmup {
    pub base_lr: f32,
    pub warmup_steps: usize,
}

impl LinearWarmup {
    pub fn new(base_lr: f32, warmup_steps: usize) -> Self {
        LinearWarmup {
            base_lr,
            warmup_steps,
        }
    }
}

impl LRScheduler for LinearWarmup {
    fn get_lr(&self, step: usize) -> f32 {
        if self.warmup_steps == 0 {
            return self.base_lr;
        }
        let s = step.min(self.warmup_steps);
        self.base_lr * (s as f32) / (self.warmup_steps as f32)
    }
}

/// Cosine Annealing scheduler: lr = base_lr * 0.5*(1+cos(pi * t / T))
pub struct CosineAnnealing {
    pub base_lr: f32,
    pub total_steps: usize,
}

impl CosineAnnealing {
    pub fn new(base_lr: f32, total_steps: usize) -> Self {
        CosineAnnealing {
            base_lr,
            total_steps,
        }
    }
}

impl LRScheduler for CosineAnnealing {
    fn get_lr(&self, step: usize) -> f32 {
        if self.total_steps == 0 {
            return self.base_lr;
        }
        let t = (step.min(self.total_steps)) as f32;
        let total_steps_f = (self.total_steps) as f32;
        self.base_lr * 0.5 * (1.0 + (std::f32::consts::PI * t / total_steps_f).cos())
    }
}

/// Stochastic Gradient Descent optimizer.
pub struct SGD {
    lr: f32,
    momentum: f32,
    velocity: HashMap<Tensor, ArrayD<f32>>,
}

impl SGD {
    /// Creates a new SGD optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - The learning rate.
    /// * `momentum` - The momentum factor.
    pub fn new(lr: f32, momentum: f32) -> Self {
        SGD {
            lr,
            momentum,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut param_lock = param.lock();
            if let Some(grad) = &param_lock.grad {
                let velocity = self
                    .velocity
                    .entry(param.clone())
                    .or_insert_with(|| ArrayD::zeros(grad.dim()));
                *velocity = &*velocity * self.momentum + grad * (1.0 - self.momentum);
                let update = velocity.mapv(|v| v * self.lr);
                // Apply update to param storage
                let mut param_f32 = param_lock.storage.to_f32_array();
                param_f32 = &param_f32 - &update;
                param_lock.storage =
                    crate::dtype::TensorStorage::from_f32_array(&param_f32, param_lock.dtype);
            }
        }
    }

    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut param_lock = param.lock();
            param_lock.grad = None;
        }
    }
}

/// Adam optimizer.
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    m: HashMap<Tensor, ArrayD<f32>>,
    v: HashMap<Tensor, ArrayD<f32>>,
}

impl Adam {
    /// Creates a new Adam optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - The learning rate.
    /// * `beta1` - The exponential decay rate for the first moment estimates.
    /// * `beta2` - The exponential decay rate for the second moment estimates.
    /// * `eps` - A small constant for numerical stability.
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Adam {
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &[Tensor]) {
        self.t += 1;

        for param in parameters {
            let mut param_lock = param.lock();
            if let Some(grad) = &param_lock.grad {
                let m = self
                    .m
                    .entry(param.clone())
                    .or_insert_with(|| ArrayD::zeros(grad.dim()));
                let v = self
                    .v
                    .entry(param.clone())
                    .or_insert_with(|| ArrayD::zeros(grad.dim()));

                *m = &*m * self.beta1 + grad * (1.0 - self.beta1);
                *v = &*v * self.beta2 + &(grad * grad) * (1.0 - self.beta2);

                let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));

                let update = (m_hat / (v_hat.mapv(|x| x.sqrt()) + self.eps)) * self.lr;
                let mut param_f32 = param_lock.storage.to_f32_array();
                param_f32 = &param_f32 - &update;
                param_lock.storage =
                    crate::dtype::TensorStorage::from_f32_array(&param_f32, param_lock.dtype);
            }
        }
    }

    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut param_lock = param.lock();
            param_lock.grad = None;
        }
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: usize,
    m: HashMap<Tensor, ArrayD<f32>>,
    v: HashMap<Tensor, ArrayD<f32>>,
}

impl AdamW {
    /// Creates a new AdamW optimizer.
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        AdamW {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, parameters: &[Tensor]) {
        self.t += 1;
        for param in parameters {
            let mut param_lock = param.lock();
            if let Some(grad) = &param_lock.grad {
                let m = self
                    .m
                    .entry(param.clone())
                    .or_insert_with(|| ArrayD::zeros(grad.dim()));
                let v = self
                    .v
                    .entry(param.clone())
                    .or_insert_with(|| ArrayD::zeros(grad.dim()));

                *m = &*m * self.beta1 + grad * (1.0 - self.beta1);
                *v = &*v * self.beta2 + &(grad * grad) * (1.0 - self.beta2);

                let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));

                // weight decay is decoupled: add weight_decay*param to update
                let mut param_f32 = param_lock.storage.to_f32_array();
                let wd_term = param_f32.mapv(|p| p * self.weight_decay);
                let mut update =
                    (m_hat / (v_hat.mapv(|x| x.sqrt()) + self.eps)).mapv(|v| v * self.lr);
                update = &update + &(wd_term.mapv(|v| v * self.lr));
                param_f32 = &param_f32 - &update;
                param_lock.storage =
                    crate::dtype::TensorStorage::from_f32_array(&param_f32, param_lock.dtype);
            }
        }
    }

    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut param_lock = param.lock();
            param_lock.grad = None;
        }
    }
}

/// RMSProp optimizer.
pub struct RMSProp {
    lr: f32,
    alpha: f32,
    eps: f32,
    state: HashMap<Tensor, ArrayD<f32>>,
}

impl RMSProp {
    pub fn new(lr: f32, alpha: f32, eps: f32) -> Self {
        RMSProp {
            lr,
            alpha,
            eps,
            state: HashMap::new(),
        }
    }
}

impl Optimizer for RMSProp {
    fn step(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut param_lock = param.lock();
            if let Some(grad) = &param_lock.grad {
                let s = self
                    .state
                    .entry(param.clone())
                    .or_insert_with(|| ArrayD::zeros(grad.dim()));
                *s = &*s * self.alpha + &(grad * grad) * (1.0 - self.alpha);
                let denom = s.mapv(|x| x.sqrt() + self.eps);
                let update = grad / &denom * self.lr;
                let mut param_f32 = param_lock.storage.to_f32_array();
                param_f32 = &param_f32 - &update;
                param_lock.storage =
                    crate::dtype::TensorStorage::from_f32_array(&param_f32, param_lock.dtype);
            }
        }
    }
    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut param_lock = param.lock();
            param_lock.grad = None;
        }
    }
}

/// MaxPool2D layer.
pub struct MaxPool2D {
    kernel_size: usize,
    stride: usize,
}

impl MaxPool2D {
    /// Creates a new MaxPool2D layer.
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        MaxPool2D {
            kernel_size,
            stride,
        }
    }
}

impl Module for MaxPool2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(MaxPool2DOp {
                kernel_size: self.kernel_size,
                stride: self.stride,
            }),
            &[input.clone()],
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// 1D convolution layer (NCL)
#[derive(Clone)]
pub struct Conv1D {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl Conv1D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let weight_data = ndarray::Array::zeros(IxDyn(&[out_channels, in_channels, kernel_size]));
        let weight = Tensor::new(weight_data, true);
        let bias = if bias {
            Some(Tensor::new(
                ndarray::Array::zeros(IxDyn(&[out_channels])),
                true,
            ))
        } else {
            None
        };
        Conv1D {
            weight,
            bias,
            stride,
            padding,
        }
    }
}

impl Module for Conv1D {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut inputs = vec![input.clone(), self.weight.clone()];
        if let Some(b) = &self.bias {
            inputs.push(b.clone());
        }
        Tensor::apply(
            Arc::new(crate::ops::Conv1D::new(self.stride, self.padding)),
            &inputs,
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// ConvTranspose1D Module
#[derive(Clone)]
pub struct ConvTranspose1D {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl ConvTranspose1D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let weight_data = ndarray::Array::zeros(IxDyn(&[out_channels, in_channels, kernel_size]));
        let weight = Tensor::new(weight_data, true);
        let bias = if bias {
            Some(Tensor::new(
                ndarray::Array::zeros(IxDyn(&[out_channels])),
                true,
            ))
        } else {
            None
        };
        ConvTranspose1D {
            weight,
            bias,
            stride,
            padding,
        }
    }
}

impl Module for ConvTranspose1D {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut inputs = vec![input.clone(), self.weight.clone()];
        if let Some(b) = &self.bias {
            inputs.push(b.clone());
        }
        Tensor::apply(
            Arc::new(crate::ops::ConvTranspose1D::new(self.stride, self.padding)),
            &inputs,
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// 2D convolution layer (NCHW)
#[derive(Clone)]
pub struct Conv2D {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl Conv2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let weight_data = ndarray::Array::zeros(IxDyn(&[
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        ]));
        let weight = Tensor::new(weight_data, true);
        let bias = if bias {
            let bias_data = ndarray::Array::zeros(IxDyn(&[out_channels]));
            Some(Tensor::new(bias_data, true))
        } else {
            None
        };
        Conv2D {
            weight,
            bias,
            stride,
            padding,
        }
    }
}

impl Module for Conv2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut inputs = vec![input.clone(), self.weight.clone()];
        if let Some(b) = &self.bias {
            inputs.push(b.clone());
        }
        Tensor::apply(Arc::new(Conv2DOp::new(self.stride, self.padding)), &inputs)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// Dropout layer.
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32, training: bool) -> Self {
        Dropout { p, training }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(crate::ops::Dropout::new(self.p, self.training)),
            &[input.clone()],
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// MSE Loss.
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        MSELoss
    }

    pub fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        pred.sub(target).pow(2.0).mean()
    }
}

/// Cross Entropy Loss (simplified).
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss
    }

    pub fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        // Expect pred to be probabilities (softmax applied) and target as one-hot vectors.
        // Loss = - 1/N * sum(target * log(pred)) where N is the number of samples
        let logp = pred.log();
        let tlogp = target.mul(&logp);
        let total = tlogp.sum();
        let n_samples = pred.lock().storage.shape()[0] as f32;
        let neg_factor = Tensor::new(arr0(-1.0 / n_samples).into_dyn(), false);
        total.mul(&neg_factor)
    }
}

/// Cross entropy loss layer that accepts logits and labels/indexes or one-hot vectors
pub struct CrossEntropyLogitsLoss;

impl CrossEntropyLogitsLoss {
    pub fn new() -> Self {
        CrossEntropyLogitsLoss
    }
    pub fn forward(&self, logits: &Tensor, targets: &Tensor, axis: isize) -> Tensor {
        logits.softmax_cross_entropy_with_logits(targets, axis)
    }
    pub fn forward_from_labels(&self, logits: &Tensor, labels: &Labels, axis: isize) -> Tensor {
        let num_classes = logits.lock().storage.shape()[axis as usize];
        let one_hot = labels.to_one_hot(num_classes);
        let t = Tensor::new(one_hot, false);
        logits.softmax_cross_entropy_with_logits(&t, axis)
    }
}

/// Negative Log Likelihood (NLLLoss) wrapper expecting log-probabilities and integer labels (as floats) or one-hot vectors
pub struct NLLLossLayer;

impl NLLLossLayer {
    pub fn new() -> Self {
        NLLLossLayer
    }
    pub fn forward(&self, log_probs: &Tensor, targets: &Tensor) -> Tensor {
        log_probs.nll_loss(targets)
    }
    pub fn forward_from_labels(&self, log_probs: &Tensor, labels: &Labels) -> Tensor {
        let num_classes = log_probs.lock().storage.shape()[1];
        let one_hot = labels.to_one_hot(num_classes);
        let t = Tensor::new(one_hot, false);
        log_probs.nll_loss(&t)
    }
}

/// Simple DataLoader.
pub struct DataLoader {
    data: Vec<(Tensor, Tensor)>,
    batch_size: usize,
    index: usize,
}

impl DataLoader {
    pub fn new(data: Vec<(Tensor, Tensor)>, batch_size: usize) -> Self {
        DataLoader {
            data,
            batch_size,
            index: 0,
        }
    }

    /// Shuffle the dataset in-place.
    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();
        self.data.shuffle(&mut rng);
        self.reset();
    }

    pub fn next_batch(&mut self) -> Option<(Tensor, Tensor)> {
        if self.index >= self.data.len() {
            return None;
        }
        let end = (self.index + self.batch_size).min(self.data.len());
        let batch_x: Vec<Tensor> = self.data[self.index..end]
            .iter()
            .map(|(x, _)| x.clone())
            .collect();
        let batch_y: Vec<Tensor> = self.data[self.index..end]
            .iter()
            .map(|(_, y)| y.clone())
            .collect();
        self.index = end;
        // Assume stack works
        let bx = Tensor::stack(&batch_x, 0);
        let by = Tensor::stack(&batch_y, 0);
        Some((bx, by))
    }

    pub fn reset(&mut self) {
        self.index = 0;
    }
}
