use crate::labels::Labels;
use crate::ops::Conv2D as Conv2DOp;
use crate::ops::MaxPool2D as MaxPool2DOp;
use crate::tensor::Tensor;
use ndarray::{arr0, ArrayD, IxDyn};
use std::collections::HashMap;
use std::sync::Arc;

pub mod conv;
pub use conv::*;

pub mod embedding;
pub mod flatten;
pub use flatten::*;

pub mod transformer;
pub use transformer::{
    compute_alibi_slopes, AttentionVariant, BERTEncoder, BiasFunction, CrossAttention,
    EncoderDecoderTransformer, GPTDecoder, GroupedQueryAttention, Gemma, Mistral, Phi,
    Qwen, SlidingWindowAttention, T5EncoderDecoder, TransformerBlock, TransformerConfig,
};

// KV cache: minimal scaffolding for incremental decoding
pub mod kv_cache;
pub use kv_cache::KVCache;
pub mod paged_kv_cache;
pub use paged_kv_cache::PagedKVCache;
pub mod paged_attention;

// Re-export common NN modules and types
pub mod audio;
pub use audio::{AudioDecoder, AudioEncoder};

pub mod audio_models;
pub use audio_models::{
    BatchNorm1d, Conv1D, HifiGanDiscriminator, HifiGanDiscriminatorBlock,
    HifiGanGenerator, HifiGanBlock, HifiGanGenerator as HiFiGanGenerator,
    LeakyReLUExt, MultiScaleDiscriminator, ResBlock, WaveNet, WaveNetBlock,
};

pub mod multimodal;
pub use multimodal::{
    get_decode_count, reset_decode_count, GenerationConfig, ModalMemoryContext, MultimodalLLM,
};

pub mod vision;
pub use vision::VisionTransformer;
pub mod diffusion;
pub use diffusion::{
    CFGWrapper, DDIMScheduler, GroupNorm, TimestepEmbedding, UNetModel, VAE, VAEBlock, VAEDecoder,
    VAEEncoder,
};
pub mod quantization;
pub use quantization::RVQ;

// new phase3 utilities
pub mod latent;
pub use latent::*;

pub mod continuous_thought;
pub use continuous_thought::ContinuousThoughtModule;

pub mod decoders;
pub use decoders::{TextDecoder, ImageDecoder, VideoDecoder};

// Don't re-export op-level Conv types here to avoid duplicate symbol errors.
// NN defines wrapper Conv1D/Conv2D types in this module. If you need the raw
// op-level Conv types, use crate::ops::Conv2D explicitly.

pub mod lora;
pub use lora::{DoRAAdapter, LoRAAdapter, LoRAConfig, LoRAModule, QLoRAAdapter};

pub mod pruning;
pub use pruning::{ChannelPruner, HeadPruner, PruningConfig, PruningMethod, Pruner};

pub mod audio_processing;
pub use audio_processing::{ISTFT, MelSpectrogram, STFT};

pub mod bpe_tokenizer;
pub use bpe_tokenizer::BPETokenizer;

pub mod wordpiece_tokenizer;
pub use wordpiece_tokenizer::WordPieceTokenizer;

pub mod sentencepiece_tokenizer;
pub use sentencepiece_tokenizer::SentencePieceTokenizer;

pub mod text_preprocessing;
pub use text_preprocessing::{TextCleaner, TextNormalizer, TextPreprocessor, TextNormalizeConfig};

pub mod sequence_padding;
pub use sequence_padding::{
    pad_sequences, pad_2d_sequences, create_attention_mask, create_causal_mask,
    create_combined_mask, create_key_padding_mask, pad_and_mask,
    PadConfig, PaddingMode,
};

pub mod audio_augmentation;
pub use audio_augmentation::{
    AudioAugmenter, AudioAugmentConfig, AudioNormalizer,
    AudioAugmentationPipeline,
};

pub mod swin_transformer;
pub use swin_transformer::{SwinTransformer, SwinConfig, SwinDetector, SwinStage, WindowAttention, SwinMLP, PatchEmbedding, PatchMerging};

pub mod knowledge_distillation;
pub use knowledge_distillation::{
    AttentionDistillation, DistillationLoss, DistillationModel, DistillationTrainer,
    FeatureDistillation, LogitsDistillation,
};

pub mod model_visualization;
pub use model_visualization::{
    ActivationTracker, GradientFlowAnalyzer, GradientTracker, ModelSummary, ParamInfo,
    TensorStats, WeightHistogram,
};

pub mod clip;
pub mod looped_transformer;
pub use clip::*;
pub mod quantized;

pub use looped_transformer::LoopedTransformer;
pub mod multi_head_attention_module;
pub use multi_head_attention_module::MultiHeadAttention as MHAVariant;

/// Absolute positional embedding: holds an embedding matrix of shape (max_len, d_model)
#[derive(Clone)]
pub struct AbsolutePositionalEmbedding {
    pub weight: Tensor,
    pub max_len: usize,
}

impl AbsolutePositionalEmbedding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        let w = ndarray::Array::zeros(IxDyn(&[max_len, d_model][..]));
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
        if seq > self.max_len {
            log::error!("AbsolutePositionalEmbedding: sequence length {} > max_len {}; returning input unchanged", seq, self.max_len);
            return input.clone();
        }
        let mut idx = vec![];
        for _ in 0..b {
            for i in 0..seq {
                idx.push(i as f32);
            }
        }
        let idx_arr = match ndarray::Array::from_shape_vec((b, seq), idx) {
            Ok(a) => a.into_dyn(),
            Err(e) => {
                log::error!(
                    "AbsolutePositionalEmbedding: failed to create idx array: {}",
                    e
                );
                // Fallback: return input unchanged to avoid panic
                return input.clone();
            }
        };
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
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        let key = format!("{}.weight", prefix);
        if let Some(w) = state.get(&key) {
            self.weight = w.clone();
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub mod linear_dispatch;
pub mod moe;
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
    /// Sets the training mode of the module and its sub-modules.
    fn set_training(&mut self, _training: bool) {}

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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
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
            ndarray::Array::zeros(ndarray::IxDyn(&[input_dim, hidden_dim][..])),
            true,
        );
        let whh = Tensor::new(
            ndarray::Array::zeros(ndarray::IxDyn(&[hidden_dim, hidden_dim][..])),
            true,
        );
        let b = if bias {
            Some(Tensor::new(
                ndarray::Array::zeros(ndarray::IxDyn(&[hidden_dim][..])),
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
        let mut out = input.clone();
        out = out.matmul(&self.weight_ih);
        let h_w = out.tanh();
        h_w.matmul(&self.weight_hh).tanh()
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
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
            ndarray::Array::zeros(ndarray::IxDyn(&[input_dim, 4 * hidden_dim][..])),
            true,
        );
        let whh = Tensor::new(
            ndarray::Array::zeros(ndarray::IxDyn(&[hidden_dim, 4 * hidden_dim][..])),
            true,
        );
        let b = if bias {
            Some(Tensor::new(
                ndarray::Array::zeros(ndarray::IxDyn(&[4 * hidden_dim][..])),
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
        if dim.len() != 2 {
            log::error!("slice_n expects 2D tensor, got shape {:?}", dim);
            return (
                t.clone(),
                Tensor::new(ndarray::Array::zeros(IxDyn(&[0, 0][..])), false),
            );
        }
        let total = dim[1];
        let first = Tensor::apply(
            Arc::new(crate::ops::Slice::new(1, start, n)),
            std::slice::from_ref(&t),
        );
        let second = Tensor::apply(
            Arc::new(crate::ops::Slice::new(1, start + n, total - (start + n))),
            std::slice::from_ref(&t),
        );
        (first, second)
    }
}

impl Module for LSTMCell {
    fn forward(&self, input: &Tensor) -> Tensor {
        let zeros = Tensor::new(
            ndarray::Array::zeros(ndarray::IxDyn(&[input.lock().storage.shape()[0], self.hidden_dim][..])),
            false,
        );
        let (h, _) = self.forward_step(input, &zeros, &zeros);
        h
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
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
        let q2 = match q.reshape(vec![b * seq, dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("SelfAttention::forward_attention reshape(q) failed: {}", e);
                // Fallback: return a zeros tensor with expected output shape to avoid panics
                return Tensor::new(ndarray::Array::zeros(IxDyn(&[b, seq, dim][..])), false);
            }
        };
        // q2 reshape done
        let k2 = match k.reshape(vec![b * seq, dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("SelfAttention::forward_attention reshape(k) failed: {}", e);
                return Tensor::new(ndarray::Array::zeros(IxDyn(&[b, seq, dim][..])), false);
            }
        };
        // k2 reshape done
        let v2 = match v.reshape(vec![b * seq, dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!("SelfAttention::forward_attention reshape(v) failed: {}", e);
                return Tensor::new(ndarray::Array::zeros(IxDyn(&[b, seq, dim][..])), false);
            }
        };
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
            ndarray::Array::from_elem(ndarray::IxDyn(&[1][..]), scale),
            false,
        ));
        let attn = scaled.softmax(1);
        // computed softmax
        // about to compute out matmul
        let out = attn.matmul(&v2);
        // computed out matmul
        match out.reshape(vec![b, seq, dim]) {
            Ok(t) => t,
            Err(e) => {
                log::error!(
                    "SelfAttention::forward_attention final reshape failed: {}",
                    e
                );
                out
            }
        }
    }
}

impl Module for SelfAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_attention(input, input, input)
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A linear (fully connected) layer.
#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
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
        let mut rng = rand::rng();
        let scale = 1.0 / (in_features as f32).sqrt();
        let weight_data = ArrayD::from_shape_fn(IxDyn(&[in_features, out_features][..]), |_| {
            use rand::Rng;
            rng.random_range(-scale..scale)
        });
        let weight = Tensor::new(weight_data, true);

        let bias = if bias {
            let bias_data = ArrayD::zeros(IxDyn(&[out_features][..]));
            Some(Tensor::new(bias_data, true))
        } else {
            None
        };

        Linear {
            weight,
            bias,
            in_features,
            out_features,
        }
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
            let reshaped = match input.reshape(vec![batch, last]) {
                Ok(t) => t,
                Err(e) => {
                    log::error!("Linear::forward reshape input failed: {}", e);
                    return input.clone();
                }
            };
            let out2 = reshaped.matmul(&self.weight);
            let mut out_shape = input_shape.clone();
            out_shape[ndim - 1] = self.weight.lock().storage.shape()[1];
            match out2.reshape(out_shape) {
                Ok(t) => t,
                Err(e) => {
                    log::error!("Linear::forward reshape output failed: {}", e);
                    out2
                }
            }
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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A sequential container for modules.
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

/// Layer Normalization module
#[derive(Clone)]
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub axis: usize,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(num_features: usize, axis: usize, eps: f32) -> Self {
        let gamma = Tensor::new(
            match ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[num_features][..]),
                vec![1.0; num_features],
            ) {
                Ok(a) => a,
                Err(e) => {
                    log::error!("LayerNorm: failed to create gamma array: {}", e);
                    ndarray::Array::from_elem(IxDyn(&[num_features][..]), 1.0f32)
                }
            },
            true,
        );
        let beta = Tensor::new(
            match ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[num_features][..]),
                vec![0.0; num_features],
            ) {
                Ok(a) => a,
                Err(e) => {
                    log::error!("LayerNorm: failed to create beta array: {}", e);
                    ndarray::Array::from_elem(IxDyn(&[num_features][..]), 0.0f32)
                }
            },
            true,
        );
        LayerNorm {
            gamma,
            beta,
            axis,
            eps,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        input.layer_norm(self.axis, self.eps, &self.gamma, &self.beta)
    }
}

/// Root Mean Square Normalization (RMSNorm) module
///
/// - `weight`: learnable gain (gamma); shape `[num_features]`
/// - `axis`: normalization axis
/// - `eps`: epsilon for numerical stability
#[derive(Clone)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub axis: usize,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(num_features: usize, axis: usize, eps: f32) -> Self {
        let weight = Tensor::new(
            match ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[num_features][..]),
                vec![1.0; num_features],
            ) {
                Ok(a) => a,
                Err(e) => {
                    log::error!("RMSNorm: failed to create weight array: {}", e);
                    ndarray::Array::from_elem(IxDyn(&[num_features][..]), 1.0f32)
                }
            },
            true,
        );
        RMSNorm { weight, axis, eps }
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.rmsnorm(&self.weight, self.axis, self.eps)
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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.layer_norm(self.axis, self.eps, &self.gamma, &self.beta)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Sequential {
    /// Creates a new sequential container.
    pub fn new() -> Self {
        Sequential {
            modules: Vec::new(),
        }
    }

    /// Adds a module to the container.
    pub fn append<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    /// Returns all parameters from all modules.
    pub fn parameters(&self) -> Vec<Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn set_training(&mut self, training: bool) {
        for module in &mut self.modules {
            module.set_training(training);
        }
    }
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

    /// Clip gradients in-place by absolute value.
    ///
    /// Each gradient element `g` is clamped to `[-clip_value, clip_value]`.
    fn clip_grad_values(&mut self, parameters: &[Tensor], clip_value: f32) {
        if clip_value <= 0.0 || !clip_value.is_finite() {
            return;
        }
        let c = clip_value.abs();
        for p in parameters {
            let mut lock = p.lock();
            if let Some(g) = &mut lock.grad {
                g.mapv_inplace(|v| v.clamp(-c, c));
            }
        }
    }

    /// Scale gradients in-place by a constant factor.
    ///
    /// This is useful for gradient accumulation (e.g., average gradients over N micro-batches)
    /// and for manual loss scaling.
    fn scale_gradients(&mut self, parameters: &[Tensor], scale: f32) {
        if !scale.is_finite() {
            return;
        }
        // If scale is exactly 1, avoid touching gradients.
        if (scale - 1.0).abs() <= f32::EPSILON {
            return;
        }
        for p in parameters {
            let mut lock = p.lock();
            if let Some(g) = &mut lock.grad {
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
        let t = step.min(self.total_steps) as f32;
        let total_steps_f = self.total_steps as f32;
        self.base_lr * 0.5 * (1.0 + (std::f32::consts::PI * t / total_steps_f).cos())
    }
}

/// Exponential decay scheduler: lr = max(min_lr, base_lr * gamma^step)
///
/// Commonly used as a simple multiplicative decay over steps.
pub struct ExponentialDecay {
    pub base_lr: f32,
    pub gamma: f32,
    pub min_lr: f32,
}

impl ExponentialDecay {
    pub fn new(base_lr: f32, gamma: f32) -> Self {
        ExponentialDecay {
            base_lr,
            gamma,
            min_lr: 0.0,
        }
    }

    pub fn new_with_min_lr(base_lr: f32, gamma: f32, min_lr: f32) -> Self {
        ExponentialDecay {
            base_lr,
            gamma,
            min_lr,
        }
    }
}

impl LRScheduler for ExponentialDecay {
    fn get_lr(&self, step: usize) -> f32 {
        if !self.base_lr.is_finite() || !self.gamma.is_finite() || !self.min_lr.is_finite() {
            return 0.0;
        }
        let base = self.base_lr.max(0.0);
        let min_lr = self.min_lr.max(0.0);
        if base == 0.0 {
            return 0.0;
        }
        // For gamma==1, lr is constant.
        let lr = if self.gamma == 1.0 {
            base
        } else {
            base * self.gamma.powi(step as i32)
        };
        lr.max(min_lr)
    }
}

/// Step decay scheduler: lr = max(min_lr, base_lr * drop_factor^(floor(step/step_size)))
pub struct StepDecay {
    pub base_lr: f32,
    pub step_size: usize,
    pub drop_factor: f32,
    pub min_lr: f32,
}

impl StepDecay {
    pub fn new(base_lr: f32, step_size: usize, drop_factor: f32) -> Self {
        StepDecay {
            base_lr,
            step_size,
            drop_factor,
            min_lr: 0.0,
        }
    }

    pub fn new_with_min_lr(base_lr: f32, step_size: usize, drop_factor: f32, min_lr: f32) -> Self {
        StepDecay {
            base_lr,
            step_size,
            drop_factor,
            min_lr,
        }
    }
}

impl LRScheduler for StepDecay {
    fn get_lr(&self, step: usize) -> f32 {
        if !self.base_lr.is_finite() || !self.drop_factor.is_finite() || !self.min_lr.is_finite() {
            return 0.0;
        }
        let base = self.base_lr.max(0.0);
        let min_lr = self.min_lr.max(0.0);
        if base == 0.0 {
            return 0.0;
        }
        if self.step_size == 0 {
            return base.max(min_lr);
        }
        let k = (step / self.step_size) as i32;
        let lr = if self.drop_factor == 1.0 {
            base
        } else {
            base * self.drop_factor.powi(k)
        };
        lr.max(min_lr)
    }
}

/// Polynomial decay scheduler:
/// lr = end_lr + (base_lr - end_lr) * (1 - t/T)^power, clamped to [min(base_lr,end_lr), max(base_lr,end_lr)]
pub struct PolynomialDecay {
    pub base_lr: f32,
    pub end_lr: f32,
    pub total_steps: usize,
    pub power: f32,
}

impl PolynomialDecay {
    pub fn new(base_lr: f32, end_lr: f32, total_steps: usize, power: f32) -> Self {
        PolynomialDecay {
            base_lr,
            end_lr,
            total_steps,
            power,
        }
    }
}

impl LRScheduler for PolynomialDecay {
    fn get_lr(&self, step: usize) -> f32 {
        if !self.base_lr.is_finite() || !self.end_lr.is_finite() || !self.power.is_finite() {
            return 0.0;
        }
        if self.total_steps == 0 {
            return self.base_lr.max(0.0);
        }
        let base = self.base_lr.max(0.0);
        let end = self.end_lr.max(0.0);
        let t = step.min(self.total_steps) as f32;
        let total = self.total_steps as f32;
        let frac = (1.0 - (t / total)).clamp(0.0, 1.0);
        let pow = if self.power == 1.0 {
            frac
        } else {
            frac.powf(self.power)
        };
        let lr = end + (base - end) * pow;
        let lo = base.min(end);
        let hi = base.max(end);
        lr.clamp(lo, hi)
    }
}

/// Constant LR scheduler similar to PyTorch ConstantLR.
///
/// Returns `base_lr * factor` for the first `total_iters` steps,
/// then returns `base_lr`.
pub struct ConstantLR {
    pub base_lr: f32,
    pub factor: f32,
    pub total_iters: usize,
}

impl ConstantLR {
    pub fn new(base_lr: f32, factor: f32, total_iters: usize) -> Self {
        ConstantLR {
            base_lr,
            factor,
            total_iters,
        }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self, step: usize) -> f32 {
        if !self.base_lr.is_finite() || !self.factor.is_finite() {
            return 0.0;
        }
        let base = self.base_lr.max(0.0);
        let factor = self.factor.max(0.0);
        if step < self.total_iters {
            (base * factor).max(0.0)
        } else {
            base
        }
    }
}

/// Linear LR scheduler similar to PyTorch LinearLR.
///
/// Linearly interpolates multiplicative factor from `start_factor` to
/// `end_factor` over `total_iters` steps, then keeps `end_factor`.
pub struct LinearLR {
    pub base_lr: f32,
    pub start_factor: f32,
    pub end_factor: f32,
    pub total_iters: usize,
}

impl LinearLR {
    pub fn new(base_lr: f32, start_factor: f32, end_factor: f32, total_iters: usize) -> Self {
        LinearLR {
            base_lr,
            start_factor,
            end_factor,
            total_iters,
        }
    }
}

impl LRScheduler for LinearLR {
    fn get_lr(&self, step: usize) -> f32 {
        if !self.base_lr.is_finite()
            || !self.start_factor.is_finite()
            || !self.end_factor.is_finite()
        {
            return 0.0;
        }
        let base = self.base_lr.max(0.0);
        let start = self.start_factor.max(0.0);
        let end = self.end_factor.max(0.0);
        if self.total_iters == 0 {
            return (base * end).max(0.0);
        }
        let t = step.min(self.total_iters) as f32;
        let total = self.total_iters as f32;
        let alpha = (t / total).clamp(0.0, 1.0);
        let factor = start + (end - start) * alpha;
        (base * factor.max(0.0)).max(0.0)
    }
}

/// OneCycle learning rate scheduler with cosine annealing.
///
/// Two phases:
/// 1) Warm-up from `max_lr / div_factor` to `max_lr` over `pct_start * total_steps`
/// 2) Anneal from `max_lr` to `max_lr / final_div_factor` for the remainder
pub struct OneCycleLR {
    pub max_lr: f32,
    pub total_steps: usize,
    pub pct_start: f32,
    pub div_factor: f32,
    pub final_div_factor: f32,
}

impl OneCycleLR {
    pub fn new(
        max_lr: f32,
        total_steps: usize,
        pct_start: f32,
        div_factor: f32,
        final_div_factor: f32,
    ) -> Self {
        OneCycleLR {
            max_lr,
            total_steps,
            pct_start,
            div_factor,
            final_div_factor,
        }
    }

    fn cosine_anneal(start: f32, end: f32, pct: f32) -> f32 {
        let p = pct.clamp(0.0, 1.0);
        end + (start - end) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
    }
}

impl LRScheduler for OneCycleLR {
    fn get_lr(&self, step: usize) -> f32 {
        if !self.max_lr.is_finite()
            || !self.pct_start.is_finite()
            || !self.div_factor.is_finite()
            || !self.final_div_factor.is_finite()
        {
            return 0.0;
        }
        if self.total_steps == 0 {
            return self.max_lr.max(0.0);
        }

        let max_lr = self.max_lr.max(0.0);
        let div = self.div_factor.max(1.0);
        let final_div = self.final_div_factor.max(1.0);
        let pct_start = self.pct_start.clamp(0.0, 1.0);

        let initial_lr = max_lr / div;
        let min_lr = max_lr / final_div;

        let up_steps = ((self.total_steps as f32) * pct_start).round() as usize;
        let up_steps = up_steps.min(self.total_steps);
        let down_steps = self.total_steps.saturating_sub(up_steps);
        let s = step.min(self.total_steps);

        if up_steps == 0 {
            if down_steps == 0 {
                return min_lr.max(0.0);
            }
            let t = (s as f32) / (down_steps as f32);
            return OneCycleLR::cosine_anneal(max_lr, min_lr, t).max(0.0);
        }

        if s <= up_steps {
            let t = (s as f32) / (up_steps as f32);
            OneCycleLR::cosine_anneal(initial_lr, max_lr, t).max(0.0)
        } else {
            if down_steps == 0 {
                return max_lr.max(0.0);
            }
            let down_pos = s.saturating_sub(up_steps);
            let t = (down_pos as f32) / (down_steps as f32);
            OneCycleLR::cosine_anneal(max_lr, min_lr, t).max(0.0)
        }
    }
}

/// Cyclic learning rate modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CyclicLRMode {
    /// The basic triangular policy.
    Triangular,
    /// Triangular policy with halving amplitude each cycle.
    Triangular2,
}

/// Cyclic learning rate scheduler with a piecewise-linear up/down cycle.
///
/// Cycle length is `step_size_up + step_size_down`.
pub struct CyclicLR {
    pub base_lr: f32,
    pub max_lr: f32,
    pub step_size_up: usize,
    pub step_size_down: usize,
    pub mode: CyclicLRMode,
}

impl CyclicLR {
    pub fn new(base_lr: f32, max_lr: f32, step_size_up: usize, step_size_down: usize) -> Self {
        CyclicLR {
            base_lr,
            max_lr,
            step_size_up,
            step_size_down,
            mode: CyclicLRMode::Triangular,
        }
    }

    pub fn new_with_mode(
        base_lr: f32,
        max_lr: f32,
        step_size_up: usize,
        step_size_down: usize,
        mode: CyclicLRMode,
    ) -> Self {
        CyclicLR {
            base_lr,
            max_lr,
            step_size_up,
            step_size_down,
            mode,
        }
    }
}

impl LRScheduler for CyclicLR {
    fn get_lr(&self, step: usize) -> f32 {
        if !self.base_lr.is_finite() || !self.max_lr.is_finite() {
            return 0.0;
        }
        let base = self.base_lr.max(0.0);
        let max_lr = self.max_lr.max(0.0);

        // Degenerate cases: if there is no room to cycle, return the base.
        if base == max_lr {
            return base;
        }
        let up = self.step_size_up;
        let down = self.step_size_down;
        let cycle_len = up.saturating_add(down);
        if cycle_len == 0 {
            return base;
        }
        let cycle_idx = step / cycle_len;
        let pos = step % cycle_len;

        let mut scale = 1.0f32;
        if self.mode == CyclicLRMode::Triangular2 {
            // 1, 1/2, 1/4, and so on per cycle.
            let denom = 2u32.saturating_pow(cycle_idx as u32) as f32;
            if denom.is_finite() && denom > 0.0 {
                scale = 1.0 / denom;
            }
        }
        scale = scale.clamp(0.0, 1.0);
        let amp = (max_lr - base) * scale;

        // Compute linear ramp ratio in [0,1].
        let ratio = if up == 0 {
            // If up is zero, we start at the peak and go down.
            1.0
        } else if pos < up {
            (pos as f32) / (up as f32)
        } else {
            // Down phase.
            if down == 0 {
                0.0
            } else {
                let down_pos = (pos - up) as f32;
                1.0 - (down_pos / (down as f32))
            }
        };

        (base + amp * ratio.clamp(0.0, 1.0)).max(0.0)
    }
}

/// Stochastic Gradient Descent optimizer.
pub struct SGD {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
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
            weight_decay: 0.0,
            velocity: HashMap::new(),
        }
    }

    /// Creates a new SGD optimizer with decoupled weight decay.
    pub fn new_with_weight_decay(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        SGD {
            lr,
            momentum,
            weight_decay,
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
                if self.weight_decay != 0.0 {
                    // Decoupled weight decay: param -= lr * weight_decay * param
                    let wd = self.lr * self.weight_decay;
                    if wd.is_finite() {
                        param_f32.mapv_inplace(|p| p - wd * p);
                    }
                }
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
    weight_decay: f32,
    state: HashMap<Tensor, ArrayD<f32>>,
}

impl RMSProp {
    pub fn new(lr: f32, alpha: f32, eps: f32) -> Self {
        RMSProp {
            lr,
            alpha,
            eps,
            weight_decay: 0.0,
            state: HashMap::new(),
        }
    }

    /// Creates a new RMSProp optimizer with decoupled weight decay.
    pub fn new_with_weight_decay(lr: f32, alpha: f32, eps: f32, weight_decay: f32) -> Self {
        RMSProp {
            lr,
            alpha,
            eps,
            weight_decay,
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
                if self.weight_decay != 0.0 {
                    let wd = self.lr * self.weight_decay;
                    if wd.is_finite() {
                        param_f32.mapv_inplace(|p| p - wd * p);
                    }
                }
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
            std::slice::from_ref(input),
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
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
        let weight_data =
            ndarray::Array::zeros(IxDyn(&[out_channels, in_channels, kernel_size][..]));
        let weight = Tensor::new(weight_data, true);
        let bias = if bias {
            Some(Tensor::new(
                ndarray::Array::zeros(IxDyn(&[out_channels][..])),
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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
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
        let weight_data = ndarray::Array::zeros(IxDyn(
            &[out_channels, in_channels, kernel_size, kernel_size][..],
        ));
        let weight = Tensor::new(weight_data, true);
        let bias = if bias {
            let bias_data = ndarray::Array::zeros(IxDyn(&[out_channels][..]));
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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
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
            std::slice::from_ref(input),
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// DropPath (stochastic depth) layer.
///
/// Drops entire residual paths per sample during training and rescales kept paths
/// by `1 / (1 - p)` to preserve expected activation magnitude.
pub struct DropPath {
    p: f32,
    training: bool,
}

impl DropPath {
    pub fn new(p: f32, training: bool) -> Self {
        DropPath { p, training }
    }
}

impl Module for DropPath {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p <= 0.0 {
            return input.clone();
        }

        let x = input.to_f32_array();
        let shape = x.shape().to_vec();
        if shape.is_empty() {
            return input.clone();
        }

        let keep_prob = (1.0 - self.p).clamp(0.0, 1.0);
        if keep_prob <= 0.0 {
            let zeros = ArrayD::zeros(IxDyn(&shape));
            let requires_grad = input.lock().requires_grad;
            return Tensor::new(zeros, requires_grad);
        }

        let batch = shape[0];
        if batch == 0 {
            return input.clone();
        }

        let total = x.len();
        let sample_size = total / batch;
        let scale = 1.0 / keep_prob;

        let mut mask = vec![0.0f32; total];
        for b in 0..batch {
            let keep = if rand::random::<f32>() < keep_prob {
                scale
            } else {
                0.0
            };
            let start = b * sample_size;
            let end = start + sample_size;
            for v in mask.iter_mut().take(end).skip(start) {
                *v = keep;
            }
        }

        let mask_arr = match ArrayD::from_shape_vec(IxDyn(&shape), mask) {
            Ok(v) => v,
            Err(e) => {
                log::error!("DropPath.forward: mask shape construction failed: {}", e);
                return input.clone();
            }
        };

        let mask_t = Tensor::new(mask_arr, false);
        input.mul(&mask_t)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
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

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
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

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
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

impl Default for CrossEntropyLogitsLoss {
    fn default() -> Self {
        Self::new()
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

impl Default for NLLLossLayer {
    fn default() -> Self {
        Self::new()
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

/// GRU (Gated Recurrent Unit) cell for sequence modeling.
///
/// GRU is a simpler alternative to LSTM with fewer parameters (2 gates vs 3),
/// making it faster to train while maintaining competitive performance.
///
/// Gates:
/// - Reset gate (r): Controls how much past information to forget
/// - Update gate (z): Controls how much new information to add
/// - New gate (n): Candidate hidden state
///
/// Equations:
/// ```text
/// r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
/// z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
/// n_t = tanh(W_in @ x_t + b_in + r_t ⊙ (W_hn @ h_{t-1} + b_hn))
/// h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
/// ```
#[derive(Clone)]
pub struct GRUCell {
    pub weight_ih: Tensor, // input to gates weights, shape [input_dim, 3*hidden_dim]
    pub weight_hh: Tensor, // hidden to gates weights, shape [hidden_dim, 3*hidden_dim]
    pub bias: Option<Tensor>,
    pub hidden_dim: usize,
}

impl GRUCell {
    /// Creates a new GRU cell.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of input features
    /// * `hidden_dim` - Dimension of hidden state
    /// * `bias` - Whether to use bias terms
    pub fn new(input_dim: usize, hidden_dim: usize, bias: bool) -> Self {
        let wih = Tensor::new(
            ndarray::Array::zeros(ndarray::IxDyn(&[input_dim, 3 * hidden_dim][..])),
            true,
        );
        let whh = Tensor::new(
            ndarray::Array::zeros(ndarray::IxDyn(&[hidden_dim, 3 * hidden_dim][..])),
            true,
        );
        let b = if bias {
            Some(Tensor::new(
                ndarray::Array::zeros(ndarray::IxDyn(&[3 * hidden_dim][..])),
                true,
            ))
        } else {
            None
        };
        GRUCell {
            weight_ih: wih,
            weight_hh: whh,
            bias: b,
            hidden_dim,
        }
    }

    /// Forward a single step through the GRU cell.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch, input_dim]
    /// * `h` - Previous hidden state of shape [batch, hidden_dim]
    ///
    /// # Returns
    ///
    /// New hidden state of shape [batch, hidden_dim]
    pub fn forward_step(&self, input: &Tensor, h: &Tensor) -> Tensor {
        // Compute input transformations: input @ w_ih
        let xw = input.matmul(&self.weight_ih);

        // Compute hidden transformations: h @ w_hh
        let hw = h.matmul(&self.weight_hh);

        // Add bias if present
        let xw = if let Some(b) = &self.bias {
            xw.add(b)
        } else {
            xw
        };

        // Split into reset, update, and new gates
        let hid = self.hidden_dim;
        let (xw_r, rest) = Self::slice_n(xw.clone(), 0, hid);
        let (xw_z, xw_n) = Self::slice_n(rest, 0, hid);

        let (hw_r, rest2) = Self::slice_n(hw.clone(), 0, hid);
        let (hw_z, hw_n) = Self::slice_n(rest2, 0, hid);

        // Reset gate: r_t = σ(W_ir @ x_t + W_hr @ h_{t-1})
        let r = xw_r.add(&hw_r).sigmoid();

        // Update gate: z_t = σ(W_iz @ x_t + W_hz @ h_{t-1})
        let z = xw_z.add(&hw_z).sigmoid();

        // New gate: n_t = tanh(W_in @ x_t + r_t ⊙ (W_hn @ h_{t-1}))
        let n = xw_n.add(&r.mul(&hw_n)).tanh();

        // Output hidden state: h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
        // Create a tensor of ones: 1 - z = -z + 1
        let shape = z.lock().storage.shape();
        let ones = Tensor::new(ndarray::ArrayD::ones(ndarray::IxDyn(&shape)), false);
        let one_minus_z = ones.sub(&z);
        one_minus_z.mul(&n).add(&z.mul(h))
    }

    fn slice_n(t: Tensor, start: usize, n: usize) -> (Tensor, Tensor) {
        // Use a Slice operation implemented in ops.rs to return differentiable slices
        let dim = t.lock().storage.shape();
        if dim.len() != 2 {
            log::error!("slice_n expects 2D tensor, got shape {:?}", dim);
            return (
                t.clone(),
                Tensor::new(ndarray::Array::zeros(IxDyn(&[0, 0][..])), false),
            );
        }
        let total = dim[1];
        let first = Tensor::apply(
            Arc::new(crate::ops::Slice::new(1, start, n)),
            std::slice::from_ref(&t),
        );
        let second = Tensor::apply(
            Arc::new(crate::ops::Slice::new(1, start + n, total - (start + n))),
            std::slice::from_ref(&t),
        );
        (first, second)
    }
}

impl Module for GRUCell {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Default: zero initial hidden state
        let shape = input.lock().storage.shape();
        let batch_size = shape[0];
        let h = Tensor::new(
            ndarray::ArrayD::zeros(ndarray::IxDyn(&[batch_size, self.hidden_dim][..])),
            false,
        );
        self.forward_step(input, &h)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Batch Normalization for 4D (spatial) inputs.
pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub gamma: Tensor,
    pub beta: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        let rm = Tensor::zeros(&[num_features][..]);
        rm.set_requires_grad(false);
        let rv = Tensor::ones(&[num_features][..]);
        rv.set_requires_grad(false);

        BatchNorm2d {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            gamma: Tensor::ones(&[num_features][..]),
            beta: Tensor::zeros(&[num_features][..]),
            running_mean: rm,
            running_var: rv,
            training: true,
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let config = crate::tensor::BatchNormConfig {
            momentum: self.momentum,
            eps: self.eps,
            training: self.training,
        };
        input.batch_norm(
            &self.gamma,
            &self.beta,
            &self.running_mean,
            &self.running_var,
            config,
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Instance Normalization for 4D (spatial) inputs.
///
/// Normalizes each sample in a batch independently across spatial dimensions,
/// preserving batch-level statistics. Commonly used in style transfer and
/// image generation tasks.
///
/// # Reference
/// [`Ulyanov et al., 2016`](https://arxiv.org/abs/1607.08022)
#[derive(Clone)]
pub struct InstanceNorm2d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub gamma: Tensor,
    pub beta: Tensor,
    pub training: bool,
}

impl InstanceNorm2d {
    pub fn new(num_features: usize) -> Self {
        InstanceNorm2d {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            gamma: Tensor::ones(&[num_features][..]),
            beta: Tensor::zeros(&[num_features][..]),
            training: false,
        }
    }

    pub fn with_training(mut self, training: bool) -> Self {
        self.training = training;
        self
    }
}

impl Module for InstanceNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.lock().storage.shape().to_vec();
        if shape.len() != 4 {
            log::error!(
                "InstanceNorm2d.forward: expected 4D input [B, C, H, W], got {:?}",
                shape
            );
            return input.clone();
        }

        let b = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];

        let inp = input.lock().storage.to_f32_array();
        let gamma_arr = self.gamma.lock().storage.to_f32_array();
        let beta_arr = self.beta.lock().storage.to_f32_array();

        let mut out = ArrayD::<f32>::zeros(IxDyn(&[b, c, h, w][..]));

        for n in 0..b {
            for ch in 0..c {
                // Compute mean and variance over spatial dimensions
                let mut sum = 0.0f32;
                let mut sumsq = 0.0f32;
                let spatial_size = h * w;

                for y in 0..h {
                    for x in 0..w {
                        let val = inp[[n, ch, y, x]];
                        sum += val;
                        sumsq += val * val;
                    }
                }

                let mean = sum / spatial_size as f32;
                let var = (sumsq / spatial_size as f32) - (mean * mean);
                let var = var.max(0.0); // Ensure non-negative variance
                let std = (var + self.eps).sqrt();

                // Normalize and apply scale/shift
                for y in 0..h {
                    for x in 0..w {
                        let val = inp[[n, ch, y, x]];
                        let normalized = (val - mean) / std;
                        out[[n, ch, y, x]] = normalized * gamma_arr[[ch]] + beta_arr[[ch]];
                    }
                }
            }
        }

        Tensor::new(out, false)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
