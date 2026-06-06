//! Python bindings for tensor_engine via pyo3.
//!
//! Exposes the core Tensor, Linear, SGD, Adam, TransformerBlock and Tokenizer
//! types so the library can be used as a native Python extension module.
//!
//! Built when the `python_bindings` feature is enabled (default).  The
//! `#[pymodule]` entry-point is named `tensor_engine` which must match the
//! `[lib] name =` in `Cargo.toml` so that maturin generates the correct
//! `PyInit_tensor_engine` symbol.

use crate::optim::Optimizer;
use ndarray::{ArrayD, IxDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

// ────────────────────────────────────────────────────────────────────────────
// PyTensor
// ────────────────────────────────────────────────────────────────────────────

/// A multi-dimensional float tensor with automatic differentiation support.
///
/// ```python
/// import tensor_engine as te
/// t = te.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
/// ```
#[pyclass(name = "Tensor", module = "tensor_engine")]
pub struct PyTensor {
    pub inner: crate::tensor::Tensor,
}

#[pymethods]
impl PyTensor {
    /// Create a new Tensor from a flat list of floats and a shape.
    #[new]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor {
            inner: crate::tensor::Tensor::new(arr, true),
        })
    }

    /// Return the tensor data as a flat Python list.
    fn get_data(&self) -> Vec<f32> {
        self.inner.to_vec()
    }

    /// Return the tensor shape as a property.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    /// Run backpropagation from this tensor.
    fn backward(&self) {
        self.inner.backward();
    }

    /// Clear the gradient stored on this tensor.
    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    /// Return the gradient as a flat list, or `None` if not yet computed.
    fn grad(&self) -> Option<Vec<f32>> {
        self.inner
            .lock()
            .grad
            .as_ref()
            .map(|g| g.iter().cloned().collect())
    }

    /// Return whether this tensor tracks gradients.
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    /// Set whether this tensor tracks gradients.
    fn set_requires_grad(&self, rg: bool) {
        self.inner.set_requires_grad(rg);
    }

    /// Return a new tensor detached from the computation graph.
    fn detach(&self) -> Self {
        PyTensor {
            inner: self.inner.detach(),
        }
    }

    /// Reshape the tensor. Returns a new tensor.
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.inner
            .reshape(shape)
            .map(|t| PyTensor { inner: t })
            .map_err(PyValueError::new_err)
    }

    // ── element-wise ops ────────────────────────────────────────────────────

    fn add(&self, other: &PyTensor) -> Self {
        PyTensor {
            inner: self.inner.add(&other.inner),
        }
    }

    fn sub(&self, other: &PyTensor) -> Self {
        PyTensor {
            inner: self.inner.sub(&other.inner),
        }
    }

    fn mul(&self, other: &PyTensor) -> Self {
        PyTensor {
            inner: self.inner.mul(&other.inner),
        }
    }

    fn div(&self, other: &PyTensor) -> Self {
        PyTensor {
            inner: self.inner.div(&other.inner),
        }
    }

    fn pow(&self, exp: f32) -> Self {
        PyTensor {
            inner: self.inner.pow(exp),
        }
    }

    fn matmul(&self, other: &PyTensor) -> Self {
        PyTensor {
            inner: self.inner.matmul(&other.inner),
        }
    }

    fn neg(&self) -> Self {
        PyTensor {
            inner: self.inner.neg(),
        }
    }

    // ── reductions ───────────────────────────────────────────────────────────

    fn mean(&self) -> Self {
        PyTensor {
            inner: self.inner.mean(),
        }
    }

    fn sum(&self) -> Self {
        PyTensor {
            inner: self.inner.sum(),
        }
    }

    fn sum_axis(&self, axis: isize, keep_dims: bool) -> Self {
        PyTensor {
            inner: self.inner.sum_axis(axis, keep_dims),
        }
    }

    fn max(&self) -> Self {
        PyTensor {
            inner: self.inner.max(),
        }
    }

    fn min(&self) -> Self {
        PyTensor {
            inner: self.inner.min(),
        }
    }

    // ── activations ──────────────────────────────────────────────────────────

    fn relu(&self) -> Self {
        PyTensor {
            inner: self.inner.relu(),
        }
    }

    fn sigmoid(&self) -> Self {
        PyTensor {
            inner: self.inner.sigmoid(),
        }
    }

    fn tanh(&self) -> Self {
        PyTensor {
            inner: self.inner.tanh(),
        }
    }

    fn gelu(&self) -> Self {
        PyTensor {
            inner: self.inner.gelu(),
        }
    }

    fn silu(&self) -> Self {
        PyTensor {
            inner: self.inner.silu(),
        }
    }

    fn swiglu(&self) -> Self {
        PyTensor {
            inner: self.inner.swiglu(),
        }
    }

    fn softmax(&self, axis: usize) -> Self {
        PyTensor {
            inner: self.inner.softmax(axis),
        }
    }

    fn log_softmax(&self, axis: usize) -> Self {
        PyTensor {
            inner: self.inner.log_softmax(axis),
        }
    }

    fn exp(&self) -> Self {
        PyTensor {
            inner: self.inner.exp(),
        }
    }

    fn log(&self) -> Self {
        PyTensor {
            inner: self.inner.log(),
        }
    }

    // ── Python dunder arithmetic ─────────────────────────────────────────────

    fn __add__(&self, other: &PyTensor) -> Self {
        PyTensor {
            inner: self.inner.add(&other.inner),
        }
    }

    fn __sub__(&self, other: &PyTensor) -> Self {
        PyTensor {
            inner: self.inner.sub(&other.inner),
        }
    }

    fn __mul__(&self, other: &PyTensor) -> Self {
        PyTensor {
            inner: self.inner.mul(&other.inner),
        }
    }

    fn __truediv__(&self, other: &PyTensor) -> Self {
        PyTensor {
            inner: self.inner.div(&other.inner),
        }
    }

    fn __pow__(&self, exp: f32, _modulus: Option<u64>) -> Self {
        PyTensor {
            inner: self.inner.pow(exp),
        }
    }

    fn __neg__(&self) -> Self {
        PyTensor {
            inner: self.inner.neg(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?})", self.inner.shape())
    }

    fn __str__(&self) -> String {
        format!(
            "Tensor(shape={:?}, data={:?})",
            self.inner.shape(),
            self.inner.to_vec()
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PyLinear
// ────────────────────────────────────────────────────────────────────────────

/// A linear (fully connected) layer: `y = X @ W + b`.
///
/// ```python
/// layer = te.Linear(128, 64, True)
/// out   = layer.forward(x)
/// ```
#[pyclass(name = "Linear", module = "tensor_engine")]
pub struct PyLinear {
    inner: crate::nn::Linear,
}

#[pymethods]
impl PyLinear {
    #[new]
    fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        PyLinear {
            inner: crate::nn::Linear::new(in_features, out_features, bias),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyTensor {
        use crate::nn::Module;
        PyTensor {
            inner: self.inner.forward(&input.inner),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        use crate::nn::Module;
        self.inner
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, PyTensor)> {
        use crate::nn::Module;
        self.inner
            .named_parameters(prefix)
            .into_iter()
            .map(|(n, t)| (n, PyTensor { inner: t }))
            .collect()
    }

    #[getter]
    fn weight(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.weight.clone(),
        }
    }

    #[getter]
    fn bias(&self) -> Option<PyTensor> {
        self.inner
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    fn __repr__(&self) -> String {
        format!(
            "Linear(in_features={}, out_features={}, bias={})",
            self.inner.in_features,
            self.inner.out_features,
            self.inner.bias.is_some()
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PySGD
// ────────────────────────────────────────────────────────────────────────────

/// Stochastic Gradient Descent optimizer (with optional momentum).
///
/// ```python
/// opt = te.SGD(lr=0.01, momentum=0.9)
/// opt.step(model.parameters())
/// opt.zero_grad(model.parameters())
/// ```
#[pyclass(name = "SGD", module = "tensor_engine")]
pub struct PySGD {
    inner: crate::optim::SGD,
    momentum: f32,
}

#[pymethods]
impl PySGD {
    /// Create a new SGD optimizer.
    ///
    /// Parameters
    /// ----------
    /// lr : float
    ///     Learning rate.
    /// momentum : float, optional
    ///     Momentum factor (default: 0.0).
    #[new]
    #[pyo3(signature = (lr, momentum=0.0))]
    fn new(lr: f32, momentum: f32) -> Self {
        PySGD {
            inner: crate::optim::SGD::new(lr, momentum),
            momentum,
        }
    }

    /// Perform a single optimisation step.
    ///
    /// Parameters
    /// ----------
    /// params : list[Tensor]
    ///     The parameters to update (usually `model.parameters()`).
    fn step(&mut self, params: Vec<PyRef<'_, PyTensor>>) {
        let tensors: Vec<crate::tensor::Tensor> =
            params.iter().map(|p| p.inner.clone()).collect();
        self.inner.step(&tensors);
    }

    /// Zero out gradients on all provided parameters.
    fn zero_grad(&mut self, params: Vec<PyRef<'_, PyTensor>>) {
        let tensors: Vec<crate::tensor::Tensor> =
            params.iter().map(|p| p.inner.clone()).collect();
        self.inner.zero_grad(&tensors);
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.inner.lr()
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }

    fn __repr__(&self) -> String {
        format!("SGD(lr={}, momentum={})", self.inner.lr(), self.momentum)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PyAdam
// ────────────────────────────────────────────────────────────────────────────

/// Adam optimizer.
///
/// ```python
/// opt = te.Adam(lr=1e-3)
/// opt.step(model.parameters())
/// opt.zero_grad(model.parameters())
/// ```
#[pyclass(name = "Adam", module = "tensor_engine")]
pub struct PyAdam {
    inner: crate::optim::Adam,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (lr, beta1=0.9, beta2=0.999, eps=1e-8))]
    fn new(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        PyAdam {
            inner: crate::optim::Adam::new(lr, beta1, beta2, eps),
            beta1,
            beta2,
            eps,
        }
    }

    fn step(&mut self, params: Vec<PyRef<'_, PyTensor>>) {
        let tensors: Vec<crate::tensor::Tensor> =
            params.iter().map(|p| p.inner.clone()).collect();
        self.inner.step(&tensors);
    }

    fn zero_grad(&mut self, params: Vec<PyRef<'_, PyTensor>>) {
        let tensors: Vec<crate::tensor::Tensor> =
            params.iter().map(|p| p.inner.clone()).collect();
        self.inner.zero_grad(&tensors);
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.inner.lr()
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }

    fn __repr__(&self) -> String {
        format!(
            "Adam(lr={}, beta1={}, beta2={}, eps={})",
            self.inner.lr(),
            self.beta1,
            self.beta2,
            self.eps
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PyTransformerBlock
// ────────────────────────────────────────────────────────────────────────────

/// A single Transformer block (multi-head attention + feed-forward).
///
/// ```python
/// block = te.TransformerBlock(512, 2048, 8)
/// out   = block.forward(x)
/// ```
#[pyclass(name = "TransformerBlock", module = "tensor_engine")]
pub struct PyTransformerBlock {
    inner: Arc<Mutex<crate::nn::TransformerBlock>>,
}

#[pymethods]
impl PyTransformerBlock {
    /// Create a standard TransformerBlock.
    #[new]
    #[pyo3(signature = (
        d_model,
        d_ff,
        num_heads = 8,
        kv_heads = None,
        use_rope = false,
        llama_style = false,
        llama_bias = true,
        nl_oob_config = None,
        nl_oob_max_scale = None,
    ))]
    #[allow(unused_variables)]
    fn new(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: Option<usize>,
        use_rope: bool,
        llama_style: bool,
        llama_bias: bool,
        nl_oob_config: Option<&str>,
        nl_oob_max_scale: Option<f32>,
    ) -> PyResult<Self> {
        if llama_style {
            let cfg = crate::nn::TransformerConfig {
                d_model,
                d_ff,
                num_heads,
                kv_heads: kv_heads.unwrap_or(num_heads),
                use_rope,
                rope_theta: 10000.0,
                rope_scale: 1.0,
                bias: false,
            };
            crate::nn::TransformerBlock::new_llama_style(cfg)
                .map(|tb| PyTransformerBlock {
                    inner: Arc::new(Mutex::new(tb)),
                })
                .map_err(PyValueError::new_err)
        } else {
            crate::nn::TransformerBlock::new(d_model, d_ff, num_heads)
                .map(|tb| PyTransformerBlock {
                    inner: Arc::new(Mutex::new(tb)),
                })
                .map_err(PyValueError::new_err)
        }
    }

    /// Create a Llama-style TransformerBlock (RoPE + RMSNorm, SwiGLU ff).
    #[staticmethod]
    fn new_llama(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
        use_rope: bool,
    ) -> PyResult<Self> {
        let cfg = crate::nn::TransformerConfig {
            d_model,
            d_ff,
            num_heads,
            kv_heads,
            use_rope,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            bias: false,
        };
        crate::nn::TransformerBlock::new_llama_style(cfg)
            .map(|tb| PyTransformerBlock {
                inner: Arc::new(Mutex::new(tb)),
            })
            .map_err(PyValueError::new_err)
    }

    /// Run a forward pass (no KV-cache, no causal mask).
    fn forward(&self, input: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self
                .inner
                .lock()
                .unwrap()
                .forward_block_no_cache(&input.inner),
        }
    }

    /// Forward pass incorporating relative-distance bias (for NL-OOB style blocks).
    fn forward_with_distance(&self, input: &PyTensor, dist: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self
                .inner
                .lock()
                .unwrap()
                .forward_block_with_distance(&input.inner, &dist.inner),
        }
    }

    /// Collect all trainable parameters.
    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .lock()
            .unwrap()
            .parameters_impl()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }

    fn __repr__(&self) -> String {
        "TransformerBlock".to_string()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PyTokenizer
// ────────────────────────────────────────────────────────────────────────────

/// A simple BPE / word-piece tokenizer loaded from a JSON file.
///
/// ```python
/// tok = te.Tokenizer.from_json("tokenizer.json")
/// ids = tok.encode("Hello, world!")
/// txt = tok.decode(ids)
/// ```
#[pyclass(name = "Tokenizer", module = "tensor_engine")]
pub struct PyTokenizer {
    inner: crate::tokenizer::Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load tokenizer from a JSON vocabulary file.
    #[staticmethod]
    fn from_json(path: &str) -> PyResult<Self> {
        crate::tokenizer::Tokenizer::from_json(path)
            .map(|t| PyTokenizer { inner: t })
            .map_err(PyValueError::new_err)
    }

    /// Encode a string to a list of token ids.
    fn encode(&self, text: &str) -> Vec<usize> {
        self.inner.encode(text)
    }

    /// Decode a list of token ids back to a string.
    fn decode(&self, ids: Vec<usize>) -> String {
        self.inner.decode(&ids)
    }

    /// Return the vocabulary size.
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Look up the integer id for a token string, or None if not found.
    fn token_to_id(&self, token: &str) -> Option<usize> {
        self.inner.token_to_id(token)
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Module entry-point
// ────────────────────────────────────────────────────────────────────────────

/// The Python extension module.
///
/// The function name **must** match `[lib] name = "tensor_engine"` in
/// `Cargo.toml` so that maturin emits the `PyInit_tensor_engine` symbol.
#[pymodule]
pub fn tensor_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyTransformerBlock>()?;
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
