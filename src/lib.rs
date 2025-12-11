#![recursion_limit = "2048"]

//! This crate provides a tensor library with automatic differentiation.

#[cfg(feature = "python_bindings")]
use ndarray::Array;
#[cfg(feature = "python_bindings")]
use ndarray::IxDyn;
#[cfg(feature = "python_bindings")]
use pyo3::prelude::*;
#[cfg(all(feature = "python_bindings", feature = "safe_tensors"))]
use pyo3::types::PyDict;
#[cfg(all(feature = "python_bindings", feature = "with_tokenizers"))]
use tokenizers::Tokenizer as HFTokenizer;

pub mod autograd;
pub mod dtype;
pub mod io;
pub mod labels;
pub mod backend;
#[path = "nn/mod.rs"]
pub mod nn;
#[cfg(feature = "safe_tensors")]
pub use io::safetensors_loader::load_safetensors_from_bytes;
#[cfg(feature = "safe_tensors")]
pub use io::safetensors_loader::apply_kronos_bytes_to_module_bytes;
pub mod ops;
pub mod tensor;

#[cfg(feature = "python_bindings")]
use crate::labels::Labels;
#[cfg(feature = "python_bindings")]
use nn::TransformerBlock;
#[cfg(feature = "python_bindings")]
use nn::{Adam, Linear, Module, Optimizer, SGD};
#[cfg(feature = "python_bindings")]
use tensor::Tensor;

/// A Python wrapper for the `Tensor` struct.
#[cfg(feature = "python_bindings")]
#[pyclass(name = "Tensor")]
#[derive(Clone)]
struct PyTensor(Tensor);

#[cfg(all(feature = "python_bindings", feature = "with_tokenizers"))]
#[pyclass(name = "Tokenizer")]
#[derive(Clone)]
struct PyTokenizer(HFTokenizer);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyTensor {
    /// Creates a new tensor.
    ///
    /// # Arguments
    ///
    /// * `value` - The tensor's data, as a flat list of f32 values.
    /// * `shape` - The shape of the tensor.
    #[new]
    fn new(value: Vec<f32>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Self> {
        let array = Array::from_shape_vec(shape, value).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create tensor: {}",
                e
            ))
        })?;
        // Parse dtype if provided
        let dt = if let Some(s) = dtype {
            match crate::dtype::DType::parse(s) {
                Some(d) => d,
                None => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown dtype: {}",
                        s
                    )))
                }
            }
        } else {
            crate::dtype::DType::F32
        };
        Ok(PyTensor(Tensor::new_with_dtype(array.into_dyn(), true, dt)))
    }

    /// Adds two tensors.
    fn add(&self, other: &PyTensor) -> PyTensor {
        PyTensor(self.0.add(&other.0))
    }

    /// Multiplies two tensors.
    fn mul(&self, other: &PyTensor) -> PyTensor {
        PyTensor(self.0.mul(&other.0))
    }

    /// Subtracts two tensors.
    fn sub(&self, other: &PyTensor) -> PyTensor {
        PyTensor(self.0.sub(&other.0))
    }

    /// Divides two tensors.
    fn div(&self, other: &PyTensor) -> PyTensor {
        PyTensor(self.0.div(&other.0))
    }

    /// Raises a tensor to a power.
    fn pow(&self, power: f32) -> PyTensor {
        PyTensor(self.0.pow(power))
    }

    /// Applies the ReLU activation function.
    fn relu(&self) -> PyTensor {
        PyTensor(self.0.relu())
    }

    /// Applies the sigmoid activation function.
    fn sigmoid(&self) -> PyTensor {
        PyTensor(self.0.sigmoid())
    }

    /// Applies the tanh activation function.
    fn tanh(&self) -> PyTensor {
        PyTensor(self.0.tanh())
    }

    /// GELU activation function.
    fn gelu(&self) -> PyTensor {
        PyTensor(self.0.gelu())
    }

    /// Ternary quantization (project values to -1, 0, 1)
    fn ternary(&self) -> PyTensor {
        PyTensor(self.0.ternary())
    }

    fn exp(&self) -> PyTensor {
        PyTensor(self.0.exp())
    }

    /// Computes the sum of the tensor's elements.
    fn sum(&self) -> PyTensor {
        PyTensor(self.0.sum())
    }

    /// Computes the mean of the tensor's elements.
    fn mean(&self) -> PyTensor {
        PyTensor(self.0.mean())
    }

    /// Computes the maximum value of the tensor's elements.
    fn max(&self) -> PyTensor {
        PyTensor(self.0.max())
    }

    /// Computes the minimum value of the tensor's elements.
    fn min(&self) -> PyTensor {
        PyTensor(self.0.min())
    }

    /// Softmax along axis (default last axis)
    fn softmax(&self, axis: Option<isize>) -> PyResult<PyTensor> {
        let ndim = self.0.lock().storage.shape().len() as isize;
        let a = axis.unwrap_or(-1);
        let axis_norm = if a < 0 {
            (ndim + a) as usize
        } else {
            a as usize
        };
        Ok(PyTensor(self.0.softmax(axis_norm)))
    }

    /// Elementwise equality comparison.
    fn equal(&self, other: &PyTensor) -> PyTensor {
        PyTensor(self.0.equal(&other.0))
    }

    /// Elementwise greater-than comparison.
    fn greater(&self, other: &PyTensor) -> PyTensor {
        PyTensor(self.0.greater(&other.0))
    }

    /// Elementwise less-than comparison.
    fn less(&self, other: &PyTensor) -> PyTensor {
        PyTensor(self.0.less(&other.0))
    }

    /// Change dtype of this tensor (round-trip conversions emulate lower precision storage)
    fn astype(&self, dtype: &str) -> PyResult<PyTensor> {
        match crate::dtype::DType::parse(dtype) {
            Some(dt) => Ok(PyTensor(self.0.astype(dt))),
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported dtype: {}",
                dtype
            ))),
        }
    }

    /// Log-Softmax along axis (default last axis)
    fn log_softmax(&self, axis: Option<isize>) -> PyResult<PyTensor> {
        let ndim = self.0.lock().storage.shape().len() as isize;
        let a = axis.unwrap_or(-1);
        let axis_norm = if a < 0 {
            (ndim + a) as usize
        } else {
            a as usize
        };
        Ok(PyTensor(self.0.log_softmax(axis_norm)))
    }

    /// Cross-entropy with logits: expects targets as one-hot float vectors or 1D indices (float ints)
    fn cross_entropy_with_logits(
        &self,
        target: &PyTensor,
        axis: Option<isize>,
    ) -> PyResult<PyTensor> {
        let ndim = self.0.lock().storage.shape().len() as isize;
        let a = axis.unwrap_or(-1);
        let axis_norm = if a < 0 {
            (ndim + a) as usize
        } else {
            a as usize
        };
        Ok(PyTensor(
            self.0
                .cross_entropy_with_logits(&target.0, axis_norm as isize),
        ))
    }

    fn softmax_cross_entropy_with_logits(
        &self,
        target: &PyTensor,
        axis: Option<isize>,
    ) -> PyResult<PyTensor> {
        let ndim = self.0.lock().storage.shape().len() as isize;
        let a = axis.unwrap_or(-1);
        let axis_norm = if a < 0 {
            (ndim + a) as usize
        } else {
            a as usize
        };
        Ok(PyTensor(self.0.softmax_cross_entropy_with_logits(
            &target.0,
            axis_norm as isize,
        )))
    }

    fn nll_loss(&self, target: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor(self.0.nll_loss(&target.0)))
    }

    fn swiglu(&self) -> PyTensor {
        PyTensor(self.0.swiglu())
    }

    fn rmsnorm(&self, gamma: &PyTensor, axis: usize, eps: f32) -> PyResult<PyTensor> {
        Ok(PyTensor(self.0.rmsnorm(&gamma.0, axis, eps)))
    }

    /// Reshapes the tensor.
    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        match self.0.reshape(shape) {
            Ok(t) => Ok(PyTensor(t)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        }
    }

    fn permute(&self, perm: Vec<usize>) -> PyTensor {
        PyTensor(self.0.permute(perm))
    }

    fn rope(&self, num_heads: usize) -> PyTensor {
        PyTensor(self.0.rope(num_heads))
    }

    /// Transposes the tensor.
    fn transpose(&self) -> PyTensor {
        PyTensor(self.0.transpose())
    }

    /// Concatenates a list of tensors along a given axis.
    #[staticmethod]
    fn cat(tensors: Vec<PyTensor>, axis: usize) -> PyResult<PyTensor> {
        let rust_tensors: Vec<Tensor> = tensors.into_iter().map(|t| t.0).collect();
        Ok(PyTensor(Tensor::concat(&rust_tensors, axis)))
    }

    /// Stacks a list of tensors along a new axis.
    #[staticmethod]
    fn stack(tensors: Vec<PyTensor>, axis: usize) -> PyResult<PyTensor> {
        let rust_tensors: Vec<Tensor> = tensors.into_iter().map(|t| t.0).collect();
        Ok(PyTensor(Tensor::stack(&rust_tensors, axis)))
    }
    #[staticmethod]
    fn embedding_lookup(emb: PyTensor, indices: PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor(Tensor::embedding_lookup(&emb.0, &indices.0)))
    }

    #[staticmethod]
    fn kvcache_append(cache: PyTensor, newkv: PyTensor, axis: usize) -> PyResult<PyTensor> {
        Ok(PyTensor(Tensor::kvcache_append(&cache.0, &newkv.0, axis)))
    }

    /// Sets the gradient of this tensor to zero.
    fn zero_grad(&self) {
        self.0.zero_grad();
    }

    /// Detaches the tensor from the computation graph.
    fn detach(&self) -> PyTensor {
        PyTensor(self.0.detach())
    }

    /// Returns whether this tensor requires gradients.
    #[getter]
    fn requires_grad(&self) -> bool {
        self.0.requires_grad()
    }

    /// Sets whether this tensor requires gradients.
    #[setter]
    fn set_requires_grad(&mut self, requires_grad: bool) {
        self.0.set_requires_grad(requires_grad);
    }

    /// Returns the shape of the tensor.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.lock().storage.shape().to_vec()
    }

    /// Returns the dtype of the tensor.
    #[getter]
    fn dtype(&self) -> String {
        self.0.lock().dtype.as_str().to_string()
    }

    /// Performs backpropagation starting from this tensor.
    fn backward(&self) {
        self.0.backward();
    }

    fn __str__(&self) -> String {
        format!("{}", self.0.lock().storage.to_f32_array())
    }

    fn __repr__(&self) -> String {
        let tensor = self.0.lock();
        format!(
            "Tensor(data={}, shape={:?}, requires_grad={})",
            tensor.storage.to_f32_array(),
            tensor.storage.shape(),
            tensor.requires_grad
        )
    }

    /// Quantized matmul: right operand should be a quantized weight tensor (I8 storage)
    fn quantized_matmul(&self, qweight: &PyTensor) -> PyTensor {
        PyTensor(self.0.quantized_matmul(&qweight.0))
    }

    /// Quantizes the tensor as weights, returning a new quantized tensor.
    /// dtype: string name e.g. "i8_rowwise", "i8_blockwise"
    fn quantize_weights(&self, dtype: &str, block_size: Option<usize>) -> PyResult<PyTensor> {
        match crate::dtype::DType::parse(dtype) {
            Some(dt) => match self.0.quantize_weights(dt, block_size) {
                Ok(t) => Ok(PyTensor(t)),
                Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
            },
            None => Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported dtype: {}", dtype))),
        }
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        if let Ok(other_tensor) = other.extract::<PyTensor>() {
            Ok(self.add(&other_tensor))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported operand type for +",
            ))
        }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        if let Ok(other_tensor) = other.extract::<PyTensor>() {
            Ok(self.mul(&other_tensor))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported operand type for *",
            ))
        }
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        if let Ok(other_tensor) = other.extract::<PyTensor>() {
            Ok(self.sub(&other_tensor))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported operand type for -",
            ))
        }
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        if let Ok(other_tensor) = other.extract::<PyTensor>() {
            Ok(self.div(&other_tensor))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported operand type for /",
            ))
        }
    }

    fn __pow__(
        &self,
        _other: &Bound<'_, PyAny>,
        modulus: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyTensor> {
        if modulus.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Modulus not supported for **",
            ));
        }
        if let Ok(power) = _other.extract::<f32>() {
            Ok(self.pow(power))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported operand type for **",
            ))
        }
    }

    /// Returns the tensor's data as a flat list of f32 values.
    fn get_data(&self) -> Vec<f32> {
        self.0
            .lock()
            .storage
            .to_f32_array()
            .iter()
            .cloned()
            .collect()
    }

    /// Sets the tensor's data from a flat list of f32 values.
    ///
    /// # Arguments
    ///
    /// * `value` - The new data for the tensor.
    fn set_data(&self, value: Vec<f32>) -> PyResult<()> {
        let mut tensor = self.0.lock();
        let cur_shape = tensor.storage.shape().to_vec();
        let cur_len: usize = cur_shape.iter().product();
        if cur_len != value.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "New data must have the same number of elements as the tensor.",
            ));
        }
        let arr = ndarray::Array::from_shape_vec(IxDyn(&cur_shape), value).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to set data: {}", e))
        })?;
        tensor.storage = crate::dtype::TensorStorage::from_f32_array(&arr.into_dyn(), tensor.dtype);
        Ok(())
    }

    /// The gradient of the tensor, if it has one.
    fn get_grad(&self) -> PyResult<Option<Vec<f32>>> {
        let tensor = self.0.lock();
        if let Some(grad) = &tensor.grad {
            Ok(Some(grad.iter().cloned().collect()))
        } else {
            Ok(None)
        }
    }

    /// Sets the tensor's gradient to a specific vector (for testing and interop)
    fn set_grad(&self, value: Vec<f32>) -> PyResult<()> {
        let mut tensor = self.0.lock();
        let cur_shape = tensor.storage.shape().to_vec();
        let cur_len: usize = cur_shape.iter().product();
        if cur_len != value.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "New grad must have the same number of elements as the tensor.",
            ));
        }
        let arr = ndarray::Array::from_shape_vec(IxDyn(&cur_shape), value).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to set grad: {}", e))
        })?;
        tensor.grad = Some(arr);
        Ok(())
    }
}

/// Python wrapper for integer label arrays
#[cfg(feature = "python_bindings")]
#[pyclass(name = "Labels")]
struct PyLabels(Labels);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyLabels {
    #[new]
    fn new(indices: Vec<i64>) -> PyResult<Self> {
        let arr = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[indices.len()]), indices)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to create labels array: {}",
                    e
                ))
            })?;
        Ok(PyLabels(Labels::new(arr)))
    }

    fn to_one_hot(&self, num_classes: usize) -> PyResult<PyTensor> {
        let oh = self.0.to_one_hot(num_classes);
        Ok(PyTensor(Tensor::new(oh, false)))
    }
}

/// MSE loss wrapper
#[cfg(feature = "python_bindings")]
#[pyclass(name = "MSELoss")]
struct PyMSELoss(nn::MSELoss);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyMSELoss {
    #[new]
    fn new() -> Self {
        PyMSELoss(nn::MSELoss::new())
    }

    fn forward(&self, pred: &PyTensor, target: &PyTensor) -> PyTensor {
        PyTensor(self.0.forward(&pred.0, &target.0))
    }
}

/// CrossEntropy loss wrapper (logits-based)
#[cfg(feature = "python_bindings")]
#[pyclass(name = "CrossEntropyLoss")]
struct PyCrossEntropyLoss;

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyCrossEntropyLoss {
    #[new]
    fn new() -> Self {
        PyCrossEntropyLoss
    }

    /// forward accepts logits and target. target may be 1D class indices or 2D one-hot.
    fn forward(
        &self,
        logits: &PyTensor,
        target: &PyTensor,
        axis: Option<isize>,
    ) -> PyResult<PyTensor> {
        let ndim = logits.0.lock().storage.shape().len() as isize;
        let a = axis.unwrap_or(-1);
        let axis_norm = if a < 0 {
            (ndim + a) as usize
        } else {
            a as usize
        };
        Ok(PyTensor(
            logits
                .0
                .cross_entropy_with_logits(&target.0, axis_norm as isize),
        ))
    }
}

/// NLLLoss wrapper
#[cfg(feature = "python_bindings")]
#[pyclass(name = "NLLLoss")]
struct PyNLLLoss;

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyNLLLoss {
    #[new]
    fn new() -> Self {
        PyNLLLoss
    }
    fn forward(&self, log_probs: &PyTensor, target: &PyTensor) -> PyTensor {
        PyTensor(log_probs.0.nll_loss(&target.0))
    }
    fn forward_from_labels(
        &self,
        log_probs: &PyTensor,
        labels: &PyLabels,
        axis: Option<isize>,
    ) -> PyResult<PyTensor> {
        let ndim = log_probs.0.lock().storage.shape().len() as isize;
        let a = axis.unwrap_or(-1);
        let axis_norm = if a < 0 {
            (ndim + a) as usize
        } else {
            a as usize
        };
        Ok(PyTensor(
            log_probs.0.nll_loss(&Tensor::new(
                labels
                    .0
                    .to_one_hot(log_probs.0.lock().storage.shape()[axis_norm])
                    .into_dyn(),
                false,
            )),
        ))
    }
}

/// SoftmaxCrossEntropy wrapper (combined op)
#[cfg(feature = "python_bindings")]
#[pyclass(name = "SoftmaxCrossEntropyLoss")]
struct PySoftmaxCrossEntropyLoss;

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PySoftmaxCrossEntropyLoss {
    #[new]
    fn new() -> Self {
        PySoftmaxCrossEntropyLoss
    }
    fn forward(
        &self,
        logits: &PyTensor,
        target: &PyTensor,
        axis: Option<isize>,
    ) -> PyResult<PyTensor> {
        let ndim = logits.0.lock().storage.shape().len() as isize;
        let a = axis.unwrap_or(-1);
        let axis_norm = if a < 0 {
            (ndim + a) as usize
        } else {
            a as usize
        };
        Ok(PyTensor(logits.0.softmax_cross_entropy_with_logits(
            &target.0,
            axis_norm as isize,
        )))
    }
    fn forward_from_labels(
        &self,
        logits: &PyTensor,
        labels: &PyLabels,
        axis: Option<isize>,
    ) -> PyResult<PyTensor> {
        let ndim = logits.0.lock().storage.shape().len() as isize;
        let a = axis.unwrap_or(-1);
        let axis_norm = if a < 0 {
            (ndim + a) as usize
        } else {
            a as usize
        };
        Ok(PyTensor(
            logits.0.softmax_cross_entropy_with_logits(
                &Tensor::new(
                    labels
                        .0
                        .to_one_hot(logits.0.lock().storage.shape()[axis_norm])
                        .into_dyn(),
                    false,
                ),
                axis_norm as isize,
            ),
        ))
    }
}

/// CrossEntropyLogits layer wrapper
#[cfg(feature = "python_bindings")]
#[pyclass(name = "CrossEntropyLogitsLoss")]
struct PyCrossEntropyLogitsLoss(nn::CrossEntropyLogitsLoss);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyCrossEntropyLogitsLoss {
    #[new]
    fn new() -> Self {
        PyCrossEntropyLogitsLoss(nn::CrossEntropyLogitsLoss::new())
    }
    fn forward(
        &self,
        logits: &PyTensor,
        target: &PyTensor,
        axis: Option<isize>,
    ) -> PyResult<PyTensor> {
        let ndim = logits.0.lock().storage.shape().len() as isize;
        let a = axis.unwrap_or(-1);
        let axis_norm = if a < 0 {
            (ndim + a) as usize
        } else {
            a as usize
        };
        Ok(PyTensor(self.0.forward(
            &logits.0,
            &target.0,
            axis_norm as isize,
        )))
    }
    fn forward_from_labels(
        &self,
        logits: &PyTensor,
        labels: &PyLabels,
        axis: Option<isize>,
    ) -> PyResult<PyTensor> {
        let ndim = logits.0.lock().storage.shape().len() as isize;
        let a = axis.unwrap_or(-1);
        let axis_norm = if a < 0 {
            (ndim + a) as usize
        } else {
            a as usize
        };
        Ok(PyTensor(self.0.forward_from_labels(
            &logits.0,
            &labels.0,
            axis_norm as isize,
        )))
    }
}

/// A Python wrapper for the `Linear` layer.
#[cfg(feature = "python_bindings")]
#[pyclass(name = "Linear")]
struct PyLinear(Linear);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyLinear {
    #[new]
    fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        PyLinear(Linear::new(in_features, out_features, bias))
    }

    fn forward(&self, input: &PyTensor) -> PyTensor {
        PyTensor(self.0.forward(&input.0))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.0.parameters().into_iter().map(PyTensor).collect()
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, PyTensor)> {
        self.0.named_parameters(prefix).into_iter().map(|(n, t)| (n, PyTensor(t))).collect()
    }

    #[getter]
    fn weight(&self) -> PyTensor {
        PyTensor(self.0.weight.clone())
    }

    #[getter]
    fn bias(&self) -> Option<PyTensor> {
        self.0.bias.as_ref().map(|b| PyTensor(b.clone()))
    }
}

/// TransformerBlock Python wrapper
#[cfg(feature = "python_bindings")]
#[pyclass(name = "TransformerBlock")]
struct PyTransformerBlock(TransformerBlock);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyTransformerBlock {
    #[new]
    fn new(
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: Option<usize>,
        use_rope: Option<bool>,
        nl_oob_config: Option<&str>,
        nl_oob_max_scale: Option<f32>,
    ) -> Self {
        let kv = kv_heads.unwrap_or(num_heads);
        let rope = use_rope.unwrap_or(false);
        if let Some(cfg) = nl_oob_config {
            let bias = match cfg {
                "logarithmic" | "log" | "0" => crate::nn::transformer::BiasFunction::Logarithmic,
                "gaussian" | "1" => crate::nn::transformer::BiasFunction::Gaussian,
                _ => crate::nn::transformer::BiasFunction::Logarithmic,
            };
            let max_scale = nl_oob_max_scale.unwrap_or(2.0);
            PyTransformerBlock(TransformerBlock::new_with_nl_oob(
                d_model, d_ff, num_heads, bias, max_scale,
            ))
        } else {
            PyTransformerBlock(TransformerBlock::new_with_kv_and_rope(
                d_model, d_ff, num_heads, kv, rope,
            ))
        }
    }

    fn forward(&self, input: &PyTensor) -> PyTensor {
        PyTensor(self.0.forward_block(&input.0))
    }

    fn forward_with_distance(&self, input: &PyTensor, distance: &PyTensor) -> PyTensor {
        PyTensor(self.0.forward_block_with_distance(&input.0, &distance.0))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.0.parameters().into_iter().map(PyTensor).collect()
    }
}

/// A Python wrapper for the `SGD` optimizer.
#[cfg(feature = "python_bindings")]
#[pyclass(name = "SGD")]
struct PySGD(SGD);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PySGD {
    #[new]
    fn new(lr: f32, momentum: f32) -> Self {
        PySGD(SGD::new(lr, momentum))
    }

    fn step(&mut self, parameters: Vec<PyTensor>) {
        let rust_params: Vec<Tensor> = parameters.into_iter().map(|p| p.0).collect();
        self.0.step(&rust_params);
    }

    fn zero_grad(&mut self, parameters: Vec<PyTensor>) {
        let rust_params: Vec<Tensor> = parameters.into_iter().map(|p| p.0).collect();
        self.0.zero_grad(&rust_params);
    }

    fn cast_params(&mut self, _parameters: Vec<PyTensor>, _dtype: &str) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "SGD::cast_params is not implemented.",
        ))
    }
}

/// A Python wrapper for the `Adam` optimizer.
#[cfg(feature = "python_bindings")]
#[pyclass(name = "Adam")]
struct PyAdam(Adam);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyAdam {
    #[new]
    fn new(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        PyAdam(Adam::new(lr, beta1, beta2, eps))
    }

    fn step(&mut self, parameters: Vec<PyTensor>) {
        let rust_params: Vec<Tensor> = parameters.into_iter().map(|p| p.0).collect();
        self.0.step(&rust_params);
    }

    fn zero_grad(&mut self, parameters: Vec<PyTensor>) {
        let rust_params: Vec<Tensor> = parameters.into_iter().map(|p| p.0).collect();
        self.0.zero_grad(&rust_params);
    }

    fn cast_params(&mut self, _parameters: Vec<PyTensor>, _dtype: &str) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Adam::cast_params is not implemented.",
        ))
    }
}

#[cfg(all(feature = "python_bindings", feature = "with_tokenizers"))]
#[pymethods]
impl PyTokenizer {
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        match crate::io::tokenizers::load_tokenizer_from_file(path) {
            Ok(tok) => Ok(PyTokenizer(tok)),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(e)),
        }
    }

    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        match crate::io::tokenizers::encode_text(&self.0, text) {
            Ok(ids) => Ok(ids),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }
}

/// A Python module implemented in Rust.
#[cfg(feature = "python_bindings")]
#[pymodule]
fn tensor_engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyMSELoss>()?;
    m.add_class::<PyCrossEntropyLoss>()?;
    m.add_class::<PyNLLLoss>()?;
    m.add_class::<PySoftmaxCrossEntropyLoss>()?;
    m.add_class::<PyCrossEntropyLogitsLoss>()?;
    m.add_class::<PyLabels>()?;
    m.add_class::<PyTransformerBlock>()?;
    m.add_class::<PyVisionTransformer>()?;
    m.add_class::<PyMultimodalLLM>()?;
    #[cfg(all(feature = "python_bindings", feature = "with_tokenizers"))]
    m.add_class::<PyTokenizer>()?;
    #[cfg(all(feature = "python_bindings", feature = "safe_tensors"))]
    m.add_function(pyo3::wrap_pyfunction!(py_load_safetensors, m)?)?;
    #[cfg(all(feature = "python_bindings", feature = "safe_tensors"))]
    m.add_function(pyo3::wrap_pyfunction!(py_load_safetensors_into_module, m)?)?;
    #[cfg(feature = "python_bindings")]
    m.add_function(pyo3::wrap_pyfunction!(py_set_cpu_backend, m)?)?;
    Ok(())
}

#[cfg(all(feature = "python_bindings", feature = "safe_tensors"))]
#[pyfunction]
fn py_load_safetensors(py: Python<'_>, bytes: Vec<u8>, transpose: bool) -> PyResult<PyObject> {
    let state = crate::io::safetensors_loader::load_safetensors_from_bytes(&bytes, transpose)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let dict = PyDict::new_bound(py);
    for (k, t) in state.into_iter() {
        let py_tensor = Py::new(py, PyTensor(t)).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create PyTensor: {}", e)))?;
        dict.set_item(k, py_tensor)?;
    }
    Ok(dict.to_object(py))
}

#[cfg(feature = "python_bindings")]
#[pyfunction(name = "set_cpu_backend")]
fn py_set_cpu_backend() -> PyResult<()> {
    crate::backend::set_cpu_backend().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(())
}

#[cfg(all(feature = "python_bindings", feature = "safe_tensors"))]
#[pyfunction]
fn py_load_safetensors_into_module(
    py: Python<'_>,
    bytes: Vec<u8>,
    transpose: bool,
    module: PyObject,
    root: Option<&str>,
) -> PyResult<()> {
    // Deserialize into state dict
    let state = crate::io::safetensors_loader::load_safetensors_from_bytes(&bytes, transpose)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    // Convert Python module to &mut dyn NN::Module via downcasting
    // Expect module to be a Python wrapper around a Rust Module; extract inner Rust object
    let root = root.unwrap_or("");
    // Since PyObject can be any Python class, try to downcast to known wrappers by name
    // We'll attempt common module wrappers like PyTransformerBlock
    use std::borrow::Cow;
    log::debug!("py_load_safetensors_into_module: about to inspect module type");
    let binding = module.bind(py).get_type();
    let type_name: Cow<'_, str> = binding.name().unwrap_or(Cow::Borrowed(""));
    log::debug!("py_load_safetensors_into_module: module type = {}", type_name);
    if type_name == "TransformerBlock" {
        // pyo3::PyTryFrom is deprecated, prefer using extract directly on PyObject
        // Borrow the TransformerBlock mutably from Python, then call loader
        log::debug!("py_load_safetensors_into_module: about to extract module as PyTransformerBlock");
        // Ensure we don't hold any Tensor locks while extracting the Python ref
        let mut py_ref: pyo3::PyRefMut<PyTransformerBlock> = module.extract(py).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Invalid module type: {}", e))
        })?;
        log::debug!("py_load_safetensors_into_module: successfully extracted PyTransformerBlock");
        // Now we can apply the state dict to the inner module
        log::debug!("py_load_safetensors_into_module: about to apply state dict to module");
        let res = crate::io::safetensors_loader::apply_state_dict_to_module(
            &mut py_ref.0,
            &state,
            root,
        );
        log::debug!("py_load_safetensors_into_module: apply_state_dict_to_module returned");
        res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    } else if type_name == "VisionTransformer" {
        log::debug!("py_load_safetensors_into_module: about to extract module as PyVisionTransformer");
        let mut py_ref: pyo3::PyRefMut<PyVisionTransformer> = module.extract(py).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Invalid module type: {}", e))
        })?;
        log::debug!("py_load_safetensors_into_module: successfully extracted PyVisionTransformer");
        let res = crate::io::safetensors_loader::apply_state_dict_to_module(&mut py_ref.0, &state, root);
        res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    } else if type_name == "MultimodalLLM" {
        log::debug!("py_load_safetensors_into_module: about to extract module as PyMultimodalLLM");
        let mut py_ref: pyo3::PyRefMut<PyMultimodalLLM> = module.extract(py).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Invalid module type: {}", e))
        })?;
        log::debug!("py_load_safetensors_into_module: successfully extracted PyMultimodalLLM");
        let res = crate::io::safetensors_loader::apply_state_dict_to_module(&mut py_ref.0, &state, root);
        res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    } else {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "py_load_safetensors_into_module: Unsupported module type '{}'",
            type_name
        )))
    }
}

// VisionTransformer Python wrapper
#[cfg(feature = "python_bindings")]
#[pyclass(name = "VisionTransformer")]
struct PyVisionTransformer(nn::VisionTransformer);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyVisionTransformer {
    #[new]
    fn new(c: usize, patch_size: usize, d_model: usize, d_ff: usize, num_heads: usize, depth: usize, max_len: usize) -> Self {
        PyVisionTransformer(nn::VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, max_len))
    }

    fn forward(&self, input: &PyTensor) -> PyTensor {
        PyTensor(self.0.forward(&input.0))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.0.parameters().into_iter().map(PyTensor).collect()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, PyTensor)> {
        self.0.named_parameters(prefix).into_iter().map(|(n, t)| (n, PyTensor(t))).collect()
    }
}

// Multimodal LLM Python wrapper
#[cfg(feature = "python_bindings")]
#[pyclass(name = "MultimodalLLM")]
struct PyMultimodalLLM(nn::MultimodalLLM);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyMultimodalLLM {
    #[new]
    fn new(vision: PyVisionTransformer, vocab_size: usize, d_model: usize, d_ff: usize, num_heads: usize, depth: usize) -> Self {
        PyMultimodalLLM(nn::MultimodalLLM::new(vision.0, vocab_size, d_model, d_ff, num_heads, depth))
    }

    #[staticmethod]
    fn from_config(py_config: &pyo3::types::PyAny) -> PyResult<Self> {
        // Accept a Python dict or a string path to a JSON file or JSON-like mapping
        let dict: pyo3::types::PyResult<&pyo3::types::PyDict> = py_config.downcast::<pyo3::types::PyDict>();
        let dict = if let Ok(d) = dict {
            d
        } else {
            // Try to interpret as a string path to a JSON file
            if let Ok(path) = py_config.extract::<&str>() {
                let json_str = std::fs::read_to_string(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read config file: {}", e)))?;
                let gil = Python::acquire_gil();
                let py = gil.python();
                let pyobj = py.run(&format!("import json\njson.loads('''{}''')", json_str.replace("'","\'")), None, None);
                // fallback: parse via json.loads into a Python dict
                let py_dict = py.import("json").unwrap().call_method1("loads", (json_str,)).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed parse json: {}", e)))?;
                py_dict.downcast::<pyo3::types::PyDict>()?
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err("from_config expects a dict or path string"));
            }
        };
        let c = dict.get_item("c").and_then(|v| v.extract::<usize>().ok()).unwrap_or(3);
        let patch_size = dict.get_item("patch_size").and_then(|v| v.extract::<usize>().ok()).unwrap_or(16);
        let d_model = dict.get_item("d_model").and_then(|v| v.extract::<usize>().ok()).unwrap_or(768);
        let d_ff = dict.get_item("d_ff").and_then(|v| v.extract::<usize>().ok()).unwrap_or(d_model*4);
        let num_heads = dict.get_item("num_heads").and_then(|v| v.extract::<usize>().ok()).unwrap_or(12);
        let depth = dict.get_item("depth").and_then(|v| v.extract::<usize>().ok()).unwrap_or(12);
        let max_len = dict.get_item("max_len").and_then(|v| v.extract::<usize>().ok()).unwrap_or(1024);
        let vocab_size = dict.get_item("vocab_size").and_then(|v| v.extract::<usize>().ok()).unwrap_or(50000);

        let vision = nn::VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, max_len);
        Ok(PyMultimodalLLM(nn::MultimodalLLM::new(vision, vocab_size, d_model, d_ff, num_heads, depth)))
    }

    fn forward(&self, images: &PyTensor, input_ids: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor(self.0.forward(&images.0, &input_ids.0)))
    }

    fn vision_forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor(self.0.vision_encoder.forward(&input.0)))
    }

    #[getter]
    fn text_embedding(&self) -> PyResult<PyTensor> {
        Ok(PyTensor(self.0.text_embedding.clone()))
    }

    #[setter]
    fn set_text_embedding(&mut self, emb: PyTensor) {
        self.0.text_embedding = emb.0;
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.0.parameters().into_iter().map(PyTensor).collect()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, PyTensor)> {
        self.0.named_parameters(prefix).into_iter().map(|(n, t)| (n, PyTensor(t))).collect()
    }

    /// Export the module parameters to a SafeTensors bytes object (Python `bytes`).
    fn save_state_dict<'py>(&self, py: Python<'py>) -> PyResult<&'py pyo3::types::PyBytes> {
        #[cfg(feature = "safe_tensors")]
        {
            match crate::io::safetensors_loader::save_module_to_safetensors_bytes(&self.0) {
                Ok(bytes) => Ok(pyo3::types::PyBytes::new(py, &bytes)),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
            }
        }
        #[cfg(not(feature = "safe_tensors"))]
        {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("safetensors feature not enabled"))
        }
    }

    /// Save module parameters to a file path in SafeTensors format.
    fn save_state_dict_to_path(&self, path: &str) -> PyResult<()> {
        #[cfg(feature = "safe_tensors")]
        {
            match crate::io::safetensors_loader::save_module_to_safetensors_bytes(&self.0) {
                Ok(bytes) => {
                    std::fs::write(path, bytes).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to write file: {}", e)))?;
                    Ok(())
                }
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
            }
        }
        #[cfg(not(feature = "safe_tensors"))]
        {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("safetensors feature not enabled"))
        }
    }

    /// Load state dict bytes (SafeTensors or Kronos) into this module.
    fn load_state_dict(&mut self, py: Python<'_>, bytes: Vec<u8>, transpose: bool, root: Option<&str>) -> PyResult<()> {
        let root_s = root.unwrap_or("");
        match crate::io::safetensors_loader::apply_safetensors_bytes_to_module_bytes(&mut self.0, &bytes, transpose, root_s) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// Load module parameters from the given SafeTensors file path.
    fn load_state_dict_from_path(&mut self, path: &str, transpose: bool, root: Option<&str>) -> PyResult<()> {
        let bytes = std::fs::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read file: {}", e)))?;
        let root_s = root.unwrap_or("");
        match crate::io::safetensors_loader::apply_safetensors_bytes_to_module_bytes(&mut self.0, &bytes, transpose, root_s) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }
}
