#![recursion_limit = "2048"]

//! This crate provides a tensor library with automatic differentiation.

#[cfg(feature = "python_bindings")]
use ndarray::Array;
#[cfg(feature = "python_bindings")]
use ndarray::IxDyn;
#[cfg(feature = "python_bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python_bindings")]
use pyo3::types::PyDict;
#[cfg(all(feature = "python_bindings", feature = "with_tokenizers"))]
use tokenizers::Tokenizer as HFTokenizer;

pub mod autograd;
pub mod backend;
pub mod dtype;
pub mod io;
pub mod labels;
#[path = "nn/mod.rs"]
pub mod nn;
#[cfg(feature = "safe_tensors")]
pub use io::safetensors_loader::apply_kronos_bytes_to_module_bytes;
#[cfg(feature = "safe_tensors")]
pub use io::safetensors_loader::load_safetensors_from_bytes;
pub mod ops;
pub mod tensor;
pub mod tokenizer;

#[cfg(feature = "python_bindings")]
use crate::labels::Labels;
#[cfg(feature = "python_bindings")]
use nn::Llama;
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

#[cfg(all(feature = "python_bindings", feature = "native_tokenizer"))]
#[pyclass(name = "Tokenizer")]
#[derive(Clone)]
struct PyNativeTokenizer(crate::tokenizer::Tokenizer);

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

    /// Python slice/index access: support tuple indices/numpy-like slicing and negative indices.
    fn __getitem__(&self, idx: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        use ndarray::{Axis, IxDyn, SliceInfo, SliceInfoElem};
        use pyo3::types::{PySlice, PyTuple};

        let arr = self.0.lock().storage.to_f32_array();
        let ndim = arr.ndim();
        let _py = idx.py();

        // Build list of slice elements
        let mut elems: Vec<SliceInfoElem> = Vec::new();
        // Track axes that were indexed with an integer, so we can remove the axis after slicing
        let mut int_axes: Vec<usize> = Vec::new();

        // Helper to push full slice for remaining dims
        let push_full_slices = |elems: &mut Vec<SliceInfoElem>, start: usize| {
            for _ in start..ndim {
                elems.push((..).into());
            }
        };

        if let Ok(tup) = idx.downcast::<PyTuple>() {
            let items_len = tup.len();
            for (axis, item) in tup.iter().enumerate() {
                if axis >= ndim {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "too many indices for tensor",
                    ));
                }
                if item.is_none() {
                    elems.push((..).into());
                    continue;
                }
                if let Ok(py_slice) = item.downcast::<PySlice>() {
                    // PySlice::indices requires a Python length (c_long) and returns PySliceIndices.
                    // Use the axis length for slice computations (not the number of dimensions)
                    let axis_len = arr.shape()[axis] as isize;
                    let si = py_slice.indices(axis_len.try_into().map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err("invalid axis length")
                    })?)?;
                    let start = si.start as isize;
                    let stop = si.stop as isize;
                    let step = si.step as isize;
                    if step != 1 {
                        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                            "slicing with step != 1 not supported yet",
                        ));
                    }
                    // Convert to usize ranges
                    let s = start;
                    let e = stop;
                    // Clip to 0..ndim
                    let s_u = if s < 0 {
                        (axis_len + s) as usize
                    } else {
                        s as usize
                    };
                    let e_u = if e < 0 {
                        (axis_len + e) as usize
                    } else {
                        e as usize
                    };
                    elems.push((s_u..e_u).into());
                } else if let Ok(i) = item.extract::<isize>() {
                    let mut ii = i;
                    if ii < 0 {
                        ii += arr.shape()[axis] as isize;
                    }
                    if ii < 0 || ii as usize >= arr.shape()[axis] {
                        return Err(pyo3::exceptions::PyIndexError::new_err(
                            "index out of bounds",
                        ));
                    }
                    let start = ii as usize;
                    elems.push((start..start + 1).into());
                    int_axes.push(axis);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "unsupported index type",
                    ));
                }
            }
            if items_len < ndim {
                push_full_slices(&mut elems, items_len);
            }
        } else {
            // Single index or slice
            if let Ok(py_slice) = idx.downcast::<PySlice>() {
                let axis_len = arr.shape()[0] as isize;
                let si = py_slice.indices(axis_len.try_into().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("invalid axis length")
                })?)?;
                let start = si.start as isize;
                let stop = si.stop as isize;
                let step = si.step as isize;
                if step != 1 {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                        "slicing with step != 1 not supported yet",
                    ));
                }
                let s = if start < 0 {
                    (axis_len + start) as usize
                } else {
                    start as usize
                };
                let e = if stop < 0 {
                    (axis_len + stop) as usize
                } else {
                    stop as usize
                };
                elems.push((s..e).into());
                push_full_slices(&mut elems, 1);
            } else if let Ok(i) = idx.extract::<isize>() {
                let mut ii = i;
                if ii < 0 {
                    ii += arr.shape()[0] as isize;
                }
                if ii < 0 || ii as usize >= arr.shape()[0] {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "index out of bounds",
                    ));
                }
                let start = ii as usize;
                elems.push((start..start + 1).into());
                push_full_slices(&mut elems, 1);
                int_axes.push(0);
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "unsupported index type",
                ));
            }
        }

        // Fill any missing dims (should already be filled) but safe check
        let cur_len = elems.len();
        if cur_len < ndim {
            push_full_slices(&mut elems, cur_len);
        }

        // Build SliceInfo
        let slice_info: SliceInfo<_, IxDyn, IxDyn> = unsafe {
            match SliceInfo::new(elems) {
                Ok(s) => s,
                Err(e) => {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "invalid slice: {}",
                        e
                    )))
                }
            }
        };

        // Slice and convert back to a Tensor
        let mut out_arr = arr.slice(slice_info).to_owned().into_dyn();

        // Remove axes for integer indexing to mimic Python semantics
        int_axes.sort();
        for axis in int_axes.iter().rev() {
            out_arr = out_arr.remove_axis(Axis(*axis));
        }

        Ok(PyTensor(crate::tensor::Tensor::new(out_arr, false)))
    }

    /// Python assignment operator: supports assigning a scalar or another Tensor into a slice.
    fn __setitem__(&mut self, idx: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        use ndarray::{IxDyn, SliceInfo, SliceInfoElem};
        use pyo3::types::{PySlice, PyTuple};

        // Build slice_info similarly to __getitem__
        let arr_shape;
        {
            let lock = self.0.lock();
            arr_shape = lock.storage.shape();
        }
        let ndim = arr_shape.len();
        let mut elems: Vec<SliceInfoElem> = Vec::new();
        // parse idx like in __getitem__
        if let Ok(tup) = idx.downcast::<PyTuple>() {
            let items_len = tup.len();
            for (axis, item) in tup.iter().enumerate() {
                if axis >= ndim {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "too many indices for tensor",
                    ));
                }
                if item.is_none() {
                    elems.push((..).into());
                    continue;
                }
                if let Ok(py_slice) = item.downcast::<PySlice>() {
                    let axis_len = arr_shape[axis] as isize;
                    let si = py_slice.indices(axis_len.try_into().map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err("invalid axis length")
                    })?)?;
                    if si.step != 1 {
                        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                            "slicing with step != 1 not supported yet",
                        ));
                    }
                    let s = si.start as isize;
                    let e = si.stop as isize;
                    let s_u = if s < 0 {
                        (axis_len + s) as usize
                    } else {
                        s as usize
                    };
                    let e_u = if e < 0 {
                        (axis_len + e) as usize
                    } else {
                        e as usize
                    };
                    elems.push((s_u..e_u).into());
                } else if let Ok(i) = item.extract::<isize>() {
                    let mut ii = i;
                    if ii < 0 {
                        ii += arr_shape[axis] as isize;
                    }
                    if ii < 0 || ii as usize >= arr_shape[axis] {
                        return Err(pyo3::exceptions::PyIndexError::new_err(
                            "index out of bounds",
                        ));
                    }
                    let start = ii as usize;
                    elems.push((start..start + 1).into());
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "unsupported index type",
                    ));
                }
            }
            if items_len < ndim {
                for _ in items_len..ndim {
                    elems.push((..).into());
                }
            }
        } else {
            // single index or slice
            if let Ok(py_slice) = idx.downcast::<PySlice>() {
                let axis_len = arr_shape[0] as isize;
                let si = py_slice.indices(axis_len.try_into().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("invalid axis length")
                })?)?;
                if si.step != 1 {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                        "slicing with step != 1 not supported yet",
                    ));
                }
                let s = si.start as isize;
                let e = si.stop as isize;
                let s_u = if s < 0 {
                    (axis_len + s) as usize
                } else {
                    s as usize
                };
                let e_u = if e < 0 {
                    (axis_len + e) as usize
                } else {
                    e as usize
                };
                elems.push((s_u..e_u).into());
                for _ in 1..ndim {
                    elems.push((..).into());
                }
            } else if let Ok(i) = idx.extract::<isize>() {
                let mut ii = i;
                if ii < 0 {
                    ii += arr_shape[0] as isize;
                }
                if ii < 0 || ii as usize >= arr_shape[0] {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "index out of bounds",
                    ));
                }
                let start = ii as usize;
                elems.push((start..start + 1).into());
                for _ in 1..ndim {
                    elems.push((..).into());
                }
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "unsupported index type",
                ));
            }
        }

        // Build SliceInfo
        let slice_info: SliceInfo<_, IxDyn, IxDyn> = unsafe {
            match SliceInfo::new(elems) {
                Ok(s) => s,
                Err(e) => {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "invalid slice: {}",
                        e
                    )))
                }
            }
        };

        // Convert RHS to f32 ArrayD
        let rhs_arr = if let Ok(py_tensor) = value.extract::<PyTensor>() {
            py_tensor.0.lock().storage.to_f32_array()
        } else if let Ok(f) = value.extract::<f32>() {
            ndarray::ArrayD::from_elem(IxDyn(&[1usize]), f)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "value must be a Tensor or float",
            ));
        };

        // Assign into the specified slice: if storage is F32, modify in-place, otherwise convert and write back
        let mut lock = self.0.lock();
        match &mut lock.storage {
            crate::dtype::TensorStorage::F32(ref mut arr) => {
                // check shape compatibility
                let view_shape = arr.slice(slice_info.clone()).to_owned().shape().to_vec();
                let view_len: usize = view_shape.iter().product();
                let rhs_len = rhs_arr.len();
                if rhs_arr.shape() != view_shape.as_slice() && rhs_len != 1 && rhs_len != view_len {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "assigned value has incompatible shape",
                    ));
                }
                if rhs_len == 1 {
                    arr.slice_mut(slice_info).fill(rhs_arr[[0]]);
                } else {
                    // If rhs is flattened and matches the view length, reshape to view and assign
                    if rhs_arr.shape() != view_shape.as_slice() {
                        let reshaped = rhs_arr.to_shape(IxDyn(&view_shape)).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "assignment reshape failed: {}",
                                e
                            ))
                        })?;
                        arr.slice_mut(slice_info).assign(&reshaped);
                    } else {
                        arr.slice_mut(slice_info).assign(&rhs_arr);
                    }
                }
            }
            _ => {
                // Convert entire storage to f32, perform assignment, then convert back
                let mut arrf = lock.storage.to_f32_array();
                let view_shape = arrf.slice(slice_info.clone()).to_owned().shape().to_vec();
                let view_len: usize = view_shape.iter().product();
                let rhs_len = rhs_arr.len();
                if rhs_arr.shape() != view_shape.as_slice() && rhs_len != 1 && rhs_len != view_len {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "assigned value has incompatible shape",
                    ));
                }
                if rhs_len == 1 {
                    arrf.slice_mut(slice_info).fill(rhs_arr[[0]]);
                } else {
                    if rhs_arr.shape() != view_shape.as_slice() {
                        let reshaped = rhs_arr.to_shape(IxDyn(&view_shape)).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "assignment reshape failed: {}",
                                e
                            ))
                        })?;
                        arrf.slice_mut(slice_info).assign(&reshaped);
                    } else {
                        arrf.slice_mut(slice_info).assign(&rhs_arr);
                    }
                }
                // Reassign to storage preserving dtype
                lock.storage = crate::dtype::TensorStorage::from_f32_array(&arrf, lock.dtype);
            }
        }
        Ok(())
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

    fn rope(&self, num_heads: usize, theta: Option<f32>) -> PyTensor {
        let t = theta.unwrap_or(10000.0);
        PyTensor(self.0.rope(num_heads, t))
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

    /// Return a (flat_list, shape) tuple for easy numpy construction.
    fn to_numpy_tuple(&self) -> (Vec<f32>, Vec<usize>) {
        (self.get_data(), self.shape())
    }

    /// Return single scalar value; errors if tensor is not scalar.
    fn item(&self) -> PyResult<f32> {
        let data = self.get_data();
        if data.len() != 1 {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "item() called on non-scalar Tensor",
            ))
        } else {
            Ok(data[0])
        }
    }

    /// Move/convert tensor dtype or device. device currently only supports 'cpu'.
    fn to(&self, dtype: Option<&str>, device: Option<&str>) -> PyResult<PyTensor> {
        let mut t = self.clone();
        if let Some(d) = dtype {
            t = t.astype(d)?;
        }
        if let Some(dev) = device {
            if dev != "cpu" {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Only 'cpu' device supported currently",
                ));
            }
        }
        Ok(t)
    }

    /// CPU no-op
    fn cpu(&self) -> PyTensor {
        self.clone()
    }

    /// CUDA not supported yet
    fn cuda(&self) -> PyResult<PyTensor> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "CUDA device not supported in this build",
        ))
    }

    /// Remove axes of length 1. `axis` optional; if provided, remove that axis only.
    fn squeeze(&self, axis: Option<isize>) -> PyResult<PyTensor> {
        let arr = self.0.lock().storage.to_f32_array();
        let mut shape = arr.shape().to_vec();
        if let Some(a) = axis {
            let ndim = shape.len() as isize;
            let mut ax = a;
            if ax < 0 { ax += ndim; }
            if ax < 0 || ax >= ndim { return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("axis out of range")); }
            if shape[ax as usize] != 1 { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("cannot squeeze axis with size != 1")); }
            shape.remove(ax as usize);
        } else {
            shape.retain(|&d| d != 1);
            if shape.is_empty() { shape = vec![1]; }
        }
        let flat = self.get_data();
        let arr2 = ndarray::Array::from_shape_vec(IxDyn(&shape), flat).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("squeeze reshape failed: {}", e)))?;
        Ok(PyTensor(crate::tensor::Tensor::new(arr2.into_dyn(), self.0.lock().requires_grad)))
    }

    /// Insert an axis of size 1 at `axis` (supports negative indexing)
    fn unsqueeze(&self, axis: isize) -> PyResult<PyTensor> {
        let arr = self.0.lock().storage.to_f32_array();
        let ndim = arr.ndim() as isize;
        let mut ax = axis;
        if ax < 0 { ax += ndim + 1; }
        if ax < 0 || ax > ndim { return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("axis out of range")); }
        let mut shape = arr.shape().to_vec();
        shape.insert(ax as usize, 1usize);
        let flat = self.get_data();
        let arr2 = ndarray::Array::from_shape_vec(IxDyn(&shape), flat).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("unsqueeze reshape failed: {}", e)))?;
        Ok(PyTensor(crate::tensor::Tensor::new(arr2.into_dyn(), self.0.lock().requires_grad)))
    }

    /// Alias for reshape/view
    fn view(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        match self.0.reshape(shape) {
            Ok(t) => Ok(PyTensor(t)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        }
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

    /// Matrix multiply (2D matmul) exposed to Python as `a.matmul(b)`.
    fn matmul(&self, other: &PyTensor) -> PyTensor {
        PyTensor(self.0.matmul(&other.0))
    }

    /// Batched matmul exposed to Python as `a.batched_matmul(b)`.
    fn batched_matmul(&self, other: &PyTensor) -> PyTensor {
        PyTensor(self.0.batched_matmul(&other.0))
    }

    /// Return flattened f32 values, shape, and dtype string for inspection.
    fn to_numpy(&self) -> (Vec<f32>, Vec<usize>, String) {
        let arr = self.0.lock().storage.to_f32_array();
        let flat = arr.iter().cloned().collect::<Vec<f32>>();
        let shape = arr.shape().to_vec();
        let dtype = self.0.lock().dtype.as_str().to_string();
        (flat, shape, dtype)
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
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported dtype: {}",
                dtype
            ))),
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

#[cfg(all(feature = "python_bindings", feature = "native_tokenizer"))]
#[pymethods]
impl PyNativeTokenizer {
    /// Load tokenizer from a `tokenizer.json` produced by HuggingFace tokenizers
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        match crate::tokenizer::Tokenizer::from_json(path) {
            Ok(t) => Ok(PyNativeTokenizer(t)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        }
    }

    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        Self::new(path)
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        self.0.encode(text)
    }

    fn decode(&self, ids: Vec<usize>) -> String {
        self.0.decode(&ids)
    }

    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Option<usize> {
        self.0.token_to_id(token)
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
#[derive(Clone)]
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
        self.0
            .named_parameters(prefix)
            .into_iter()
            .map(|(n, t)| (n, PyTensor(t)))
            .collect()
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
        llama_style: Option<bool>,
        llama_bias: Option<bool>,
    ) -> Self {
        let kv = kv_heads.unwrap_or(num_heads);
        let rope = use_rope.unwrap_or(false);
        let llama = llama_style.unwrap_or(false);
        let bias = llama_bias.unwrap_or(true);
        if let Some(cfg) = nl_oob_config {
            let cfg_val = match cfg {
                "logarithmic" | "log" | "0" => crate::nn::BiasFunction::Logarithmic,
                "gaussian" | "1" => crate::nn::BiasFunction::Gaussian,
                _ => crate::nn::BiasFunction::Logarithmic,
            };
            let max_scale = nl_oob_max_scale.unwrap_or(2.0);
            PyTransformerBlock(
                TransformerBlock::new_with_nl_oob(d_model, d_ff, num_heads, cfg_val, max_scale)
                    .expect("create transformer block with nl_oob"),
            )
        } else if llama {
            // Use LLaMA-style block constructor
            PyTransformerBlock(
                TransformerBlock::new_llama_style(d_model, d_ff, num_heads, kv, rope, bias)
                    .expect("create llama style block"),
            )
        } else {
            PyTransformerBlock(
                TransformerBlock::new_with_kv_and_rope(d_model, d_ff, num_heads, kv, rope)
                    .expect("create transformer block with kv and rope"),
            )
        }
    }

    fn forward(&mut self, input: &PyTensor) -> PyTensor {
        PyTensor(self.0.forward_block(&input.0))
    }

    fn forward_with_distance(&self, input: &PyTensor, distance: &PyTensor) -> PyTensor {
        PyTensor(self.0.forward_block_with_distance(&input.0, &distance.0))
    }

    fn debug_forward(&self, input: &PyTensor) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let map = self.0.forward_block_debug(&input.0);
            let dict = PyDict::new_bound(py);
            for (k, v) in map.into_iter() {
                let py_t = Py::new(py, PyTensor(v)).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create PyTensor: {}",
                        e
                    ))
                })?;
                dict.set_item(k, py_t)?;
            }
            Ok(dict.to_object(py))
        })
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.0.parameters().into_iter().map(PyTensor).collect()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, PyTensor)> {
        self.0
            .named_parameters(prefix)
            .into_iter()
            .map(|(n, t)| (n, PyTensor(t)))
            .collect()
    }
}

/// Llama Python wrapper
#[cfg(feature = "python_bindings")]
#[pyclass(name = "Llama")]
struct PyLlama(Llama);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyLlama {
    #[new]
    fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        d_ff: usize,
        num_heads: usize,
        kv_heads: usize,
    ) -> pyo3::PyResult<Self> {
        match Llama::new(vocab_size, d_model, num_layers, d_ff, num_heads, kv_heads) {
            Ok(l) => Ok(PyLlama(l)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyTensor {
        PyTensor(self.0.forward(&input.0))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.0.parameters().into_iter().map(PyTensor).collect()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, PyTensor)> {
        self.0
            .named_parameters(prefix)
            .into_iter()
            .map(|(n, t)| (n, PyTensor(t)))
            .collect()
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

    fn clip_gradients(&mut self, parameters: Vec<PyTensor>, max_norm: f32) {
        let rust_params: Vec<Tensor> = parameters.into_iter().map(|p| p.0).collect();
        self.0.clip_gradients(&rust_params, max_norm);
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

    fn clip_gradients(&mut self, parameters: Vec<PyTensor>, max_norm: f32) {
        let rust_params: Vec<Tensor> = parameters.into_iter().map(|p| p.0).collect();
        self.0.clip_gradients(&rust_params, max_norm);
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

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        match crate::io::tokenizers::decode_tokens(&self.0, &ids) {
            Ok(s) => Ok(s),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }

    fn vocab_size(&self) -> usize {
        self.0.get_vocab_size(true)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id).map(|s| s.to_string())
    }
}

#[cfg(all(feature = "python_bindings", feature = "vision"))]
#[pyclass(name = "ImageTextDataLoader")]
struct PyImageTextDataLoader(crate::io::image_text_dataloader::ImageTextDataLoader);

#[cfg(all(feature = "python_bindings", feature = "audio"))]
#[pyclass(name = "AudioEncoder")]
struct PyAudioEncoder {
    inner: Option<crate::nn::AudioEncoder>,
}

#[cfg(all(feature = "python_bindings", feature = "audio"))]
#[pymethods]
impl PyAudioEncoder {
    #[new]
    fn new(in_channels: usize, hidden: usize, layers: usize) -> Self {
        PyAudioEncoder {
            inner: Some(crate::nn::AudioEncoder::new(in_channels, hidden, layers)),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let enc = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("AudioEncoder has been moved into a model")
        })?;
        Ok(PyTensor(enc.forward(&input.0)))
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let enc = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("AudioEncoder has been moved into a model")
        })?;
        Ok(enc.parameters().into_iter().map(PyTensor).collect())
    }
}

#[cfg(all(feature = "python_bindings", feature = "vision"))]
#[pymethods]
impl PyImageTextDataLoader {
    #[new]
    fn new(
        manifest_path: &str,
        image_w: u32,
        image_h: u32,
        batch_size: usize,
        shuffle: bool,
        augment: bool,
        parallel: bool,
    ) -> PyResult<Self> {
        match crate::io::image_text_dataloader::ImageTextDataLoader::new_from_manifest(
            manifest_path,
            (image_w, image_h),
            batch_size,
            shuffle,
            augment,
            parallel,
        ) {
            Ok(l) => Ok(PyImageTextDataLoader(l)),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(e)),
        }
    }

    fn num_batches(&self) -> usize {
        self.0.num_batches()
    }

    fn load_batch(&self, batch_idx: usize) -> PyResult<(Vec<PyTensor>, Vec<String>)> {
        match self.0.load_batch(batch_idx) {
            Ok((images, captions)) => {
                let py_images: Vec<PyTensor> = images.into_iter().map(PyTensor).collect();
                Ok((py_images, captions))
            }
            Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(e)),
        }
    }

    fn shuffle_in_place(&mut self) {
        self.0.shuffle_in_place();
    }

    #[cfg(all(
        feature = "python_bindings",
        feature = "vision",
        feature = "with_tokenizers"
    ))]
    fn load_batch_tokenized(
        &self,
        batch_idx: usize,
        tokenizer: &PyTokenizer,
    ) -> PyResult<(Vec<PyTensor>, Vec<Vec<u32>>)> {
        match self.0.load_batch_tokenized(batch_idx, &tokenizer.0) {
            Ok((images, tokenized)) => {
                let py_images: Vec<PyTensor> = images.into_iter().map(PyTensor).collect();
                Ok((py_images, tokenized))
            }
            Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(e)),
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
    m.add_class::<PyLlama>()?;
    m.add_class::<PyVisionTransformer>()?;
    m.add_class::<PyMultimodalLLM>()?;
    #[cfg(feature = "python_bindings")]
    m.add_class::<PyModalMemoryContext>()?;
    #[cfg(all(feature = "python_bindings", feature = "with_tokenizers"))]
    m.add_class::<PyTokenizer>()?;
    #[cfg(all(feature = "python_bindings", feature = "vision"))]
    m.add_class::<PyImageTextDataLoader>()?;
    #[cfg(all(feature = "python_bindings", feature = "audio"))]
    m.add_class::<PyAudioEncoder>()?;
    #[cfg(all(feature = "python_bindings", feature = "safe_tensors"))]
    m.add_function(pyo3::wrap_pyfunction!(py_load_safetensors, m)?)?;
    #[cfg(all(feature = "python_bindings", feature = "safe_tensors"))]
    m.add_function(pyo3::wrap_pyfunction!(py_load_safetensors_into_module, m)?)?;

    // Expose helper functions for parity testing (no torch needed): py_matmul, py_batched_matmul
    #[pyfunction]
    fn py_matmul(py: Python<'_>, a: PyObject, b: PyObject) -> PyResult<PyTensor> {
        let at: PyTensor = a.extract(py).map_err(|e| pyo3::exceptions::PyTypeError::new_err(format!("py_matmul: expected Tensor objects: {}", e)))?;
        let bt: PyTensor = b.extract(py).map_err(|e| pyo3::exceptions::PyTypeError::new_err(format!("py_matmul: expected Tensor objects: {}", e)))?;
        Ok(PyTensor(at.0.matmul(&bt.0)))
    }
    m.add_function(pyo3::wrap_pyfunction!(py_matmul, m)?)?;

    #[pyfunction]
    fn py_batched_matmul(py: Python<'_>, a: PyObject, b: PyObject) -> PyResult<PyTensor> {
        let at: PyTensor = a.extract(py).map_err(|e| pyo3::exceptions::PyTypeError::new_err(format!("py_batched_matmul: expected Tensor objects: {}", e)))?;
        let bt: PyTensor = b.extract(py).map_err(|e| pyo3::exceptions::PyTypeError::new_err(format!("py_batched_matmul: expected Tensor objects: {}", e)))?;
        Ok(PyTensor(at.0.batched_matmul(&bt.0)))
    }
    m.add_function(pyo3::wrap_pyfunction!(py_batched_matmul, m)?)?;

    // FlashAttentionRef helper: run the FlashAttentionRef op on three tensors (q,k,v) with given head_dim
    #[pyfunction]
    fn py_flash_attention_ref(py: Python<'_>, q: PyObject, k: PyObject, v: PyObject, head_dim: usize) -> PyResult<PyTensor> {
        let qt: PyTensor = q.extract(py).map_err(|e| pyo3::exceptions::PyTypeError::new_err(format!("py_flash_attention_ref: expected Tensor objects: {}", e)))?;
        let kt: PyTensor = k.extract(py).map_err(|e| pyo3::exceptions::PyTypeError::new_err(format!("py_flash_attention_ref: expected Tensor objects: {}", e)))?;
        let vt: PyTensor = v.extract(py).map_err(|e| pyo3::exceptions::PyTypeError::new_err(format!("py_flash_attention_ref: expected Tensor objects: {}", e)))?;
        // Build op and run via Tensor::apply
        let op: std::sync::Arc<dyn crate::ops::Operation + Send + Sync> = std::sync::Arc::new(crate::ops::FlashAttentionRef::new(head_dim));
        let out = crate::tensor::Tensor::apply(op, &[qt.0.clone(), kt.0.clone(), vt.0.clone()]);
        Ok(PyTensor(out))
    }
    m.add_function(pyo3::wrap_pyfunction!(py_flash_attention_ref, m)?)?;

    // Helper: extract a PyTensor's flattened f32 values, shape, and dtype string for external parity checks
    #[pyfunction]
    fn py_tensor_to_flat(py: Python<'_>, py_tensor: PyObject) -> PyResult<(Vec<f32>, Vec<usize>, String)> {
        // Attempt to extract the PyTensor wrapper and then return its f32 data and shape
        let pt: PyTensor = py_tensor.extract(py).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("py_tensor_to_flat: expected Tensor object: {}", e))
        })?;
        let arr = pt.0.lock().storage.to_f32_array();
        let flat = arr.iter().cloned().collect::<Vec<f32>>();
        let shape = arr.shape().to_vec();
        let dtype = pt.0.lock().dtype.as_str().to_string();
        Ok((flat, shape, dtype))
    }
    m.add_function(pyo3::wrap_pyfunction!(py_tensor_to_flat, m)?)?;
    #[cfg(feature = "python_bindings")]
    m.add_function(pyo3::wrap_pyfunction!(py_set_cpu_backend, m)?)?;
    #[cfg(feature = "python_bindings")]
    m.add_function(pyo3::wrap_pyfunction!(py_set_cuda_backend, m)?)?;
    Ok(())
}

#[cfg(feature = "python_bindings")]
#[pyfunction(name = "set_cuda_backend")]
fn py_set_cuda_backend() -> PyResult<()> {
    match crate::backend::set_cuda_backend() {
        Ok(_) => Ok(()),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

#[cfg(all(feature = "python_bindings", feature = "safe_tensors"))]
#[pyfunction]
fn py_load_safetensors(py: Python<'_>, bytes: Vec<u8>, transpose: bool) -> PyResult<PyObject> {
    let state = crate::io::safetensors_loader::load_safetensors_from_bytes(&bytes, transpose)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let dict = PyDict::new_bound(py);
    for (k, t) in state.into_iter() {
        let py_tensor = Py::new(py, PyTensor(t)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create PyTensor: {}", e))
        })?;
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
    log::debug!(
        "py_load_safetensors_into_module: module type = {}",
        type_name
    );
    // Handle both "TransformerBlock" and "builtins.TransformerBlock" (PyO3 behavior varies)
    if type_name == "TransformerBlock" || type_name.ends_with(".TransformerBlock") {
        // pyo3::PyTryFrom is deprecated, prefer using extract directly on PyObject
        // Borrow the TransformerBlock mutably from Python, then call loader
        log::debug!(
            "py_load_safetensors_into_module: about to extract module as PyTransformerBlock"
        );
        // Ensure we don't hold any Tensor locks while extracting the Python ref
        let mut py_ref: pyo3::PyRefMut<PyTransformerBlock> = module.extract(py).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Invalid module type: {}", e))
        })?;
        log::debug!("py_load_safetensors_into_module: successfully extracted PyTransformerBlock");
        // Now we can apply the state dict to the inner module
        log::debug!("py_load_safetensors_into_module: about to apply state dict to module");
        let res =
            crate::io::safetensors_loader::apply_state_dict_to_module(&mut py_ref.0, &state, root);
        log::debug!("py_load_safetensors_into_module: apply_state_dict_to_module returned");
        res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    } else if type_name == "VisionTransformer" || type_name.ends_with(".VisionTransformer") {
        log::debug!(
            "py_load_safetensors_into_module: about to extract module as PyVisionTransformer"
        );
        let mut py_ref: pyo3::PyRefMut<PyVisionTransformer> = module.extract(py).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Invalid module type: {}", e))
        })?;
        log::debug!("py_load_safetensors_into_module: successfully extracted PyVisionTransformer");
        let res =
            crate::io::safetensors_loader::apply_state_dict_to_module(&mut py_ref.0, &state, root);
        res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    } else if type_name == "MultimodalLLM" || type_name.ends_with(".MultimodalLLM") {
        log::debug!("py_load_safetensors_into_module: about to extract module as PyMultimodalLLM");
        let mut py_ref: pyo3::PyRefMut<PyMultimodalLLM> = module.extract(py).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Invalid module type: {}", e))
        })?;
        log::debug!("py_load_safetensors_into_module: successfully extracted PyMultimodalLLM");
        let res =
            crate::io::safetensors_loader::apply_state_dict_to_module(&mut py_ref.0, &state, root);
        res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    } else if type_name == "Llama" || type_name == "builtins.Llama" {
        log::debug!("py_load_safetensors_into_module: about to extract module as PyLlama");
        let mut py_ref: pyo3::PyRefMut<PyLlama> = module.extract(py).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Invalid module type: {}", e))
        })?;
        log::debug!("py_load_safetensors_into_module: successfully extracted PyLlama");
        let res =
            crate::io::safetensors_loader::apply_state_dict_to_module(&mut py_ref.0, &state, root);
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
#[derive(Clone)]
struct PyVisionTransformer(nn::VisionTransformer);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyVisionTransformer {
    #[new]
    fn new(
        c: usize,
        patch_size: usize,
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        depth: usize,
        max_len: usize,
    ) -> PyResult<Self> {
        match nn::VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, max_len) {
            Ok(v) => Ok(PyVisionTransformer(v)),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyTensor {
        PyTensor(self.0.forward(&input.0))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.0.parameters().into_iter().map(PyTensor).collect()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, PyTensor)> {
        self.0
            .named_parameters(prefix)
            .into_iter()
            .map(|(n, t)| (n, PyTensor(t)))
            .collect()
    }
}

// Multimodal LLM Python wrapper
#[cfg(feature = "python_bindings")]
#[pyclass(name = "MultimodalLLM", unsendable)]
struct PyMultimodalLLM(nn::MultimodalLLM);

#[cfg(feature = "python_bindings")]
#[pyclass(name = "ModalMemoryContext")]
struct PyModalMemoryContext(nn::ModalMemoryContext);

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyMultimodalLLM {
    #[new]
    fn new(
        vision: PyVisionTransformer,
        vocab_size: usize,
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        depth: usize,
    ) -> PyResult<Self> {
        match nn::MultimodalLLM::new(vision.0, vocab_size, d_model, d_ff, num_heads, depth) {
            Ok(m) => Ok(PyMultimodalLLM(m)),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }

    #[staticmethod]
    fn from_config(py_config: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Accept a Python dict or a string path to a JSON file or JSON-like mapping
        let dict = py_config.downcast::<pyo3::types::PyDict>();
        let py_get_usize = |d: &Bound<'_, PyDict>, key: &str, default: usize| -> usize {
            match d.get_item(key) {
                Ok(Some(v)) => v.extract::<usize>().unwrap_or(default),
                Ok(None) => default,
                Err(_) => default,
            }
        };
        let (c, patch_size, d_model, d_ff, num_heads, depth, max_len, vocab_size) = if let Ok(d) =
            dict
        {
            (
                py_get_usize(d, "c", 3),
                py_get_usize(d, "patch_size", 16),
                py_get_usize(d, "d_model", 768),
                py_get_usize(d, "d_ff", 768 * 4),
                py_get_usize(d, "num_heads", 12),
                py_get_usize(d, "depth", 12),
                py_get_usize(d, "max_len", 1024),
                py_get_usize(d, "vocab_size", 50000),
            )
        } else if let Ok(path) = py_config.extract::<&str>() {
            let json_str = std::fs::read_to_string(path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to read config file: {}", e))
            })?;
            Python::with_gil(
                |py| -> PyResult<(usize, usize, usize, usize, usize, usize, usize, usize)> {
                    let py_dict = py
                        .import_bound("json")
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "Failed to import python json module: {}",
                                e
                            ))
                        })?
                        .call_method1("loads", (json_str,))
                        .map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Failed parse json: {}",
                                e
                            ))
                        })?;
                    let d = py_dict.downcast::<pyo3::types::PyDict>()?;
                    Ok((
                        (match d.get_item("c") {
                            Ok(Some(v)) => v.extract::<usize>().unwrap_or(3),
                            Ok(None) => 3,
                            Err(_) => 3,
                        }),
                        (match d.get_item("patch_size") {
                            Ok(Some(v)) => v.extract::<usize>().unwrap_or(16),
                            Ok(None) => 16,
                            Err(_) => 16,
                        }),
                        (match d.get_item("d_model") {
                            Ok(Some(v)) => v.extract::<usize>().unwrap_or(768),
                            Ok(None) => 768,
                            Err(_) => 768,
                        }),
                        (match d.get_item("d_ff") {
                            Ok(Some(v)) => v.extract::<usize>().unwrap_or(768 * 4),
                            Ok(None) => 768 * 4,
                            Err(_) => 768 * 4,
                        }),
                        (match d.get_item("num_heads") {
                            Ok(Some(v)) => v.extract::<usize>().unwrap_or(12),
                            Ok(None) => 12,
                            Err(_) => 12,
                        }),
                        (match d.get_item("depth") {
                            Ok(Some(v)) => v.extract::<usize>().unwrap_or(12),
                            Ok(None) => 12,
                            Err(_) => 12,
                        }),
                        (match d.get_item("max_len") {
                            Ok(Some(v)) => v.extract::<usize>().unwrap_or(1024),
                            Ok(None) => 1024,
                            Err(_) => 1024,
                        }),
                        (match d.get_item("vocab_size") {
                            Ok(Some(v)) => v.extract::<usize>().unwrap_or(50000),
                            Ok(None) => 50000,
                            Err(_) => 50000,
                        }),
                    ))
                },
            )?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "from_config expects a dict or path string",
            ));
        };

        let vision =
            nn::VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, max_len)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let mm = nn::MultimodalLLM::new(vision, vocab_size, d_model, d_ff, num_heads, depth)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(PyMultimodalLLM(mm))
    }

    fn forward(&mut self, images: &PyTensor, input_ids: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor(nn::MultimodalLLM::forward(&mut self.0, &images.0, &input_ids.0)))
    }

    /// Prefill images and an optional text prefix into a memory context for decoding.
    ///
    /// Args:
    ///     images: Image tensor [B, C, H, W]
    ///     input_ids: Optional prefix token ids tensor [B, seq]
    ///
    /// Returns:
    ///     A `ModalMemoryContext` containing cached image/text hidden states that can be passed to `decode_step`.
    fn prefill(
        &mut self,
        images: &PyTensor,
        input_ids: Option<&PyTensor>,
    ) -> PyResult<PyModalMemoryContext> {
        let ids_ref = input_ids.map(|p| &p.0);
        match self.0.prefill(&images.0, ids_ref) {
            Ok(mem) => Ok(PyModalMemoryContext(mem)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// Decode step: append `new_input_ids` to memory context and return logits and an updated memory context.
    ///
    /// This method appends new token ids to the supplied memory context and returns logits for the new sequence,
    /// together with an updated `ModalMemoryContext` that contains the expanded cache.
    fn decode_step(
        &mut self,
        memory: &PyModalMemoryContext,
        new_input_ids: &PyTensor,
    ) -> PyResult<(PyTensor, PyModalMemoryContext)> {
        match self.0.decode_step(&memory.0, &new_input_ids.0) {
            Ok((logits, new_mem)) => Ok((PyTensor(logits), PyModalMemoryContext(new_mem))),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// Compute logits for the current memory context without appending tokens.
    ///
    /// Useful for calculating the next token probability after a `prefill` call.
    fn logits_from_memory(&self, memory: &PyModalMemoryContext) -> PyResult<PyTensor> {
        match self.0.logits_from_memory(&memory.0) {
            Ok(logits) => Ok(PyTensor(logits)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// Return top candidate tokens and normalized probabilities for the next step, after applying
    /// temperature, top_k and top_p truncation. Useful for testing and deterministic inspection.
    fn sample_candidates_from_memory(
        &self,
        memory: &PyModalMemoryContext,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> PyResult<Vec<(usize, f32)>> {
        match self
            .0
            .sample_candidates(&memory.0, temperature, top_k, top_p)
        {
            Ok(cands) => Ok(cands),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// Attach a linear projector module to the model. Accepts a pre-constructed `Linear` instance.
    fn set_projector_linear(&mut self, linear: PyLinear) {
        self.0.set_projector_linear(linear.0);
    }

    /// Set an MLP projector with a hidden dimension.
    fn set_projector_mlp(&mut self, hidden_dim: usize) {
        self.0.set_projector_mlp(hidden_dim);
    }

    /// Attach an audio encoder.
    #[cfg(feature = "audio")]
    fn set_audio_encoder(&mut self, encoder: &mut PyAudioEncoder) -> PyResult<()> {
        let enc = encoder.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "AudioEncoder has already been moved into a model",
            )
        })?;
        self.0.set_audio_encoder(enc);
        Ok(())
    }

    /// Generate tokens with sampling or beam search.
    #[pyo3(signature = (images, prefix = None, max_len = 512, temperature = 1.0, top_k = None, top_p = None, beam_size = 1)
    )]
    fn generate(
        &mut self,
        images: &PyTensor,
        prefix: Option<&PyTensor>,
        max_len: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        beam_size: usize,
    ) -> PyResult<Vec<usize>> {
        let prefix_ref = prefix.map(|p| &p.0);
        match self.0.generate(
            &images.0,
            prefix_ref,
            max_len,
            temperature,
            top_k,
            top_p,
            beam_size,
        ) {
            Ok(seq) => Ok(seq),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// Batched generation API. If beam_size > 1 uses batched beam search; otherwise performs sampling for each batch item.
    #[pyo3(signature = (images, prefix = None, max_len = 512, temperature = 1.0, top_k = None, top_p = None, beam_size = 1, length_penalty = 1.0, eos_token = None)
    )]
    fn generate_batch(
        &mut self,
        images: &PyTensor,
        prefix: Option<&PyTensor>,
        max_len: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        beam_size: usize,
        length_penalty: f32,
        eos_token: Option<usize>,
    ) -> PyResult<Vec<Vec<usize>>> {
        let prefix_ref = prefix.map(|p| &p.0);
        match self.0.generate_batch(
            &images.0,
            prefix_ref,
            max_len,
            temperature,
            top_k,
            top_p,
            beam_size,
            length_penalty,
            eos_token,
        ) {
            Ok(seq) => Ok(seq),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// Beam search for a given ModalMemoryContext with options for length_penalty and EOS.
    fn beam_search_with_options(
        &mut self,
        memory: &PyModalMemoryContext,
        max_len: usize,
        beam_size: usize,
        length_penalty: f32,
        eos_token: Option<usize>,
    ) -> PyResult<Vec<usize>> {
        match self.0.beam_search_with_options(
            &memory.0,
            max_len,
            beam_size,
            length_penalty,
            eos_token,
        ) {
            Ok(seq) => Ok(seq),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
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
        self.0
            .named_parameters(prefix)
            .into_iter()
            .map(|(n, t)| (n, PyTensor(t)))
            .collect()
    }

    /// Export the module parameters to a SafeTensors bytes object (Python `bytes`).
    fn save_state_dict<'py>(&self, _py: Python<'py>) -> PyResult<&'py pyo3::types::PyBytes> {
        #[cfg(feature = "safe_tensors")]
        {
            match crate::io::safetensors_loader::save_module_to_safetensors_bytes(&self.0) {
                Ok(bytes) => {
                    #[allow(deprecated)]
                    {
                        Ok(pyo3::types::PyBytes::new(_py, &bytes))
                    }
                }
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
            }
        }
        #[cfg(not(feature = "safe_tensors"))]
        {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "safetensors feature not enabled",
            ))
        }
    }

    /// Save module parameters to a file path in SafeTensors format.
    fn save_state_dict_to_path(&self, _path: &str) -> PyResult<()> {
        #[cfg(feature = "safe_tensors")]
        {
            match crate::io::safetensors_loader::save_module_to_safetensors_bytes(&self.0) {
                Ok(bytes) => {
                    std::fs::write(_path, bytes).map_err(|e| {
                        pyo3::exceptions::PyIOError::new_err(format!("Failed to write file: {}", e))
                    })?;
                    Ok(())
                }
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
            }
        }
        #[cfg(not(feature = "safe_tensors"))]
        {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "safetensors feature not enabled",
            ))
        }
    }

    /// Load state dict bytes (SafeTensors or Kronos) into this module.
    fn load_state_dict(
        &mut self,
        _py: Python<'_>,
        bytes: Vec<u8>,
        transpose: bool,
        root: Option<&str>,
    ) -> PyResult<()> {
        let root_s = root.unwrap_or("");
        // Silence unused variable warnings if safe_tensors feature isn't enabled
        #[cfg(not(feature = "safe_tensors"))]
        {
            let _ = &bytes;
            let _ = &transpose;
            let _ = &root_s;
        }
        #[cfg(feature = "safe_tensors")]
        {
            match crate::io::safetensors_loader::apply_safetensors_bytes_to_module_bytes(
                &mut self.0,
                &bytes,
                transpose,
                root_s,
            ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
            }
        }
        #[cfg(not(feature = "safe_tensors"))]
        {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "safetensors feature not enabled",
            ))
        }
    }

    /// Load module parameters from the given SafeTensors file path.
    fn load_state_dict_from_path(
        &mut self,
        path: &str,
        transpose: bool,
        root: Option<&str>,
    ) -> PyResult<()> {
        #[cfg(not(feature = "safe_tensors"))]
        let _ = path;
        #[cfg(feature = "safe_tensors")]
        let bytes = std::fs::read(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to read file: {}", e))
        })?;
        let root_s = root.unwrap_or("");
        #[cfg(not(feature = "safe_tensors"))]
        {
            let _ = &transpose;
            let _ = &root_s;
        }
        #[cfg(feature = "safe_tensors")]
        {
            match crate::io::safetensors_loader::apply_safetensors_bytes_to_module_bytes(
                &mut self.0,
                &bytes,
                transpose,
                root_s,
            ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
            }
        }
        #[cfg(not(feature = "safe_tensors"))]
        {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "safetensors feature not enabled",
            ))
        }
    }

    /// Cast the module parameters to a specified dtype string (e.g., 'f16'). Uses dtype features if enabled.
    fn cast_params(&mut self, dtype: &str) -> PyResult<()> {
        if let Some(dt) = crate::dtype::DType::parse(dtype) {
            let params = self.parameters();
            for p in params {
                let converted = p.astype(dt.as_str())?;
                let mut lock = p.0.lock();
                let converted_lock = converted.0.lock();
                lock.storage = converted_lock.storage.clone();
                lock.dtype = dt;
            }
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown dtype: {}",
                dtype
            )))
        }
    }
}

#[cfg(feature = "python_bindings")]
#[pymethods]
impl PyModalMemoryContext {
    fn encoding(&self) -> PyResult<PyTensor> {
        Ok(PyTensor(self.0.encoding.clone()))
    }
    fn prefill_image_tokens(&self) -> PyResult<usize> {
        Ok(self.0.prefill_image_tokens)
    }
    fn modality(&self) -> PyResult<String> {
        Ok(self.0.modality.clone())
    }
}
