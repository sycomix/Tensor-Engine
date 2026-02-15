use crate::dtype::{DType, TensorStorage};

use crate::ops::{
    Add, BinaryCrossEntropy, BinaryCrossEntropyWithLogits, Concat, CrossEntropyLogits, Div,
    EmbeddingLookup, KVCacheAppend, LayerNorm, Log, LogSoftmax, MatMul, Mean, Mul, NLLLoss,
    Operation, PermuteAxes, Pow, RMSNorm, ReLU, RoPE, Sigmoid, Softmax, SoftmaxCrossEntropyLogits,
    Stack, Sub, Sum, SwiGLU, Tanh, TopK,
};
use ndarray::{ArrayD, IxDyn};
use std::sync::{Arc, Mutex, MutexGuard};

/// `TensorData` contains the actual data of a tensor, along with metadata for automatic differentiation.
pub struct TensorData {
    /// The tensor's data, stored as a dynamically-dimensioned array.
    pub storage: TensorStorage,
    /// The gradient of the tensor, if it has one.
    pub grad: Option<ArrayD<f32>>,
    /// The operation that created this tensor, if any.
    pub creator: Option<Arc<dyn Operation + Send + Sync>>,
    /// The input tensors that were used to create this tensor.
    pub inputs: Vec<Tensor>,
    /// Whether this tensor requires a gradient.
    pub requires_grad: bool,
    /// Data type indicator for storage/representation purposes (MVP: data stays f32 but dtype captures intended storage semantics)
    pub dtype: DType,
}

/// A multi-dimensional array (tensor) that supports automatic differentiation.
///
/// Tensors are the fundamental data structure in `tensor_engine`. They can be created from scratch
/// or as the result of operations on other tensors. If a tensor is created from operations on other
/// tensors that have `requires_grad = true`, then it will also have `requires_grad = true` and will
/// be part of a computation graph.
#[derive(Clone)]
pub struct Tensor(Arc<Mutex<TensorData>>);

impl Tensor {
    /// Creates a new tensor.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor's data.
    /// * `requires_grad` - Whether this tensor should have a gradient.
    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> Self {
        Tensor(Arc::new(Mutex::new(TensorData {
            storage: TensorStorage::from_f32_array(&data, DType::F32),
            grad: None,
            creator: None,
            inputs: vec![],
            requires_grad,
            dtype: DType::F32,
        })))
    }

    /// Creates a new tensor of ones with the given shape.
    pub fn ones(shape: &[usize]) -> Self {
        Self::new(ArrayD::ones(ndarray::IxDyn(shape)), true)
    }

    /// Creates a new tensor of zeros with the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        Self::new(ArrayD::zeros(ndarray::IxDyn(shape)), true)
    }

    /// Creates a new tensor using memory from a pool.
    ///
    /// This is more efficient for frequently allocated/deallocated tensors
    /// as it reuses memory instead of going to the system allocator.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor's data.
    /// * `requires_grad` - Whether this tensor should have a gradient.
    /// * `pool` - The memory pool to use for allocation.
    pub fn new_pooled(
        data: ArrayD<f32>,
        requires_grad: bool,
        pool: &crate::memory_pool::TensorPool,
    ) -> Self {
        // Allocate buffer from pool for potential future use
        // For now, integrate by pre-warming the pool with tensor-sized allocations
        let elem_count = data.len();
        let byte_size = elem_count * std::mem::size_of::<f32>();

        // Pre-allocate a pooled buffer - this helps warm the cache for future similar allocations
        let _pooled_buf = pool.allocate(byte_size);

        // Create the tensor normally - the pool will be used for future operations
        log::debug!(
            "Created pooled tensor with {} bytes ({} f32 elements)",
            byte_size,
            elem_count
        );

        Tensor(Arc::new(Mutex::new(TensorData {
            storage: TensorStorage::from_f32_array(&data, DType::F32),
            grad: None,
            creator: None,
            inputs: vec![],
            requires_grad,
            dtype: DType::F32,
        })))
    }

    /// Creates a new tensor of zeros using memory from a pool.
    ///
    /// This is more efficient for frequently allocated/deallocated tensors.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `pool` - The memory pool to use for allocation.
    pub fn zeros_pooled(shape: &[usize], pool: &crate::memory_pool::TensorPool) -> Self {
        let elem_count: usize = shape.iter().product();
        let byte_size = elem_count * std::mem::size_of::<f32>();

        // Pre-allocate from pool to warm cache
        let _pooled_buf = pool.allocate_zeroed(byte_size);

        log::debug!("Created pooled zero tensor with shape {:?}", shape);
        Self::new(ArrayD::zeros(ndarray::IxDyn(shape)), true)
    }

    /// Creates a new tensor of ones using memory from a pool.
    ///
    /// This is more efficient for frequently allocated/deallocated tensors.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `pool` - The memory pool to use for allocation.
    pub fn ones_pooled(shape: &[usize], pool: &crate::memory_pool::TensorPool) -> Self {
        let elem_count: usize = shape.iter().product();
        let byte_size = elem_count * std::mem::size_of::<f32>();

        // Pre-allocate from pool to warm cache
        let _pooled_buf = pool.allocate(byte_size);

        log::debug!("Created pooled ones tensor with shape {:?}", shape);
        Self::new(ArrayD::ones(ndarray::IxDyn(shape)), true)
    }

    /// Create a new tensor with an explicit dtype. For MVP, this will store the dtype but the underlying
    /// data remains `ArrayD<f32>`. We perform a round-trip conversion for non-f32 types to emulate reduced precision.
    pub fn new_with_dtype(data: ArrayD<f32>, requires_grad: bool, dtype: DType) -> Self {
        log::info!(
            "Creating tensor with dtype {} (MVP: storage remains f32)",
            dtype
        );
        let t = Tensor::new(data.clone(), requires_grad);
        if dtype != DType::F32 {
            // Perform a round-trip conversion: f32 -> (f16/bf16/f8) -> f32 to emulate precision loss.
            let converted = match dtype {
                DType::F32 => t.lock().storage.to_f32_array(),
                DType::F16 => {
                    #[cfg(feature = "dtype_f16")]
                    {
                        let f16arr =
                            crate::dtype::f16_helpers::to_f16(&t.lock().storage.to_f32_array());
                        crate::dtype::f16_helpers::from_f16(&f16arr)
                    }
                    #[cfg(not(feature = "dtype_f16"))]
                    {
                        // If the feature isn't enabled, fallback to no-op but mark dtype
                        t.lock().storage.to_f32_array()
                    }
                }
                DType::BF16 => {
                    #[cfg(feature = "dtype_bf16")]
                    {
                        let bf =
                            crate::dtype::f16_helpers::to_bf16(&t.lock().storage.to_f32_array());
                        crate::dtype::f16_helpers::from_bf16(&bf)
                    }
                    #[cfg(not(feature = "dtype_bf16"))]
                    {
                        t.lock().storage.to_f32_array()
                    }
                }
                DType::F8 => {
                    // Emulate f8 quantization
                    let arr = t.lock().storage.to_f32_array();
                    let (q, scale) = crate::dtype::f8::quantize_to_f8(&arr);
                    crate::dtype::f8::dequantize_from_f8(&q, scale, arr.shape())
                }
                DType::I8 => {
                    let arr = t.lock().storage.to_f32_array();
                    let (q, scale) = crate::dtype::int8::quantize_to_i8(&arr);
                    crate::dtype::int8::dequantize_from_i8(&q, scale, arr.shape())
                }
                DType::I8Rowwise => {
                    let arr = t.lock().storage.to_f32_array();
                    let converted = match crate::dtype::int8::quantize_rowwise_to_i8(&arr) {
                        Ok((q, scales)) => {
                            crate::dtype::int8::dequantize_from_i8_rowwise(&q, &scales, arr.shape())
                        }
                        Err(e) => {
                            log::error!("astype I8Rowwise quantization failed: {}", e);
                            arr.clone()
                        }
                    };
                    converted
                }
                DType::I8Blockwise => {
                    let arr = t.lock().storage.to_f32_array();
                    let block_size = 32usize; // default block size
                    let converted =
                        match crate::dtype::int8::quantize_blockwise_to_i8(&arr, block_size) {
                            Ok((q, scales)) => crate::dtype::int8::dequantize_from_i8_blockwise(
                                &q,
                                &scales,
                                arr.shape(),
                                block_size,
                            ),
                            Err(e) => {
                                log::error!("astype I8Blockwise quantization failed: {}", e);
                                arr.clone()
                            }
                        };
                    converted
                }
                DType::U8 => {
                    let arr = t.lock().storage.to_f32_array();
                    arr.mapv(|x| x as u8 as f32)
                }
            };
            let mut lock = t.lock();
            lock.storage = TensorStorage::from_f32_array(&converted, dtype);
            lock.dtype = dtype;
        }
        log::debug!("new_with_dtype: dtype set to {:?}", dtype);
        t
    }

    /// Applies an operation to a set of input tensors.
    ///
    /// This is the primary way that computation graphs are constructed.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to apply.
    /// * `inputs` - The input tensors.
    pub fn apply(op: Arc<dyn Operation + Send + Sync>, inputs: &[Tensor]) -> Tensor {
        let requires_grad = inputs.iter().any(|t| t.lock().requires_grad);
        // Determine output shape, supporting broadcasting for element-wise ops.
        let out_shape: Vec<usize> = if op.as_any().is::<Sum>() || op.as_any().is::<Mean>() {
            vec![] // scalar
        } else if op.as_any().is::<crate::ops::Concat>() || op.as_any().is::<crate::ops::Stack>() {
            // Concat/Stack manage their own shapes in ops implementations; default to first input
            inputs[0].lock().storage.shape().to_vec()
        } else {
            // Generic element-wise broadcast across inputs
            fn broadcast_shape_from(shapes: &[Vec<usize>]) -> Result<Vec<usize>, String> {
                let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
                let mut result = vec![1usize; max_ndim];
                for s in shapes {
                    for (i, &dim) in s.iter().rev().enumerate() {
                        let ridx = max_ndim - 1 - i;
                        let cur = result[ridx];
                        if cur == 1 {
                            result[ridx] = dim;
                        } else if dim == 1 {
                            /* keep cur */
                        } else if cur == dim {
                            /* ok */
                        } else {
                            return Err(format!("Cannot broadcast shapes: {:?}", shapes));
                        }
                    }
                }
                Ok(result)
            }

            let shapes: Vec<Vec<usize>> = inputs
                .iter()
                .map(|t| t.lock().storage.shape().to_vec())
                .collect();
            match broadcast_shape_from(&shapes) {
                Ok(s) => s,
                Err(_e) => inputs[0].lock().storage.shape().to_vec(),
            }
        };

        let mut data = ArrayD::zeros(IxDyn(&out_shape[..]));
        op.forward(inputs, &mut data);

        Tensor(Arc::new(Mutex::new(TensorData {
            storage: TensorStorage::from_f32_array(&data, DType::F32),
            grad: None,
            creator: Some(op),
            inputs: inputs.to_vec(),
            requires_grad,
            dtype: DType::F32,
        })))
    }

    /// Apply a quantized matmul operation: left operand is f32, right operand is int8/quantized Tensor.
    pub fn quantized_matmul(&self, qweight: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(crate::ops::QuantizedMatMul::new()),
            &[self.clone(), qweight.clone()][..],
        )
    }

    /// Quantize weights (2D tensor) into the specified dtype storage format.
    /// Supports DType::I8, DType::I8Rowwise, and DType::I8Blockwise.
    pub fn quantize_weights(
        &self,
        dtype: DType,
        block_size: Option<usize>,
    ) -> Result<Tensor, String> {
        let arr = self.lock().storage.to_f32_array();
        if arr.ndim() != 2 {
            return Err("quantize_weights expects a 2D matrix".to_string());
        }
        match dtype {
            DType::I8 => {
                let (bytes, scale) = crate::dtype::int8::quantize_to_i8(&arr);
                let td = Tensor(Arc::new(Mutex::new(TensorData {
                    storage: crate::dtype::TensorStorage::I8(bytes, scale, arr.shape().to_vec()),
                    grad: None,
                    creator: None,
                    inputs: vec![],
                    requires_grad: self.lock().requires_grad,
                    dtype: DType::I8,
                })));
                Ok(td)
            }
            DType::I8Rowwise => {
                let (bytes, scales) = crate::dtype::int8::quantize_rowwise_to_i8(&arr)?;
                let td = Tensor(Arc::new(Mutex::new(TensorData {
                    storage: crate::dtype::TensorStorage::I8Rowwise(
                        bytes,
                        scales,
                        arr.shape().to_vec(),
                    ),
                    grad: None,
                    creator: None,
                    inputs: vec![],
                    requires_grad: self.lock().requires_grad,
                    dtype: DType::I8Rowwise,
                })));
                Ok(td)
            }
            DType::I8Blockwise => {
                let block = block_size.unwrap_or(32usize);
                let (bytes, scales) = crate::dtype::int8::quantize_blockwise_to_i8(&arr, block)?;
                let td = Tensor(Arc::new(Mutex::new(TensorData {
                    storage: crate::dtype::TensorStorage::I8Blockwise(
                        bytes,
                        scales,
                        arr.shape().to_vec(),
                        block,
                    ),
                    grad: None,
                    creator: None,
                    inputs: vec![],
                    requires_grad: self.lock().requires_grad,
                    dtype: DType::I8Blockwise,
                })));
                Ok(td)
            }
            _ => {
                // For other dtypes, fallback to new_with_dtype round-trip conversion
                let t = self.clone();
                let arr = t.lock().storage.to_f32_array();
                t.lock().storage = crate::dtype::TensorStorage::from_f32_array(&arr, dtype);
                t.lock().dtype = dtype;
                Ok(t)
            }
        }
    }

    /// Validate that this tensor represents a well-formed 2D quantized weight matrix.
    ///
    /// This is an ergonomic safety check for inference and benchmarks.
    /// Supported storages: `I8`, `I8Rowwise`, `I8Blockwise`.
    pub fn validate_quantized_weights_2d(&self) -> Result<(), String> {
        let guard = self.lock();
        let _qm = guard.storage.try_as_quantized_matrix_2d()?;
        Ok(())
    }

    /// Public helper: compute broadcasted shape from a slice of shapes (Vec<usize>). Returns Err on incompatible shapes.
    pub fn broadcast_shapes(shapes: &[Vec<usize>]) -> Result<Vec<usize>, String> {
        let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut result = vec![1usize; max_ndim];
        for s in shapes {
            for (i, &dim) in s.iter().rev().enumerate() {
                let ridx = max_ndim - 1 - i;
                let cur = result[ridx];
                if cur == 1 {
                    result[ridx] = dim;
                } else if dim == 1 {
                    // keep cur
                } else if cur == dim {
                    // ok
                } else {
                    return Err(format!("Cannot broadcast shapes: {:?}", shapes));
                }
            }
        }
        Ok(result)
    }

    /// Return a new `Tensor` with the desired dtype. This performs a round-trip conversion for
    /// non-f32 types to emulate precision loss while keeping in-memory data as f32 (MVP behavior).
    pub fn astype(&self, dtype: DType) -> Tensor {
        log::debug!("astype called: {:?} -> {:?}", self.lock().dtype, dtype);
        let (data, req_grad) = {
            let lock = self.lock();
            (lock.storage.to_f32_array(), lock.requires_grad)
        };
        Tensor::new_with_dtype(data, req_grad, dtype)
    }

    /// Adds two tensors.
    pub fn add(&self, other: &Tensor) -> Tensor {
        Tensor::apply(Arc::new(Add), &[self.clone(), other.clone()][..])
    }

    /// Multiplies two tensors.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        Tensor::apply(Arc::new(Mul), &[self.clone(), other.clone()][..])
    }

    /// Subtracts two tensors.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        Tensor::apply(Arc::new(Sub), &[self.clone(), other.clone()][..])
    }

    /// Divides two tensors.
    pub fn div(&self, other: &Tensor) -> Tensor {
        Tensor::apply(Arc::new(Div), &[self.clone(), other.clone()][..])
    }

    /// Performs matrix multiplication.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        Tensor::apply(Arc::new(MatMul), &[self.clone(), other.clone()][..])
    }

    /// Batched matrix multiplication: a [batch,m,k] @ b [batch,k,n] -> out [batch,m,n]
    pub fn batched_matmul(&self, other: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(crate::ops::BatchedMatMul::new()),
            &[self.clone(), other.clone()][..],
        )
    }

    /// Raises a tensor to a power.
    pub fn pow(&self, power: f32) -> Tensor {
        Tensor::apply(Arc::new(Pow(power)), std::slice::from_ref(self))
    }

    /// Element-wise natural exponent e^x
    pub fn exp(&self) -> Tensor {
        Tensor::apply(Arc::new(crate::ops::Exp), std::slice::from_ref(self))
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Tensor {
        // Use imported `Log` symbol (avoids unused-import warnings for `Log` in module imports)
        Tensor::apply(Arc::new(Log), std::slice::from_ref(self))
    }

    /// RMSNorm: input x and scale gamma
    pub fn rmsnorm(&self, gamma: &Tensor, axis: usize, eps: f32) -> Tensor {
        Tensor::apply(
            Arc::new(RMSNorm::new(axis, eps)),
            &[self.clone(), gamma.clone()][..],
        )
    }

    /// SwiGLU: split last axis into two and apply SwiGLU activation
    pub fn swiglu(&self) -> Tensor {
        Tensor::apply(Arc::new(SwiGLU::new()), std::slice::from_ref(self))
    }

    /// Embedding lookup: Embedding matrix (vocab, dim) + indices -> gathered Embedding
    pub fn embedding_lookup(emb: &Tensor, indices: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(EmbeddingLookup::new()),
            &[emb.clone(), indices.clone()][..],
        )
    }

    /// Elementwise equality comparison. Returns tensor of 0.0/1.0 floats.
    pub fn equal(&self, other: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(crate::ops::Equal),
            &[self.clone(), other.clone()][..],
        )
    }

    /// Elementwise greater-than comparison. Returns tensor of 0.0/1.0 floats.
    pub fn greater(&self, other: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(crate::ops::Greater),
            &[self.clone(), other.clone()][..],
        )
    }

    /// Elementwise less-than comparison. Returns tensor of 0.0/1.0 floats.
    pub fn less(&self, other: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(crate::ops::Less),
            &[self.clone(), other.clone()][..],
        )
    }

    /// KvCache append: concat cache and new_kv along axis
    pub fn kvcache_append(cache: &Tensor, new_kv: &Tensor, axis: usize) -> Tensor {
        Tensor::apply(
            Arc::new(KVCacheAppend::new(axis)),
            &[cache.clone(), new_kv.clone()][..],
        )
    }

    /// Negates the tensor (multiply by -1 scalar).
    pub fn neg(&self) -> Tensor {
        let scalar = Tensor::new(ArrayD::from_elem(IxDyn(&[][..]), -1.0), false);
        Tensor::apply(Arc::new(Mul), &[self.clone(), scalar][..])
    }

    /// Applies the ReLU activation function.
    pub fn relu(&self) -> Tensor {
        Tensor::apply(Arc::new(ReLU), std::slice::from_ref(self))
    }

    /// Applies the ternary quantization operation (project to -1/0/1 with STE)
    pub fn ternary(&self) -> Tensor {
        Tensor::apply(Arc::new(crate::ops::Ternary), std::slice::from_ref(self))
    }

    /// Applies the sigmoid activation function.
    pub fn sigmoid(&self) -> Tensor {
        Tensor::apply(Arc::new(Sigmoid), std::slice::from_ref(self))
    }

    /// Applies the tanh activation function.
    pub fn tanh(&self) -> Tensor {
        Tensor::apply(Arc::new(Tanh), std::slice::from_ref(self))
    }

    /// GELU activation function (Gaussian Error Linear Unit)
    pub fn gelu(&self) -> Tensor {
        Tensor::apply(Arc::new(crate::ops::GELU), std::slice::from_ref(self))
    }

    /// Applies the SiLU (Swish) activation function.
    pub fn silu(&self) -> Tensor {
        Tensor::apply(Arc::new(crate::ops::SiLU), std::slice::from_ref(self))
    }

    /// Computes the sum of the tensor's elements.
    pub fn sum(&self) -> Tensor {
        Tensor::apply(Arc::new(Sum), std::slice::from_ref(self))
    }

    /// Computes the mean of the tensor's elements.
    pub fn mean(&self) -> Tensor {
        Tensor::apply(Arc::new(Mean), std::slice::from_ref(self))
    }

    /// Computes the maximum value of the tensor's elements.
    pub fn max(&self) -> Tensor {
        Tensor::apply(Arc::new(crate::ops::Max), std::slice::from_ref(self))
    }

    /// Computes the minimum value of the tensor's elements.
    pub fn min(&self) -> Tensor {
        Tensor::apply(Arc::new(crate::ops::Min), std::slice::from_ref(self))
    }

    /// Element-wise softmax along the specified axis (default last axis)
    pub fn softmax(&self, axis: usize) -> Tensor {
        Tensor::apply(Arc::new(Softmax::new(axis)), std::slice::from_ref(self))
    }

    /// Stable log-softmax along the specified axis
    pub fn log_softmax(&self, axis: usize) -> Tensor {
        Tensor::apply(Arc::new(LogSoftmax::new(axis)), std::slice::from_ref(self))
    }

    /// Upsample a 4D NCHW tensor using nearest-neighbor upsampling by integer scale.
    pub fn upsample_nearest2d(&self, scale: usize) -> Tensor {
        Tensor::apply(
            Arc::new(crate::ops::UpSampleNearest2D::new(scale)),
            std::slice::from_ref(self),
        )
    }

    /// Cross-entropy with logits (logits + targets), targets may be a vector of indices (float ints) or one-hot vectors.
    /// `axis` may be negative to index from the right (e.g., -1). Pass axis as signed integer.
    pub fn cross_entropy_with_logits(&self, target: &Tensor, axis: isize) -> Tensor {
        let ndim = self.lock().storage.shape().to_vec().len() as isize;
        let axis_norm = if axis < 0 {
            (ndim + axis) as usize
        } else {
            axis as usize
        };
        Tensor::apply(
            Arc::new(CrossEntropyLogits::new(axis_norm)),
            &[self.clone(), target.clone()][..],
        )
    }

    /// TopK: selects the largest k elements along the last dimension.
    /// Returns a tensor of shape [..., 2*k] where the first half of the last dim are values
    /// and the second half are indices (as floats).
    pub fn topk(&self, k: usize) -> Tensor {
        Tensor::apply(Arc::new(TopK::new(k)), std::slice::from_ref(self))
    }

    /// Combined softmax and cross-entropy for logits to avoid extra allocations.
    /// `axis` may be negative to index from the right (e.g., -1).
    pub fn softmax_cross_entropy_with_logits(&self, target: &Tensor, axis: isize) -> Tensor {
        let ndim = self.lock().storage.shape().to_vec().len() as isize;
        let axis_norm = if axis < 0 {
            (ndim + axis) as usize
        } else {
            axis as usize
        };
        Tensor::apply(
            Arc::new(SoftmaxCrossEntropyLogits::new(axis_norm)),
            &[self.clone(), target.clone()][..],
        )
    }

    /// NLLLoss expects log-probabilities (log_softmax output) and integer label indices (as floats) or one-hot.
    pub fn nll_loss(&self, target: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(NLLLoss::new()),
            &[self.clone(), target.clone()][..],
        )
    }

    /// Binary Cross Entropy (element-wise).
    /// Inputs: self (probabilities), target (0..1).
    pub fn binary_cross_entropy(&self, target: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(BinaryCrossEntropy::new()),
            &[self.clone(), target.clone()][..],
        )
    }

    /// Binary Cross Entropy with Logits (element-wise).
    /// Inputs: self (logits), target (0..1).
    /// Numerically stable.
    pub fn binary_cross_entropy_with_logits(&self, target: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(BinaryCrossEntropyWithLogits::new()),
            &[self.clone(), target.clone()][..],
        )
    }

    /// Layer normalization along axis with learnable gamma and beta tensors.
    pub fn layer_norm(&self, axis: usize, eps: f32, gamma: &Tensor, beta: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(LayerNorm::new(axis, eps)),
            &[self.clone(), gamma.clone(), beta.clone()][..],
        )
    }
}

/// Configuration for batch normalization parameters.
#[derive(Clone, Debug)]
pub struct BatchNormConfig {
    pub momentum: f32,
    pub eps: f32,
    pub training: bool,
}

impl Default for BatchNormConfig {
    fn default() -> Self {
        Self {
            momentum: 0.1,
            eps: 1e-5,
            training: true,
        }
    }
}

impl Tensor {
    /// Batch normalization over the mini-batch (assumes [B, C, ...] format).
    pub fn batch_norm(
        &self,
        gamma: &Tensor,
        beta: &Tensor,
        running_mean: &Tensor,
        running_var: &Tensor,
        config: BatchNormConfig,
    ) -> Tensor {
        let momentum = config.momentum;
        let eps = config.eps;
        let training = config.training;
        Tensor::apply(
            Arc::new(crate::ops::BatchNorm::new(momentum, eps, training)),
            &[
                self.clone(),
                gamma.clone(),
                beta.clone(),
                running_mean.clone(),
                running_var.clone(),
            ][..],
        )
    }

    /// Batch normalization with individual parameters (for backward compatibility).
    pub fn batch_norm_with_params(
        &self,
        gamma: &Tensor,
        beta: &Tensor,
        running_mean: &Tensor,
        running_var: &Tensor,
        momentum: f32,
        eps: f32,
        training: bool,
    ) -> Tensor {
        let config = BatchNormConfig {
            momentum,
            eps,
            training,
        };
        self.batch_norm(gamma, beta, running_mean, running_var, config)
    }

    /// Reshapes the tensor.
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor, String> {
        // Validate target shape first to produce same error semantics
        let lock = self.lock();
        let data_clone = lock.storage.to_f32_array();
        let _requires_grad = lock.requires_grad;
        drop(lock);
        // Validate shape
        match data_clone.to_shape(shape.clone()) {
            Ok(_) => Ok(Tensor::apply(
                Arc::new(crate::ops::Reshape::new(shape)),
                std::slice::from_ref(self),
            )),
            Err(e) => Err(format!(
                "Cannot reshape tensor from {:?} to {:?}: {}",
                data_clone.shape(),
                shape,
                e
            )),
        }
    }

    /// Transposes the tensor.
    pub fn transpose(&self) -> Tensor {
        let lock = self.lock();
        let data = lock.storage.to_f32_array().reversed_axes();
        let requires_grad = lock.requires_grad;
        drop(lock);
        Tensor::new(data, requires_grad)
    }

    /// Permute axes of the tensor by a permutation vector
    pub fn permute(&self, perm: Vec<usize>) -> Tensor {
        Tensor::apply(Arc::new(PermuteAxes::new(perm)), std::slice::from_ref(self))
    }

    /// Apply rotary positional embeddings (RoPE) along the last axis, splitting the last axis into `num_heads` heads.
    /// `theta` controls the base frequency (LLaMA uses large theta like 500000.0).
    pub fn rope(&self, num_heads: usize, theta: f32, scale: f32, offset: usize) -> Tensor {
        Tensor::apply(
            Arc::new(RoPE::new(num_heads, theta, scale, offset)),
            std::slice::from_ref(self),
        )
    }

    /// Concatenates a list of tensors along a given axis.
    pub fn concat(tensors: &[Tensor], axis: usize) -> Tensor {
        Tensor::apply(Arc::new(Concat(axis)), tensors)
    }

    /// Stacks a list of tensors along a new axis.
    pub fn stack(tensors: &[Tensor], axis: usize) -> Tensor {
        Tensor::apply(Arc::new(Stack(axis)), tensors)
    }

    /// Locks the tensor's data for reading or writing.
    pub fn lock(&self) -> MutexGuard<'_, TensorData> {
        match self.0.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Returns a copy of the underlying data as an `ArrayD<f32>`, converting if needed.
    /// For MVP this is a helpful abstraction since internal storage remains f32.
    pub fn to_f32_array(&self) -> ArrayD<f32> {
        self.lock().storage.to_f32_array()
    }

    /// Returns the tensor's storage dtype
    pub fn dtype(&self) -> DType {
        self.lock().dtype
    }

    /// Sets the gradient of this tensor to zero.
    pub fn zero_grad(&self) {
        let mut lock = self.lock();
        lock.grad = None;
    }

    /// Detaches the tensor from the computation graph.
    pub fn detach(&self) -> Tensor {
        let lock = self.lock();
        Tensor::new_with_dtype(lock.storage.to_f32_array(), false, lock.dtype)
    }

    /// Returns whether this tensor requires gradients.
    pub fn requires_grad(&self) -> bool {
        self.lock().requires_grad
    }

    /// Sets whether this tensor requires gradients.
    pub fn set_requires_grad(&self, requires_grad: bool) {
        let mut lock = self.lock();
        lock.requires_grad = requires_grad;
    }

    /// Performs backpropagation starting from this tensor.
    ///
    /// This will compute the gradients of all tensors in the computation graph that have
    /// `requires_grad = true`.
    pub fn backward(&self) {
        crate::autograd::AutogradEngine::new().backward(self);
    }

    /// Builds a topological sort of the computation graph.
    pub fn build_topo(
        &self,
        visited: &mut std::collections::HashSet<*const Mutex<TensorData>>,
        topo_order: &mut Vec<Tensor>,
    ) {
        let ptr = Arc::as_ptr(&self.0);
        if !visited.contains(&ptr) {
            visited.insert(ptr);
            for input in &self.lock().inputs {
                input.build_topo(visited, topo_order);
            }
            topo_order.push(self.clone());
        }
    }
}

// Implement Deref to allow treating Tensor like Arc<Mutex<TensorData>>
use std::ops::Deref;

impl Deref for Tensor {
    type Target = Arc<Mutex<TensorData>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Implement PartialEq for Tensors
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

// Implement Eq for Tensors
impl Eq for Tensor {}

// Implement Hash for Tensors
use std::hash::{Hash, Hasher};

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_build_topo_simple_chain() {
        // a -> b -> c
        let a = Tensor::new_with_dtype(ArrayD::from_elem(IxDyn(&[1][..]), 1.0), false, DType::F32);
        let b = Tensor::new_with_dtype(ArrayD::from_elem(IxDyn(&[1][..]), 2.0), false, DType::F32);
        let c = Tensor::new_with_dtype(ArrayD::from_elem(IxDyn(&[1][..]), 3.0), false, DType::F32);

        // set dependencies
        b.lock().inputs = vec![a.clone()];
        c.lock().inputs = vec![b.clone()];

        let mut visited: HashSet<*const Mutex<TensorData>> = HashSet::new();
        let mut topo: Vec<Tensor> = Vec::new();
        c.build_topo(&mut visited, &mut topo);
        // topo should contain a,b,c in that order
        let ids: Vec<*const Mutex<TensorData>> = topo.iter().map(|t| Arc::as_ptr(&t.0)).collect();
        assert_eq!(ids.len(), 3);
        assert_eq!(Arc::as_ptr(&topo[0].0), Arc::as_ptr(&a.0));
        assert_eq!(Arc::as_ptr(&topo[1].0), Arc::as_ptr(&b.0));
        assert_eq!(Arc::as_ptr(&topo[2].0), Arc::as_ptr(&c.0));
    }
}
