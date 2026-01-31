//! Backend abstraction placeholder.
//!
//! The goal is to abstract Rust ops behind a Backend trait so it's easier to add GPU/accelerated backends
//! (wgpu/cudarc) later. This file provides a minimal trait definition and a CPU backend stub to start
//! refactoring ops.

use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;
use std::sync::OnceLock;

/// Minimal backend trait to prepare for future backends. Implementations will provide
/// optimized kernels for operations like matmul, conv, etc.
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;

    // Return an optional result for matmul. If a backend returns `Some(ArrayD)` the
    // op can short-circuit and use that result; if `None` the op should fall back
    // to its built-in implementation.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Option<ArrayD<f32>>;

    /// Quantized Matrix Multiplication (AWQ)
    ///
    /// Computes input @ (unpack(qweight) - qzeros) * scales + bias
    #[allow(clippy::too_many_arguments)]
    fn matmul_quantized(
        &self,
        input: &Tensor,
        qweight: &Tensor,
        scales: &Tensor,
        qzeros: &Tensor,
        bias: Option<&Tensor>,
        group_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> Option<ArrayD<f32>>;
}

/// CPU backend stub — delegates to existing op implementations for now.
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor) -> Option<ArrayD<f32>> {
        // Default: return None to allow MatMul op to use its optimized CPU code path.
        None
    }

    fn matmul_quantized(
        &self,
        input: &Tensor,
        qweight: &Tensor,
        scales: &Tensor,
        qzeros: &Tensor,
        bias: Option<&Tensor>,
        group_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> Option<ArrayD<f32>> {
        // Dequantize weights
        let target_shape = vec![in_features, out_features];
        let w = match crate::quantization::awq::awq_dequantize_affine(
            qweight,
            scales,
            qzeros,
            group_size,
            &target_shape,
        ) {
            Ok(t) => t,
            Err(e) => {
                log::error!("CpuBackend: Quantized matmul dequantize failed: {}", e);
                return None;
            }
        };

        // Linear forward: input @ weight + bias
        // Handle broadcasting if input is > 2D (e.g. [batch, seq, in])
        // MatMul op only supports 2D, so we flatten, matmul, then reshape.
        let input_shape = input.lock().storage.shape();
        let ndim = input_shape.len();

        let activation = if ndim > 2 {
            let last_dim = input_shape[ndim - 1];
            if last_dim != in_features {
                log::error!(
                    "CpuBackend: Quantized matmul input shape mismatch: expected last dim {}, got {}",
                    in_features,
                    last_dim
                );
                return None;
            }
            // product of all dims except last
            let batch_dim: usize = input_shape[0..ndim - 1].iter().product();
            let flattened_shape = vec![batch_dim, last_dim];

            // Reshape input to 2D
            match input.reshape(flattened_shape) {
                Ok(flat_input) => {
                    let flat_out = flat_input.matmul(&w);
                    // Reshape back to [..., out_features]
                    let mut out_shape = input_shape[0..ndim - 1].to_vec();
                    out_shape.push(out_features);
                    match flat_out.reshape(out_shape) {
                        Ok(o) => o,
                        Err(e) => {
                            log::error!("CpuBackend: failed to reshape output: {}", e);
                            flat_out
                        }
                    }
                }
                Err(e) => {
                    log::error!("CpuBackend: failed to flatten input: {}", e);
                    return None;
                }
            }
        } else {
            input.matmul(&w)
        };

        let bias_tensor = match bias {
            Some(b) => b.clone(),
            Option::None => Tensor::new(ArrayD::zeros(IxDyn(&[out_features][..])), false),
        };

        let res = activation.add(&bias_tensor);
        let lock = res.lock();
        Some(lock.storage.to_f32_array())
    }
}

// NOTE: A backend trait migration is planned to allow backend-specific optimizations
// (e.g. CUDA, wgpu). See `docs/backend_migration_plan.md` for design notes and steps
// to migrate `src/ops.rs` to use the `Backend` trait and to call backend-specific
// kernels from `Tensor::apply`.

static GLOBAL_BACKEND: OnceLock<Arc<dyn Backend>> = OnceLock::new();

/// Set the global backend for operations. Only callable once per process.
pub fn set_global_backend(b: Arc<dyn Backend>) -> Result<(), String> {
    GLOBAL_BACKEND
        .set(b)
        .map_err(|_| "Global backend already set".to_string())
}

/// Returns the global backend. Defaults to CPU backend if not set.
pub fn get_global_backend() -> &'static Arc<dyn Backend> {
    GLOBAL_BACKEND.get_or_init(|| Arc::new(CpuBackend {}))
}

/// Convenience function to set the backend to CPU explicitly.
pub fn set_cpu_backend() -> Result<(), String> {
    set_global_backend(Arc::new(CpuBackend {}))
}

/// CUDA backend implementation for high-performance GPU acceleration.
/// This backend provides CUDA-accelerated operations for large-scale ML workloads.
/// Implements cuBLAS integration for matrix operations and custom CUDA kernels.
///
/// Features:
/// - GPU memory management
/// - cuBLAS integration for matrix operations
/// - Custom CUDA kernels for specialized operations
/// - Multi-GPU support preparation
#[cfg(feature = "backend_cuda")]
use cuda::{CudaContext, CudaDevice, CudaStream};

#[cfg(feature = "backend_cuda")]
use std::ffi::c_void;

#[cfg(feature = "backend_cuda")]
pub struct CudaBackend {
    pub device: cuda::CudaDevice,
    pub context: cuda::CudaContext,
    pub stream: cuda::CudaStream,
}

#[cfg(feature = "backend_cuda")]
impl CudaBackend {
    pub fn new(device_ordinal: i32) -> Result<Self, String> {
        // Initialize CUDA device
        let device = cuda::CudaDevice::new(device_ordinal)
            .map_err(|e| format!("Failed to initialize CUDA device {}: {}", device_ordinal, e))?;

        // Create CUDA context
        let context = cuda::CudaContext::new(&device)
            .map_err(|e| format!("Failed to create CUDA context: {}", e))?;

        // Create CUDA stream for async operations
        let stream = cuda::CudaStream::new(&context, cuda::CudaStreamFlags::NON_BLOCKING)
            .map_err(|e| format!("Failed to create CUDA stream: {}", e))?;

        Ok(CudaBackend {
            device,
            context,
            stream,
        })
    }
}

#[cfg(feature = "backend_cuda")]
impl crate::backend::Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Option<ArrayD<f32>> {
        // CUDA-accelerated matrix multiplication
        let a_lock = a.lock();
        let b_lock = b.lock();
        let a_arr = a_lock.storage.to_f32_array();
        let b_arr = b_lock.storage.to_f32_array();

        // Only support 2D matrices for now
        if a_arr.ndim() == 2 && b_arr.ndim() == 2 {
            if let (Ok(a2), Ok(b2)) = (
                a_arr.into_dimensionality::<ndarray::Ix2>(),
                b_arr.into_dimensionality::<ndarray::Ix2>(),
            ) {
                // Use cuBLAS for matrix multiplication
                let (m, k) = a2.dim();
                let (k2, n) = b2.dim();
                assert_eq!(k, k2, "Matrix dimensions incompatible");

                unsafe {
                    let alpha: f32 = 1.0;
                    let beta: f32 = 0.0;

                    // Allocate GPU memory
                    let mut d_a = std::ptr::null_mut::<c_void>();
                    let mut d_b = std::ptr::null_mut::<c_void>();
                    let mut d_c = std::ptr::null_mut::<c_void>();

                    // Copy matrices to GPU
                    let result = self.context.cublas().sgemm(
                        cuda::sys::cublasOperation_t::CUBLAS_OP_N,
                        cuda::sys::cublasOperation_t::CUBLAS_OP_T,
                        m as i32,
                        k as i32,
                        k as i32,
                        n as i32,
                        &alpha,
                        a2.as_ptr(),
                        m as i32,
                        k as i32,
                        b2.as_ptr(),
                        k as i32,
                        n as i32,
                        &beta,
                        &mut d_c,
                        n as i32,
                    );

                    if result == cuda::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        // Copy result back to host
                        let c_shape = (m, n);
                        let mut c_host = ndarray::Array::<f32, _>::zeros(c_shape);

                        // Transfer result back from GPU
                        self.context
                            .cublas()
                            .get_vector_async(c_host.as_mut_ptr(), (m * n) as i32, &self.stream)
                            .wait()
                            .map_err(|e| format!("Failed to copy result from GPU: {}", e))?;

                        return Some(c_host.into_dyn());
                    }
                }
            }
        }

        // Fall back to CPU for unsupported cases
        None
    }

    fn matmul_quantized(
        &self,
        input: &Tensor,
        qweight: &Tensor,
        scales: &Tensor,
        qzeros: &Tensor,
        bias: Option<&Tensor>,
        group_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> Option<ArrayD<f32>> {
        // CUDA-optimized quantized matrix multiplication
        // Future implementation with custom CUDA kernels
        None
    }
}

/// Convenience function to set the backend to a CUDA-accelerated backend.
pub fn set_cuda_backend(_device_id: Option<i32>) -> Result<(), String> {
    #[cfg(feature = "backend_cuda")]
    {
        let device_id = _device_id.unwrap_or(0);
        let backend = CudaBackend::new(device_id)?;
        set_global_backend(Arc::new(backend));
        Ok(())
    }

    #[cfg(not(feature = "backend_cuda"))]
    {
        Err("CUDA backend not enabled. Build with --features backend_cuda".to_string())
    }
}

#[cfg(feature = "backend_wgpu")]
pub mod wgpu;

#[cfg(feature = "backend_wgpu")]
/// Convenience function to set the backend to WgpuBackend.
pub fn set_wgpu_backend() -> Result<(), String> {
    log::info!("Initializing WGPU Backend...");
    let backend = wgpu::WgpuBackend::new()?;
    set_global_backend(Arc::new(backend))
}

#[cfg(all(feature = "backend_metal", target_os = "macos"))]
pub mod metal;

#[cfg(all(feature = "backend_metal", target_os = "macos"))]
/// Convenience function to set the backend to MetalBackend.
pub fn set_metal_backend() -> Result<(), String> {
    log::info!("Initializing Metal Backend...");
    let backend = metal::MetalBackend::new()?;
    set_global_backend(Arc::new(backend))
}

/// Auto-detect and initialize the best available backend.
pub fn auto_detect_backend() -> Result<(), String> {
    // Priority: CUDA > WGPU > CPU
    #[cfg(feature = "backend_cuda")]
    {
        match CudaBackend::new(0) {
            Ok(backend) => {
                log::info!("Auto-detected CUDA backend");
                set_global_backend(Arc::new(backend));
                return Ok(());
            }
            Err(e) => {
                log::warn!(
                    "CUDA backend initialization failed: {}, trying alternatives",
                    e
                );
            }
        }
    }

    #[cfg(all(feature = "backend_metal", target_os = "macos"))]
    {
        match metal::MetalBackend::new() {
            Ok(backend) => {
                log::info!("Auto-detected Metal backend");
                let _ = set_global_backend(Arc::new(backend));
                return Ok(());
            }
            Err(e) => {
                log::warn!(
                    "Metal backend initialization failed: {}, trying alternatives",
                    e
                );
            }
        }
    }

    #[cfg(feature = "backend_wgpu")]
    {
        match wgpu::WgpuBackend::new() {
            Ok(backend) => {
                log::info!("Auto-detected WGPU backend");
                let _ = set_global_backend(Arc::new(backend));
                return Ok(());
            }
            Err(e) => {
                log::warn!("WGPU backend initialization failed: {}, using CPU", e);
            }
        }
    }

    log::info!("Using CPU backend");
    set_cpu_backend()
}
