//! CUDA kernel launchers for Tensor Engine.
//!
//! This module provides functions to load and execute PTX kernels.

use cudarc::driver::CudaContext;
use cudarc::nvrtc;
use cudarc::driver::PushKernelArg;
use std::sync::Arc;

/// PTX source for matrix multiplication kernel (optimized version).
pub const MATMUL_PTX: &str = include_str!("cuda_kernels/optimized.ptx");

/// PTX source for element-wise addition kernel.
pub const ADD_PTX: &str = r#"
.version 8.0
.target sm_80
.address_bits 64

.extern "C" add_kernel(void* A, void* B, void* C, int N, void* OUT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float* a = (float*)A;
        float* b = (float*)B;
        float* out = (float*)OUT;
        out[idx] = a[idx] + b[idx];
    }
}
"#;

/// PTX source for ReLU activation kernel.
pub const RELU_PTX: &str = r#"
.version 8.0
.target sm_80
.address_bits 64

.extern "C" relu_kernel(void* X, void* OUT, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float* x = (float*)X;
        float* out = (float*)OUT;
        out[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}
"#;

/// PTX source for softmax kernel.
pub const SOFTMAX_PTX: &str = r#"
.version 8.0
.target sm_80
.address_bits 64

.extern "C" softmax_kernel(void* X, void* OUT, int ROWS, int COLS) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < ROWS) {
        float* x = (float*)X + row * COLS;
        float* out = (float*)OUT + row * COLS;
        
        float max_val = x[0];
        for (int i = 1; i < COLS; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
        
        float sum = 0.0f;
        for (int i = 0; i < COLS; i++) {
            out[i] = exp(x[i] - max_val);
            sum += out[i];
        }
        
        if (sum > 0.0f) {
            for (int i = 0; i < COLS; i++) {
                out[i] /= sum;
            }
        }
    }
}
"#;

/// Compile PTX source and load it into a module on the given CUDA context.
fn compile_and_load(ctx: &Arc<CudaContext>, ptx_src: &str) -> Result<Arc<cudarc::driver::CudaModule>, String> {
    let ptx = nvrtc::compile_ptx(ptx_src).map_err(|e| format!("NVPTX compile failed: {:?}", e))?;
    ctx.load_module(ptx).map_err(|e| format!("Module load failed: {:?}", e))
}

/// Launch matrix multiplication kernel (2D).
pub fn matmul_2d(
    ctx: &Arc<CudaContext>,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    output: &mut [f32],
) -> Result<(), String> {
    let module = compile_and_load(ctx, MATMUL_PTX)?;
    
    // load_function takes &self: &Arc<Self>, so we can pass &module directly
    let func = module.load_function("matmul_kernel")
        .map_err(|e| format!("Get func failed: {:?}", e))?;

    let stream = ctx.default_stream();
    
    // Launch configuration
    let threads_per_block = 256u32;
    let total_elements = (m * n) as u64;
    let num_blocks = ((total_elements + (threads_per_block as u64) - 1) / (threads_per_block as u64)) as u32;
    
    unsafe {
        // Convert pointers to usize for kernel arguments
        let a_ptr = a.as_ptr() as usize;
        let b_ptr = b.as_ptr() as usize;
        let out_ptr = output.as_mut_ptr() as usize;
        let m_val = m as i32;
        let n_val = n as i32;
        let k_val = k as i32;
        
        stream.launch_builder(&func)
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&out_ptr)
            .arg(&m_val)
            .arg(&n_val)
            .arg(&k_val)
            .arg(&out_ptr)
            .arg(&0usize) // shared_mem as usize (null pointer = 0)
            .launch(cudarc::driver::LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: 0u32,
            })
            .map_err(|e| format!("Kernel launch failed: {:?}", e))?;
    }

    stream.synchronize().map_err(|e| format!("Stream sync failed: {:?}", e))?;
    Ok(())
}

/// Launch matrix multiplication kernel (3D batched).
pub fn matmul_3d(
    ctx: &Arc<CudaContext>,
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    output: &mut [f32],
) -> Result<(), String> {
    for i in 0..batch {
        let a_start = i * m * k;
        let b_start = i * k * n;
        let out_start = i * m * n;
        
        matmul_2d(
            ctx,
            &a[a_start..a_start + m * k],
            &b[b_start..b_start + k * n],
            m, n, k,
            &mut output[out_start..out_start + m * n],
        )?;
    }
    
    Ok(())
}

/// Check if CUDA is available on this system.
pub fn is_cuda_available() -> bool {
    CudaContext::new(0).is_ok()
}
