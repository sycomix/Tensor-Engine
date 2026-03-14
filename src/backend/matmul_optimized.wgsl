// Optimized WGSL Compute Shader for Matrix Multiplication (MatMul)
// Features: Tiled computation, shared memory simulation, and optimized access patterns

struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    tile_size: u32, // Typically 16 or 32 for optimal performance
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: MatMulParams;

// Tile size for computation
const TILE_SIZE: u32 = 16u;

/// Optimized matrix multiplication with tiling
/// Uses shared memory simulation through thread-local registers
/// Computes C = A * B where:
/// - A is [m x k]
/// - B is [k x n]  
/// - C is [m x n]
@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn matmul_optimized(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    // Check bounds - skip work items outside matrix dimensions
    if row >= params.m || col >= params.n {
        return;
    }

    var acc: f32 = 0.0;

    // Tiled computation to improve cache utilization
    for (var tile_start: u32 = 0u; tile_start < params.k; tile_start += TILE_SIZE) {
        let tile_end = min(tile_start + TILE_SIZE, params.k);
        
        // Load tiles into thread-local registers
        var a_tile: array<f32, TILE_SIZE>;
        var b_tile: array<f32, TILE_SIZE>;

        for (var i: u32 = 0u; i < tile_end - tile_start; i++) {
            let k_idx = tile_start + i;
            
            // Load from A with row-major access pattern
            a_tile[i] = a[row * params.k + k_idx];
            
            // Transpose B for column-major access (better cache behavior)
            b_tile[i] = b[k_idx * params.n + col];
        }

        // Compute partial dot product for this tile
        for (var i: u32 = 0u; i < TILE_SIZE; i++) {
            if (i < tile_end - tile_start) {
                acc += a_tile[i] * b_tile[i];
            }
        }
    }

    // Write result to output matrix
    c[row * params.n + col] = acc;
}

/// Batched matrix multiplication for LLM inference
/// Handles multiple matrices simultaneously with shared parameters
@compute @workgroup_size(64, 1, 1)
fn matmul_batched(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z;
    let row = global_id.x;
    let col = global_id.y;

    // Check bounds - skip work items outside matrix dimensions
    if row >= params.m || col >= params.n {
        return;
    }

    var acc: f32 = 0.0;

    // Compute dot product for this batch element
    for (var i: u32 = 0u; i < params.k; i++) {
        let a_idx = batch_idx * (params.m * params.k) + row * params.k + i;
        let b_idx = batch_idx * (params.k * params.n) + i * params.n + col;
        
        acc += a[a_idx] * b[b_idx];
    }

    // Write result to output matrix
    let c_idx = batch_idx * (params.m * params.n) + row * params.n + col;
    c[c_idx] = acc;
}

/// Matrix multiplication with fused bias and activation
/// Computes: C = gelu(A @ B + bias)
struct MatMulBiasActivationParams {
    m: u32,
    n: u32,
    k: u32,
    has_bias: bool,  // Whether to add bias
    has_activation: bool, // Whether to apply activation function
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>; // Optional
@group(0) @binding(3) var<uniform> params: MatMulBiasActivationParams;

fn gelu(x: f32) -> f32 {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = 0.7978845608028654; // sqrt(2/pi)
    let cbrt_coef = 0.044715;
    
    return 0.5 * x * (1.0 + tanh(sqrt_2_over_pi * (x + cbrt_coef * x * x * x)));
}

@compute @workgroup_size(64, 1, 1)
fn matmul_fused(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if row >= params.m || col >= params.n {
        return;
    }

    var acc: f32 = 0.0;

    // Compute dot product
    for (var i: u32 = 0u; i < params.k; i++) {
        let a_idx = row * params.k + i;
        let b_idx = i * params.n + col;
        
        acc += a[a_idx] * b[b_idx];
    }

    // Add bias if present
    if params.has_bias && col < params.n {
        acc += bias[col];
    }

    // Apply activation function if requested
    if params.has_activation {
        acc = gelu(acc);
    }

    // Write result to output matrix
    let c_idx = row * params.n + col;
    c[c_idx] = acc;
}
