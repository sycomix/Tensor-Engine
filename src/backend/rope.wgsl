// Optimized WGSL Compute Shader for Rotary Positional Embeddings (RoPE)
// Used in Llama/Mistral architectures for positional encoding
// Applies rotation based on position and frequency to each pair of dimensions

struct RopeParams {
    seq_len: u32,        // Sequence length
    head_dim: u32,       // Dimension per attention head (must be even)
    freq_offset: u32,    // Offset for frequency calculation
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> freqs: array<f32>; // Precomputed frequencies
@group(0) @binding(2) var<storage, write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: RopeParams;

/// Rotary Positional Embedding (RoPE) computation
/// Applies rotation matrix to each pair of dimensions based on position
/// Formula: For position p and dimension pair (d, d+1):
///   x'_d = x_d * cos(p * theta_d) - x_{d+1} * sin(p * theta_d)
///   x'_{d+1} = x_d * sin(p * theta_d) + x_{d+1} * cos(p * theta_d)
@compute @workgroup_size(64, 1, 1)
fn rope_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let pos = global_id.y;
    let dim_pair_start = global_id.z * 2u;

    if pos >= params.seq_len || dim_pair_start + 1u >= params.head_dim {
        return;
    }

    // Get the frequency for this dimension pair
    let freq_idx = dim_pair_start / 2u;
    
    if freq_idx >= params.freqs.len() {
        log::warn!("RoPE: Frequency index {} out of bounds", freq_idx);
        return;
    }

    let theta = params.freqs[freq_idx];

    // Compute rotation angle based on position
    let angle = pos * theta;
    
    let cos_theta = cos(angle);
    let sin_theta = sin(angle);

    // Get the two values to rotate (from input tensor)
    // Input layout: [batch, seq_len, head_dim]
    let base_idx = batch_idx * params.seq_len * params.head_dim + pos * params.head_dim;
    
    let x_i = input[base_idx + dim_pair_start];
    let x_i1 = input[base_idx + dim_pair_start + 1u];

    // Apply rotation: [cos θ -sin θ; sin θ cos θ]
    let rotated_0 = x_i * cos_theta - x_i1 * sin_theta;
    let rotated_1 = x_i * sin_theta + x_i1 * cos_theta;

    // Write rotated values to output
    output[base_idx + dim_pair_start] = rotated_0;
    output[base_idx + dim_pair_start + 1u] = rotated_1;
}

/// Parallel RoPE with optimized memory access patterns
/// Uses strided access for better cache utilization
@compute @workgroup_size(64, 1, 1)
fn rope_parallel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let pos = global_id.y;
    
    if pos >= params.seq_len {
        return;
    }

    // Each thread processes multiple dimension pairs with strided access
    let stride = 64u;
    let dim_pair_start = global_id.z * 2u + (global_id.x % stride) * 2u;

    if dim_pair_start + 1u >= params.head_dim {
        return;
    }

    // Get the frequency for this dimension pair
    let freq_idx = dim_pair_start / 2u;
    
    if freq_idx >= params.freqs.len() {
        log::warn!("RoPE Parallel: Frequency index {} out of bounds", freq_idx);
        return;
    }

    let theta = params.freqs[freq_idx];

    // Compute rotation angle based on position
    let angle = pos * theta;
    
    let cos_theta = cos(angle);
    let sin_theta = sin(angle);

    // Get the two values to rotate (from input tensor)
    let base_idx = batch_idx * params.seq_len * params.head_dim + pos * params.head_dim;
    
    var x_i: f32 = 0.0;
    var x_i1: f32 = 0.0;

    // Load with bounds checking for strided access
    if dim_pair_start < params.head_dim {
        x_i = input[base_idx + dim_pair_start];
    }
    
    if dim_pair_start + 1u < params.head_dim {
        x_i1 = input[base_idx + dim_pair_start + 1u];
    }

    // Apply rotation: [cos θ -sin θ; sin θ cos θ]
    let rotated_0 = x_i * cos_theta - x_i1 * sin_theta;
    let rotated_1 = x_i * sin_theta + x_i1 * cos_theta;

    // Write rotated values to output with bounds checking
    if dim_pair_start < params.head_dim {
        output[base_idx + dim_pair_start] = rotated_0;
    }
    
    if dim_pair_start + 1u < params.head_dim {
        output[base_idx + dim_pair_start + 1u] = rotated_1;
    }
}

/// RoPE with numerical stability and overflow protection
@compute @workgroup_size(64, 1, 1)
fn rope_safe(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let pos = global_id.y;
    let dim_pair_start = global_id.z * 2u;

    if pos >= params.seq_len || dim_pair_start + 1u >= params.head_dim {
        return;
    }

    // Get the frequency for this dimension pair
    let freq_idx = dim_pair_start / 2u;
    
    if freq_idx >= params.freqs.len() {
        log::warn!("RoPE Safe: Frequency index {} out of bounds", freq_idx);
        return;
    }

    let theta = params.freqs[freq_idx];

    // Compute rotation angle with overflow protection
    let angle = pos * theta;
    
    // Clamp angle to prevent numerical issues for very large positions
    let clamped_angle = min(abs(angle), 1e6) * sign(angle);
    
    let cos_theta = cos(clamped_angle);
    let sin_theta = sin(clamped_angle);

    // Get the two values to rotate (from input tensor)
    let base_idx = batch_idx * params.seq_len * params.head_dim + pos * params.head_dim;
    
    var x_i: f32 = 0.0;
    var x_i1: f32 = 0.0;

    // Load with bounds checking and overflow protection
    if dim_pair_start < params.head_dim {
        let loaded_val = input[base_idx + dim_pair_start];
        
        if is_finite(loaded_val) && !is_nan(loaded_val) {
            x_i = min(abs(loaded_val), 1e6) * sign(loaded_val);
        } else {
            log::warn!("RoPE Safe: Non-finite value detected at index {}", base_idx + dim_pair_start);
            output[base_idx + dim_pair_start] = 0.0;
            return;
        }
    }
    
    if dim_pair_start + 1u < params.head_dim {
        let loaded_val = input[base_idx + dim_pair_start + 1u];
        
        if is_finite(loaded_val) && !is_nan(loaded_val) {
            x_i1 = min(abs(loaded_val), 1e6) * sign(loaded_val);
        } else {
            log::warn!("RoPE Safe: Non-finite value detected at index {}", base_idx + dim_pair_start + 1u);
            output[base_idx + dim_pair_start + 1u] = 0.0;
            return;
        }
    }

    // Apply rotation: [cos θ -sin θ; sin θ cos θ]
    let rotated_0 = x_i * cos_theta - x_i1 * sin_theta;
    let rotated_1 = x_i * sin_theta + x_i1 * cos_theta;

    // Clamp results to prevent overflow
    output[base_idx + dim_pair_start] = min(abs(rotated_0), 1e6) * sign(rotated_0);
    output[base_idx + dim_pair_start + 1u] = min(abs(rotated_1), 1e6) * sign(rotated_1);
}

/// RoPE with pre-computed rotation matrices for better performance
/// Uses lookup tables for cos/sin values when positions are limited
struct RopeWithLookupParams {
    seq_len: u32,
    head_dim: u32,
    freq_offset: u32,
    use_lookup_table: bool, // Whether to use pre-computed rotation matrices
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> freqs: array<f32>;
@group(0) @binding(2) var<storage, read> cos_table: array<f32>; // Pre-computed cosine values
@group(0) @binding(3) var<storage, read> sin_table: array<f32>; // Pre-computed sine values
@group(0) @binding(4) var<uniform> params: RopeWithLookupParams;

fn apply_rotation_with_lookup(x_i: f32, x_i1: f32, cos_val: f32, sin_val: f32) -> vec2<f32> {
    let rotated = vec2<f32>(
        x_i * cos_val - x_i1 * sin_val,
        x_i * sin_val + x_i1 * cos_val
    );
    
    return rotated;
}

@compute @workgroup_size(64, 1, 1)
fn rope_with_lookup(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let pos = global_id.y;
    let dim_pair_start = global_id.z * 2u;

    if pos >= params.seq_len || dim_pair_start + 1u >= params.head_dim {
        return;
    }

    // Get the frequency for this dimension pair
    let freq_idx = dim_pair_start / 2u;
    
    if freq_idx >= params.freqs.len() {
        log::warn!("RoPE with Lookup: Frequency index {} out of bounds", freq_idx);
        return;
    }

    // Use pre-computed rotation values from lookup table
    let table_idx = pos % (params.seq_len * 1024u); // Assume table size is seq_len * 1024
    
    if table_idx >= params.cos_table.len() || table_idx >= params.sin_table.len() {
        log::warn!("RoPE with Lookup: Table index {} out of bounds", table_idx);
        
        // Fallback to on-the-fly computation
        let theta = params.freqs[freq_idx];
        let angle = pos * theta;
        
        let cos_theta = cos(angle);
        let sin_theta = sin(angle);

        let base_idx = batch_idx * params.seq_len * params.head_dim + pos * params.head_dim;
        
        var x_i: f32 = 0.0;
        var x_i1: f32 = 0.0;

        if dim_pair_start < params.head_dim {
            let loaded_val = input[base_idx + dim_pair_start];
            if is_finite(loaded_val) && !is_nan(loaded_val) {
                x_i = min(abs(loaded_val), 1e6) * sign(loaded_val);
            }
        }
        
        if dim_pair_start + 1u < params.head_dim {
            let loaded_val = input[base_idx + dim_pair_start + 1u];
            if is_finite(loaded_val) && !is_nan(loaded_val) {
                x_i1 = min(abs(loaded_val), 1e6) * sign(loaded_val);
            }
        }

        let rotated = apply_rotation_with_lookup(x_i, x_i1, cos_theta, sin_theta);
        
        if dim_pair_start < params.head_dim {
            output[base_idx + dim_pair_start] = min(abs(rotated.x), 1e6) * sign(rotated.x);
        }
        
        if dim_pair_start + 1u < params.head_dim {
            output[base_idx + dim_pair_start + 1u] = min(abs(rotated.y), 1e6) * sign(rotated.y);
        }

        return;
    }

    // Use pre-computed rotation values
    let cos_val = cos_table[table_idx];
    let sin_val = sin_table[table_idx];

    // Get the two values to rotate (from input tensor)
    let base_idx = batch_idx * params.seq_len * params.head_dim + pos * params.head_dim;
    
    var x_i: f32 = 0.0;
    var x_i1: f32 = 0.0;

    if dim_pair_start < params.head_dim {
        let loaded_val = input[base_idx + dim_pair_start];
        
        if is_finite(loaded_val) && !is_nan(loaded_val) {
            x_i = min(abs(loaded_val), 1e6) * sign(loaded_val);
        } else {
            log::warn!("RoPE with Lookup: Non-finite value detected at index {}", base_idx + dim_pair_start);
            output[base_idx + dim_pair_start] = 0.0;
            return;
        }
    }
    
    if dim_pair_start + 1u < params.head_dim {
        let loaded_val = input[base_idx + dim_pair_start + 1u];
        
        if is_finite(loaded_val) && !is_nan(loaded_val) {
            x_i1 = min(abs(loaded_val), 1e6) * sign(loaded_val);
        } else {
            log::warn!("RoPE with Lookup: Non-finite value detected at index {}", base_idx + dim_pair_start + 1u);
            output[base_idx + dim_pair_start + 1u] = 0.0;
            return;
        }
    }

    // Apply rotation using pre-computed values
    let rotated = apply_rotation_with_lookup(x_i, x_i1, cos_val, sin_val);

    // Write rotated values to output with bounds checking and overflow protection
    if dim_pair_start < params.head_dim {
        output[base_idx + dim_pair_start] = min(abs(rotated.x), 1e6) * sign(rotated.x);
    }
    
    if dim_pair_start + 1u < params.head_dim {
        output[base_idx + dim_pair_start + 1u] = min(abs(rotated.y), 1e6) * sign(rotated.y);
    }
}
