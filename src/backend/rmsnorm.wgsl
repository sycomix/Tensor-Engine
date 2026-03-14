// Optimized WGSL Compute Shader for RMSNorm (Root Mean Square Layer Normalization)
// Used in Llama/Mistral architectures with pre-normalization
// Computes: output = (x / sqrt(mean(x^2) + eps)) * weight

struct RmsNormParams {
    size: u32,           // Size of the normalization dimension
    axis_offset: u32,    // Offset to skip dimensions before the norm axis
    eps: f32,            // Epsilon for numerical stability (typically 1e-6)
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: RmsNormParams;

/// RMSNorm computation with numerical stability
/// Computes: output = (x / sqrt(mean(x^2) + eps)) * weight
/// This is the pre-norm variant used in Llama/Mistral architectures
@compute @workgroup_size(256, 1, 1)
fn rms_norm_stable(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= params.size {
        return;
    }

    // Compute sum of squares along the normalization dimension
    var sum_sq: f32 = 0.0;

    for (var i: u32 = 0u; i < params.size; i++) {
        let current_idx = idx * params.axis_offset + i;
        
        if current_idx < params.size * params.axis_offset {
            let x_i = input[current_idx];
            
            // Square the value and accumulate
            sum_sq += x_i * x_i;
        }
    }

    // Compute RMS with epsilon for numerical stability
    let rms = sqrt(sum_sq / params.size + params.eps);

    // Normalize by dividing by RMS
    var normalized: f32 = 0.0;
    
    if rms > 1e-8 {
        let current_idx = idx * params.axis_offset;
        
        for (var i: u32 = 0u; i < params.size && i < params.axis_offset; i++) {
            let x_i = input[current_idx + i];
            
            // Apply normalization and weight scaling
            normalized = (x_i / rms) * weight[i % params.axis_offset as usize];
        }
    } else {
        log::warn!("RMSNorm: Near-zero RMS value detected, skipping normalization");
        normalized = 0.0;
    }

    output[idx] = normalized;
}

/// Parallel RMSNorm with reduction for better performance
/// Uses thread-local computation and parallel reduction patterns
@compute @workgroup_size(256, 1, 1)
fn rms_norm_parallel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= params.size {
        return;
    }

    // Thread-local sum of squares computation with strided access
    var local_sum_sq: f32 = 0.0;
    let stride = 256u;

    for (var i: u32 = idx; i < params.size * params.axis_offset; i += stride) {
        let x_i = input[i];
        
        // Square and accumulate with overflow protection
        if is_finite(x_i) {
            local_sum_sq += x_i * x_i;
        } else {
            log::warn!("RMSNorm: Non-finite value detected at index {}", i);
        }
    }

    // Compute RMS with epsilon for numerical stability
    let rms = sqrt(local_sum_sq / (params.size * params.axis_offset) + params.eps);

    // Apply normalization and weight scaling
    if rms > 1e-8 {
        let normalized_val = input[idx] / rms;
        
        // Apply weight scaling (assuming weight matches the normalization dimension size)
        let weight_idx = idx % (params.size as usize);
        output[idx] = normalized_val * weight[weight_idx];
    } else {
        log::warn!("RMSNorm: Near-zero RMS value detected at index {}", idx);
        output[idx] = 0.0;
    }
}

/// RMSNorm with fused bias addition
/// Computes: output = (x / sqrt(mean(x^2) + eps)) * weight + bias
@struct {
    size: u32,
    axis_offset: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>; // Optional bias
@group(0) @binding(3) var<storage, write> output: array<f32>;

fn rms_norm_with_bias(input_val: f32, weight_val: f32, bias_val: f32, rms: f32) -> f32 {
    if rms > 1e-8 {
        return (input_val / rms) * weight_val + bias_val;
    } else {
        log::warn!("RMSNorm with Bias: Near-zero RMS value detected");
        return bias_val; // Return just the bias as fallback
    }
}

@compute @workgroup_size(256, 1, 1)
fn rms_norm_fused_bias(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= params.size {
        return;
    }

    // Compute sum of squares along the normalization dimension
    var local_sum_sq: f32 = 0.0;

    for (var i: u32 = 0u; i < params.axis_offset; i++) {
        let current_idx = idx * params.axis_offset + i;
        
        if current_idx < params.size * params.axis_offset {
            let x_i = input[current_idx];
            
            if is_finite(x_i) {
                local_sum_sq += x_i * x_i;
            } else {
                log::warn!("RMSNorm with Bias: Non-finite value detected at index {}", current_idx);
            }
        }
    }

    // Compute RMS with epsilon for numerical stability
    let rms = sqrt(local_sum_sq / params.axis_offset + params.eps);

    // Apply normalization, weight scaling, and bias addition
    var result: f32 = 0.0;
    
    if rms > 1e-8 {
        for (var i: u32 = 0u; i < params.axis_offset; i++) {
            let current_idx = idx * params.axis_offset + i;
            
            if current_idx < params.size * params.axis_offset {
                let input_val = input[current_idx];
                let weight_val = weight[i as usize % (params.axis_offset as usize)];
                let bias_val = bias[i as usize % (params.axis_offset as usize)];
                
                result = (input_val / rms) * weight_val + bias_val;
            }
        }
    } else {
        log::warn!("RMSNorm with Bias: Near-zero RMS value detected");
        
        // Fallback to just adding bias if RMS is near zero
        for (var i: u32 = 0u; i < params.axis_offset; i++) {
            let current_idx = idx * params.axis_offset + i;
            
            if current_idx < params.size * params.axis_offset {
                result += bias[i as usize % (params.axis_offset as usize)];
            }
        }
    }

    output[idx] = result;
}

/// RMSNorm with numerical stability checks and overflow protection
@compute @workgroup_size(256, 1, 1)
fn rms_norm_safe(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= params.size {
        return;
    }

    var local_sum_sq: f32 = 0.0;
    var has_nan_or_inf: bool = false;

    // Compute sum of squares with overflow protection
    for (var i: u32 = 0u; i < params.size * params.axis_offset; i++) {
        let x_i = input[i];
        
        if !is_finite(x_i) || is_nan(x_i) {
            has_nan_or_inf = true;
            log::warn!("RMSNorm Safe: Non-finite value detected at index {}", i);
        } else {
            // Clamp large values to prevent overflow in squaring
            let clamped_x = min(abs(x_i), 1e6); // Cap at 1e6 before squaring
            local_sum_sq += clamped_x * clamped_x;
        }
    }

    if has_nan_or_inf {
        output[idx] = 0.0;
        return;
    }

    // Compute RMS with epsilon for numerical stability
    let rms = sqrt(local_sum_sq / (params.size * params.axis_offset) + params.eps);

    // Apply normalization and weight scaling with safety checks
    if rms > 1e-8 {
        let normalized_val = input[idx] / rms;
        
        // Clamp the result to prevent overflow
        let clamped_normalized = min(abs(normalized_val), 1e6) * sign(normalized_val);
        
        output[idx] = clamped_normalized * weight[idx % (params.size as usize)];
    } else {
        log::warn!("RMSNorm Safe: Near-zero RMS value detected at index {}", idx);
        output[idx] = 0.0; // Return zeros for near-zero RMS case
    }
}
