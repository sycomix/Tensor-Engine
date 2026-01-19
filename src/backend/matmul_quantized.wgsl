// matmul_quantized.wgsl
// Computes C = A * (Unpack(Q) - Z) * S + B
// A: [M, K] (f32)
// Q: [K, N/2] (u8 packed 4-bit, row-major)
// S: [K, N/G] (f32 scales)
// Z: [K, N/G] (f32 zeros)
// B: [N] (f32 bias)
// C: [M, N] (f32)

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> Q: array<u32>; // Packed as u32 (4 bytes = 8 nibbles) for alignment

@group(0) @binding(2)
var<storage, read> S: array<f32>;

@group(0) @binding(3)
var<storage, read> Z: array<f32>;

@group(0) @binding(4)
var<storage, read> B: array<f32>;

@group(0) @binding(5)
var<storage, read_write> C: array<f32>;

struct Params {
    M: u32,
    K: u32, // Input features
    N: u32, // Output features
    G: u32, // Group size
}

@group(0) @binding(6)
var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y; // M index
    let col = global_id.x; // N index

    if (row >= params.M || col >= params.N) {
        return;
    }

    var sum = 0.0;
    
    // Iterate over K
    for (var k = 0u; k < params.K; k = k + 1u) {
        let a_idx = row * params.K + k;
        let a_val = A[a_idx];

        // Retrieve quantized weight Q[k, col]
        // Q is likely stored in [K, N] logic, but packed.
        // Assuming [K, N] row-major logical layout.
        // Packed Layout: Each byte holds 2 values.
        //   Index formula for byte: (k * N + col) / 2
        //   Nibble selector: (k * N + col) % 2
        //   But we bind as u32 array for alignment.
        //   Byte index = idx / 2.
        //   U32 index = idx / 8.
        
        let linear_idx = k * params.N + col;
        let u32_idx = linear_idx / 8u;
        let shift = (linear_idx % 8u) * 4u;
        
        let packed_word = Q[u32_idx];
        let nibble = (packed_word >> shift) & 0xFu;
        let q_val = f32(nibble);
        
        // Retrieve Scale & Zero
        // Shape [K, N/G] -> Row-major
        // Index = k * (N/G) + (col / G)
        // BUT scales/zeros are usually Block-wise k-major?
        // Let's assume standard AWQ packing: scales/zeros are [N, K/G]? Or [K/G, N]?
        // In our CPU impl: scales/zeros are [K, N] LOGICAL but broadcasted from [K, N/G].
        // Wait, Cpu logic: 
        //   s_slice[row_offset_sz + g] where row_offset_sz = r * k_groups -> r is row(in_features=K index), g is col/G
        //   So scales are [K, N/G] physically.
        
        let group_idx = col / params.G;
        let scale_cols = params.N / params.G;
        let sz_idx = k * scale_cols + group_idx;
        
        let s_val = S[sz_idx];
        let z_val = Z[sz_idx];
        
        let w_val = (q_val - z_val) * s_val;
        
        sum = sum + a_val * w_val;
    }

    // Add bias
    // Bias is [N] or [1, N] broadcasted
    // If bias is strictly [1, N] or [N], index is just col.
    // However, if bias is passed as Option, the caller logic handles providing a zero-tensor if None.
    // The binding always exists.
    
    let bias_val = B[col];
    
    let c_idx = row * params.N + col;
    C[c_idx] = sum + bias_val;
}
