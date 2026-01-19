@group(0) @binding(0)
var<storage, read> buffer_a: array<f32>;

@group(0) @binding(1)
var<storage, read> buffer_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> buffer_c: array<f32>;

struct Params {
    M: u32,
    K: u32,
    N: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= params.M || col >= params.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        let a_idx = row * params.K + k;
        let b_idx = k * params.N + col; // Assuming row-major B
        sum = sum + buffer_a[a_idx] * buffer_b[b_idx];
    }

    let c_idx = row * params.N + col;
    buffer_c[c_idx] = sum;
}
