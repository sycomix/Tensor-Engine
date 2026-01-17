use ndarray::IxDyn;
use tensor_engine::nn::MultiHeadAttention;
use tensor_engine::tensor::Tensor;

#[test]
fn test_mha_with_kv_mismatch_shapes() {
    // Simulate a model where k_proj weight was saved with shape (kv_heads*head_dim, d_model)
    // For d_model=3072, num_heads=24, head_dim=128, kv_heads=8 -> kv_heads*head_dim = 1024
    let d_model = 3072usize;
    let num_heads = 24usize;
    let kv_heads = 8usize;
    let mha = MultiHeadAttention::new_with_kv_and_rope(
        d_model, num_heads, kv_heads, false, 10000.0, 1.0, true,
    );
    // Create a fake input: batch=1, seq=2, d_model
    let input = Tensor::new(ndarray::Array::zeros(IxDyn(&[1, 2, d_model])), false);
    // Replace k weight with a transposed-like shape [1024, 3072]
    let bad_k = Tensor::new(
        ndarray::Array::zeros(IxDyn(&[kv_heads * (d_model / num_heads), d_model])),
        false,
    );
    // Assign into linear_k directly (simulating loaded transposed weight)
    let mut mha2 = mha.clone();
    mha2.linear_k.weight = bad_k;
    // Run forward; should not panic and should log debug messages about shapes
    let _ = mha2.forward_with_causal(&input, false, None);
}
