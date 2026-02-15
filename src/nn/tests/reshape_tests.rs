use crate::tensor::Tensor;
use ndarray::Array;

pub fn reshape_for_multihead(
    t: &Tensor,
    num_heads: usize,
    seq: usize,
    head_dim: usize,
    b: usize, // Note: argument order might need checking, let's match the usage in test
) -> Result<Tensor, String> {
    // The usage in test is: reshape_for_multihead(&t, 2, 3, 4, 2);
    // arguments: t, num_heads, seq, head_dim, b ?
    // implementation in transformer.rs was: t, b, seq, num_heads, head_dim
    // Let's check the test usage vs implementation.
    // Test: reshape_for_multihead(&t, 2, 3, 4, 2) where t has 24 elements.
    // 2*3*4*2 = 48 elements.
    // The implementation:
    // fn reshape_for_multihead(t, b, seq, num_heads, head_dim)
    // t.reshape(vec![b, seq, num_heads, head_dim])

    // Let's copy the exact implementation from transformer.rs.
    let r = t.reshape(vec![b, seq, num_heads, head_dim])?;
    let p = r.permute(vec![0, 2, 1, 3]);
    p.reshape(vec![b * num_heads, seq, head_dim])
}

#[test]
fn reshape_for_multihead_returns_err_on_size_mismatch() {
    // create a tensor with only 24 elements
    let t = Tensor::new(
        Array::from_shape_vec((2, 3, 4), vec![0.0f32; 24])
            .unwrap()
            .into_dyn(),
        false,
    );
    // request reshape that requires 48 elements (2*3*4*2)
    let res = reshape_for_multihead(&t, 2, 3, 4, 2);
    assert!(
        res.is_err(),
        "Expected reshape_for_multihead to return Err on size mismatch"
    );
}
