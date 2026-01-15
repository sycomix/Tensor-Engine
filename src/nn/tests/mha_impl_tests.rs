use crate::nn::MultiHeadAttention;

#[test]
fn mha_module_impl_delegates_to_impl_methods() {
    let d_model = 8usize;
    let num_heads = 2usize;
    let mha = MultiHeadAttention::new(d_model, num_heads);
    // Compare parameter lengths to ensure no infinite recursion and proper delegation
    let p = mha.parameters();
    let p_impl = mha.parameters_impl();
    assert_eq!(
        p.len(),
        p_impl.len(),
        "parameters() should delegate to parameters_impl"
    );

    let named = mha.named_parameters("layer");
    let named_impl = mha.named_parameters_impl("layer");
    assert_eq!(
        named.len(),
        named_impl.len(),
        "named_parameters() should delegate to named_parameters_impl"
    );
}
