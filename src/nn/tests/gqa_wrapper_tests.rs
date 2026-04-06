#![cfg(test)]

use crate::nn::GroupedQueryAttention;
use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn gqa_wrapper_forward_shape_and_head_config() {
    let b = 2;
    let seq = 5;
    let d_model = 16;
    let num_heads = 8;
    let kv_heads = 2;

    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32 * 0.001).collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        true,
    );

    let gqa = GroupedQueryAttention::new(
        d_model,
        num_heads,
        kv_heads,
        true,
        10000.0,
        1.0,
        true,
    )
    .expect("create GroupedQueryAttention");

    assert_eq!(gqa.mha.num_heads, num_heads);
    assert_eq!(gqa.mha.kv_heads, kv_heads);

    let out = gqa.forward(&x);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);

    let out_causal = gqa.forward_with_causal(&x, true, Some(0));
    assert_eq!(out_causal.lock().storage.shape(), &[b, seq, d_model]);
}

#[test]
fn gqa_wrapper_rejects_invalid_head_ratio() {
    let res = GroupedQueryAttention::new(16, 6, 4, false, 10000.0, 1.0, true);
    assert!(res.is_err(), "expected constructor error for invalid head ratio");
    if let Err(err) = res {
        assert!(
            err.contains("divisible"),
            "expected divisibility error, got: {}",
            err
        );
    }
}
