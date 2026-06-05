#![cfg(test)]

use crate::nn::CrossAttention;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn cross_attention_wrapper_forward_shape() {
    let b = 2;
    let q_seq = 3;
    let kv_seq = 5;
    let d_model = 16;

    let query_data: Vec<f32> = (0..(b * q_seq * d_model))
        .map(|i| i as f32 * 0.01)
        .collect();
    let context_data: Vec<f32> = (0..(b * kv_seq * d_model))
        .map(|i| i as f32 * 0.005)
        .collect();

    let query = Tensor::new(
        Array::from_shape_vec((b, q_seq, d_model), query_data)
            .unwrap()
            .into_dyn(),
        true,
    );
    let context = Tensor::new(
        Array::from_shape_vec((b, kv_seq, d_model), context_data)
            .unwrap()
            .into_dyn(),
        true,
    );

    let cross = CrossAttention::new(d_model, 8, 2, true, 10000.0, 1.0, true)
        .expect("create CrossAttention");

    let out = cross.forward_cross(&query, &context, None);
    assert_eq!(out.lock().storage.shape(), &[b, q_seq, d_model]);
}

#[test]
fn cross_attention_wrapper_rejects_invalid_kv_ratio() {
    let res = CrossAttention::new(16, 6, 4, false, 10000.0, 1.0, true);
    assert!(
        res.is_err(),
        "expected constructor error for invalid head ratio"
    );
    if let Err(err) = res {
        assert!(
            err.contains("divisible"),
            "expected divisibility error, got: {}",
            err
        );
    }
}
