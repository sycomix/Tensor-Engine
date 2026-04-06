#![cfg(test)]

use crate::nn::{AttentionVariant, MultiHeadAttention, SlidingWindowAttention};
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn sliding_window_wrapper_forward_shape() {
    let b = 2usize;
    let seq = 6usize;
    let d_model = 16usize;
    let num_heads = 4usize;

    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32 * 0.01).collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        true,
    );

    let swa = SlidingWindowAttention::new(d_model, num_heads, num_heads, 2, true, 10000.0, 1.0, true)
        .expect("create SlidingWindowAttention");
    let out = swa.forward_with_causal(&x, true, None);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);

    match swa.mha.attention_variant {
        AttentionVariant::SlidingWindow { window_size } => assert_eq!(window_size, 2),
        _ => panic!("expected sliding window variant"),
    }
}

#[test]
fn sliding_window_restricts_outputs_vs_causal_baseline() {
    let b = 1usize;
    let seq = 8usize;
    let d_model = 16usize;
    let num_heads = 4usize;

    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32 * 0.005).collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        false,
    );

    let mut mha = MultiHeadAttention::new(d_model, num_heads);
    let baseline = mha.forward_with_causal(&x, true, None, None);

    mha.set_attention_variant(AttentionVariant::SlidingWindow { window_size: 1 });
    let local = mha.forward_with_causal(&x, true, None, None);

    assert_eq!(baseline.lock().storage.shape(), local.lock().storage.shape());
    assert_ne!(
        baseline.lock().storage.to_f32_array(),
        local.lock().storage.to_f32_array(),
        "windowed causal attention should differ from full causal attention"
    );
}

#[test]
fn sliding_window_matches_causal_baseline_when_window_covers_context() {
    let b = 1usize;
    let seq = 5usize;
    let d_model = 8usize;
    let num_heads = 2usize;

    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32 * 0.02).collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        false,
    );

    let mut mha = MultiHeadAttention::new(d_model, num_heads);
    let baseline = mha.forward_with_causal(&x, true, None, None);

    mha.set_attention_variant(AttentionVariant::SlidingWindow { window_size: seq });
    let local = mha.forward_with_causal(&x, true, None, None);

    assert_eq!(
        baseline.lock().storage.to_f32_array(),
        local.lock().storage.to_f32_array(),
        "window >= sequence length should match standard causal attention"
    );
}
