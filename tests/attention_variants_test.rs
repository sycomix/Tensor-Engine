use tensor_engine::nn::{AttentionVariant, Module, MultiHeadAttention};
use tensor_engine::tensor::Tensor;
use ndarray::IxDyn;

// Compare baseline attention vs FlashAttentionRef and ChunkedAttention
#[test]
fn test_flash_ref_and_chunked_match_baseline() {
    let d_model = 8usize;
    let num_heads = 2usize;
    let seq = 3usize;
    // build MHAs
    let mut base = MultiHeadAttention::new(d_model, num_heads);
    let mut flash = MultiHeadAttention::new(d_model, num_heads);
    let mut chunked = MultiHeadAttention::new(d_model, num_heads);
    // set variants
    flash.set_attention_variant(AttentionVariant::FlashRef);
    chunked.set_attention_variant(AttentionVariant::Chunked { chunk_size: 2 });

    // set deterministic weights: use small sequential values for Q/K/V and O
    let mut weight_vals = vec![];
    for i in 0..(d_model * d_model) { weight_vals.push((i as f32) * 0.01); }
    let weight_arr = ndarray::Array::from_shape_vec(IxDyn(&[d_model, d_model]), weight_vals.clone()).unwrap();
    let w_t = Tensor::new(weight_arr.into_dyn(), false);
    // bias
    let bias_arr = ndarray::Array::from_shape_vec(IxDyn(&[d_model]), vec![0.0f32; d_model]).unwrap();
    let b_t = Tensor::new(bias_arr.into_dyn(), false);
    // assign same weights to all Q/K/V/O layers of the MHAs
    for m in [&mut base, &mut flash, &mut chunked].iter_mut() {
        {
            let mut lk = m.linear_q.weight.lock();
            lk.storage = w_t.lock().storage.clone();
        }
        {
            let mut lk = m.linear_k.weight.lock();
            lk.storage = w_t.lock().storage.clone();
        }
        {
            let mut lk = m.linear_v.weight.lock();
            lk.storage = w_t.lock().storage.clone();
        }
        {
            let mut lk = m.linear_o.weight.lock();
            lk.storage = w_t.lock().storage.clone();
            let mut lb = m.linear_o.bias.as_ref().unwrap().lock();
            lb.storage = b_t.lock().storage.clone();
        }
    }

    // create simple input: batch 1, seq x d_model
    let mut in_vals = vec![];
    for i in 0..(1 * seq * d_model) {
        in_vals.push(((i % d_model) as f32) * 0.01 + 0.001);
    }
    let inp = Tensor::new(ndarray::Array::from_shape_vec(IxDyn(&[1, seq, d_model]), in_vals).unwrap().into_dyn(), false);

    // compute outputs
    let out_base = base.forward(&inp);
    let out_flash = flash.forward(&inp);
    let out_chunked = chunked.forward(&inp);

    let a = out_base.lock().storage.to_f32_array();
    let b = out_flash.lock().storage.to_f32_array();
    let c = out_chunked.lock().storage.to_f32_array();
    // assert shapes equal
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.shape(), c.shape());
    // numeric approx equality: check element differences within small epsilon
    for ((ai, bi), ci) in a.iter().zip(b.iter()).zip(c.iter()) {
        let basev = *ai;
        let flashv = *bi;
        let chunkv = *ci;
        let diff1 = (basev - flashv).abs();
        let diff2 = (basev - chunkv).abs();
        assert!(diff1 < 1e-4 || diff1.is_nan() == false, "Flash mismatch: {} vs {} diff {}", basev, flashv, diff1);
        assert!(diff2 < 1e-4 || diff2.is_nan() == false, "Chunked mismatch: {} vs {} diff {}", basev, chunkv, diff2);
    }
}
