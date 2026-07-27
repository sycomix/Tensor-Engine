use crate::dtype::DType;
use crate::nn::paged_kv_cache::{PagedCacheConfig, PagedKVCache};
use crate::nn::MultiHeadAttention;
use crate::tensor::Tensor;
use ndarray::IxDyn;

#[test]
fn test_paged_attention_integration() {
    // Setup
    let d_model = 32;
    let num_heads = 4;
    let head_dim = d_model / num_heads; // 8

    let mha = MultiHeadAttention::new(d_model, num_heads);

    let config = PagedCacheConfig {
        block_size: 16,
        num_layers: 1,
        num_heads,
        head_dim,
        dtype: DType::F32,
        device: "cpu".to_string(),
    };

    let cache = PagedKVCache::new(config, 10);
    let seq_id = 999;
    cache.add_sequence(seq_id);

    // Simulate decoding step
    // Input: [batch=1, seq=1, d_model]
    let input_data = vec![0.5f32; d_model]; // simple uniform input
    let x = Tensor::new(
        ndarray::Array::from_shape_vec(IxDyn(&[1, 1, d_model][..]), input_data).unwrap(),
        false,
    );

    // 1. Run forward_with_paged_cache (first token)
    let out1 = mha.forward_with_paged_cache(&x, &cache, &[seq_id][..]);

    // Check output shape
    assert_eq!(out1.lock().storage.shape(), &[1, 1, d_model]);

    // Check that cache grew
    {
        let seqs = cache.sequences.lock().unwrap();
        let meta = seqs.get(&seq_id).unwrap();
        assert_eq!(meta.context_len, 1);
    }

    // 2. Run again (second token)
    let out2 = mha.forward_with_paged_cache(&x, &cache, &[seq_id][..]);

    // Check cache grew again
    {
        let seqs = cache.sequences.lock().unwrap();
        let meta = seqs.get(&seq_id).unwrap();
        assert_eq!(meta.context_len, 2);
    }

    // Compare output logic?
    // Hard to compare exact values without deterministic weights, but it shouldn't panic
    // and should produce finite values.
    let out_arr = out2.to_f32_array();
    assert!(out_arr.iter().all(|x| x.is_finite()));
}
