use crate::nn::KVCache;
use crate::nn::TransformerBlock;
use crate::tensor::Tensor;

#[test]
fn transformer_block_sets_kv_cache_on_forward() {
    let d_model = 8usize;
    let num_heads = 2usize;
    let kv_heads = 2usize;
    let mut block = TransformerBlock::new_with_kv_and_rope(
        d_model, 16, num_heads, kv_heads, false, 10000.0, 1.0, true,
    )
    .expect("create block");
    block.set_kv_cache(KVCache::new());

    let seq = 2usize;
    let mut data = Vec::new();
    for i in 0..(1 * seq * d_model) {
        data.push((i % 7) as f32 + 0.1);
    }
    let arr = ndarray::Array::from_shape_vec((1, seq, d_model), data)
        .unwrap()
        .into_dyn();
    let x = Tensor::new(arr.clone(), false);

    // forward should populate packed storage in the per-layer cache
    let _out = block.forward_block(&x);

    let cache = block.kv_cache_clone();
    assert!(cache.is_some());
    let cache = cache.unwrap();
    assert!(cache.has_packed());
    let pk = cache.packed_keys().unwrap();
    let shape = pk.lock().storage.shape().to_vec();
    assert_eq!(shape, vec![1, seq, d_model]);
}
