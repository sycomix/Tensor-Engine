use crate::nn::KVCache;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn kv_cache_basic_append_and_len() {
    // create a tiny tensor for keys/values
    let k = Array::from_elem((1, 1), 0.5f32).into_dyn();
    let v = Array::from_elem((1, 1), 1.5f32).into_dyn();

    let tk = Tensor::new(k, false);
    let tv = Tensor::new(v, false);

    let mut cache = KVCache::new();
    assert!(cache.is_empty());
    cache.append(tk.clone(), tv.clone());
    assert_eq!(cache.len(), 1);
    cache.append(tk, tv);
    assert_eq!(cache.len(), 2);
    cache.clear();
    assert!(cache.is_empty());
}
