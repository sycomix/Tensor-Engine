use crate::tensor::Tensor;
use crate::nn::KVCache;
use ndarray::Array;

#[test]
fn append_packed_concatenates_along_seq() {
    // create packed keys: shape (batch=1, seq=2, dim=4)
    let a_k = Array::from_shape_vec((1, 2, 4), (0..8).map(|i| i as f32).collect()).unwrap().into_dyn();
    let a_v = Array::from_shape_vec((1, 2, 4), (100..108).map(|i| i as f32).collect()).unwrap().into_dyn();
    let b_k = Array::from_shape_vec((1, 3, 4), (8..20).map(|i| i as f32).collect()).unwrap().into_dyn();
    let b_v = Array::from_shape_vec((1, 3, 4), (200..212).map(|i| i as f32).collect()).unwrap().into_dyn();

    let ta_k = Tensor::new(a_k, false);
    let ta_v = Tensor::new(a_v, false);
    let tb_k = Tensor::new(b_k, false);
    let tb_v = Tensor::new(b_v, false);

    let mut cache = KVCache::new();
    assert!(!cache.has_packed());
    cache.append_packed(&ta_k, &ta_v).unwrap();
    assert!(cache.has_packed());
    // append second packed chunk
    cache.append_packed(&tb_k, &tb_v).unwrap();

    let keys = cache.packed_keys().unwrap();
    let vals = cache.packed_values().unwrap();

    let k_arr = keys.lock().storage.to_f32_array();
    let v_arr = vals.lock().storage.to_f32_array();

    assert_eq!(k_arr.shape(), &[1, 5, 4]);
    assert_eq!(v_arr.shape(), &[1, 5, 4]);
    // check first and last elements
    assert_eq!(k_arr[[0, 0, 0]], 0.0);
    assert_eq!(k_arr[[0, 4, 3]], 19.0);
    assert_eq!(v_arr[[0, 0, 0]], 100.0);
    assert_eq!(v_arr[[0, 4, 3]], 211.0);
}

#[test]
fn append_packed_rejects_shape_mismatch() {
    let a_k = Array::from_shape_vec((1, 2, 4), (0..8).map(|i| i as f32).collect()).unwrap().into_dyn();
    let a_v = Array::from_shape_vec((1, 2, 4), (100..108).map(|i| i as f32).collect()).unwrap().into_dyn();
    // mismatch batch
    let b_k = Array::from_shape_vec((2, 1, 4), (8..16).map(|i| i as f32).collect()).unwrap().into_dyn();
    let b_v = Array::from_shape_vec((2, 1, 4), (200..208).map(|i| i as f32).collect()).unwrap().into_dyn();

    let ta_k = Tensor::new(a_k, false);
    let ta_v = Tensor::new(a_v, false);
    let tb_k = Tensor::new(b_k, false);
    let tb_v = Tensor::new(b_v, false);

    let mut cache = KVCache::new();
    cache.append_packed(&ta_k, &ta_v).unwrap();
    let res = cache.append_packed(&tb_k, &tb_v);
    assert!(res.is_err());
}
