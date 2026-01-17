use crate::nn::KVCache;
use crate::nn::MultiHeadAttention;
use crate::tensor::Tensor;
use ndarray::s;
use ndarray::Array;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > tol {
            return false;
        }
    }
    true
}

#[test]
fn mha_incremental_matches_full_decode() {
    let d_model = 8usize;
    let num_heads = 2usize;
    let kv_heads = 2usize;
    let mha = MultiHeadAttention::new_with_kv_and_rope(
        d_model, num_heads, kv_heads, false, 10000.0, 1.0, true,
    );

    // build input sequence (batch=1, seq=4)
    let seq = 4usize;
    let mut data = Vec::new();
    for i in 0..(1 * seq * d_model) {
        data.push((i % 13) as f32 + 0.1);
    }
    let arr = Array::from_shape_vec((1, seq, d_model), data)
        .unwrap()
        .into_dyn();
    let x_full = Tensor::new(arr.clone(), false);

    // full decode
    let full_out = mha.forward_with_causal(&x_full, true, None);
    let full_arr = full_out.lock().storage.to_f32_array();

    // incremental decode
    let mut cache = KVCache::new();
    let mut pieces: Vec<f32> = Vec::new();
    for t in 0..seq {
        // slice arr to shape (1,1,d_model)
        let slice = arr.slice(s![0..1, t..t + 1, ..]).to_owned().into_dyn();
        let x_t = Tensor::new(slice, false);
        let out_t = mha.forward_with_caching(&x_t, true, None, Some(&mut cache));
        let arr_t = out_t.lock().storage.to_f32_array();
        // arr_t shape (1,1,d_model)
        for i in 0..d_model {
            pieces.push(arr_t[[0, 0, i]]);
        }
    }
    // reshape pieces into (1, seq, d_model)
    let inc_arr = Array::from_shape_vec((1, seq, d_model), pieces)
        .unwrap()
        .into_dyn();

    let full_flat = full_arr.iter().cloned().collect::<Vec<f32>>();
    let inc_flat = inc_arr.iter().cloned().collect::<Vec<f32>>();

    assert!(approx_eq(&full_flat, &inc_flat, 1e-5));
}
