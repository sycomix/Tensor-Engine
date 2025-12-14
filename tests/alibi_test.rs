use ndarray::IxDyn;
use tensor_engine::nn::{Module, TransformerBlock};
use tensor_engine::tensor::Tensor;

#[test]
fn test_alibi_zero_slopes_no_effect() {
    let mut block = TransformerBlock::new(4, 8, 2);
    // input tensor: batch=1, seq=4, d_model=4
    let data = vec![0.0f32; 1 * 4 * 4];
    let arr = ndarray::Array::from_shape_vec(IxDyn(&[1, 4, 4]), data).unwrap().into_dyn();
    let t = Tensor::new(arr, false);
    // baseline output
    let baseline = block.forward(&t);
    // with ALiBi but zero slopes
    block.mha.use_alibi = true;
    block.mha.alibi_slopes = Some(vec![0.0f32; block.mha.num_heads]);
    let with_alibi = block.forward(&t);
    assert_eq!(baseline.lock().storage.to_f32_array(), with_alibi.lock().storage.to_f32_array());
}

#[test]
fn test_compute_alibi_slopes_and_bias_tensor() {
    use tensor_engine::nn::compute_alibi_slopes;
    let n_heads = 4usize;
    let slopes = compute_alibi_slopes(n_heads);
    assert_eq!(slopes.len(), n_heads);
    // monotonic decreasing slopes
    for i in 1..n_heads { assert!(slopes[i] <= slopes[i - 1]); }
    // build bias tensor for batch=1, seq=4
    let b = 1usize;
    let seq = 4usize;
    let mut bias_arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[b * n_heads, seq, seq]));
    for batch in 0..b {
        for h in 0..n_heads {
            let slope = slopes[h];
            for i in 0..seq {
                for j in 0..seq {
                    let dist = (j as isize - i as isize) as f32;
                    bias_arr[[batch * n_heads + h, i, j]] = -slope * dist;
                }
            }
        }
    }
    // sanity check first entry (head 0, i=0, j=1) == -slope * 1
    assert_eq!(bias_arr[[0, 0, 1]], -slopes[0] * 1.0f32);
}
