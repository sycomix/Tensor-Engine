use crate::nn::BiasFunction;
use crate::nn::MultiHeadAttention;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn debug_forward_with_distance_shapes() {
    let b = 1;
    let seq = 64;
    let d_model = 128;
    let num_heads = 8;
    let x_data: Vec<f32> = (0..(b * seq * d_model)).map(|i| i as f32 * 0.001).collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        true,
    );
    let mut dist_data: Vec<f32> = Vec::with_capacity(seq * seq);
    for i in 0..seq {
        for j in 0..seq {
            dist_data.push((i as isize - j as isize).abs() as f32);
        }
    }
    let dist = Tensor::new(
        Array::from_shape_vec((seq, seq), dist_data)
            .unwrap()
            .into_dyn(),
        false,
    );
    // Time reshape directly to check for hangs
    let tstart = std::time::Instant::now();
    let _dist_r = dist.clone().reshape(vec![1, 1, seq, seq]).unwrap();
    let tsim = tstart.elapsed();
    println!("dist reshape elapsed: {:?}", tsim);

    let mha =
        MultiHeadAttention::new_with_nl_oob(d_model, num_heads, BiasFunction::Logarithmic, 2.0);
    let start = std::time::Instant::now();
    let out = mha.forward_with_distance(&x, &dist);
    let dur = start.elapsed();
    println!("mha forward_with_distance elapsed: {:?}", dur);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d_model]);
}
