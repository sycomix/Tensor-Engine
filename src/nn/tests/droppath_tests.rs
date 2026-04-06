use crate::nn::{DropPath, Module};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

#[test]
fn droppath_eval_mode_is_identity() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        false,
    );

    let layer = DropPath::new(0.5, false);
    let y = layer.forward(&x);

    let xs = x.to_f32_array();
    let ys = y.to_f32_array();
    assert_eq!(xs.shape(), ys.shape());
    let xv = xs.as_slice().unwrap();
    let yv = ys.as_slice().unwrap();
    for i in 0..xv.len() {
        assert!((xv[i] - yv[i]).abs() < 1e-6);
    }
}

#[test]
fn droppath_training_drops_or_scales_per_sample() {
    let x = Tensor::new(ArrayD::from_elem(IxDyn(&[4, 3, 2][..]), 1.0), false);

    let layer = DropPath::new(0.5, true);
    let y = layer.forward(&x);
    let ya = y.to_f32_array();

    let sample_size = 3 * 2;
    let s = ya.as_slice().unwrap();

    for b in 0..4 {
        let start = b * sample_size;
        let first = s[start];
        // For p=0.5 and input=1, each sample should be all 0.0 or all 2.0.
        assert!(first.abs() < 1e-6 || (first - 2.0).abs() < 1e-6);
        for i in 1..sample_size {
            assert!((s[start + i] - first).abs() < 1e-6);
        }
    }
}

#[test]
fn droppath_zero_probability_is_identity_even_training() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![0.2, -0.3, 1.2, 2.4]).unwrap(),
        false,
    );

    let layer = DropPath::new(0.0, true);
    let y = layer.forward(&x);

    let xs = x.to_f32_array();
    let ys = y.to_f32_array();
    let xv = xs.as_slice().unwrap();
    let yv = ys.as_slice().unwrap();
    for i in 0..xv.len() {
        assert!((xv[i] - yv[i]).abs() < 1e-6);
    }
}
