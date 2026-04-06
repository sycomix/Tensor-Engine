use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

#[test]
fn test_fft_forward_delta_signal() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4][..]), vec![1.0, 0.0, 0.0, 0.0]).unwrap(),
        false,
    );

    let y = x.fft();
    let ya = y.to_f32_array();

    assert_eq!(ya.shape(), &[4, 2]);
    let s = ya.as_slice().unwrap();
    for k in 0..4 {
        assert!((s[k * 2] - 1.0).abs() < 1e-5);
        assert!((s[k * 2 + 1] - 0.0).abs() < 1e-5);
    }
}

#[test]
fn test_fft_ifft_roundtrip_real_signal() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4][..]), vec![0.5, -1.0, 2.0, 3.5]).unwrap(),
        false,
    );

    let y = x.fft();
    let z = y.ifft();
    let za = z.to_f32_array();
    let zs = za.as_slice().unwrap();
    let xs = x.to_f32_array();
    let xsv = xs.as_slice().unwrap();

    for i in 0..4 {
        assert!((zs[i] - xsv[i]).abs() < 1e-4);
    }
}

#[test]
fn test_fft_backward_runs_and_shapes_match() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4][..]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        true,
    );

    let y = x.fft();
    let loss = y.sum();
    loss.backward();

    let gx = x.lock().grad.clone().unwrap();
    assert_eq!(gx.shape(), &[4]);
}

#[test]
fn test_rfft_forward_shape_and_basic_values() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4][..]), vec![1.0, 0.0, 0.0, 0.0]).unwrap(),
        false,
    );

    let y = x.rfft();
    let ya = y.to_f32_array();
    assert_eq!(ya.shape(), &[3, 2]);

    let s = ya.as_slice().unwrap();
    assert!((s[0] - 1.0).abs() < 1e-5);
    assert!((s[1] - 0.0).abs() < 1e-5);
    assert!((s[2] - 1.0).abs() < 1e-5);
    assert!((s[3] - 0.0).abs() < 1e-5);
    assert!((s[4] - 1.0).abs() < 1e-5);
    assert!((s[5] - 0.0).abs() < 1e-5);
}

#[test]
fn test_rfft_irfft_roundtrip_real_signal() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4][..]), vec![0.5, -1.0, 2.0, 3.5]).unwrap(),
        false,
    );

    let y = x.rfft();
    let z = y.irfft();
    let za = z.to_f32_array();
    let zs = za.as_slice().unwrap();
    let xs = x.to_f32_array();
    let xsv = xs.as_slice().unwrap();

    for i in 0..4 {
        assert!((zs[i] - xsv[i]).abs() < 1e-4);
    }
}
