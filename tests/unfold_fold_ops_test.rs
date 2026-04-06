use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

#[test]
fn test_unfold2d_forward_values() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3, 3][..]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap(),
        false,
    );

    let y = x.unfold2d(2, 2, 1, 0);
    let ya = y.to_f32_array();
    assert_eq!(ya.shape(), &[1, 4, 4]);

    // Columns (l=0..3): [1,2,4,5], [2,3,5,6], [4,5,7,8], [5,6,8,9]
    assert!((ya[[0, 0, 0]] - 1.0).abs() < 1e-6);
    assert!((ya[[0, 1, 0]] - 2.0).abs() < 1e-6);
    assert!((ya[[0, 2, 0]] - 4.0).abs() < 1e-6);
    assert!((ya[[0, 3, 0]] - 5.0).abs() < 1e-6);

    assert!((ya[[0, 0, 3]] - 5.0).abs() < 1e-6);
    assert!((ya[[0, 1, 3]] - 6.0).abs() < 1e-6);
    assert!((ya[[0, 2, 3]] - 8.0).abs() < 1e-6);
    assert!((ya[[0, 3, 3]] - 9.0).abs() < 1e-6);
}

#[test]
fn test_unfold2d_backward_patch_counts() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3, 3][..]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap(),
        true,
    );

    let y = x.unfold2d(2, 2, 1, 0);
    let loss = y.sum();
    loss.backward();

    let gx = x.lock().grad.clone().unwrap();
    let s = gx.as_slice().unwrap();

    let expected = [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
    for i in 0..9 {
        assert!((s[i] - expected[i]).abs() < 1e-6);
    }
}

#[test]
fn test_fold2d_forward_no_overlap_roundtrip() {
    let x = Tensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 4, 4][..]),
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
        )
        .unwrap(),
        false,
    );

    let cols = x.unfold2d(2, 2, 2, 0);
    let recon = cols.fold2d(4, 4, 2, 2, 2, 0);
    let ra = recon.to_f32_array();
    let rs = ra.as_slice().unwrap();
    let xs = x.to_f32_array();
    let xsv = xs.as_slice().unwrap();

    for i in 0..16 {
        assert!((rs[i] - xsv[i]).abs() < 1e-6);
    }
}

#[test]
fn test_fold2d_backward_simple() {
    let cols = Tensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 4, 4][..]),
            vec![
                1.0, 2.0, 5.0, 6.0,
                3.0, 4.0, 7.0, 8.0,
                9.0, 10.0, 13.0, 14.0,
                11.0, 12.0, 15.0, 16.0,
            ],
        )
        .unwrap(),
        true,
    );

    let out = cols.fold2d(4, 4, 2, 2, 2, 0);
    let loss = out.sum();
    loss.backward();

    let gcols = cols.lock().grad.clone().unwrap();
    let s = gcols.as_slice().unwrap();
    for &v in s {
        assert!((v - 1.0).abs() < 1e-6);
    }
}
