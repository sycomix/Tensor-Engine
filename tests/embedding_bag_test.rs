use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

#[test]
fn test_embedding_bag_forward_sum_mode() {
    // emb shape [5, 3]
    let emb = Tensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[5, 3][..]),
            vec![
                1.0, 1.0, 1.0, // 0
                2.0, 2.0, 2.0, // 1
                3.0, 3.0, 3.0, // 2
                4.0, 4.0, 4.0, // 3
                5.0, 5.0, 5.0, // 4
            ],
        )
        .unwrap(),
        false,
    );

    // bags: [1,2], [3], [0,4]
    let indices = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[5][..]), vec![1.0, 2.0, 3.0, 0.0, 4.0]).unwrap(),
        false,
    );
    let offsets = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![0.0, 2.0, 3.0]).unwrap(),
        false,
    );

    let out = Tensor::embedding_bag(&emb, &indices, &offsets);
    let oa = out.to_f32_array();
    assert_eq!(oa.shape(), &[3, 3]);
    let s = oa.as_slice().unwrap();

    // [1,2] -> [5,5,5]
    assert!((s[0] - 5.0).abs() < 1e-6);
    assert!((s[1] - 5.0).abs() < 1e-6);
    assert!((s[2] - 5.0).abs() < 1e-6);
    // [3] -> [4,4,4]
    assert!((s[3] - 4.0).abs() < 1e-6);
    assert!((s[4] - 4.0).abs() < 1e-6);
    assert!((s[5] - 4.0).abs() < 1e-6);
    // [0,4] -> [6,6,6]
    assert!((s[6] - 6.0).abs() < 1e-6);
    assert!((s[7] - 6.0).abs() < 1e-6);
    assert!((s[8] - 6.0).abs() < 1e-6);
}

#[test]
fn test_embedding_bag_backward_accumulates_grad_to_embeddings() {
    let emb = Tensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[5, 2][..]),
            vec![
                1.0, 1.0, // 0
                2.0, 2.0, // 1
                3.0, 3.0, // 2
                4.0, 4.0, // 3
                5.0, 5.0, // 4
            ],
        )
        .unwrap(),
        true,
    );

    // bags: [1,2], [3], [0,4]
    let indices = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[5][..]), vec![1.0, 2.0, 3.0, 0.0, 4.0]).unwrap(),
        false,
    );
    let offsets = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![0.0, 2.0, 3.0]).unwrap(),
        false,
    );

    let out = Tensor::embedding_bag(&emb, &indices, &offsets);
    let loss = out.sum();
    loss.backward();

    let g = emb.lock().grad.clone().unwrap();
    let s = g.as_slice().unwrap();

    // each selected row appears once, each feature gets +1 from loss=sum(out)
    // row 0
    assert!((s[0] - 1.0).abs() < 1e-6);
    assert!((s[1] - 1.0).abs() < 1e-6);
    // row 1
    assert!((s[2] - 1.0).abs() < 1e-6);
    assert!((s[3] - 1.0).abs() < 1e-6);
    // row 2
    assert!((s[4] - 1.0).abs() < 1e-6);
    assert!((s[5] - 1.0).abs() < 1e-6);
    // row 3
    assert!((s[6] - 1.0).abs() < 1e-6);
    assert!((s[7] - 1.0).abs() < 1e-6);
    // row 4
    assert!((s[8] - 1.0).abs() < 1e-6);
    assert!((s[9] - 1.0).abs() < 1e-6);
}
