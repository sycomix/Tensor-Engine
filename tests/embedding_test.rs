use ndarray::{ArrayD, IxDyn};
use tensor_engine::nn::{
    embedding::{AdaptiveEmbedding, SparseEmbedding},
    Module,
};
use tensor_engine::tensor::Tensor;

#[test]
fn test_sparse_embedding_forward() {
    let emb = SparseEmbedding::new(10, 4);
    // Indices: [1, 3]
    let data = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 3.0]).unwrap();
    let x = Tensor::new(data, true);

    let out = emb.forward(&x);
    let out_shape = out.to_f32_array().shape().to_vec();
    assert_eq!(out_shape, vec![2, 4]);

    // Check backward
    let loss = out.sum();
    loss.backward();

    // Check gradients on weight
    assert!(emb.weight.lock().grad.is_some());
}

#[test]
fn test_adaptive_embedding_forward() {
    // Vocab 20. Cutoffs [10]. d_model = 8. div = 2.0.
    // Head: 0..10 -> dim 8.
    // Tail: 10..20 -> dim 4 -> proj 8.

    let adap = AdaptiveEmbedding::new(20, 8, vec![10], 2.0);

    // Input indices: mixture of head (2, 5) and tail (12, 15)
    // [2, 12, 5, 15]
    let data = ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0, 12.0, 5.0, 15.0]).unwrap();
    let x = Tensor::new(data, true);

    let out = adap.forward(&x);
    let out_shape = out.to_f32_array().shape().to_vec();
    assert_eq!(out_shape, vec![4, 8]);

    // Check gradients
    let loss = out.sum();
    loss.backward();

    // Head grads
    assert!(adap.head.weight.lock().grad.is_some());
    // Tail grads
    assert!(adap.tail[0].0.weight.lock().grad.is_some()); // tail emb
                                                          // Check projection grads via parameters
    let proj_params = adap.tail[0].1.parameters();
    assert!(proj_params.iter().any(|p| p.lock().grad.is_some()));
}
