use ndarray::{ArrayD, IxDyn};
use tensor_engine::tensor::Tensor;

fn tensor(shape: &[usize], values: Vec<f32>, requires_grad: bool) -> Tensor {
    Tensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
        requires_grad,
    )
}

fn assert_values(actual: &Tensor, expected: &[f32]) {
    let values = actual.to_f32_array();
    assert_eq!(values.len(), expected.len());
    for (index, (&actual, &expected)) in values.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn embedding_lookup_supports_multidimensional_indices_and_accumulates_repeats() {
    let embedding = tensor(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], true);
    let indices = tensor(&[2, 2], vec![2.0, 0.0, 2.0, 1.0], true);

    let output = Tensor::embedding_lookup(&embedding, &indices);
    assert_eq!(output.shape(), vec![2, 2, 2]);
    assert_values(&output, &[5.0, 6.0, 1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);

    let weights = tensor(
        &[2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        false,
    );
    output.mul(&weights).sum().backward();

    let embedding_grad = embedding.lock().grad.clone().unwrap();
    assert_eq!(
        embedding_grad.as_slice().unwrap(),
        &[3.0, 4.0, 7.0, 8.0, 6.0, 8.0]
    );
    let indices_grad = indices.lock().grad.clone().unwrap();
    assert!(indices_grad.iter().all(|&value| value == 0.0));
}

#[test]
fn embedding_bag_accumulates_repeated_rows_and_has_zero_metadata_gradients() {
    let embedding = tensor(&[4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
    let indices = tensor(&[5], vec![1.0, 1.0, 2.0, 3.0, 1.0], true);
    let offsets = tensor(&[2], vec![0.0, 3.0], true);
    let output = Tensor::embedding_bag(&embedding, &indices, &offsets);
    assert_values(&output, &[11.0, 14.0, 10.0, 12.0]);

    let weights = tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0], false);
    output.mul(&weights).sum().backward();

    let embedding_grad = embedding.lock().grad.clone().unwrap();
    assert_eq!(
        embedding_grad.as_slice().unwrap(),
        &[0.0, 0.0, 5.0, 8.0, 1.0, 2.0, 3.0, 4.0]
    );
    assert!(indices
        .lock()
        .grad
        .clone()
        .unwrap()
        .iter()
        .all(|&value| value == 0.0));
    assert!(offsets
        .lock()
        .grad
        .clone()
        .unwrap()
        .iter()
        .all(|&value| value == 0.0));
}

#[test]
fn unfold_padding_and_fold_overlap_have_exact_gradients() {
    let input = tensor(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0], true);
    let columns = input.unfold2d(2, 2, 1, 1);
    assert_eq!(columns.shape(), vec![1, 4, 9]);
    assert_values(
        &columns,
        &[
            0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0,
            0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0,
            0.0, 0.0,
        ],
    );
    columns.sum().backward();
    assert_eq!(
        input.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[4.0, 4.0, 4.0, 4.0]
    );

    let columns = tensor(&[1, 4, 4], (1..=16).map(|v| v as f32).collect(), true);
    let folded = columns.fold2d(3, 3, 2, 2, 1, 0);
    assert_values(
        &folded,
        &[1.0, 7.0, 6.0, 12.0, 34.0, 22.0, 11.0, 27.0, 16.0],
    );
    let weights = tensor(
        &[1, 1, 3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        false,
    );
    folded.mul(&weights).sum().backward();
    assert_eq!(
        columns.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[1.0, 2.0, 4.0, 5.0, 2.0, 3.0, 5.0, 6.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0]
    );
}

#[test]
fn kvcache_append_splits_weighted_gradient_on_nonleading_axis() {
    let cache = tensor(&[1, 2, 2], vec![1.0, 2.0, 3.0, 4.0], true);
    let new_values = tensor(&[1, 1, 2], vec![5.0, 6.0], true);
    let output = Tensor::kvcache_append(&cache, &new_values, 1);
    assert_eq!(output.shape(), vec![1, 3, 2]);
    assert_values(&output, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let weights = tensor(&[1, 3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false);
    output.mul(&weights).sum().backward();
    assert_eq!(
        cache.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        new_values.lock().grad.clone().unwrap().as_slice().unwrap(),
        &[5.0, 6.0]
    );
}

#[test]
fn structural_ops_reject_invalid_public_inputs() {
    let embedding = tensor(&[2, 2], vec![1.0; 4], false);
    let fractional = tensor(&[1], vec![0.5], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Tensor::embedding_lookup(&embedding, &fractional)
    }))
    .is_err());

    let image = tensor(&[1, 1, 2, 2], vec![1.0; 4], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        image.unfold2d(3, 3, 1, 0)
    }))
    .is_err());

    let columns = tensor(&[1, 4, 3], vec![1.0; 12], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        columns.fold2d(3, 3, 2, 2, 1, 0)
    }))
    .is_err());

    let cache = tensor(&[1, 2, 2], vec![1.0; 4], false);
    let incompatible = tensor(&[2, 1, 2], vec![1.0; 4], false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Tensor::kvcache_append(&cache, &incompatible, 1)
    }))
    .is_err());
}
