use ndarray::{arr0, arr1, arr2, ArrayD};
use tensor_engine::tensor::Tensor;

fn assert_values(actual: &ArrayD<f32>, expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(*actual, *expected, "mismatch at flat index {index}");
    }
}

#[test]
fn where_select_broadcasts_and_reduces_branch_gradients() {
    let condition = Tensor::new(arr2(&[[1.0], [0.0]]).into_dyn(), true);
    let x = Tensor::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn(), true);
    let y = Tensor::new(arr0(10.0).into_dyn(), true);
    let output = Tensor::where_select(&condition, &x, &y);

    assert_values(&output.to_f32_array(), &[1.0, 2.0, 3.0, 10.0, 10.0, 10.0]);
    output.sum().backward();
    assert_values(condition.lock().grad.as_ref().unwrap(), &[0.0, 0.0]);
    assert_values(x.lock().grad.as_ref().unwrap(), &[1.0, 1.0, 1.0]);
    assert_values(y.lock().grad.as_ref().unwrap(), &[3.0]);
}

#[test]
fn masked_fill_routes_gradient_only_through_unmasked_values() {
    let input = Tensor::new(arr1(&[1.0, 2.0, 3.0, 4.0]).into_dyn(), true);
    let mask = Tensor::new(arr1(&[0.0, 1.0, 0.0, 1.0]).into_dyn(), false);
    let output = input.masked_fill(&mask, -5.0);

    assert_values(&output.to_f32_array(), &[1.0, -5.0, 3.0, -5.0]);
    output.sum().backward();
    assert_values(input.lock().grad.as_ref().unwrap(), &[1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn masked_scatter_routes_row_major_source_and_base_gradients() {
    let base = Tensor::new(arr1(&[1.0, 2.0, 3.0, 4.0]).into_dyn(), true);
    let mask = Tensor::new(arr1(&[0.0, 1.0, 0.0, 1.0]).into_dyn(), false);
    let source = Tensor::new(arr1(&[10.0, 20.0, 30.0]).into_dyn(), true);
    let output = base.masked_scatter(&mask, &source);

    assert_values(&output.to_f32_array(), &[1.0, 10.0, 3.0, 20.0]);
    output.sum().backward();
    assert_values(base.lock().grad.as_ref().unwrap(), &[1.0, 0.0, 1.0, 0.0]);
    assert_values(source.lock().grad.as_ref().unwrap(), &[1.0, 1.0, 0.0]);
}

#[test]
fn index_select_and_gather_accumulate_repeated_index_gradients() {
    let selected_input = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(), true);
    let selected_indices = Tensor::new(arr1(&[2.0, 2.0, 0.0]).into_dyn(), false);
    let selected = selected_input.index_select(1, &selected_indices);
    assert_values(&selected.to_f32_array(), &[3.0, 3.0, 1.0, 6.0, 6.0, 4.0]);
    selected.sum().backward();
    assert_values(
        selected_input.lock().grad.as_ref().unwrap(),
        &[1.0, 0.0, 2.0, 1.0, 0.0, 2.0],
    );

    let gathered_input = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(), true);
    let gather_indices = Tensor::new(arr2(&[[2.0, 2.0], [0.0, 1.0]]).into_dyn(), false);
    let gathered = gathered_input.gather(1, &gather_indices);
    assert_values(&gathered.to_f32_array(), &[3.0, 3.0, 4.0, 5.0]);
    gathered.sum().backward();
    assert_values(
        gathered_input.lock().grad.as_ref().unwrap(),
        &[0.0, 0.0, 2.0, 1.0, 1.0, 0.0],
    );
}

#[test]
fn scatter_last_write_wins_in_forward_and_backward() {
    let base = Tensor::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn(), true);
    let index = Tensor::new(arr2(&[[1.0, 1.0, 2.0]]).into_dyn(), false);
    let source = Tensor::new(arr2(&[[10.0, 20.0, 30.0]]).into_dyn(), true);
    let output = base.scatter(1, &index, &source);
    let weights = Tensor::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn(), false);

    assert_values(&output.to_f32_array(), &[1.0, 20.0, 30.0]);
    output.mul(&weights).sum().backward();
    assert_values(base.lock().grad.as_ref().unwrap(), &[1.0, 0.0, 0.0]);
    assert_values(source.lock().grad.as_ref().unwrap(), &[0.0, 2.0, 3.0]);
}

#[test]
fn scatter_add_routes_gradient_to_every_additive_source() {
    let base = Tensor::new(arr2(&[[0.0, 0.0, 0.0]]).into_dyn(), true);
    let index = Tensor::new(arr2(&[[1.0, 1.0, 2.0]]).into_dyn(), false);
    let source = Tensor::new(arr2(&[[10.0, 20.0, 30.0]]).into_dyn(), true);
    let output = base.scatter_add(1, &index, &source);
    let weights = Tensor::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn(), false);

    assert_values(&output.to_f32_array(), &[0.0, 30.0, 30.0]);
    output.mul(&weights).sum().backward();
    assert_values(base.lock().grad.as_ref().unwrap(), &[1.0, 2.0, 3.0]);
    assert_values(source.lock().grad.as_ref().unwrap(), &[2.0, 2.0, 3.0]);
}

#[test]
#[should_panic(expected = "gather indices must be finite non-negative integers")]
fn gather_rejects_fractional_indices() {
    let input = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), false);
    let index = Tensor::new(arr1(&[0.5]).into_dyn(), false);
    let _ = input.gather(0, &index);
}
