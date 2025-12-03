use ndarray::IxDyn;
use tensor_engine::nn::{Linear, Module, Sequential};
use tensor_engine::tensor::Tensor;

use std::collections::HashMap;

#[test]
fn test_default_load_state_dict_applies_named_parameters() {
    // Build a Sequential with a single Linear
    let lin = Linear::new(2, 3, true);
    let mut seq = Sequential::new().add(lin);

    // Create new weight and bias tensors with known values
    let weight_arr = ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32; 6]).unwrap();
    let bias_arr = ndarray::Array::from_shape_vec(IxDyn(&[3]), vec![0.5f32; 3]).unwrap();
    let w = Tensor::new(weight_arr.into_dyn(), false);
    let b = Tensor::new(bias_arr.into_dyn(), false);

    // Build state dict using default named parameter keys: `seqparam0`, `seqparam1` ...
    let names = seq.named_parameters("seq");
    assert!(
        names.len() >= 2,
        "Sequential expected at least two parameters"
    );
    let mut state: HashMap<String, Tensor> = HashMap::new();
    // Map first param to weight and second param to bias
    state.insert(names[0].0.clone(), w.clone());
    state.insert(names[1].0.clone(), b.clone());

    // Apply state dict using module's default load_state_dict (should use named_parameters)
    seq.load_state_dict(&state, "seq").unwrap();

    // Now check that named parameters were updated
    let after = seq.named_parameters("seq");
    let p0 = after[0].1.lock().storage.to_f32_array();
    let p1 = after[1].1.lock().storage.to_f32_array();
    assert_eq!(p0, w.lock().storage.to_f32_array());
    assert_eq!(p1, b.lock().storage.to_f32_array());
}
