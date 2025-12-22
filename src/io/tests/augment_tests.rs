use crate::io::safetensors_loader::augment_state_dict_for_compat;
use crate::tensor::Tensor;
use ndarray::ArrayD;
use std::collections::HashMap;

#[test]
fn augment_state_dict_handles_missing_up_proj() {
    let mut map: HashMap<String, Tensor> = HashMap::new();
    // create a dummy gate tensor
    let gate = Tensor::new(ndarray::Array::from_elem((2, 2), 1.0).into_dyn(), false);
    map.insert("layer.mlp.gate_proj.weight".to_string(), gate);
    // call augment and ensure it doesn't panic and returns Ok
    let res = augment_state_dict_for_compat(map);
    assert!(res.is_ok());
    let out = res.unwrap();
    // since up_proj was missing, linear1.weight should not have been inserted
    assert!(!out.contains_key("layer.linear1.weight"));
}
