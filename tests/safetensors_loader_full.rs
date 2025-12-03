#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use safetensors::tensor::serialize as st_serialize;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use safetensors::tensor::Dtype as STDtype;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use safetensors::tensor::TensorView as STTensorView;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use tensor_engine::io::safetensors_loader::{
    apply_state_dict_to_module, load_safetensors_from_bytes,
};
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use tensor_engine::nn::MultiHeadAttention;

#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
#[test]
fn test_safetensors_full_state_dict_load() {
    use std::collections::HashMap;
    // We'll create a tiny state dict for a single linear weight
    // We create a MultiHeadAttention and set weight to zeros, then load a safe tensor
    let mut mha = MultiHeadAttention::new(4, 1);
    // create raw bytes for a 2D weight shape (out, in) 4x4
    let shape = vec![4usize, 4usize];
    let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let mut bytes: Vec<u8> = Vec::new();
    for f in data.iter() {
        bytes.extend(&f.to_le_bytes());
    }
    // Build safetensors map
    let mut tensors = HashMap::new();
    let st_view = STTensorView::new(STDtype::F32, shape.clone(), &bytes).unwrap();
    tensors.insert("mha.linear_q.weight".to_string(), st_view);
    // Serialize using safetensors serialize API
    let bytes_st = st_serialize(&tensors, None).unwrap();
    // Deserialize using our loader
    let map = load_safetensors_from_bytes(&bytes_st, true).unwrap();
    // Apply to mha with root prefix 'mha'
    apply_state_dict_to_module(&mut mha, &map, "mha").unwrap();
    // Verify that linear_q weight equals loaded data
    let w = mha.linear_q.weight.lock().storage.to_f32_array();
    let loaded = map
        .get("mha.linear_q.weight")
        .unwrap()
        .lock()
        .storage
        .to_f32_array();
    assert_eq!(w, loaded);
}
