use tensor_engine::io::pytorch_loader;

#[test]
fn missing_file_returns_error() {
    let res = pytorch_loader::load_torch_state_dict_to_map("nonexistent_file.pt", false);
    assert!(res.is_err());
    if let Err(msg) = res {
        assert!(!msg.is_empty(), "Error message should not be empty");
    } else {
        panic!("Expected error for missing path");
    }
}

#[test]
fn safetensors_fallback_returns_error() {
    // Path without .pt extension — no fallback possible
    let res = pytorch_loader::load_torch_state_dict_to_map("nonexistent_file", false);
    assert!(res.is_err());
}
