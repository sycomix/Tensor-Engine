#![cfg(feature = "with_tch")]
use tensor_engine::io::pytorch_loader::loader;
use base64::Engine;

#[test]
fn missing_file_returns_error() {
    let res = loader::load_torch_state_dict_to_map("nonexistent_file.pt", false);
    assert!(res.is_err());
    if let Err(msg) = res {
        assert!(msg.contains("Use examples/convert_torch_to_safetensors.py") || msg.contains("Failed to load via VarStore"));
    } else {
        panic!("Expected error for missing path");
    }
}

#[test]
fn cmodule_extraction_via_python_jit() {
    // This test attempts to create a TorchScript CModule via Python torch.jit and validates that
    // loader::load_torch_state_dict_to_map can extract parameters. It is a best-effort test that
    // will be skipped if Python or torch are not available on the system running tests.
    use std::process::Command;

    // Prefer using the checked-in TorchScript fixture if present (avoid relying on Python).
    let fixture_pt = "tests/assets/simple_linear.pt";
    if !std::path::Path::new(fixture_pt).exists() {
        // Try decode the baked base64 fixture file if present
        let b64file = "tests/assets/simple_linear.pt.b64";
        if std::path::Path::new(b64file).exists() {
            let encoded = std::fs::read_to_string(b64file).expect("failed to read b64 asset");
            let decoded = base64::engine::general_purpose::STANDARD.decode(&encoded).expect("failed to base64-decode fixture");
            std::fs::write(fixture_pt, &decoded).expect("failed to write decoded fixture");
        } else {
            // if baked fixture doesn't exist, try to generate using python+torch if available
            let check = Command::new("python").arg("-c").arg("import torch; print(torch.__version__)").output();
            if check.is_err() || !check.unwrap().status.success() {
                eprintln!("Skipping cmodule_extraction_via_python_jit: no fixture and python + torch not available");
                return;
            }
            let out_path = fixture_pt;
            let py_script = format!(r#"import sys
import torch
import torch.nn as nn
class Simple(nn.Module):
    def __init__(self):
        super(Simple,self).__init__()
        self.l = nn.Linear(4,2)
    def forward(self,x):
        return self.l(x)
model=Simple()
traced=torch.jit.trace(model, torch.randn(1,4))
traced.save(sys.argv[1])
"#);
            let out = Command::new("python").arg("-c").arg(py_script).arg(out_path).output();
            if out.is_err() || !out.unwrap().status.success() {
                eprintln!("Skipping cmodule_extraction_via_python_jit: failed to generate torchscript module");
                return;
            }
        }
    }
    // Now load with loader and assert parameters exist
    let res = loader::load_torch_state_dict_to_map(fixture_pt, false);
    if res.is_err() {
        let msg = res.err().unwrap();
        panic!("Expected to load generated cmodule but got error: {}", msg);
    }
    let map = res.unwrap();
    // Expect typical linear parameters: 'l.weight' and 'l.bias'
    assert!(map.get("l.weight").is_some() || map.get("linear.weight").is_some(), "weight not found in state dict");
    assert!(map.get("l.bias").is_some() || map.get("linear.bias").is_some(), "bias not found in state dict");
    // No cleanup; we used a checked-in fixture
}

#[test]
fn cmodule_nested_state_dict_extraction() {
    use std::process::Command;
    let fixture_pt = "tests/assets/simple_linear_nested.pt";
    if !std::path::Path::new(fixture_pt).exists() {
        let b64file = "tests/assets/simple_linear_nested.pt.b64";
        if std::path::Path::new(b64file).exists() {
            let encoded = std::fs::read_to_string(b64file).expect("failed to read nested b64 asset");
            let decoded = base64::engine::general_purpose::STANDARD.decode(&encoded).expect("failed to base64-decode nested fixture");
            std::fs::write(fixture_pt, &decoded).expect("failed to write decoded nested fixture");
        } else {
            let check = Command::new("python").arg("-c").arg("import torch; print(torch.__version__)").output();
            if check.is_err() || !check.unwrap().status.success() {
                eprintln!("Skipping cmodule_nested_state_dict_extraction: python + torch not available");
                return;
            }
        // Generate a model that overrides state_dict to return a nested dict
        let out_path = fixture_pt;
        let py_script = format!(r#"import sys
import torch
import torch.nn as nn
class SimpleNested(nn.Module):
    def __init__(self):
        super(SimpleNested,self).__init__()
        self.l = nn.Linear(4,2)
    def forward(self,x):
        return self.l(x)
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        return {{'nested': sd}}
model=SimpleNested()
traced=torch.jit.trace(model, torch.randn(1,4))
traced.save(sys.argv[1])
"#);
        let out = Command::new("python").arg("-c").arg(py_script).arg(out_path).output();
        if out.is_err() || !out.unwrap().status.success() {
            eprintln!("Skipping cmodule_nested_state_dict_extraction: failed to generate torchscript module");
            return;
        }
        }
    }
    let res = loader::load_torch_state_dict_to_map(fixture_pt, false);
    if res.is_err() {
        let msg = res.err().unwrap();
        panic!("Expected to load generated nested cmodule but got error: {}", msg);
    }
    let map = res.unwrap();
    assert!(map.get("nested.l.weight").is_some() || map.get("nested.l.bias").is_some(), "nested parameters not found");
}

#[test]
fn cmodule_state_dict_list_pairs_extraction() {
    use std::process::Command;
    let fixture_pt = "tests/assets/simple_linear_pairs.pt";
    if !std::path::Path::new(fixture_pt).exists() {
        let b64file = "tests/assets/simple_linear_pairs.pt.b64";
        if std::path::Path::new(b64file).exists() {
            let encoded = std::fs::read_to_string(b64file).expect("failed to read pairs b64 asset");
            let decoded = base64::engine::general_purpose::STANDARD.decode(&encoded).expect("failed to base64-decode pairs fixture");
            std::fs::write(fixture_pt, &decoded).expect("failed to write decoded pairs fixture");
        } else {
            let check = Command::new("python").arg("-c").arg("import torch; print(torch.__version__)").output();
            if check.is_err() || !check.unwrap().status.success() {
                eprintln!("Skipping cmodule_state_dict_list_pairs_extraction: python + torch not available");
                return;
            }
        // Generate a model that returns a list of (name, tensor) pairs for state_dict
        let out_path = fixture_pt;
        let py_script = format!(r#"import sys
import torch
import torch.nn as nn
class PairState(nn.Module):
    def __init__(self):
        super(PairState,self).__init__()
        self.l = nn.Linear(4,2)
    def forward(self,x):
        return self.l(x)
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        return list(sd.items())
model=PairState()
traced=torch.jit.trace(model, torch.randn(1,4))
traced.save(sys.argv[1])
"#);
        let out = Command::new("python").arg("-c").arg(py_script).arg(out_path).output();
        if out.is_err() || !out.unwrap().status.success() {
            eprintln!("Skipping cmodule_state_dict_list_pairs_extraction: failed to generate torchscript module");
            return;
        }
        }
    }
    let res = loader::load_torch_state_dict_to_map(fixture_pt, false);
    if res.is_err() {
        let msg = res.err().unwrap();
        panic!("Expected to load generated pairs cmodule but got error: {}", msg);
    }
    let map = res.unwrap();
    assert!(map.get("l.weight").is_some() || map.get("linear.weight").is_some(), "weight not found in state dict list pairs");
}

#[test]
fn cmodule_state_dict_hashmap_extraction() {
    use std::process::Command;
    let fixture_pt = "tests/assets/simple_linear_hashmap.pt";
    if !std::path::Path::new(fixture_pt).exists() {
        let b64file = "tests/assets/simple_linear_hashmap.pt.b64";
        if std::path::Path::new(b64file).exists() {
            let encoded = std::fs::read_to_string(b64file).expect("failed to read hashmap b64 asset");
            let decoded = base64::engine::general_purpose::STANDARD.decode(&encoded).expect("failed to base64-decode hashmap fixture");
            std::fs::write(fixture_pt, &decoded).expect("failed to write decoded hashmap fixture");
        } else {
            let check = Command::new("python").arg("-c").arg("import torch; print(torch.__version__)").output();
            if check.is_err() || !check.unwrap().status.success() {
                eprintln!("Skipping cmodule_state_dict_hashmap_extraction: python + torch not available");
                return;
            }
        // Generate a model that returns a dict with aliased keys (hash-like structure)
        let out_path = fixture_pt;
        let py_script = format!(r#"import sys
import torch
import torch.nn as nn
class HashMapState(nn.Module):
    def __init__(self):
        super(HashMapState,self).__init__()
        self.l = nn.Linear(4,2)
    def forward(self,x):
        return self.l(x)
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        return {{'a.weight': sd['l.weight'], 'a.bias': sd['l.bias']}}
model=HashMapState()
traced=torch.jit.trace(model, torch.randn(1,4))
traced.save(sys.argv[1])
"#);
        let out = Command::new("python").arg("-c").arg(py_script).arg(out_path).output();
        if out.is_err() || !out.unwrap().status.success() {
            eprintln!("Skipping cmodule_state_dict_hashmap_extraction: failed to generate torchscript module");
            return;
        }
        }
    }
    let res = loader::load_torch_state_dict_to_map(fixture_pt, false);
    if res.is_err() {
        let msg = res.err().unwrap();
        panic!("Expected to load generated hashmap cmodule but got error: {}", msg);
    }
    let map = res.unwrap();
    assert!(map.get("a.weight").is_some() || map.get("a.bias").is_some(), "aliased hashmap parameters not found");
}
