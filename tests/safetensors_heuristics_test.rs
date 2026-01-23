use std::collections::HashMap;
use tensor_engine::tensor::Tensor;
use tensor_engine::nn::Module;
use tensor_engine::io::safetensors_loader::apply_state_dict_to_module;
use std::any::Any;

// Mock Module to simulate Llama structure
struct MockLlamaLayer {
    pub self_attn_q: Tensor, // "self_attn.q_proj.weight"
    pub mlp_gate: Tensor,    // "mlp.gate_proj.weight"
    pub qweight: Tensor,     // "linear.qweight"
    pub qzeros: Tensor,      // "linear.qzeros"
    pub scales: Tensor,      // "linear.scales"
}

impl MockLlamaLayer {
    fn new() -> Self {
        MockLlamaLayer {
            self_attn_q: Tensor::zeros(&[10, 10]),
            mlp_gate: Tensor::zeros(&[10, 10]),
            qweight: Tensor::zeros(&[10, 10]),
            qzeros: Tensor::zeros(&[10, 10]),
            scales: Tensor::zeros(&[10, 10]),
        }
    }
}

impl Module for MockLlamaLayer {
    fn forward(&self, _: &Tensor) -> Tensor { Tensor::zeros(&[1]) }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (format!("{}.self_attn.q_proj.weight", prefix), self.self_attn_q.clone()),
            (format!("{}.mlp.gate_proj.weight", prefix), self.mlp_gate.clone()),
            (format!("{}.linear.qweight", prefix), self.qweight.clone()),
            (format!("{}.linear.qzeros", prefix), self.qzeros.clone()),
            (format!("{}.linear.scales", prefix), self.scales.clone()),
        ]
    }
    // Minimal impl
    fn load_state_dict(&mut self, _: &HashMap<String, Tensor>, _: &str) -> Result<(), String> {
        // Return Ok to proceed to fallback logic in apply_state_dict_to_module
        Ok(())
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

struct MockModel {
    layers: Vec<MockLlamaLayer>,
}

impl MockModel {
    fn new() -> Self {
        MockModel {
            layers: vec![MockLlamaLayer::new(), MockLlamaLayer::new()],
        }
    }
}

impl Module for MockModel {
    fn forward(&self, _: &Tensor) -> Tensor { Tensor::zeros(&[1]) }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut p = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            p.extend(layer.named_parameters(&format!("{}.layers.{}", prefix, i)));
        }
        p
    }
    fn load_state_dict(&mut self, _: &HashMap<String, Tensor>, _: &str) -> Result<(), String> { Ok(()) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

#[test]
fn test_prefix_discovery_heuristics() {
    let mut model = MockModel::new();
    let mut state_dict = HashMap::new();

    // Create a mismatch:
    // Module expects: "model.layers.0.self_attn.q_proj.weight"
    // Checkpoint has: "layers.0.self_attn.q_proj.weight" (Prefix "layers.0" vs "model.layers.0")
    
    // 1. Fill state dict with "short" keys
    let t = Tensor::ones(&[10, 10]);
    state_dict.insert("layers.0.self_attn.q_proj.weight".to_string(), t.clone());
    
    // 2. Also put qweight/qzeros/scales with that same short prefix
    state_dict.insert("layers.0.linear.qweight".to_string(), t.clone());
    state_dict.insert("layers.0.linear.qzeros".to_string(), t.clone());
    state_dict.insert("layers.0.linear.scales".to_string(), t.clone());

    // 3. And for layer 1
    state_dict.insert("layers.1.self_attn.q_proj.weight".to_string(), t.clone());
    state_dict.insert("layers.1.linear.qweight".to_string(), t.clone());
    // Note: Missing qzeros/scales for layer 1 to test diagnostics? Or just test they don't crash.

    // Apply
    // We pass root "model" to simulate loading "model" substructure
    apply_state_dict_to_module(&mut model, &state_dict, "model").expect("Apply failed");

    // Verify assignment by checking if tensors are Ones (modified) instead of Zeros (init)
    let p0 = model.layers[0].self_attn_q.lock().storage.to_f32_array();
    assert_eq!(p0[[0,0]], 1.0, "Layer 0 self_attn_q should be assigned via heuristics");
    
    let p0_qw = model.layers[0].qweight.lock().storage.to_f32_array();
    assert_eq!(p0_qw[[0,0]], 1.0, "Layer 0 qweight should be assigned via heuristics");

    let p1 = model.layers[1].self_attn_q.lock().storage.to_f32_array();
    assert_eq!(p1[[0,0]], 1.0, "Layer 1 self_attn_q should be assigned via heuristics (learned prefix)");
}
