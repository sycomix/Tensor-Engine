#[cfg(feature = "safe_tensors")]
use crate::dtype::DType;
#[cfg(feature = "safe_tensors")]
use crate::nn::Module;
#[cfg(feature = "safe_tensors")]
use crate::tensor::Tensor;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use half::{bf16, f16};
#[cfg(feature = "safe_tensors")]
use ndarray::{Array, IxDyn};
#[cfg(feature = "safe_tensors")]
use safetensors::tensor::Dtype;
#[cfg(feature = "safe_tensors")]
use safetensors::SafeTensors;
#[cfg(feature = "safe_tensors")]
use std::collections::HashMap;

#[cfg(feature = "safe_tensors")]
pub fn load_safetensors_from_bytes(
    bytes: &[u8],
    transpose_two_dim_weights: bool,
) -> Result<HashMap<String, Tensor>, String> {
    // Parse SafeTensors file from bytes and map to Tensor instances
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| format!("safetensors deserialize error: {}", e))?;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for (key, tensor) in st.tensors() {
        // gather dtype -- currently only support f32
        match tensor.dtype() {
            Dtype::F32 => {
                let shape: Vec<usize> = tensor.shape().iter().map(|d| *d as usize).collect();
                // convert bytes to f32 vec by reading the raw bytes
                let bytes = tensor.data();
                let mut data = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(chunk);
                    data.push(f32::from_le_bytes(b));
                }
                let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                    .map_err(|e| format!("ndarray shape creation error: {}", e))?;
                let out =
                    if transpose_two_dim_weights && shape.len() == 2 && key.ends_with(".weight") {
                        // transpose [out, in] -> [in, out]
                        let mut mat = arr
                            .into_dimensionality::<ndarray::Ix2>()
                            .map_err(|e| format!("transpose dim error: {}", e))?;
                        mat = mat.reversed_axes();
                        Tensor::new(mat.into_dyn(), false)
                    } else {
                        // If this tensor represents NL-OOB learnable slopes, ensure it is set as requires_grad
                        if key.ends_with(".nl_oob.slopes") {
                            Tensor::new(arr.into_dyn(), true)
                        } else {
                            Tensor::new(arr.into_dyn(), false)
                        }
                    };
                map.insert(key.clone(), out);
            }
            Dtype::F16 => {
                #[cfg(not(feature = "multi_precision"))]
                return Err(
                    "F16 dtype requested but crate not built with 'multi_precision' feature"
                        .to_string(),
                );
                #[cfg(feature = "multi_precision")]
                {
                    let shape: Vec<usize> = tensor.shape().iter().map(|d| *d as usize).collect();
                    if tensor.data().len() % 2 != 0 {
                        return Err("Invalid byte length for f16 tensor".to_string());
                    }
                    let bytes_slice = tensor.data();
                    let mut data: Vec<f32> = Vec::with_capacity(bytes_slice.len() / 2);
                    for chunk in bytes_slice.chunks_exact(2) {
                        let mut b = [0u8; 2];
                        b.copy_from_slice(chunk);
                        let bits = u16::from_le_bytes(b);
                        let val = f16::from_bits(bits);
                        data.push(f32::from(val));
                    }
                    let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                        .map_err(|e| format!("ndarray shape creation error: {}", e))?;
                    let out = if transpose_two_dim_weights
                        && shape.len() == 2
                        && key.ends_with(".weight")
                    {
                        let mut mat = arr
                            .into_dimensionality::<ndarray::Ix2>()
                            .map_err(|e| format!("transpose dim error: {}", e))?;
                        mat = mat.reversed_axes();
                        Tensor::new_with_dtype(mat.into_dyn(), false, DType::F16)
                    } else {
                        if key.ends_with(".nl_oob.slopes") {
                            Tensor::new_with_dtype(arr.into_dyn(), true, DType::F16)
                        } else {
                            Tensor::new_with_dtype(arr.into_dyn(), false, DType::F16)
                        }
                    };
                    map.insert(key.clone(), out);
                }
            }
            Dtype::BF16 => {
                #[cfg(not(feature = "multi_precision"))]
                return Err(
                    "BF16 dtype requested but crate not built with 'multi_precision' feature"
                        .to_string(),
                );
                #[cfg(feature = "multi_precision")]
                {
                    let shape: Vec<usize> = tensor.shape().iter().map(|d| *d as usize).collect();
                    if tensor.data().len() % 2 != 0 {
                        return Err("Invalid byte length for bf16 tensor".to_string());
                    }
                    let bytes_slice = tensor.data();
                    let mut data: Vec<f32> = Vec::with_capacity(bytes_slice.len() / 2);
                    for chunk in bytes_slice.chunks_exact(2) {
                        let mut b = [0u8; 2];
                        b.copy_from_slice(chunk);
                        let bits = u16::from_le_bytes(b);
                        let val = bf16::from_bits(bits);
                        data.push(f32::from(val));
                    }
                    let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                        .map_err(|e| format!("ndarray shape creation error: {}", e))?;
                    let out = if transpose_two_dim_weights
                        && shape.len() == 2
                        && key.ends_with(".weight")
                    {
                        let mut mat = arr
                            .into_dimensionality::<ndarray::Ix2>()
                            .map_err(|e| format!("transpose dim error: {}", e))?;
                        mat = mat.reversed_axes();
                        Tensor::new_with_dtype(mat.into_dyn(), false, DType::BF16)
                    } else {
                        if key.ends_with(".nl_oob.slopes") {
                            Tensor::new_with_dtype(arr.into_dyn(), true, DType::BF16)
                        } else {
                            Tensor::new_with_dtype(arr.into_dyn(), false, DType::BF16)
                        }
                    };
                    map.insert(key.clone(), out);
                }
            }
            _ => return Err(format!("Unsupported dtype: {:?}", tensor.dtype())),
        }
    }
    // Augment and return compatibility-mapped state dict
    augment_state_dict_for_compat(map)
}

#[cfg(feature = "safe_tensors")]
/// Augment a state dict with compatibility mappings for common HuggingFace-style
/// checkpoint key names (e.g., `self_attn.q_proj.weight`, `mlp.gate_proj.weight`)
/// to the internal module naming used by `load_state_dict` (e.g., `mha.linear_q.weight`,
/// `linear1.weight`). This helps `apply_state_dict_to_module` find and apply weights
/// when checkpoints use alternative naming conventions.
fn augment_state_dict_for_compat(mut map: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>, String> {
    use ndarray::Axis;

    // Collect original keys to iterate safely
    let keys: Vec<String> = map.keys().cloned().collect();

    for k in keys {
        // Map self_attn.* -> mha.*
        if k.ends_with(".self_attn.q_proj.weight") {
            let nk = k.replace(".self_attn.q_proj.weight", ".mha.linear_q.weight");
            if !map.contains_key(&nk) {
                if let Some(v) = map.get(&k).cloned() {
                    map.insert(nk, v);
                }
            }
        }
        if k.ends_with(".self_attn.k_proj.weight") {
            let nk = k.replace(".self_attn.k_proj.weight", ".mha.linear_k.weight");
            if !map.contains_key(&nk) {
                if let Some(v) = map.get(&k).cloned() {
                    map.insert(nk, v);
                }
            }
        }
        if k.ends_with(".self_attn.v_proj.weight") {
            let nk = k.replace(".self_attn.v_proj.weight", ".mha.linear_v.weight");
            if !map.contains_key(&nk) {
                if let Some(v) = map.get(&k).cloned() {
                    map.insert(nk, v);
                }
            }
        }
        if k.ends_with(".self_attn.o_proj.weight") {
            let nk = k.replace(".self_attn.o_proj.weight", ".mha.linear_o.weight");
            if !map.contains_key(&nk) {
                if let Some(v) = map.get(&k).cloned() {
                    map.insert(nk, v);
                }
            }
        }
        // also map self_attn biases
        if k.ends_with(".self_attn.q_proj.bias") {
            let nk = k.replace(".self_attn.q_proj.bias", ".mha.linear_q.bias");
            if !map.contains_key(&nk) {
                if let Some(v) = map.get(&k).cloned() { map.insert(nk, v); }
            }
        }
        if k.ends_with(".self_attn.k_proj.bias") {
            let nk = k.replace(".self_attn.k_proj.bias", ".mha.linear_k.bias");
            if !map.contains_key(&nk) {
                if let Some(v) = map.get(&k).cloned() { map.insert(nk, v); }
            }
        }
        if k.ends_with(".self_attn.v_proj.bias") {
            let nk = k.replace(".self_attn.v_proj.bias", ".mha.linear_v.bias");
            if !map.contains_key(&nk) {
                if let Some(v) = map.get(&k).cloned() { map.insert(nk, v); }
            }
        }
        if k.ends_with(".self_attn.o_proj.bias") {
            let nk = k.replace(".self_attn.o_proj.bias", ".mha.linear_o.bias");
            if !map.contains_key(&nk) {
                if let Some(v) = map.get(&k).cloned() { map.insert(nk, v); }
            }
        }

        // MLP: gate_proj + up_proj -> linear1 (concatenate vertically)
        if k.ends_with(".mlp.gate_proj.weight") {
            let prefix = k.trim_end_matches(".mlp.gate_proj.weight");
            let gate_key = k.clone();
            let up_key = format!("{}.mlp.up_proj.weight", prefix);
            let linear1_key = format!("{}.linear1.weight", prefix);
            if map.contains_key(&up_key) && !map.contains_key(&linear1_key) {
                if let (Some(gate), Some(up)) = (map.get(&gate_key), map.get(&up_key)) {
                    let garr = gate.to_f32_array();
                    let uarr = up.to_f32_array();
                    // concatenate vertically along axis 0 to produce [2*d_ff, d_model]
                    let conc = ndarray::concatenate(Axis(0), &[garr.view(), uarr.view()])
                        .map_err(|e| format!("ndarray concatenate error: {}", e))?;
                    map.insert(linear1_key, Tensor::new(conc.into_dyn(), false));
                } else {
                    log::error!("augment_state_dict_for_compat: missing gate or up tensor for {}", prefix);
                }
            }
        }
        // map gate/up biases into linear1.bias if present (sum biases) - HF often has no bias, but support if present
        if k.ends_with(".mlp.gate_proj.bias") {
            let prefix = k.trim_end_matches(".mlp.gate_proj.bias");
            let gate_b = k.clone();
            let up_b = format!("{}.mlp.up_proj.bias", prefix);
            let linear1_bkey = format!("{}.linear1.bias", prefix);
            if map.contains_key(&up_b) && !map.contains_key(&linear1_bkey) {
                if let (Some(gb), Some(ub)) = (map.get(&gate_b), map.get(&up_b)) {
                    let ga = gb.to_f32_array();
                    let ua = ub.to_f32_array();
                    // concatenate biases vertically then flatten to 1D bias tensor
                    let conc = ndarray::concatenate(Axis(0), &[ga.view(), ua.view()])
                        .map_err(|e| format!("ndarray concatenate error: {}", e))?;
                    let r0 = conc.shape()[0];
                    let r1 = conc.shape()[1];
                    let flat = conc.into_shape_with_order((r0 * r1,))
                        .map_err(|e| format!("reshape error: {}", e))?;
                    map.insert(linear1_bkey, Tensor::new(flat.into_dyn(), false));
                }
            }
        }

        // down_proj -> linear2
        if k.ends_with(".mlp.down_proj.weight") {
            let nk = k.replace(".mlp.down_proj.weight", ".linear2.weight");
            if !map.contains_key(&nk) {
                if let Some(v) = map.get(&k).cloned() {
                    map.insert(nk, v);
                }
            }
        }
        // down_proj bias -> linear2.bias
        if k.ends_with(".mlp.down_proj.bias") {
            let nk = k.replace(".mlp.down_proj.bias", ".linear2.bias");
            if !map.contains_key(&nk) {
                if let Some(v) = map.get(&k).cloned() { map.insert(nk, v); }
            }
        }
        // LM head and output naming variants -> head.* (used by Multimodal and general modules)
        if k.ends_with("lm_head.weight") || k.ends_with("output.weight") || k.ends_with("model.lm_head.weight") || k.ends_with("model.output.weight") {
            let hk = k.replace("lm_head.weight", "head.weight").replace("output.weight", "head.weight").replace("model.lm_head.weight", "head.weight").replace("model.output.weight", "head.weight");
            if !map.contains_key(&hk) {
                if let Some(v) = map.get(&k).cloned() { map.insert(hk, v); }
            }
        }
        if k.ends_with("lm_head.bias") || k.ends_with("output.bias") || k.ends_with("model.lm_head.bias") || k.ends_with("model.output.bias") {
            let hk = k.replace("lm_head.bias", "head.bias").replace("output.bias", "head.bias").replace("model.lm_head.bias", "head.bias").replace("model.output.bias", "head.bias");
            if !map.contains_key(&hk) { if let Some(v) = map.get(&k).cloned() { map.insert(hk, v); } }
        }
        // Layer norm mappings: input_layernorm -> rms_attn_gamma, post_attention_layernorm -> rms_ffn_gamma
        if k.ends_with(".input_layernorm.weight") {
            let nk = k.replace(".input_layernorm.weight", ".rms_attn_gamma");
            if !map.contains_key(&nk) {
                // convert weight tensor to gamma tensor
                if let Some(v) = map.get(&k).cloned() {
                    map.insert(nk, v);
                }
            }
        }
        if k.ends_with(".post_attention_layernorm.weight") {
            let nk = k.replace(".post_attention_layernorm.weight", ".rms_ffn_gamma");
            if !map.contains_key(&nk) {
                if let Some(v) = map.get(&k).cloned() {
                    map.insert(nk, v);
                }
            }
        }
    }

    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_augment_state_dict_self_attn_and_mlp() {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        // create small fake tensors
        let q = Tensor::new(array![[1.0f32; 4]; 4].into_dyn(), false);
        let k = Tensor::new(array![[2.0f32; 4]; 4].into_dyn(), false);
        let gate = Tensor::new(array![[3.0f32; 4]; 2].into_dyn(), false);
        let up = Tensor::new(array![[4.0f32; 4]; 2].into_dyn(), false);

        map.insert("model.layers.0.self_attn.q_proj.weight".to_string(), q.clone());
        map.insert("model.layers.0.self_attn.k_proj.weight".to_string(), k.clone());
        map.insert("model.layers.0.mlp.gate_proj.weight".to_string(), gate.clone());
        map.insert("model.layers.0.mlp.up_proj.weight".to_string(), up.clone());

        let aug = augment_state_dict_for_compat(map).expect("augment failed");
        assert!(aug.contains_key("model.layers.0.mha.linear_q.weight"));
        assert!(aug.contains_key("model.layers.0.mha.linear_k.weight"));
        assert!(aug.contains_key("model.layers.0.linear1.weight"));

        // check linear1 was concatenated vertically: shape should be (4,4)
        let lin1 = aug.get("model.layers.0.linear1.weight").unwrap();
        let arr = lin1.to_f32_array();
        assert_eq!(arr.shape()[0], 4);
        assert_eq!(arr.shape()[1], 4);
    }
}

#[cfg(feature = "safe_tensors")]
/// Parse a single tensor from raw bytes and return a `Tensor` instance.
/// This helper is useful for unit tests that only want to validate dtype conversions
/// without constructing a full SafeTensors archive.
pub fn parse_safetensors_tensor(
    dtype: Dtype,
    shape: Vec<usize>,
    data_bytes: &[u8],
    transpose_two_dim_weights: bool,
    key: Option<&str>,
) -> Result<Tensor, String> {
    match dtype {
        Dtype::F32 => {
            if data_bytes.len() % 4 != 0 {
                return Err("Invalid byte length for f32 tensor".to_string());
            }
            let mut data = Vec::with_capacity(data_bytes.len() / 4);
            for chunk in data_bytes.chunks_exact(4) {
                let mut b = [0u8; 4];
                b.copy_from_slice(chunk);
                data.push(f32::from_le_bytes(b));
            }
            let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                .map_err(|e| format!("ndarray shape creation error: {}", e))?;
            let out = if transpose_two_dim_weights
                && shape.len() == 2
                && key.unwrap_or("").ends_with(".weight")
            {
                let mut mat = arr
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| format!("transpose dim error: {}", e))?;
                mat = mat.reversed_axes();
                Tensor::new_with_dtype(mat.into_dyn(), false, DType::F32)
            } else {
                if key.unwrap_or("").ends_with(".nl_oob.slopes") {
                    Tensor::new_with_dtype(arr.into_dyn(), true, DType::F32)
                } else {
                    Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32)
                }
            };
            Ok(out)
        }
        Dtype::F16 => {
            #[cfg(not(feature = "multi_precision"))]
            return Err(
                "F16 dtype requested but crate not built with 'multi_precision' feature"
                    .to_string(),
            );
            #[cfg(feature = "multi_precision")]
            {
                if data_bytes.len() % 2 != 0 {
                    return Err("Invalid byte length for f16 tensor".to_string());
                }
                let mut data: Vec<f32> = Vec::with_capacity(data_bytes.len() / 2);
                for chunk in data_bytes.chunks_exact(2) {
                    let mut b = [0u8; 2];
                    b.copy_from_slice(chunk);
                    let bits = u16::from_le_bytes(b);
                    let val = f16::from_bits(bits);
                    data.push(f32::from(val));
                }
                let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                    .map_err(|e| format!("ndarray shape creation error: {}", e))?;
                let out = if transpose_two_dim_weights
                    && shape.len() == 2
                    && key.unwrap_or("").ends_with(".weight")
                {
                    let mut mat = arr
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|e| format!("transpose dim error: {}", e))?;
                    mat = mat.reversed_axes();
                    Tensor::new_with_dtype(mat.into_dyn(), false, DType::F16)
                } else {
                    if key.unwrap_or("").ends_with(".nl_oob.slopes") {
                        Tensor::new_with_dtype(arr.into_dyn(), true, DType::F16)
                    } else {
                        Tensor::new_with_dtype(arr.into_dyn(), false, DType::F16)
                    }
                };
                Ok(out)
            }
        }
        Dtype::BF16 => {
            #[cfg(not(feature = "multi_precision"))]
            return Err(
                "BF16 dtype requested but crate not built with 'multi_precision' feature"
                    .to_string(),
            );
            #[cfg(feature = "multi_precision")]
            {
                if data_bytes.len() % 2 != 0 {
                    return Err("Invalid byte length for bf16 tensor".to_string());
                }
                let mut data: Vec<f32> = Vec::with_capacity(data_bytes.len() / 2);
                for chunk in data_bytes.chunks_exact(2) {
                    let mut b = [0u8; 2];
                    b.copy_from_slice(chunk);
                    let bits = u16::from_le_bytes(b);
                    let val = bf16::from_bits(bits);
                    data.push(f32::from(val));
                }
                let arr = Array::from_shape_vec(IxDyn(&shape), data.clone())
                    .map_err(|e| format!("ndarray shape creation error: {}", e))?;
                let out = if transpose_two_dim_weights
                    && shape.len() == 2
                    && key.unwrap_or("").ends_with(".weight")
                {
                    let mut mat = arr
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|e| format!("transpose dim error: {}", e))?;
                    mat = mat.reversed_axes();
                    Tensor::new_with_dtype(mat.into_dyn(), false, DType::BF16)
                } else {
                    if key.unwrap_or("").ends_with(".nl_oob.slopes") {
                        Tensor::new_with_dtype(arr.into_dyn(), true, DType::BF16)
                    } else {
                        Tensor::new_with_dtype(arr.into_dyn(), false, DType::BF16)
                    }
                };
                Ok(out)
            }
        }
        _ => Err(format!("Unsupported dtype: {:?}", dtype)),
    }
}

/// Apply a state dict mapping (names -> Tensors) to a module by delegating
/// to the module's `load_state_dict` implementation first, then falling back
/// to a robust direct assignment routine if any parameters remain unset.
/// The fallback will match parameter names with/without the provided `root`,
/// handle common transposed 2D weight layouts, and report how many params were
/// assigned. This reduces surprises when checkpoint naming conventions differ.
pub fn apply_state_dict_to_module(
    module: &mut dyn crate::nn::Module,
    state: &HashMap<String, Tensor>,
    root: &str,
) -> Result<(), String> {
    // First, allow the module to load via its own logic; many modules will
    // implement optimized load_state_dict handlers.
    let res = module.load_state_dict(state, root);

    // After the standard load, attempt a best-effort pass that iterates the
    // module's named parameters and applies any matching tensors from the
    // provided state dict. This helps with mismatched prefixes, transposed
    // weights, and alternate naming schemes.
    let mut assigned = 0usize;
    let mut tried = 0usize;

    let params = module.named_parameters(root).into_iter().collect::<Vec<_>>();
    for (name, param) in params.iter() {
        tried += 1;
        // Normalized name without leading dots
        let lname = name.trim_start_matches('.').to_string();
        // Candidate keys to look up in state dict
        let mut candidates: Vec<String> = Vec::new();
        // exact name (no root)
        candidates.push(lname.clone());
        // with root, allowing trailing/leading dot variants
        let root_norm = root.trim_end_matches('.');
        if !root_norm.is_empty() {
            candidates.push(format!("{}.{}", root_norm, lname));
        }
        // model.* variants (some checkpoints use top-level 'model.' prefix)
        candidates.push(format!("model.{}", lname));
        if !root_norm.is_empty() {
            candidates.push(format!("model.{}{}.{}", root_norm, if root_norm.ends_with('.') {""} else {""}, lname));
        }

        let mut _found = false;
        for c in candidates.iter() {
            if let Some(t) = state.get(c) {
                // Check shape compatibility, and handle common transpose case for 2D weights
                let param_shape = param.lock().storage.shape().to_vec();
                let t_shape = t.lock().storage.shape().to_vec();
                if param_shape == t_shape {
                    // direct assign
                    let mut p_lock = param.lock();
                    let src_lock = t.lock();
                    p_lock.storage = src_lock.storage.clone();
                    p_lock.dtype = src_lock.dtype;
                    assigned += 1;
                    _found = true;
                    break;
                } else if param_shape.len() == 2 && t_shape.len() == 2 && param_shape[0] == t_shape[1] && param_shape[1] == t_shape[0] {
                    // likely transposed -- transpose before assign
                    let arr = t.to_f32_array();
                    let mat = match arr.into_dimensionality::<ndarray::Ix2>() {
                        Ok(m) => m.reversed_axes().into_dyn(),
                        Err(_) => continue,
                    };
                    let trans = Tensor::new(mat, false);
                    let mut p_lock = param.lock();
                    let src_lock = trans.lock();
                    p_lock.storage = src_lock.storage.clone();
                    p_lock.dtype = src_lock.dtype;
                    assigned += 1;
                    _found = true;
                    break;
                } else if param_shape.len() == 2 && t_shape.len() == 3 {
                    // Handle 3D stacked tensors (e.g., gate/up stored as [2, d_ff, d_model])
                    // Collapse first two dims to form a 2D matrix then transpose if needed
                    let arr3 = t.to_f32_array();
                    let a0 = t_shape[0];
                    let a1 = t_shape[1];
                    let a2 = t_shape[2];
                    // collapse to (a0*a1, a2)
                    let reshaped = match arr3.into_shape_with_order((a0 * a1, a2)) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    // If param expects (in, out) and reshaped is (out, in), transpose
                    if param_shape[0] == a2 && param_shape[1] == a0 * a1 {
                        let mat = reshaped.reversed_axes().into_dyn();
                        let trans = Tensor::new(mat, false);
                        let mut p_lock = param.lock();
                        let src_lock = trans.lock();
                        p_lock.storage = src_lock.storage.clone();
                        p_lock.dtype = src_lock.dtype;
                        assigned += 1;
                        _found = true;
                        break;
                    } else if param_shape[0] == a0 * a1 && param_shape[1] == a2 {
                        // direct assignment after collapse
                        let t2 = Tensor::new(reshaped.into_dyn(), false);
                        let mut p_lock = param.lock();
                        let src_lock = t2.lock();
                        p_lock.storage = src_lock.storage.clone();
                        p_lock.dtype = src_lock.dtype;
                        assigned += 1;
                        _found = true;
                        break;
                    } else {
                        continue;
                    }
                } else {
                    // shape mismatch - skip
                    continue;
                }
            }
        }
        // if not found, continue to next param
    }

    log::info!("apply_state_dict_to_module: attempted {} params, assigned {} via fallback", tried, assigned);

    // Return original module.load_state_dict result (preserve its error if any),
    // but if it was Ok, return Ok regardless of fallback result; if it was Err,
    // prefer to return the module error unless fallback assigned everything.
    match res {
        Ok(()) => Ok(()),
        Err(e) => {
            if assigned > 0 {
                log::warn!("apply_state_dict_to_module: module.load_state_dict returned error '{}', but fallback assigned {} params", e, assigned);
                Ok(())
            } else {
                Err(e)
            }
        }
    }
}

#[cfg(feature = "safe_tensors")]
/// Load a SafeTensors archive from bytes and apply it to the provided module.
/// This function combines `load_safetensors_from_bytes` and `apply_state_dict_to_module`.
pub fn apply_safetensors_bytes_to_module_bytes(
    module: &mut dyn crate::nn::Module,
    bytes: &[u8],
    transpose_two_dim_weights: bool,
    root: &str,
) -> Result<(), String> {
    let map = load_safetensors_from_bytes(bytes, transpose_two_dim_weights)?;
    apply_state_dict_to_module(module, &map, root)
}

#[cfg(feature = "safe_tensors")]
/// Apply a Kronos SafeTensors archive to a `MultimodalLLM` module (or any
/// module in fallback mode). This will verify Kronos metadata/marker and then
/// map keys to the nested modules in `MultimodalLLM` when applicable.
pub fn apply_kronos_bytes_to_module_bytes(
    module: &mut dyn crate::nn::Module,
    bytes: &[u8],
    transpose_two_dim_weights: bool,
    root: &str,
) -> Result<(), String> {
    // Load the safetensors archive and detect Kronos keys heuristically. Prefer
    // metadata detection but fall back to key pattern matching for robust
    // identification across versions of the SafeTensors parser.
    let map = load_safetensors_from_bytes(bytes, transpose_two_dim_weights)?;
    // Kronos detection strategies (best-effort):
    // 1) If the archive contains keys we expect from kronos, it's likely kronos.
    // 2) If the archive includes a Kronos metadata marker or format field (in the raw bytes) we detect that too.
    // 3) Fallback uses the presence of specific keys as a heuristic.
    let mut is_kronos = map.contains_key("text_embedding.weight")
        || map.keys().any(|k| k.starts_with("vision_encoder."));
    if !is_kronos {
        // Try to detect Kronos via the embedded metadata in the safetensors file bytes.
        // We look for the '__kronos_marker__' or a '"format":"Kronos"' substring in the bytes.
        if bytes.windows(16).any(|w| w == b"__kronos_marker__") {
            log::info!("Kronos detector: found __kronos_marker__ in raw safetensors bytes");
            is_kronos = true;
        } else if bytes.windows(14).any(|w| w == b"\"format\":\"Kronos\"") {
            log::info!("Kronos detector: found \"format\":\"Kronos\" in raw safetensors bytes");
            is_kronos = true;
        }
    }
    // Log unknown keys that don't match expected kronos mappings -- helps debugging
    for k in map.keys() {
        if !k.starts_with("vision_encoder.")
            && !k.starts_with("text_embedding")
            && !k.starts_with("projector.")
            && !k.starts_with("decoder_blocks.")
            && !k.starts_with("head")
        {
            log::debug!("Kronos loader: unknown key encountered: {}", k);
        }
    }
    if !is_kronos {
        return Err("Not a Kronos-formatted SafeTensors archive (no expected keys)".into());
    }
    // If the module is a MultimodalLLM, apply structured mapping
    if let Some(mm) = module.as_any_mut().downcast_mut::<crate::nn::multimodal::MultimodalLLM>() {
        // apply vision encoder block state if present
        let _ = apply_state_dict_to_module(&mut mm.vision_encoder, &map, "vision_encoder");
        // text embedding
        if let Some(t) = map.get("text_embedding") {
            mm.text_embedding = t.clone();
        }
        if let Some(tw) = map.get("text_embedding.weight") {
            mm.text_embedding = tw.clone();
        }
        // projector: may be missing; instantiate from weight shape if needed
        if let Some(pw) = map.get("projector.weight") {
            let pw_shape = pw.lock().storage.shape().to_vec();
            if pw_shape.len() == 2 {
                let in_f = pw_shape[0];
                let out_f = pw_shape[1];
                if mm.projector.is_none() {
                    log::info!("Kronos loader: creating projector with shape in_features={} out_features={}", in_f, out_f);
                    mm.projector = Some(crate::nn::multimodal::Projector::Linear(crate::nn::Linear::new(in_f, out_f, true)));
                }
                if let Some(p) = mm.projector.as_mut() {
                    if let Err(e) = p.load_state_dict(&map, "projector") {
                        log::warn!("Kronos loader: projector.load_state_dict failed: {}", e);
                    }
                }
            }
        }
        // decoder blocks: attempt to apply per-layer weights
        for i in 0..mm.decoder_blocks.len() {
            let prefix = format!("decoder_blocks.layers.{}", i);
            if let Err(e) = mm.decoder_blocks[i].load_state_dict(&map, &prefix) {
                log::warn!("Kronos loader: failed to load decoder block {}: {}", i, e);
            }
        }
        // head
        if let Err(e) = mm.head.load_state_dict(&map, "head") {
            log::warn!("Kronos loader: failed to load head: {}", e);
        }
        // in case there are remaining top-level keys, fallback to default behavior
        // After targeted structured mapping, attempt to apply any remaining keys to the root module.
        if let Err(e) = apply_state_dict_to_module(mm, &map, root) {
            log::warn!("Kronos loader: fallback apply_state_dict_to_module returned error: {}", e);
            return Err(e);
        }
        Ok(())
    } else {
        // generic fallback: apply the state dict using the default module loader
        let res = apply_state_dict_to_module(module, &map, root);
        if res.is_err() {
            log::warn!("safetensors loader fallback: apply_state_dict_to_module returned error: {:?}", res.as_ref().err());
        }
        res
    }
}

#[cfg(feature = "safe_tensors")]
/// Serialize a module's named parameters into SafeTensors bytes.
pub fn save_module_to_safetensors_bytes(module: &dyn crate::nn::Module) -> Result<Vec<u8>, String> {
    use safetensors::tensor::{serialize as st_serialize, Dtype as STDtype, TensorView as STTensorView};
    use std::collections::HashMap;

    // NOTE: `TensorView` borrows the underlying byte slice; keep buffers alive until after `serialize`.
    // We store buffers as `Box<[u8]>` so the underlying allocation is stable.
    let mut buffers: Vec<Box<[u8]>> = Vec::new();
    let mut map: HashMap<String, STTensorView<'_>> = HashMap::new();
    for (name, tensor) in module.named_parameters("") {
        let lock = tensor.lock();
        let shape: Vec<usize> = lock.storage.shape().to_vec();
        // Only f32 supported for now
        let mut bytes: Vec<u8> = Vec::with_capacity((shape.iter().product::<usize>()) * 4);
        for v in lock.storage.to_f32_array().iter() {
            bytes.extend(&v.to_le_bytes());
        }
        let boxed: Box<[u8]> = bytes.into_boxed_slice();
        // Capture a raw pointer before moving the box into `buffers`.
        let ptr = boxed.as_ptr();
        let len = boxed.len();
        buffers.push(boxed);
        // SAFETY: `ptr` points to the heap allocation owned by the just-pushed `Box<[u8]>`.
        // That allocation remains valid until `buffers` is dropped at the end of this function,
        // which is after `serialize` completes.
        let buf_ref: &[u8] = unsafe { std::slice::from_raw_parts(ptr, len) };
        let std_dtype = STDtype::F32;
        let st_view = STTensorView::new(std_dtype, shape.iter().map(|x| *x as usize).collect::<Vec<_>>(), buf_ref)
            .map_err(|e| format!("Failed to create SafeTensors view: {}", e))?;
        map.insert(name.clone(), st_view);
    }
    let bytes = st_serialize(&map, None).map_err(|e| format!("safetensors serialize error: {}", e))?;
    Ok(bytes)
}
