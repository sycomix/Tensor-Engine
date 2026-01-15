/// Optional Torch loader using the `tch` crate. This is a best-effort implementation that attempts
/// to load a saved `VarStore`-style checkpoint using tch and falls back to instructing the
/// caller to use the Python converter if unsupported. Enable with `--features with_tch`.
#[cfg(feature = "with_tch")]
pub mod loader {
    use crate::dtype::DType;
    use crate::tensor::Tensor;
    use ndarray::ArrayD;
    use std::collections::HashMap;
    use tch::{nn, Device, IValue, Tensor as TchTensor};

    fn tch_tensor_to_ndarray_f32(t: &TchTensor) -> Result<ArrayD<f32>, String> {
        let t_cpu = t.to(Device::Cpu);
        // Make contiguous and force float32 kind
        let t_f32 = t_cpu.to_kind(tch::Kind::Float);
        // Get shape
        let shape = t_f32.size();
        let shape_usize: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        // Chunked extraction: avoid making a single large Vec for huge tensors.
        let numel: usize = shape_usize.iter().product();
        let mut out_vec: Vec<f32> = Vec::with_capacity(numel);
        let flat = t_f32.reshape(&[numel as i64]);
        // Choose chunk size in number of elements (e.g., 1M floats per chunk)
        let chunk: usize = 1_000_000usize;
        let mut offset: usize = 0usize;
        while offset < numel {
            let len = (numel - offset).min(chunk);
            let slice = flat.narrow(0, offset as i64, len as i64);
            let chunk_data: Vec<f32> = Vec::<f32>::try_from(slice)
                .map_err(|e| format!("Failed to extract tensor chunk: {}", e))?;
            out_vec.extend(chunk_data);
            offset += len;
        }
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape_usize), out_vec)
            .map_err(|e| format!("ndarray shape creation error: {}", e))
    }

    fn normalize_key(key: &str) -> String {
        // Remove common python module prefix used in DataParallel / wrapping
        if key.starts_with("module.") {
            key[7..].to_string()
        } else {
            key.to_string()
        }
    }

    // We avoid complex recursive parsing due to IValue ownership/variant limitations.
    // Instead, parse top-level Vec<(IValue, IValue)> state dict entries and insert tensors.
    fn try_insert_ivalue_into_map(map: &mut HashMap<String, Tensor>, key: String, iv: IValue, transpose_two_dim_weights: bool) -> Result<(), String> {
        // no local alias for IValue required
        match iv {
            IValue::Tensor(tt) => {
                let arr = tch_tensor_to_ndarray_f32(&tt)?;
                let t = if transpose_two_dim_weights && arr.ndim() == 2 && key.ends_with(".weight") {
                    let mut mat = arr.into_dimensionality::<ndarray::Ix2>().map_err(|e| format!("transpose dim error: {}", e))?;
                    mat = mat.reversed_axes();
                    Tensor::new_with_dtype(mat.into_dyn(), false, DType::F32)
                } else {
                    Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32)
                };
                map.insert(key, t);
            }
            IValue::GenericDict(entries) => {
                for (k_iv, v_iv) in entries.into_iter() {
                    if let Ok(kstr) = String::try_from(k_iv) {
                        let nested_key = if key.is_empty() { kstr.clone() } else { format!("{}.{}", key, kstr) };
                        try_insert_ivalue_into_map(map, normalize_key(&nested_key), v_iv, transpose_two_dim_weights)?;
                    }
                }
            }
            IValue::Tuple(tup) => {
                // iterate tuple elements and use numeric index suffix
                for (i, item) in tup.into_iter().enumerate() {
                    let nested_key = if key.is_empty() { format!("{}", i) } else { format!("{}[{}]", key, i) };
                    let _ = try_insert_ivalue_into_map(map, nested_key, item, transpose_two_dim_weights);
                }
            }
            // 'List' variants are represented as Tuple in current tch::IValue implementations
            _ => { /* ignore other variants */ }
        }
        Ok(())
    }

    fn process_state_iv_into_map(map: &mut HashMap<String, Tensor>, state_iv: IValue, transpose_two_dim_weights: bool) -> Result<(), String> {
        match state_iv {
            IValue::GenericDict(entries) => {
                for (k_iv, v_iv) in entries.into_iter() {
                    if let Ok(kstr) = String::try_from(k_iv) {
                        let norm_key = normalize_key(&kstr);
                        let _ = try_insert_ivalue_into_map(map, norm_key, v_iv, transpose_two_dim_weights);
                    }
                }
                return Ok(());
            }
            // Tuple fallback is handled by Vec<(IValue,IValue)> try_from below; do not duplicate handling here.
            // Fallback: try to convert the IValue directly into a Vec<(String, TchTensor)> or HashMap
            other => {
                // Fallback: no direct Vec<(String,TchTensor)> or HashMap<String,TchTensor> conversions
                // Try Vec<(IValue, IValue)> last - this consumes the IValue
                // Avoid converting the whole IValue into a Vec which would consume and copy
                // the entire state_dict, particularly problematic for large models. Instead,
                // iterate over Tuple/List entries in-place to stream entries into the map.
                match other {
                    IValue::Tuple(items) => {
                        for item in items.into_iter() {
                            if let IValue::Tuple(pair) = item {
                                if pair.len() == 2 {
                                    let mut iter = pair.into_iter();
                                    if let (Some(k_iv), Some(v_iv)) = (iter.next(), iter.next()) {
                                        if let Ok(kstr) = String::try_from(k_iv) {
                                            let norm_key = normalize_key(&kstr);
                                            let _ = try_insert_ivalue_into_map(map, norm_key, v_iv, transpose_two_dim_weights);
                                        }
                                    } else {
                                        // malformed pair: skip
                                        continue;
                                    }
                                }
                            }
                        }
                        return Ok(());
                    }
                    _ => {
                        // Fallthrough: we cannot iterate, attempt TryFrom<Vec> as last resort
                        if let Ok(entries) = Vec::<(IValue, IValue)>::try_from(other) {
                            for (k_iv, v_iv) in entries.into_iter() {
                                if let Ok(kstr) = String::try_from(k_iv) {
                                    let norm_key = normalize_key(&kstr);
                                    let _ = try_insert_ivalue_into_map(map, norm_key, v_iv, transpose_two_dim_weights);
                                }
                            }
                            return Ok(());
                        }
                    }
                }
                Ok(())
            }
        }
    }

    pub fn load_torch_state_dict_to_map(path: &str, transpose_two_dim_weights: bool) -> Result<HashMap<String, Tensor>, String> {
        // Attempt to load using VarStore: this will succeed for state dicts saved by tch's VarStore
        let device = Device::Cpu;
        let mut vs = nn::VarStore::new(device);

        if let Err(e) = vs.load(path) {
            // If load failed, try TorchScript module and extract parameters (best-effort)
            match tch::CModule::load(path) {
                Ok(m) => {
                    let mut map: HashMap<String, Tensor> = HashMap::new();
                    // First try the convenience APIs: named_parameters() and named_buffers()
                    match m.named_parameters() {
                        Ok(params) => {
                            for (name, p) in params.into_iter() {
                                let arr = tch_tensor_to_ndarray_f32(&p)?;
                                let t = if transpose_two_dim_weights && arr.ndim() == 2 && name.ends_with(".weight") {
                                    let mut mat = arr.into_dimensionality::<ndarray::Ix2>().map_err(|e| format!("transpose dim error: {}", e))?;
                                    mat = mat.reversed_axes();
                                    Tensor::new_with_dtype(mat.into_dyn(), false, DType::F32)
                                } else {
                                    Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32)
                                };
                                map.insert(name, t);
                            }
                        }
                        Err(_e) => {
                            // named_parameters not available or failed; we'll try a best-effort state_dict call below.
                        }
                    }
                    // named_buffers is not available on CModule (removed) - rely on state_dict() below to include buffers
                    // Try to extract state_dict() via IValue if available; this returns a dict mapping
                    // strings to tensors and will include buffers. Use it to add any missing entries.
                    if let Ok(state_iv) = m.method_is::<IValue>("state_dict", &[]) {
                        let _ = process_state_iv_into_map(&mut map, state_iv, transpose_two_dim_weights);
                    }
                    if map.is_empty() {
                        return Err(format!("tch load succeeded but no parameters/buffers found in {}. Use examples/convert_torch_to_safetensors.py", path));
                    }
                    return Ok(map);
                }
                Err(e2) => {
                    return Err(format!("Failed to load via VarStore and TorchScript loader: {}, {}. Use examples/convert_torch_to_safetensors.py", e, e2));
                }
            }
        } else {
            // if varstore load succeeded but contains no variables, try CModule fallback
            if vs.variables().len() == 0 {
                // Prefer TorchScript path
                match tch::CModule::load(path) {
                    Ok(m) => {
                        let mut map: HashMap<String, Tensor> = HashMap::new();
                        match m.named_parameters() {
                            Ok(params) => {
                                for (name, p) in params.into_iter() {
                                    let arr = tch_tensor_to_ndarray_f32(&p)?;
                                    let t = if transpose_two_dim_weights && arr.ndim() == 2 && name.ends_with(".weight") {
                                        let mut mat = arr.into_dimensionality::<ndarray::Ix2>().map_err(|e| format!("transpose dim error: {}", e))?;
                                        mat = mat.reversed_axes();
                                        Tensor::new_with_dtype(mat.into_dyn(), false, DType::F32)
                                    } else {
                                        Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32)
                                    };
                                    map.insert(name, t);
                                }
                            }
                            Err(_e) => {}
                        }
                        if let Ok(state_iv) = m.method_is::<IValue>("state_dict", &[]) {
                            let _ = process_state_iv_into_map(&mut map, state_iv, transpose_two_dim_weights);
                        }
                        if map.is_empty() {
                            return Err(format!("tch load succeeded but no parameters/buffers found in {}. Use examples/convert_torch_to_safetensors.py", path));
                        }
                        return Ok(map);
                    }
                    Err(e2) => {
                        return Err(format!("VarStore load succeeded but no variables found; TorchScript fallback failed: {}. Use examples/convert_torch_to_safetensors.py", e2));
                    }
                }
            }
        }
        // Convert varstore variables to our Map
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for (name, var) in vs.variables() {
            let arr = tch_tensor_to_ndarray_f32(&var)?;
            let t = if transpose_two_dim_weights && arr.ndim() == 2 && name.ends_with(".weight") {
                let mut mat = arr.into_dimensionality::<ndarray::Ix2>().map_err(|e| format!("transpose dim error: {}", e))?;
                mat = mat.reversed_axes();
                Tensor::new_with_dtype(mat.into_dyn(), false, DType::F32)
            } else {
                Tensor::new_with_dtype(arr.into_dyn(), false, DType::F32)
            };
            map.insert(name, t);
        }
        if map.is_empty() {
            Err("VarStore load succeeded but no variables found; use converter script for general PyTorch pickled state dicts".to_string())
        } else {
            Ok(map)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::collections::HashMap;
        use tch::IValue;

        #[test]
        #[cfg(feature = "with_tch")]
        fn process_state_handles_malformed_tuple_pairs() {
            let mut map: HashMap<String, Tensor> = HashMap::new();
            // Construct a malformed tuple where inner pair has length 1
            let inner = IValue::Tuple(vec![IValue::from("only_key")]);
            let outer = IValue::Tuple(vec![inner]);
            let res = process_state_iv_into_map(&mut map, outer, false);
            assert!(res.is_ok());
            assert!(map.is_empty());
        }
    }
}

#[cfg(not(feature = "with_tch"))]
pub mod loader {
    use crate::tensor::Tensor;
    use std::collections::HashMap;

    pub fn load_torch_state_dict_to_map(_path: &str, _transpose_two_dim_weights: bool) -> Result<HashMap<String, Tensor>, String> {
        Err("tch feature not enabled; enable with --features with_tch".to_string())
    }
}
