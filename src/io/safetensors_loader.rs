#[cfg(feature = "safe_tensors")]
use crate::dtype::DType;
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
use crate::nn::Module;

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
    Ok(map)
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
/// to the module's `load_state_dict` implementation. This will return an error
/// if the module fails to load a parameter.
pub fn apply_state_dict_to_module(
    module: &mut dyn crate::nn::Module,
    state: &HashMap<String, Tensor>,
    root: &str,
) -> Result<(), String> {
    module.load_state_dict(state, root)
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
        if map.get("projector.weight").is_some() {
            let pw = map.get("projector.weight").unwrap();
            let pw_shape = pw.lock().storage.shape().to_vec();
            if pw_shape.len() == 2 {
                let in_f = pw_shape[0];
                let out_f = pw_shape[1];
                if mm.projector.is_none() {
                    log::info!("Kronos loader: creating projector with shape in_features={} out_features={}", in_f, out_f);
                    mm.projector = Some(crate::nn::Linear::new(in_f, out_f, true));
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
