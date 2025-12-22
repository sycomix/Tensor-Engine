#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use safetensors::tensor::serialize as st_serialize;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use safetensors::tensor::Dtype as STDtype;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use safetensors::tensor::TensorView as STTensorView;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use std::collections::HashMap;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use tensor_engine::io::safetensors_loader::{apply_kronos_bytes_to_module_bytes, load_safetensors_from_bytes};
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
use tensor_engine::nn::MultimodalLLM;
#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
#[test]
fn test_kronos_loader_text_embedding() {
    // Build a tiny MultimodalLLM
    let vision = tensor_engine::nn::VisionTransformer::new(3, 16, 8, 32, 2, 1, 8).expect("create vision transformer");
    let mut model = MultimodalLLM::new(vision, 8, 8, 16, 2, 1);
    // Create text_embedding tensor bytes 8x8
    let shape = vec![8usize, 8usize];
    let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    let mut bytes: Vec<u8> = Vec::new();
    for f in data.iter() { bytes.extend(&f.to_le_bytes()); }
    let mut tensors = HashMap::new();
    let st_view = STTensorView::new(STDtype::F32, shape.clone(), &bytes).unwrap();
    tensors.insert("text_embedding.weight".to_string(), st_view);
    // Create Kronos metadata
    let mut meta = HashMap::new();
    meta.insert("__kronos_marker__".to_string(), "4B524F4E".to_string());
    meta.insert("format".to_string(), "Kronos".to_string());
    let bytes_st = st_serialize(&tensors, Some(meta)).unwrap();
    // Apply Kronos loader
    let res = apply_kronos_bytes_to_module_bytes(&mut model, &bytes_st, true, "");
    assert!(res.is_ok());
    // Verify text_embedding updated
    let loaded = load_safetensors_from_bytes(&bytes_st, true).unwrap();
    assert!(loaded.get("text_embedding.weight").is_some());
    let tm = loaded.get("text_embedding.weight").unwrap().lock().storage.to_f32_array();
    let me = model.text_embedding.lock().storage.to_f32_array();
    assert_eq!(tm, me);
}

#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
#[test]
fn test_kronos_loader_projector_and_vision_head_decoder() {
    // Build a tiny MultimodalLLM
    let vision = tensor_engine::nn::VisionTransformer::new(3, 2, 8, 16, 2, 1, 8).expect("create vision transformer");
    let mut model = MultimodalLLM::new(vision, 8, 8, 16, 2, 1);
    // Build tensors for projector, vision encoder patch conv, head, and decoder linear1
    let mut tensors = HashMap::new();
    // projector weight 8x8
    let shape_proj = vec![8usize, 8usize];
    let data_proj: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    let mut bytes_proj: Vec<u8> = Vec::new();
    for f in data_proj.iter() { bytes_proj.extend(&f.to_le_bytes()); }
    let st_proj = STTensorView::new(STDtype::F32, shape_proj.clone(), &bytes_proj).unwrap();
    tensors.insert("projector.weight".to_string(), st_proj);

    // vision encoder patch_embed conv weight: shape [out, in, k, k]
    let shape_vis = vec![8usize, 3usize, 2usize, 2usize];
    let data_vis: Vec<f32> = (0..(8 * 3 * 2 * 2)).map(|i| (i as f32) * 0.05).collect();
    let mut bytes_vis: Vec<u8> = Vec::new();
    for f in data_vis.iter() { bytes_vis.extend(&f.to_le_bytes()); }
    tensors.insert("vision_encoder.patch_embed.conv.weight".to_string(), STTensorView::new(STDtype::F32, shape_vis.clone(), &bytes_vis).unwrap());

    // head weight: shape [d_model, vocab]
    let shape_head = vec![8usize, 8usize];
    let data_head: Vec<f32> = (0..64).map(|i| (i as f32) * 0.2).collect();
    let mut bytes_head: Vec<u8> = Vec::new();
    for f in data_head.iter() { bytes_head.extend(&f.to_le_bytes()); }
    tensors.insert("head.weight".to_string(), STTensorView::new(STDtype::F32, shape_head.clone(), &bytes_head).unwrap());

    // decoder block 0 linear1 weight: prefix as decoder_blocks.layers.0.linear1.weight
    let shape_dec = vec![8usize, 16usize];
    let data_dec: Vec<f32> = (0..(8 * 16)).map(|i| (i as f32) * 0.3).collect();
    let mut bytes_dec: Vec<u8> = Vec::new();
    for f in data_dec.iter() { bytes_dec.extend(&f.to_le_bytes()); }
    tensors.insert("decoder_blocks.layers.0.linear1.weight".to_string(), STTensorView::new(STDtype::F32, shape_dec.clone(), &bytes_dec).unwrap());

    // Kronos metadata
    let mut meta = HashMap::new();
    meta.insert("__kronos_marker__".to_string(), "4B524F4E".to_string());
    meta.insert("format".to_string(), "Kronos".to_string());
    let bytes_st = st_serialize(&tensors, Some(meta)).unwrap();
    // Apply Kronos loader
    let res = apply_kronos_bytes_to_module_bytes(&mut model, &bytes_st, true, "");
    assert!(res.is_ok());
    // Verify projector created and loaded
    assert!(model.projector.is_some());
    if let Some(p) = model.projector.as_ref() {
        match p {
            tensor_engine::nn::multimodal::Projector::Linear(l) => {
                let pw = l.weight.lock().storage.to_f32_array();
                let mut arr_proj = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape_proj), data_proj.clone()).unwrap();
                arr_proj = arr_proj.reversed_axes(); // loader transposes 2D weights when transpose flag is true
                assert_eq!(pw, arr_proj);
            }
            tensor_engine::nn::multimodal::Projector::MLP(_) => panic!("expected Linear projector for projector.weight"),
        }
    }
    // Verify vision encoder conv weight loaded
    let pw = model.vision_encoder.patch_embed.conv.weight.lock().storage.to_f32_array();
    let arr_vis = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape_vis), data_vis.clone()).unwrap();
    assert_eq!(pw, arr_vis);
    // Verify head weight
    let hw = model.head.weight.lock().storage.to_f32_array();
    let mut arr_head = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape_head), data_head.clone()).unwrap();
    arr_head = arr_head.reversed_axes();
    assert_eq!(hw, arr_head);
    // Verify decoder linear1 weight
    let lw = model.decoder_blocks[0].linear1.weight.lock().storage.to_f32_array();
    let mut arr_dec = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape_dec), data_dec.clone()).unwrap();
    arr_dec = arr_dec.reversed_axes();
    assert_eq!(lw, arr_dec);
}

#[cfg(all(feature = "safe_tensors", feature = "multi_precision"))]
#[test]
fn test_apply_safetensors_bytes_dtype_conversion() {
    // Build trivial map with a f16 and bf16 typed tensors and verify load_safetensors_from_bytes preserves reported dtype.
    let shape = vec![4usize, 4usize];
    let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
    let mut bytes_f16: Vec<u8> = Vec::new();
    let mut bytes_bf16: Vec<u8> = Vec::new();
    // Encode as f16 and bf16 using helper conversion
    #[cfg(feature = "multi_precision")]
    {
        for f in data.iter() {
            // f16: convert to half representation
            let f16 = half::f16::from_f32(*f);
            bytes_f16.extend(&f16.to_bits().to_le_bytes());
            let bf = half::bf16::from_f32(*f);
            bytes_bf16.extend(&bf.to_bits().to_le_bytes());
        }
    }
    let mut tensors = HashMap::new();
    tensors.insert("f16.weight".to_string(), STTensorView::new(STDtype::F16, shape.clone(), &bytes_f16).unwrap());
    tensors.insert("bf16.weight".to_string(), STTensorView::new(STDtype::BF16, shape.clone(), &bytes_bf16).unwrap());
    let bytes_st = st_serialize(&tensors, None).unwrap();
    let map = load_safetensors_from_bytes(&bytes_st, true).unwrap();
    let t1 = map.get("f16.weight").unwrap();
    let t2 = map.get("bf16.weight").unwrap();
    assert_eq!(t1.lock().dtype, tensor_engine::dtype::DType::F16);
    assert_eq!(t2.lock().dtype, tensor_engine::dtype::DType::BF16);
}

