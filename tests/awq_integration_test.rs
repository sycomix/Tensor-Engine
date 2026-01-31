use ndarray::IxDyn;

use tensor_engine::dtype::DType;
use tensor_engine::nn::linear_dispatch::LinearLayer;
use tensor_engine::nn::quantized::QuantizedLinear;
use tensor_engine::nn::transformer_cleaned::{TransformerBlock, TransformerConfig};

use tensor_engine::tensor::Tensor;

#[test]
fn test_mixed_precision_transformer_block() {
    let d_model = 32;
    let d_ff = 64; // hidden_pixels
                   // linear1 in Llama style usually has out_features = 2 * d_ff (gate + up)
                   // let linear1 = Linear::new(d_model, hidden_pixels * 2);
    let out_features = d_ff * 2; // 128
    let num_heads = 4;
    let kv_heads = 4;

    // 1. Create block (defaults to F32)
    let mut block = TransformerBlock::new_llama_style(TransformerConfig {
        d_model,
        d_ff,
        num_heads,
        kv_heads,
        use_rope: true,
        rope_theta: 10000.0,
        rope_scale: 1.0,
        bias: false,
    })
    .expect("create block");

    // 2. Verify linear1 is F32
    if let LinearLayer::F32(ref _l) = block.linear1 {
        // ok
    } else {
        panic!("Expected linear1 to be F32 initially");
    }

    // 3. Create a QuantizedLinear layer to replace linear1
    let group_size = 32;
    let in_feat = d_model; // 32
    let out_feat = out_features; // 128

    // qweight: Packed (in, out/2).
    // Wait, awq_dequantize_affine unpacks to (in, out) or (out, in)?
    // The implementation of awq_dequantize_affine unpacks packed buffer to `target_shape`.
    // In QuantizedLinear::forward, target_shape = vec![self.in_features, self.out_features].
    // So it expects unpacked shape (in, out).
    // The packed buffer must hold enough bytes: (in * out) / 2.
    // Shape of packed tensor is loosely checked by unpack_4bit_u8 (just size check).
    // But we should give it reasonable shape.
    // Let's make packed tensor flat for simplicity or (in, out/2).
    let packed_len = (in_feat * out_feat) / 2;
    let packed_data_f32: Vec<f32> = vec![0xAB as f32; packed_len]; // 0xA=10, 0xB=11.
    let packed_arr =
        ndarray::ArrayD::from_shape_vec(IxDyn(&[packed_len][..]), packed_data_f32).unwrap();

    // Create U8 tensor
    let qweight = Tensor::new_with_dtype(packed_arr, false, DType::U8);

    // Scales: (in, out/G).
    // In QuantizedLinear::forward, target is (in, out).
    // So scales should be (in, out/G)?
    // awq_dequantize_affine says:
    // s_arr shape: (N, K/G). if target is (N, K).
    // So (in, out/group_size).
    let scales_shape = vec![in_feat, out_feat / group_size]; // [32, 128/32] = [32, 4]
    let scales_data = vec![1.0f32; 32 * 4];
    let scales = Tensor::new(
        ndarray::ArrayD::from_shape_vec(IxDyn(&scales_shape), scales_data).unwrap(),
        false,
    );

    // Zeros: same shape as scales, but F32 (unpacked).
    let zeros_data = vec![0.0f32; 32 * 4];
    let qzeros = Tensor::new(
        ndarray::ArrayD::from_shape_vec(IxDyn(&scales_shape), zeros_data).unwrap(),
        false,
    );

    // Bias: optional. Llama usually has no bias for linear layers?
    // But QuantizedLinear supports it. Let's provide None to be like Llama linear.
    let bias = None;

    let qlinear =
        QuantizedLinear::new(qweight, qzeros, scales, bias, in_feat, out_feat, group_size);

    // 4. Replace linear1
    block.linear1 = LinearLayer::Quantized(qlinear);

    // 5. Run forward
    let batch = 1;
    let seq = 2;
    let input_arr = ndarray::Array::from_shape_fn(IxDyn(&[batch, seq, d_model][..]), |_| 0.1f32);
    let x = Tensor::new(input_arr, false);

    let out = block.forward_block(&x, None);

    // 6. Verify output shape
    // Output of block should be (batch, seq, d_model)
    let out_shape = out.lock().storage.shape().to_vec();
    assert_eq!(out_shape, vec![batch, seq, d_model]);

    // Ensure output is not all zeros (unless our weights made it so)
    // weights are roughly 10/11 * 1.0 = ~10. input 0.1.
    // So output should be non-zero.
    let out_data = out.lock().storage.to_f32_array();
    let sum = out_data.sum();
    assert!(
        sum.abs() > 0.001,
        "Output sum should be non-zero, got {}",
        sum
    );

    println!(
        "Mixed precision block forward successful. Output sum: {}",
        sum
    );
}
