#[cfg(feature = "backend_wgpu")]
#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use tensor_engine::backend::set_wgpu_backend;
    use tensor_engine::dtype::DType;
    // TensorStorage not needed if we use to_f32_array
    use tensor_engine::nn::quantized::QuantizedLinear;
    use tensor_engine::nn::Module;
    use tensor_engine::tensor::Tensor;

    #[test]
    fn test_wgpu_quantized_matmul_simple() {
        // 1. Initialize WGPU Backend
        if let Err(e) = set_wgpu_backend() {
            println!("Skipping WGPU test: {}", e);
            return;
        }

        // 2. Setup Quantized Linear Layer
        let in_features = 64;
        let out_features = 64;
        let group_size = 32;

        // Create dummy quantized weights
        // qweight: [64, 32] (packed u8)
        let qweight_data: Vec<u8> = (0..in_features * out_features / 2)
            .map(|i| (i % 255) as u8)
            .collect();

        // Tensor::new expects f32, so we convert u8->f32 and use new_with_dtype(..., U8) which converts storage to U8
        let qweight_f32: Vec<f32> = qweight_data.iter().map(|&x| x as f32).collect();
        let qweight = Tensor::new_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[in_features, out_features / 2]), qweight_f32).unwrap(),
            false,
            DType::U8,
        );

        // Scales: [64, 2]
        let scales_data: Vec<f32> = (0..in_features * (out_features / group_size))
            .map(|i| (i % 10) as f32 * 0.1)
            .collect();
        let scales = Tensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[in_features, out_features / group_size]),
                scales_data,
            )
                .unwrap(),
            false,
        );

        // Zeros: [64, 2]
        let zeros_data: Vec<f32> = (0..in_features * (out_features / group_size))
            .map(|i| (i % 5) as f32)
            .collect();
        let qzeros = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[in_features, out_features / group_size]), zeros_data)
                .unwrap(),
            false,
        );

        // Bias
        let bias_data: Vec<f32> = (0..out_features).map(|i| i as f32 * 0.01).collect();
        let bias = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[out_features]), bias_data).unwrap(),
            false,
        );

        let layer = QuantizedLinear::new(
            qweight.clone(), // Clone since we access qweight later
            qzeros.clone(),
            scales.clone(),
            Some(bias.clone()),
            in_features,
            out_features,
            group_size,
        );

        // 3. Input
        let input_data: Vec<f32> = (0..8 * in_features)
            .map(|i| (i % 10) as f32 * 0.1)
            .collect();
        let input = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[8, in_features]), input_data).unwrap(),
            false,
        );

        // 4. Run on WGPU
        let output_wgpu = layer.forward(&input);
        let output_wgpu_arr = output_wgpu.to_f32_array();

        println!(
            "WGPU Output Sample: {:?}",
            output_wgpu_arr.slice(ndarray::s![0, ..5])
        );

        // 5. Run on CPU (Reference - Local Implementation using public API)
        // Convert input to Array2
        let input_arr = input.to_f32_array();
        let input_2d = input_arr.into_dimensionality::<ndarray::Ix2>().unwrap();

        let mut w_dequant = ArrayD::<f32>::zeros(IxDyn(&[in_features, out_features]));

        // Access data using to_f32_array()
        // qweight was stored as U8, but to_f32_array converts it to f32.
        // We cast back to u8 for logic.
        let q_float = qweight.to_f32_array();
        let scales_arr = scales.to_f32_array();
        let qzeros_arr = qzeros.to_f32_array();

        // Simple dequant loop
        for i in 0..in_features {
            for j in 0..out_features {
                // qweight has shape [in, out/2]
                let q_idx = j / 2;
                // Access using IxDyn
                let byte = q_float[IxDyn(&[i, q_idx])] as u8;

                // AWQ unpacking:
                let nibble = if j % 2 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };

                let g = j / group_size;
                let s = scales_arr[IxDyn(&[i, g])];
                let z = qzeros_arr[IxDyn(&[i, g])];

                w_dequant[[i, j]] = (nibble as f32 - z) * s;
            }
        }

        // Matmul
        let w_2d = w_dequant.into_dimensionality::<ndarray::Ix2>().unwrap();
        let out_ref = input_2d.dot(&w_2d);

        // Add bias
        let bias_arr = bias.to_f32_array();
        let output_cpu_arr = out_ref + bias_arr;

        println!(
            "Link Output Sample (Local Ref): {:?}",
            output_cpu_arr.slice(ndarray::s![0, ..5])
        );

        // 6. Compare
        let diff = &output_wgpu_arr - &output_cpu_arr;
        let mse = diff.mapv(|x| x * x).sum() / diff.len() as f32;

        println!("MSE: {}", mse);

        // Tolerance: f32 precision drift between CPU (f64 accum sometimes?) and GPU (f32)
        // WGSL generic compute usually f32.
        assert!(mse < 1e-4, "MSE too high: {}", mse);
    }
}
