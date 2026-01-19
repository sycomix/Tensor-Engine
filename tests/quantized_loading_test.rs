#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use std::collections::HashMap;
    use tensor_engine::nn::{linear_dispatch::LinearLayer, Module, TransformerBlock};
    use tensor_engine::tensor::Tensor;

    #[test]
    fn test_quantized_loading_transformer() {
        // 1. Create a TransformerBlock (F32 by default)
        let mut block = TransformerBlock::new(32, 64, 4).expect("new block"); // d_model=32, d_ff=64, heads=4

        // 2. Create a "fake" state dict with quantized keys for one of the headers (e.g. mha.linear_q)
        // TransformerBlock maps "mha" field to "self_attn" prefix in load_state_dict.
        let prefix = "self_attn.linear_q";
        let mut state_dict = HashMap::new();

        // Quantized Linear requires: qweight, qzeros, scales.
        // d_model=32 -> in_features=32, out_features=32.
        // Let's assume group_size=32 for simplicity so output_features/group_size = 1.
        // qweight shape: [in=32, out/2=16] (packed 4bit u8)
        let qweight = Tensor::new(ArrayD::zeros(IxDyn(&[32, 16][..])), false);
        // qzeros shape: [in=32, out/group_size=1] (f32)
        let qzeros = Tensor::new(ArrayD::zeros(IxDyn(&[32, 1][..])), false);
        // scales shape: [in=32, out/group_size=1] (f32)
        let scales = Tensor::new(ArrayD::zeros(IxDyn(&[32, 1][..])), false);

        state_dict.insert(format!("{}.qweight", prefix), qweight);
        state_dict.insert(format!("{}.qzeros", prefix), qzeros);
        state_dict.insert(format!("{}.scales", prefix), scales);

        // 3. Load state dict
        // We need to pass the prefix "mha.linear_q" actually, but TransformerBlock.load_state_dict takes prefix for itself.
        // So we call block.load_state_dict(state, "block") and the block calls mha.linear_q with "block.mha.linear_q"

        // Let's adjust keys to full path
        let mut full_state_dict = HashMap::new();
        for (k, v) in state_dict {
            full_state_dict.insert(format!("test.{}", k), v);
        }

        block
            .load_state_dict(&full_state_dict, "test")
            .expect("load_state_dict");

        // 4. Verify that linear_q is now Quantized
        match &block.mha.linear_q {
            LinearLayer::Quantized(_) => {
                println!("Successfully upgraded to QuantizedLinear");
            }
            LinearLayer::F32(_) => {
                panic!("Failed to switch to QuantizedLinear!");
            }
        }

        // 5. Verify other layers are still F32 (e.g. linear_k)
        match &block.mha.linear_k {
            LinearLayer::F32(_) => {}
            _ => panic!("linear_k should remain F32"),
        }
    }
}
