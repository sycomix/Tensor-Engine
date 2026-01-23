#[cfg(test)]
#[cfg(feature = "hf_compat")]
mod tests {
    use hf_compat::token_sampler::TokenSampler;
    use ndarray::Array;
    use tensor_engine::hf_bridge::sample_from_tensor;
    use tensor_engine::tensor::Tensor;

    #[test]
    fn test_sample_from_tensor() {
        // vocab size 3, logits shape (3,1)
        let arr = Array::from_shape_vec((3, 1), vec![0.1f32, 10.0f32, 0.2f32])
            .unwrap()
            .into_dyn();
        let t = Tensor::new(arr, false);
        let mut map = std::collections::BTreeMap::new();
        map.insert("a".to_string(), 0usize);
        map.insert("b".to_string(), 1usize);
        map.insert("c".to_string(), 2usize);
        // create tokenizer directly
        let tokenizer = hf_compat::tokenizer::Tokenizer { vocab: map };
        let sampler = TokenSampler::new().top_k(2).top_p(0.9);
        let (id, prob) =
            sample_from_tensor(&sampler, &t, &tokenizer, &[]).expect("sampling failed");
        assert!(prob >= 0.0 && prob <= 1.0);
        assert!(id < 3usize);
    }
}
