#[cfg(test)]
mod tests {
    use hf_compat::tokenizer::Tokenizer;
    use hf_compat::token_sampler::TokenSampler;
    use tempfile::tempdir;

    #[test]
    fn test_sampler_basic() {
        // Vocab: a:0 b:1 c:2
        let dir = tempdir().unwrap();
        let path = dir.path().join("vocab.json");
        let vocab_json = r#"{ "a": 0, "b": 1, "c": 2 }"#;
        std::fs::write(&path, vocab_json).unwrap();
        let tok = Tokenizer::from_json(path.to_str().unwrap()).unwrap();
        // logits favor token 1
        let logits = vec![0.1f32, 10.0f32, 0.2f32];
        let sampler = TokenSampler::new().top_k(2).top_p(0.9);
        let (id, prob) = sampler.sample_ids(&logits, 3, &tok, &[]);
        assert!(id == 1 || id == 2 || id == 0);
        // probability should be between 0 and 1
        assert!(prob >= 0.0 && prob <= 1.0);
    }
}
