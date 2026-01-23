#[cfg(test)]
mod tests {
    use hf_compat::tokenizer::Tokenizer;
    // no extra imports needed
    use tempfile::tempdir;

    #[test]
    fn test_tokenizer_json_load_and_tokenize() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vocab.json");
        let vocab_json = r#"{ "hello": 1, "world": 2, "!": 3 }"#;
        std::fs::write(&path, vocab_json).unwrap();
        let tok = Tokenizer::from_json(path.to_str().unwrap()).unwrap();
        let ids = tok.tokenize_to_ids("hello world!");
        assert_eq!(ids.len(), 3);
    }
}
