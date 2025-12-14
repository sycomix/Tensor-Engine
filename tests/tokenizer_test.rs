#![allow(unused)]
#[cfg(feature = "with_tokenizers")]
mod tokenizer_tests {
    use tensor_engine::io::tokenizers::{encode_text, load_tokenizer_from_file};

    #[test]
    fn test_tokenizer_load_and_encode() {
        // This test assumes an example tokenizer file exists. We'll use a small fallback
        // by building a tiny tokenizer JSON in test data; but for now, test load failure gracefully.
        let res = load_tokenizer_from_file("nonexistent.json");
        assert!(res.is_err());
    }
}
