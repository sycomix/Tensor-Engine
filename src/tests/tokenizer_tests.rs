use crate::tokenizer::Tokenizer;

#[test]
fn load_llama_tokenizer_basic() {
    // Use the example tokenizer.json included in repo
    let path = "../examples/Llama-3.2-1B/tokenizer.json";
    let tok = Tokenizer::from_json(path).expect("Failed to load tokenizer.json");
    assert!(tok.vocab_size() > 1000, "vocab too small");
    // Basic encode/decode sanity: special tokens should be preserved
    let ids = tok.encode("<|begin_of_text|>");
    assert!(!ids.is_empty(), "encode returned empty for special token");
    let s = tok.decode(&ids);
    assert!(s.contains("<|begin_of_text|>"), "decode did not contain special token");
}
