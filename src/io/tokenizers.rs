//! Feature-gated Rust wrapper around Hugging Face Tokenizers crate.
#![allow(dead_code)]

#[cfg(feature = "with_tokenizers")]
use tokenizers::Tokenizer;

#[cfg(feature = "with_tokenizers")]
use std::path::Path;

#[cfg(feature = "with_tokenizers")]
pub fn load_tokenizer_from_file(path: &str) -> Result<Tokenizer, String> {
    Tokenizer::from_file(Path::new(path))
        .map_err(|e| format!("Failed to load tokenizer {}: {}", path, e))
}

#[cfg(feature = "with_tokenizers")]
pub fn encode_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    tokenizer
        .encode(text, true)
        .map_err(|e| format!("Tokenization error: {}", e))
        .map(|encoding| encoding.get_ids().to_vec())
}

#[cfg(feature = "with_tokenizers")]
pub fn decode_tokens(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String, String> {
    // The tokenizers decode API expects a slice of u32 or Vec<u32>. We'll pass a slice directly.
    tokenizer
        .decode(ids.to_vec(), true)
        .map_err(|e| format!("Token decoding error: {}", e))
}

#[cfg(not(feature = "with_tokenizers"))]
pub fn load_tokenizer_from_file(_path: &str) -> Result<(), String> { Err("Tokenizers feature not enabled".into()) }

#[cfg(not(feature = "with_tokenizers"))]
pub fn encode_text(_t: &(), _text: &str) -> Result<Vec<u32>, String> { Err("Tokenizers feature not enabled".into()) }

#[cfg(not(feature = "with_tokenizers"))]
pub fn decode_tokens(_t: &(), _ids: &[u32]) -> Result<String, String> { Err("Tokenizers feature not enabled".into()) }
