//! Feature-gated Rust wrapper around Hugging Face Tokenizers crate.

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
    tokenizer
        .decode(ids, true)
        .map_err(|e| format!("Token decoding error: {}", e))
}

#[cfg(not(feature = "with_tokenizers"))]
pub fn load_tokenizer_from_file(_path: &str) -> Result<(), String> { Err("Tokenizers feature not enabled".into()) }

#[cfg(not(feature = "with_tokenizers"))]
pub fn encode_text(_t: &(), _text: &str) -> Result<Vec<u32>, String> { Err("Tokenizers feature not enabled".into()) }

#[cfg(not(feature = "with_tokenizers"))]
pub fn decode_tokens(_t: &(), _ids: &[u32]) -> Result<String, String> { Err("Tokenizers feature not enabled".into()) }


#[cfg(test)]
mod tests {
    #[cfg(not(feature = "with_tokenizers"))]
    #[test]
    fn not_with_tokenizers_returns_err() {
        assert!(super::load_tokenizer_from_file("nonexistent").is_err());
        assert!(super::encode_text(&(), "hello").is_err());
        assert!(super::decode_tokens(&(), &[1, 2, 3]).is_err());
    }

    #[cfg(feature = "with_tokenizers")]
    #[test]
    fn with_tokenizers_invalid_path_errs() {
        // When feature is enabled, loading a missing file should still return Err
        assert!(super::load_tokenizer_from_file("nonexistent").is_err());
    }
} 
