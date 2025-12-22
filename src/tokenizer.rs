use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

#[cfg(feature = "with_tokenizers")]
use tokenizers::Tokenizer as HFTokenizer;

/// A tokenizer wrapper. Prefer the `tokenizers` crate when available (feature
/// "with_tokenizers") for exact compatibility with HuggingFace tokenizers.
/// Fall back to a pragmatic longest-match implementation for demo purposes.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    /// If the HF tokenizers crate was enabled and the file parsed successfully,
    /// this will be Some(tokenizer) and we delegate encode/decode to it.
    #[cfg(feature = "with_tokenizers")]
    pub hf: Option<HFTokenizer>,

    /// token -> id (fallback)
    pub vocab: HashMap<String, usize>,
    /// id -> token (fallback)
    pub inv_vocab: HashMap<usize, String>,
}

impl Tokenizer {
    /// Load tokenizer from a JSON file produced by HuggingFace `tokenizers`.
    /// When `with_tokenizers` feature is enabled, attempt to parse with the
    /// tokenizers crate (recommended). Otherwise fall back to a simple JSON
    /// vocabulary loader.
    pub fn from_json(path: &str) -> Result<Self, String> {
        let mut vocab_map: HashMap<String, usize> = HashMap::new();
        let mut inv: HashMap<usize, String> = HashMap::new();

        #[cfg(feature = "with_tokenizers")]
        {
            // Try the HF tokenizers loader first; if it fails, fall back to JSON parse
            match HFTokenizer::from_file(path) {
                Ok(tok) => return Ok(Tokenizer { hf: Some(tok), vocab: HashMap::new(), inv_vocab: HashMap::new() }),
                Err(e) => {
                    log::warn!("tokenizers::Tokenizer::from_file failed: {} - falling back to JSON vocab parse", e);
                    // fallthrough to JSON parse
                }
            }
        }

        let mut f = File::open(path).map_err(|e| format!("Failed to open tokenizer file: {}", e))?;
        let mut s = String::new();
        f.read_to_string(&mut s).map_err(|e| format!("Failed to read tokenizer file: {}", e))?;

        let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        // Try common locations for vocab mapping
        if let Some(model) = v.get("model") {
            if let Some(obj) = model.get("vocab") {
                if let Some(map) = obj.as_object() {
                    for (k, val) in map.iter() {
                        if let Some(id) = val.as_u64() {
                            vocab_map.insert(k.clone(), id as usize);
                        }
                    }
                }
            }
        }

        // Also try top-level "vocab"
        if vocab_map.is_empty() {
            if let Some(obj) = v.get("vocab") {
                if let Some(map) = obj.as_object() {
                    for (k, val) in map.iter() {
                        if let Some(id) = val.as_u64() {
                            vocab_map.insert(k.clone(), id as usize);
                        }
                    }
                }
            }
        }

        // Try to collect added_tokens if present (ensure special tokens available)
        if let Some(added) = v.get("added_tokens") {
            if let Some(arr) = added.as_array() {
                for tok in arr.iter() {
                    if let (Some(content), Some(idv)) = (tok.get("content"), tok.get("id")) {
                        if let (Some(s), Some(id)) = (content.as_str(), idv.as_u64()) {
                            vocab_map.insert(s.to_string(), id as usize);
                        }
                    }
                }
            }
        }

        // Fallback: if no vocab map found, but a "tokens" array exists, use that
        if vocab_map.is_empty() {
            if let Some(tokens) = v.get("tokens") {
                if let Some(arr) = tokens.as_array() {
                    for (i, item) in arr.iter().enumerate() {
                        if let Some(s) = item.as_str() {
                            vocab_map.insert(s.to_string(), i);
                        }
                    }
                }
            }
        }

        if vocab_map.is_empty() {
            return Err("No vocab mapping could be found in tokenizer file".to_string());
        }

        for (k, &v) in vocab_map.iter() {
            inv.insert(v, k.clone());
        }

        Ok(Tokenizer { #[cfg(feature = "with_tokenizers")] hf: None, vocab: vocab_map, inv_vocab: inv })
    }

    /// Encode text to token ids. Prefer HF tokenizer if available; otherwise
    /// fallback to greedy longest-match.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        #[cfg(feature = "with_tokenizers")]
        {
            if let Some(tok) = &self.hf {
                if let Ok(output) = tok.encode(text, true) {
                    return output.get_ids().iter().map(|&i| i as usize).collect();
                }
            }
        }
        // Fallback greedy tokenizer
        let mut out: Vec<usize> = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut i = 0;
        while i < n {
            // Try decreasing lengths from remaining length down to 1
            let mut matched = false;
            let mut max_len = 0usize;
            let mut max_id: usize = 0;
            // Limit match length to remaining chars
            for j in (i+1..=n).rev() {
                let slice: String = chars[i..j].iter().collect();
                if let Some(id) = self.vocab.get(&slice) {
                    matched = true;
                    max_len = j - i;
                    max_id = *id;
                    break;
                }
            }
            if matched {
                out.push(max_id);
                i += max_len;
            } else {
                // If no multi-char token matches, try single-char
                let s: String = chars[i..i+1].iter().collect();
                if let Some(id) = self.vocab.get(&s) {
                    out.push(*id);
                } else {
                    // Unknown token: as a fallback, push 0 if present or skip
                    if let Some(z) = self.vocab.get("<|unknown|>") {
                        out.push(*z);
                    }
                }
                i += 1;
            }
        }
        out
    }

    /// Decode token ids back to a string. Prefer HF tokenizer decoding when
    /// available (preserves special tokens and normalization), else concatenate
    /// the token strings from the fallback vocab.
    pub fn decode(&self, ids: &[usize]) -> String {
        #[cfg(feature = "with_tokenizers")]
        {
            if let Some(tok) = &self.hf {
                // tokenizers expects u32 ids
                let ids_u32: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
                if let Ok(s) = tok.decode(&ids_u32, false) {
                    return s;
                }
            }
        }
        let mut s = String::new();
        for &id in ids {
            if let Some(tok) = self.inv_vocab.get(&id) {
                s.push_str(tok);
            }
        }
        s
    }

    /// Number of entries in the vocabulary
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Map token string to id, if present in vocabulary
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.vocab.get(token).cloned()
    }
}
