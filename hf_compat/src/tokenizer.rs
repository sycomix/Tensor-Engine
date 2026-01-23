use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JSONError(#[from] serde_json::Error),
    #[error("Unknown piece type")] 
    UnknownPieceType,
}

/// A small tokenizer compatible with the reference "longest-match" approach.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    pub vocab: BTreeMap<String, usize>,
}

impl Tokenizer {
    /// Load tokenizer from a JSON file with a mapping token -> id.
    pub fn from_json(path: &str) -> Result<Self, TokenizerError> {
        let mut f = File::open(path)?;
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        let v: BTreeMap<String, usize> = serde_json::from_str(&s)?;
        Ok(Tokenizer { vocab: v })
    }

    /// Greedy longest match tokenization into pieces (strings)
    pub fn tokenize_to_pieces<'a, S: AsRef<str>>(&self, s_in: S) -> Vec<&'a str> {
        // Note: For simplicity and safety we operate on owned String slices.
        let _s: String = s_in.as_ref().to_string();
        // This function is a compatibility shim; prefer `tokenize_to_ids` which returns ids.
        Vec::new()
    }

    /// Tokenize and return token ids using greedy matching by decreasing substring length.
    pub fn tokenize_to_ids(&self, s: &str) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        let mut i = 0;
        while i < n {
            let mut matched = false;
            let mut max_len = 0usize;
            let mut max_id: usize = 0;
            for j in (i + 1..=n).rev() {
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
                let s1: String = chars[i..i + 1].iter().collect();
                if let Some(id) = self.vocab.get(&s1) {
                    out.push(*id);
                }
                i += 1;
            }
        }
        out
    }
}