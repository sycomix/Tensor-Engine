//! Byte Pair Encoding (BPE) tokenizer implementation.
//!
//! BPE is a subword tokenization algorithm that iteratively merges the most
//! frequent character pairs. It's used in GPT-2, RoBERTa, and many other models.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// BPE merge rule: (pair, new_symbol)
#[derive(Clone, Debug, PartialEq)]
pub struct MergeRule {
    pub left: String,
    pub right: String,
    pub new_symbol: String,
}

/// BPE tokenizer state
#[derive(Clone)]
pub struct BPEState {
    /// Vocabulary: symbol -> id
    pub vocab: HashMap<String, usize>,
    /// Reverse vocabulary: id -> symbol
    pub reverse_vocab: HashMap<usize, String>,
    /// Merge rules in order of application
    pub merges: Vec<MergeRule>,
    /// Unknown token id
    pub unk_id: usize,
    /// Pad token id
    pub pad_id: usize,
    /// BOS (beginning of sequence) token id
    pub bos_id: usize,
    /// EOS (end of sequence) token id
    pub eos_id: usize,
}

impl BPEState {
    /// Create a new BPE state from a vocabulary and merge rules.
    pub fn new(
        vocab: HashMap<String, usize>,
        merges: Vec<MergeRule>,
        unk_id: usize,
        pad_id: usize,
        bos_id: usize,
        eos_id: usize,
    ) -> Self {
        let reverse_vocab: HashMap<usize, String> = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        BPEState {
            vocab,
            reverse_vocab,
            merges,
            unk_id,
            pad_id,
            bos_id,
            eos_id,
        }
    }

    /// Get the ID for a symbol, returning unk_id if not found.
    pub fn get_id(&self, symbol: &str) -> usize {
        *self.vocab.get(symbol).unwrap_or(&self.unk_id)
    }

    /// Get the symbol for an ID, returning "<unk>" if not found.
    pub fn get_symbol(&self, id: usize) -> String {
        self.reverse_vocab.get(&id).cloned().unwrap_or_else(|| "<unk>".to_string())
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        // Split text into characters
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return vec![self.bos_id, self.eos_id];
        }

        // Initial word pieces with special boundary marker
        let mut word_pieces: Vec<String> = chars.iter().map(|c| c.to_string()).collect();

        // Apply merge rules
        for merge in &self.merges {
            let mut i = 0;
            while i < word_pieces.len() - 1 {
                if word_pieces[i] == merge.left && word_pieces[i + 1] == merge.right {
                    word_pieces.splice(i..=i + 1, vec![merge.new_symbol.clone()]);
                } else {
                    i += 1;
                }
            }
        }

        // Convert to IDs
        word_pieces.iter().map(|s| self.get_id(s)).collect()
    }

    /// Decode token IDs into text.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| self.get_symbol(id))
            .collect::<Vec<String>>()
            .join(" ")
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// BPE tokenizer builder
pub struct BPETokenizerBuilder {
    vocab: HashMap<String, usize>,
    merges: Vec<MergeRule>,
    unk_token: String,
    pad_token: String,
    bos_token: String,
    eos_token: String,
}

impl BPETokenizerBuilder {
    /// Create a new BPE tokenizer builder.
    pub fn new() -> Self {
        BPETokenizerBuilder {
            vocab: HashMap::new(),
            merges: Vec::new(),
            unk_token: "<unk>".to_string(),
            pad_token: "<pad>".to_string(),
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
        }
    }

    /// Set the unknown token.
    pub fn unk_token(mut self, token: &str) -> Self {
        self.unk_token = token.to_string();
        self
    }

    /// Set the pad token.
    pub fn pad_token(mut self, token: &str) -> Self {
        self.pad_token = token.to_string();
        self
    }

    /// Set the BOS token.
    pub fn bos_token(mut self, token: &str) -> Self {
        self.bos_token = token.to_string();
        self
    }

    /// Set the EOS token.
    pub fn eos_token(mut self, token: &str) -> Self {
        self.eos_token = token.to_string();
        self
    }

    /// Add a vocabulary entry.
    pub fn add_vocab(mut self, symbol: &str, id: usize) -> Self {
        self.vocab.insert(symbol.to_string(), id);
        self
    }

    /// Add a merge rule.
    pub fn add_merge(mut self, left: &str, right: &str, new_symbol: &str) -> Self {
        self.merges.push(MergeRule {
            left: left.to_string(),
            right: right.to_string(),
            new_symbol: new_symbol.to_string(),
        });
        self
    }

    /// Add multiple merge rules.
    pub fn add_merges(mut self, merges: Vec<(String, String, String)>) -> Self {
        for (left, right, new_symbol) in merges {
            self.merges.push(MergeRule {
                left,
                right,
                new_symbol,
            });
        }
        self
    }

    /// Build the BPE tokenizer.
    pub fn build(self) -> BPEState {
        let unk_id = self.vocab.get(&self.unk_token).copied().unwrap_or(0);
        let pad_id = self.vocab.get(&self.pad_token).copied().unwrap_or(1);
        let bos_id = self.vocab.get(&self.bos_token).copied().unwrap_or(2);
        let eos_id = self.vocab.get(&self.eos_token).copied().unwrap_or(3);

        BPEState::new(self.vocab, self.merges, unk_id, pad_id, bos_id, eos_id)
    }
}

impl Default for BPETokenizerBuilder {
    fn default() -> Self {
        BPETokenizerBuilder::new()
    }
}

/// SentencePiece-like tokenizer.
///
/// Uses a vocabulary of subword units and handles unknown characters
/// by falling back to character-level encoding.
#[derive(Clone)]
pub struct SentencePieceTokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    unk_id: usize,
    pad_id: usize,
    bos_id: usize,
    eos_id: usize,
    /// Minimum n-gram size for subword units
    min_ngram: usize,
    /// Maximum n-gram size for subword units
    max_ngram: usize,
}

impl SentencePieceTokenizer {
    /// Create a new SentencePiece tokenizer.
    pub fn new(
        vocab: HashMap<String, usize>,
        unk_id: usize,
        pad_id: usize,
        bos_id: usize,
        eos_id: usize,
    ) -> Self {
        let reverse_vocab: HashMap<usize, String> = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        SentencePieceTokenizer {
            vocab,
            reverse_vocab,
            unk_id,
            pad_id,
            bos_id,
            eos_id,
            min_ngram: 2,
            max_ngram: 6,
        }
    }

    /// Encode text using greedy longest-match decoding.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut matched = false;

            // Try longest match first
            for n in (self.min_ngram..=self.max_ngram).rev() {
                if remaining.len() >= n {
                    let candidate = &remaining[..n];
                    if let Some(&id) = self.vocab.get(candidate) {
                        ids.push(id);
                        remaining = &remaining[n..];
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                // Fall back to character-level
                if let Some(c) = remaining.chars().next() {
                    let char_str = c.to_string();
                    if let Some(&id) = self.vocab.get(&char_str) {
                        ids.push(id);
                    } else {
                        ids.push(self.unk_id);
                    }
                    remaining = &remaining[c.len_utf8()..];
                } else {
                    break;
                }
            }
        }

        ids
    }

    /// Decode token IDs into text.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| self.reverse_vocab.get(&id).cloned().unwrap_or_else(|| "?".to_string()))
            .collect::<Vec<String>>()
            .join("")
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Tiktoken-like tokenizer using byte-level BPE.
///
/// Handles Unicode characters by encoding them as bytes, then applying BPE merges.
#[derive(Clone)]
pub struct TiktokenTokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    merges: Vec<(String, String)>,
    unk_id: usize,
    pad_id: usize,
    bos_id: usize,
    eos_id: usize,
}

impl TiktokenTokenizer {
    /// Create a new Tiktoken tokenizer.
    pub fn new(
        vocab: HashMap<String, usize>,
        merges: Vec<(String, String)>,
        unk_id: usize,
        pad_id: usize,
        bos_id: usize,
        eos_id: usize,
    ) -> Self {
        let reverse_vocab: HashMap<usize, String> = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        TiktokenTokenizer {
            vocab,
            reverse_vocab,
            merges,
            unk_id,
            pad_id,
            bos_id,
            eos_id,
        }
    }

    /// Encode text using byte-level BPE.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        // Convert text to bytes
        let bytes: Vec<u8> = text.bytes().collect();
        if bytes.is_empty() {
            return vec![self.bos_id, self.eos_id];
        }

        // Initial tokenization: each byte is a token
        let mut tokens: Vec<String> = bytes.iter().map(|b| format!("<0x{:02X}>", b)).collect();

        // Apply merges
        let mut changed = true;
        while changed {
            changed = false;
            let mut i = 0;
            while i < tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&merged_id) = self.merges.iter().find(|(l, r)| (*l, *r) == pair) {
                    // Find the merged token in vocabulary
                    if let Some(merged_token) = self.reverse_vocab.get(&merged_id.1) {
                        tokens.splice(i..=i + 1, vec![merged_token.clone()]);
                        changed = true;
                    }
                } else {
                    i += 1;
                }
            }
        }

        // Convert to IDs
        tokens.iter().map(|t| self.vocab.get(t).copied().unwrap_or(self.unk_id)).collect()
    }

    /// Decode token IDs into text.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut result = String::new();
        for &id in ids {
            if let Some(symbol) = self.reverse_vocab.get(&id) {
                // Convert byte tokens back to bytes
                if symbol.starts_with("<0x") && symbol.ends_with('>') {
                    let hex = &symbol[3..symbol.len() - 1];
                    if let Ok(byte) = u8::from_str_radix(hex, 16) {
                        result.push(byte as char);
                    }
                } else {
                    result.push_str(symbol);
                }
            }
        }
        result
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Text preprocessing pipeline.
///
/// Applies a series of transformations to text before tokenization.
#[derive(Clone)]
pub struct TextPreprocessor {
    steps: Vec<TextPreprocessStep>,
}

#[derive(Clone)]
pub enum TextPreprocessStep {
    /// Lowercase the text
    Lowercase,
    /// Strip whitespace
    StripWhitespace,
    /// Remove punctuation
    RemovePunctuation,
    /// Normalize Unicode (NFC form)
    NormalizeUnicode,
    /// Custom regex replacement
    RegexReplace { pattern: String, replacement: String },
}

impl TextPreprocessor {
    /// Create a new text preprocessor.
    pub fn new() -> Self {
        TextPreprocessor { steps: Vec::new() }
    }

    /// Add a preprocessing step.
    pub fn add_step(mut self, step: TextPreprocessStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Add lowercase step.
    pub fn lowercase(mut self) -> Self {
        self.steps.push(TextPreprocessStep::Lowercase);
        self
    }

    /// Add strip whitespace step.
    pub fn strip_whitespace(mut self) -> Self {
        self.steps.push(TextPreprocessStep::StripWhitespace);
        self
    }

    /// Apply all preprocessing steps.
    pub fn preprocess(&self, text: &str) -> String {
        let mut result = text.to_string();
        for step in &self.steps {
            result = match step {
                TextPreprocessStep::Lowercase => result.to_lowercase(),
                TextPreprocessStep::StripWhitespace => result.split_whitespace().collect::<Vec<&str>>().join(" "),
                TextPreprocessStep::RemovePunctuation => result.chars()
                    .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                    .collect(),
                TextPreprocessStep::NormalizeUnicode => result, // MVP: no Unicode normalization
                TextPreprocessStep::RegexReplace { pattern, replacement } => {
                    // MVP: simple string replacement
                    result.replace(&pattern, &replacement)
                }
            };
        }
        result
    }
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        TextPreprocessor::new()
    }
}

/// Sequence padding and masking utilities.
pub struct SequenceUtils;

impl SequenceUtils {
    /// Pad a sequence to a target length with a pad token.
    pub fn pad_sequence(seq: &[usize], target_len: usize, pad_id: usize) -> Vec<usize> {
        let mut padded = seq.to_vec();
        while padded.len() < target_len {
            padded.push(pad_id);
        }
        padded
    }

    /// Create an attention mask for a padded sequence.
    /// 1 for real tokens, 0 for padding.
    pub fn create_attention_mask(seq: &[usize], pad_id: usize) -> Vec<f32> {
        seq.iter().map(|&id| if id == pad_id { 0.0 } else { 1.0 }).collect()
    }

    /// Create a causal mask for a sequence of given length.
    pub fn create_causal_mask(size: usize) -> Vec<Vec<f32>> {
        let mut mask = vec![vec![0.0; size]; size];
        for i in 0..size {
            for j in 0..=i {
                mask[i][j] = 1.0;
            }
        }
        mask
    }

    /// Create a padding mask for a batch of sequences.
    pub fn create_batch_mask(batch: &[Vec<usize>], pad_id: usize) -> Vec<Vec<f32>> {
        batch.iter().map(|seq| Self::create_attention_mask(seq, pad_id)).collect()
    }

    /// Find the maximum sequence length in a batch.
    pub fn max_seq_len(batch: &[Vec<usize>]) -> usize {
        batch.iter().map(|seq| seq.len()).max().unwrap_or(0)
    }

    /// Pad a batch of sequences to the same length.
    pub fn pad_batch(batch: Vec<Vec<usize>>, pad_id: usize) -> Vec<Vec<usize>> {
        let max_len = Self::max_seq_len(&batch);
        batch.iter().map(|seq| Self::pad_sequence(&seq, max_len, pad_id)).collect()
    }
}

/// Encode text using a HF tokenizers Tokenizer.
#[cfg(feature = "with_tokenizers")]
pub fn encode_text(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
) -> Result<Vec<u32>, String> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| format!("Tokenization error: {}", e))?;
    Ok(encoding.get_ids().to_vec())
}

/// Encode text using a HF tokenizers Tokenizer with padding.
#[cfg(feature = "with_tokenizers")]
pub fn encode_text_padded(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
    max_len: usize,
    padding_side: &str,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>), String> {
    let encoding = tokenizer
        .encode_batch(texts.to_vec(), false)
        .map_err(|e| format!("Tokenization error: {}", e))?;
    let mut ids_list = Vec::new();
    let mut mask_list = Vec::new();
    for enc in encoding {
        let mut ids = enc.get_ids().to_vec();
        let mask = vec![1u32; ids.len()];
        if ids.len() < max_len {
            let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(0);
            let pad_count = max_len - ids.len();
            if padding_side == "right" {
                ids.resize(max_len, pad_id as u32);
                let mut m = mask;
                m.resize(max_len, 0);
                mask_list.push(m);
            } else {
                let mut new_ids = vec![pad_id as u32; pad_count];
                new_ids.extend(ids);
                ids = new_ids;
                let mut m = vec![0u32; pad_count];
                m.extend(mask);
                mask_list.push(m);
            }
        } else {
            ids.truncate(max_len);
            mask_list.push(mask);
        }
        ids_list.push(ids);
    }
    Ok((ids_list, mask_list))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_encode_decode() {
        let mut vocab = HashMap::new();
        vocab.insert("<s>".to_string(), 0);
        vocab.insert("</s>".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);
        vocab.insert("<pad>".to_string(), 3);
        vocab.insert("h".to_string(), 4);
        vocab.insert("e".to_string(), 5);
        vocab.insert("l".to_string(), 6);
        vocab.insert("o".to_string(), 7);
        vocab.insert("he".to_string(), 8);
        vocab.insert("lo".to_string(), 9);

        let merges = vec![
            MergeRule { left: "h".to_string(), right: "e".to_string(), new_symbol: "he".to_string() },
            MergeRule { left: "l".to_string(), right: "o".to_string(), new_symbol: "lo".to_string() },
        ];

        let tokenizer = BPEState::new(vocab, merges, 2, 3, 0, 1);
        let ids = tokenizer.encode("hello");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_bpe_decode() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("ab".to_string(), 2);
        vocab.insert("<unk>".to_string(), 3);
        vocab.insert("<pad>".to_string(), 4);
        vocab.insert("<s>".to_string(), 5);
        vocab.insert("</s>".to_string(), 6);

        let tokenizer = BPEState::new(vocab, vec![], 3, 4, 5, 6);
        let text = tokenizer.decode(&[0, 1]);
        assert_eq!(text, "a b");
    }

    #[test]
    fn test_sentence_piece_encode() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("hel".to_string(), 2);
        vocab.insert("lo".to_string(), 3);
        vocab.insert("<unk>".to_string(), 4);
        vocab.insert("<pad>".to_string(), 5);
        vocab.insert("<s>".to_string(), 6);
        vocab.insert("</s>".to_string(), 7);

        let tokenizer = SentencePieceTokenizer::new(vocab, 4, 5, 6, 7);
        let ids = tokenizer.encode("hello world");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_sequence_padding() {
        let seq = vec![1, 2, 3];
        let padded = SequenceUtils::pad_sequence(&seq, 5, 0);
        assert_eq!(padded, vec![1, 2, 3, 0, 0]);
    }

    #[test]
    fn test_attention_mask() {
        let seq = vec![1, 2, 0, 3];
        let mask = SequenceUtils::create_attention_mask(&seq, 0);
        assert_eq!(mask, vec![1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_causal_mask() {
        let mask = SequenceUtils::create_causal_mask(3);
        assert_eq!(mask[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(mask[1], vec![1.0, 1.0, 0.0]);
        assert_eq!(mask[2], vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_batch_padding() {
        let batch = vec![vec![1, 2], vec![3, 4, 5], vec![6]];
        let padded = SequenceUtils::pad_batch(batch, 0);
        assert_eq!(padded[0].len(), 3);
        assert_eq!(padded[1].len(), 3);
        assert_eq!(padded[2].len(), 3);
        assert_eq!(padded[2], vec![6, 0, 0]);
    }

    #[test]
    fn test_text_preprocessor() {
        let preprocessor = TextPreprocessor::new()
            .lowercase()
            .strip_whitespace();
        let result = preprocessor.preprocess("  Hello World  ");
        assert_eq!(result, "hello world");
    }
}
