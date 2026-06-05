use std::collections::HashMap;

/// WordPiece tokenizer (used by BERT).
///
/// Implements the WordPiece subword tokenization algorithm. It uses a greedy
/// longest-match-first approach to split words into subword units from a
/// pre-built vocabulary.
///
/// # Reference
/// [`Schuster & Nakajima, 2012`](https://dl.acm.org/doi/10.3115/1258146.1258157)
///
/// # Example
/// ```
/// use tensor_engine::nn::wordpiece_tokenizer::{WordPieceTokenizer, WordPieceConfig};
///
/// let config = WordPieceConfig::default();
/// let mut tokenizer = WordPieceTokenizer::new(config);
/// tokenizer.train(vec!["hello world".to_string(), "helloworld".to_string()]);
/// let tokens = tokenizer.encode("hello world");
/// ```
#[derive(Clone)]
pub struct WordPieceTokenizer {
    vocab: HashMap<String, usize>,
    inv_vocab: HashMap<usize, String>,
    config: WordPieceConfig,
    /// Cache for encoded results
    encode_cache: HashMap<String, Vec<usize>>,
}

/// Configuration for WordPiece tokenization.
#[derive(Clone, Debug)]
pub struct WordPieceConfig {
    /// Maximum vocabulary size
    pub vocab_size: usize,
    /// Indicator for subword tokens (typically "[MASK]", "[UNK]", "[CLS]", "[SEP]")
    pub mask_token: String,
    pub unk_token: String,
    pub cls_token: String,
    pub sep_token: String,
    /// Indicator for continuation of a word (typically "##")
    pub continuation_indicator: String,
    /// Whether to lowercase text before tokenization
    pub do_lower_case: bool,
}

impl Default for WordPieceConfig {
    fn default() -> Self {
        WordPieceConfig {
            vocab_size: 30000,
            mask_token: "[MASK]".to_string(),
            unk_token: "[UNK]".to_string(),
            cls_token: "[CLS]".to_string(),
            sep_token: "[SEP]".to_string(),
            continuation_indicator: "##".to_string(),
            do_lower_case: true,
        }
    }
}

impl WordPieceTokenizer {
    /// Create a new WordPiece tokenizer with default configuration.
    pub fn new(config: WordPieceConfig) -> Self {
        let mut vocab = HashMap::new();
        let mut inv_vocab = HashMap::new();
        let mut next_id = 0;

        // Reserve IDs for special tokens
        let mask_id = next_id;
        next_id += 1;
        vocab.insert(config.mask_token.clone(), mask_id);
        inv_vocab.insert(mask_id, config.mask_token.clone());

        let unk_id = next_id;
        next_id += 1;
        vocab.insert(config.unk_token.clone(), unk_id);
        inv_vocab.insert(unk_id, config.unk_token.clone());

        let cls_id = next_id;
        next_id = next_id + 1;
        vocab.insert(config.cls_token.clone(), cls_id);
        inv_vocab.insert(cls_id, config.cls_token.clone());

        let sep_id = next_id;
        next_id = next_id + 1;
        vocab.insert(config.sep_token.clone(), sep_id);
        inv_vocab.insert(sep_id, config.sep_token.clone());

        WordPieceTokenizer {
            vocab,
            inv_vocab,
            config,
            encode_cache: HashMap::new(),
        }
    }

    /// Train the WordPiece tokenizer on a corpus of text strings.
    ///
    /// This builds a vocabulary using character frequency analysis and
    /// iteratively adds the most common character bigrams.
    pub fn train(&mut self, corpus: Vec<String>) {
        // Count character bigram frequencies
        let mut bigram_freqs: HashMap<(char, char), usize> = HashMap::new();
        let mut char_freqs: HashMap<char, usize> = HashMap::new();
        let mut word_freqs: HashMap<String, usize> = HashMap::new();

        for text in &corpus {
            let normalized = if self.config.do_lower_case {
                text.to_lowercase()
            } else {
                text.clone()
            };

            let words: Vec<String> = normalized
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();

            for word in &words {
                *word_freqs.entry(word.clone()).or_insert(0) += 1;
                for ch in word.chars() {
                    *char_freqs.entry(ch).or_insert(0) += 1;
                }
                for i in 0..word.len().saturating_sub(1) {
                    let chars: Vec<char> = word.chars().collect();
                    let pair = (chars[i], chars[i + 1]);
                    *bigram_freqs.entry(pair).or_insert(0) += 1;
                }
            }
        }

        // Initialize vocabulary with characters (sorted by frequency, descending)
        let mut chars: Vec<(char, usize)> = char_freqs.into_iter().collect();
        chars.sort_by(|a, b| b.1.cmp(&a.1));

        let max_chars = self.config.vocab_size.saturating_sub(4); // Reserve for special tokens
        for (ch, _freq) in chars.iter().take(max_chars) {
            let token = ch.to_string();
            let id = self.vocab.len();
            self.vocab.insert(token, id);
            self.inv_vocab.insert(id, ch.to_string());
        }

        // Iteratively add most frequent bigrams that consist of existing tokens
        let max_merges = self.config.vocab_size.saturating_sub(self.vocab.len());
        for _ in 0..max_merges {
            let best_bigram = bigram_freqs
                .iter()
                .filter(|&((c1, c2), _)| {
                    // Both characters must be in vocab
                    self.vocab.contains_key(&c1.to_string())
                        && self.vocab.contains_key(&c2.to_string())
                })
                .max_by_key(|(_, freq)| *freq);

            match best_bigram {
                Some(((c1, c2), _freq)) => {
                    let merged = format!("{}{}", c1, c2);
                    let id = self.vocab.len();
                    self.vocab.insert(merged.clone(), id);
                    self.inv_vocab.insert(id, merged);
                }
                None => break,
            }
        }
    }

    /// Encode a text string into token IDs.
    ///
    /// Returns a vector of token IDs. The output does not include [CLS] or [SEP] tokens.
    /// Use `encode_with_special` for sequences that need those.
    pub fn encode(&mut self, text: &str) -> Vec<usize> {
        if let Some(cached) = self.encode_cache.get(text) {
            return cached.clone();
        }

        let tokens = self.tokenize(text);
        let ids: Vec<usize> = tokens
            .iter()
            .filter_map(|t| self.vocab.get(t).copied())
            .collect();

        self.encode_cache.insert(text.to_string(), ids.clone());
        ids
    }

    /// Encode with [CLS] and [SEP] tokens prepended/appended.
    pub fn encode_with_special(&mut self, text: &str) -> Vec<usize> {
        let mut ids = vec![self.vocab[&self.config.cls_token]];
        ids.extend(self.encode(text));
        ids.push(self.vocab[&self.config.sep_token]);
        ids
    }

    /// Encode two sequences for classification tasks (e.g., sentence pairs).
    /// Format: [CLS] seq1 [SEP] seq2 [SEP]
    pub fn encode_pair(&mut self, text1: &str, text2: &str) -> Vec<usize> {
        let mut ids = vec![self.vocab[&self.config.cls_token]];
        ids.extend(self.encode(text1));
        ids.push(self.vocab[&self.config.sep_token]);
        ids.extend(self.encode(text2));
        ids.push(self.vocab[&self.config.sep_token]);
        ids
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for &id in token_ids {
            if let Some(token) = self.inv_vocab.get(&id) {
                parts.push(token.clone());
            }
        }
        let joined = parts.join("");
        // Remove continuation indicators and add spaces
        let cleaned = joined.replace(&self.config.continuation_indicator, "");
        cleaned
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Check if a token is in the vocabulary.
    pub fn contains(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    /// Get the token ID for a given token string.
    pub fn get_token_id(&self, token: &str) -> Option<usize> {
        self.vocab.get(token).copied()
    }

    /// Get the token string for a given ID.
    pub fn get_token(&self, id: usize) -> Option<String> {
        self.inv_vocab.get(&id).cloned()
    }

    /// Get the ID of the [UNK] token.
    pub fn unk_id(&self) -> usize {
        self.vocab[&self.config.unk_token]
    }

    /// Get the ID of the [CLS] token.
    pub fn cls_id(&self) -> usize {
        self.vocab[&self.config.cls_token]
    }

    /// Get the ID of the [SEP] token.
    pub fn sep_id(&self) -> usize {
        self.vocab[&self.config.sep_token]
    }

    /// Get the ID of the [MASK] token.
    pub fn mask_id(&self) -> usize {
        self.vocab[&self.config.mask_token]
    }

    /// Clear the encode cache.
    pub fn clear_cache(&mut self) {
        self.encode_cache.clear();
    }

    /// Get the configuration.
    pub fn config(&self) -> &WordPieceConfig {
        &self.config
    }

    // --- Internal helpers ---

    /// Tokenize a single word into WordPiece subwords.
    fn tokenize_word(&self, word: &str) -> Vec<String> {
        let normalized = if self.config.do_lower_case {
            word.to_lowercase()
        } else {
            word.to_string()
        };

        // If the word is in vocab, return it directly
        if self.vocab.contains_key(&normalized) {
            return vec![normalized];
        }

        let mut output: Vec<String> = Vec::new();
        let mut start = 0;
        let chars: Vec<char> = normalized.chars().collect();
        let mut subword = String::new();

        for i in 0..chars.len() {
            if i > 0 {
                subword = format!("{}{}", self.config.continuation_indicator, chars[i]);
            } else {
                subword = chars[i].to_string();
            }

            // Check if subword is in vocab
            if i < chars.len() - 1 && self.vocab.contains_key(&subword) {
                // Check if the remaining suffix could also be tokenized
                let remaining: String = chars[i + 1..].iter().collect();
                if !remaining.is_empty() && self.vocab.contains_key(&remaining) {
                    // Both current subword and remaining are in vocab, prefer longer match
                    output.push(subword.clone());
                    output.push(remaining);
                    return output;
                }
                output.push(subword.clone());
            } else if i == chars.len() - 1 {
                // Last character
                if self.vocab.contains_key(&subword) {
                    output.push(subword);
                } else {
                    // Unknown word: add [UNK]
                    output.push(self.config.unk_token.clone());
                }
            }
        }

        // If nothing was added, use [UNK]
        if output.is_empty() {
            output.push(self.config.unk_token.clone());
        }

        output
    }

    /// Tokenize text into WordPiece subwords.
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut output: Vec<String> = Vec::new();

        for word in text.split_whitespace() {
            let subwords = self.tokenize_word(word);
            output.extend(subwords);
        }

        output
    }
}

#[cfg(test)]
mod wordpiece_tests {
    use super::*;

    #[test]
    fn test_wordpiece_basic_training() {
        let config = WordPieceConfig::default();
        let mut tokenizer = WordPieceTokenizer::new(config);

        let corpus = vec![
            "hello world".to_string(),
            "hello world!".to_string(),
            "world hello".to_string(),
            "helloworld".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];

        tokenizer.train(corpus);

        assert!(tokenizer.vocab_size() > 4); // At least 4 special tokens
    }

    #[test]
    fn test_wordpiece_encode_decode() {
        let config = WordPieceConfig::default();
        let mut tokenizer = WordPieceTokenizer::new(config);

        let corpus = vec![
            "hello world".to_string(),
            "helloworld".to_string(),
            "hello".to_string(),
            "world".to_string(),
            "hi there".to_string(),
        ];

        tokenizer.train(corpus);

        let tokens = tokenizer.encode("hello world");
        assert!(!tokens.is_empty());

        let decoded = tokenizer.decode(&tokens);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_wordpiece_special_tokens() {
        let config = WordPieceConfig::default();
        let tokenizer = WordPieceTokenizer::new(config);

        // Special tokens should always be in vocab
        assert!(tokenizer.contains(&tokenizer.config().mask_token));
        assert!(tokenizer.contains(&tokenizer.config().unk_token));
        assert!(tokenizer.contains(&tokenizer.config().cls_token));
        assert!(tokenizer.contains(&tokenizer.config().sep_token));

        assert!(tokenizer.unk_id() > 0);
        assert!(tokenizer.cls_id() > 0);
        assert!(tokenizer.sep_id() > 0);
        assert!(tokenizer.mask_id() > 0);
    }

    #[test]
    fn test_wordpiece_encode_with_special() {
        let config = WordPieceConfig::default();
        let mut tokenizer = WordPieceTokenizer::new(config);

        let corpus = vec!["hello world".to_string()];
        tokenizer.train(corpus);

        let tokens = tokenizer.encode_with_special("hello");
        assert!(tokens.len() >= 2); // At least [CLS] and [SEP]
        assert_eq!(tokens[0], tokenizer.cls_id());
        assert_eq!(tokens[tokens.len() - 1], tokenizer.sep_id());
    }

    #[test]
    fn test_wordpiece_encode_pair() {
        let config = WordPieceConfig::default();
        let mut tokenizer = WordPieceTokenizer::new(config);

        let corpus = vec!["hello world".to_string(), "foo bar".to_string()];
        tokenizer.train(corpus);

        let tokens = tokenizer.encode_pair("hello", "world");
        assert!(tokens.len() >= 3); // [CLS] + [SEP] + [SEP] minimum
        assert_eq!(tokens[0], tokenizer.cls_id());
    }

    #[test]
    fn test_wordpiece_unk_token() {
        let config = WordPieceConfig::default();
        let mut tokenizer = WordPieceTokenizer::new(config);

        // Train on limited corpus
        let corpus = vec!["hello world".to_string()];
        tokenizer.train(corpus);

        // Encode a completely unknown word
        let tokens = tokenizer.encode("xyzxyzxyz");
        assert!(!tokens.is_empty());
        assert!(tokens.contains(&tokenizer.unk_id()));
    }

    #[test]
    fn test_wordpiece_cache() {
        let config = WordPieceConfig::default();
        let mut tokenizer = WordPieceTokenizer::new(config);

        let corpus = vec!["hello world".to_string()];
        tokenizer.train(corpus);

        let tokens1 = tokenizer.encode("hello world");
        let tokens2 = tokenizer.encode("hello world");
        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_wordpiece_clear_cache() {
        let config = WordPieceConfig::default();
        let mut tokenizer = WordPieceTokenizer::new(config);

        let corpus = vec!["hello world".to_string()];
        tokenizer.train(corpus);

        tokenizer.encode("hello world");
        assert!(!tokenizer.encode_cache.is_empty());

        tokenizer.clear_cache();
        assert!(tokenizer.encode_cache.is_empty());
    }

    #[test]
    fn test_wordpiece_get_set_token() {
        let config = WordPieceConfig::default();
        let mut tokenizer = WordPieceTokenizer::new(config);

        let corpus = vec!["hello world".to_string()];
        tokenizer.train(corpus);

        if let Some(token) = tokenizer.get_token(0) {
            let id = tokenizer.get_token_id(&token);
            assert_eq!(id, Some(0));
        }
    }

    #[test]
    fn test_wordpiece_lowercase() {
        let config = WordPieceConfig {
            do_lower_case: true,
            ..WordPieceConfig::default()
        };
        let mut tokenizer = WordPieceTokenizer::new(config);

        let corpus = vec!["Hello World".to_string()];
        tokenizer.train(corpus);

        // Should handle case-insensitive matching
        let tokens_lower = tokenizer.encode("hello world");
        let tokens_upper = tokenizer.encode("HELLO WORLD");
        assert_eq!(tokens_lower, tokens_upper);
    }
}
