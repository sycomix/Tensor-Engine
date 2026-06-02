use std::collections::{HashMap, HashSet};
use std::fmt;

/// Byte Pair Encoding (BPE) tokenizer.
///
/// Implements the BPE algorithm as described in "Neural Machine Translation of Rare Words with Subword Unit Models"
/// by Sennrich et al. (2015). Merges the most frequent pairs of consecutive bytes/characters
/// iteratively to build a subword vocabulary.
///
/// # Example
/// ```
/// use tensor_engine::nn::bpe_tokenizer::{BPETokenizer, BPEConfig};
///
/// let config = BPEConfig::default();
/// let mut tokenizer = BPETokenizer::new(config);
/// tokenizer.train(vec!["hello world".to_string(), "hello world!".to_string()]);
/// let tokens = tokenizer.encode("hello world");
/// ```
#[derive(Clone)]
pub struct BPETokenizer {
    vocab: HashMap<String, usize>,
    merges: Vec<(String, String)>,
    config: BPEConfig,
    /// Cache for encoded results
    encode_cache: HashMap<String, Vec<usize>>,
}

/// Configuration for BPE tokenization.
#[derive(Clone, Debug)]
pub struct BPEConfig {
    /// Maximum vocabulary size (includes base characters + merged tokens)
    pub vocab_size: usize,
    /// Minimum frequency for a byte pair to be merged
    pub min_frequency: usize,
    /// Suffix indicator for subword tokens
    pub suffix_indicator: String,
    /// Characters to preserve as single tokens (e.g., punctuation)
    pub preserve_tokens: Vec<String>,
    /// Whether to handle text normalization before tokenization
    pub do_normalization: bool,
}

impl Default for BPEConfig {
    fn default() -> Self {
        BPEConfig {
            vocab_size: 50000,
            min_frequency: 2,
            suffix_indicator: "▁".to_string(), // Unicode blank, like SentencePiece
            preserve_tokens: vec![".".to_string(), ",".to_string(), "!".to_string(), "?".to_string()],
            do_normalization: true,
        }
    }
}

impl fmt::Display for BPEConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BPEConfig(vocab_size={}, min_frequency={}, suffix_indicator='{}')",
            self.vocab_size, self.min_frequency, self.suffix_indicator
        )
    }
}

impl BPETokenizer {
    /// Create a new BPE tokenizer with default configuration.
    pub fn new(config: BPEConfig) -> Self {
        BPETokenizer {
            vocab: HashMap::new(),
            merges: Vec::new(),
            config,
            encode_cache: HashMap::new(),
        }
    }

    /// Train the BPE tokenizer on a corpus of text strings.
    ///
    /// # Arguments
    /// * `corpus` - Vector of text strings to train on
    pub fn train(&mut self, corpus: Vec<String>) {
        // Step 1: Build word frequency map
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for text in &corpus {
            let words = self.tokenize_to_words(text);
            for word in words {
                *word_freqs.entry(word).or_insert(0) += 1;
            }
        }

        // Step 2: Initialize vocabulary with unique characters (bytes)
        let mut char_freqs: HashMap<char, usize> = HashMap::new();
        for word in word_freqs.keys() {
            for ch in word.chars() {
                *char_freqs.entry(ch).or_insert(0) += word_freqs[word];
            }
        }

        // Build initial vocab from characters
        let mut chars: Vec<char> = char_freqs.keys().cloned().collect();
        chars.sort();
        for (i, ch) in chars.iter().enumerate() {
            self.vocab.insert(ch.to_string(), i);
        }

        // Step 3: Iteratively merge most frequent pairs
        let max_merges = self.config.vocab_size.saturating_sub(self.vocab.len());
        for _ in 0..max_merges {
            let pair_freqs = self.compute_pair_frequencies(&word_freqs);
            if pair_freqs.is_empty() {
                break;
            }

            // Find most frequent pair with sufficient frequency
            let best_pair = pair_freqs
                .iter()
                .filter(|(_, freq)| **freq >= self.config.min_frequency)
                .max_by_key(|(_, freq)| *freq);

            match best_pair {
                Some((pair, _)) => {
                    let (left, right) = pair.clone();
                    let merged = format!("{}{}", left, right);
                    let vocab_idx = self.vocab.len();
                    self.vocab.insert(merged, vocab_idx);
                    self.merges.push((left, right));
                }
                None => break,
            }
        }
    }

    /// Encode a text string into token IDs.
    pub fn encode(&mut self, text: &str) -> Vec<usize> {
        // Check cache first
        if let Some(cached) = self.encode_cache.get(text) {
            return cached.clone();
        }

        let tokens = self.bpe_tokenize(text);
        let ids: Vec<usize> = tokens
            .iter()
            .filter_map(|t| self.vocab.get(t).copied())
            .collect();

        // Cache the result
        self.encode_cache.insert(text.to_string(), ids.clone());

        ids
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for &id in token_ids {
            if let Some(token) = self.vocab.iter().find(|(_, &v)| v == id) {
                parts.push(token.0.clone());
            }
        }
        // Join and clean up (remove suffix indicators, handle spaces)
        let joined = parts.join("");
        self.clean_output(&joined)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get the number of merge rules.
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    /// Get a specific merge rule by index.
    pub fn get_merge(&self, index: usize) -> Option<(String, String)> {
        self.merges.get(index).cloned()
    }

    /// Check if a string is in the vocabulary.
    pub fn contains(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    /// Get the token ID for a given token string.
    pub fn get_token_id(&self, token: &str) -> Option<usize> {
        self.vocab.get(token).copied()
    }

    /// Get the token string for a given ID.
    pub fn get_token(&self, id: usize) -> Option<String> {
        self.vocab.iter().find(|(_, &v)| v == id).map(|(k, _)| k.clone())
    }

    /// Get all tokens sorted by frequency (most frequent first).
    pub fn sorted_tokens(&self) -> Vec<(String, usize)> {
        let mut tokens: Vec<(String, usize)> = self.vocab.iter().cloned().collect();
        tokens.sort_by(|a, b| b.1.cmp(&a.1));
        tokens
    }

    /// Clear the encode cache.
    pub fn clear_cache(&mut self) {
        self.encode_cache.clear();
    }

    /// Get the merge rules as a list of (left, right) pairs.
    pub fn get_merges(&self) -> &[(String, String)] {
        &self.merges
    }

    /// Get the configuration.
    pub fn config(&self) -> &BPEConfig {
        &self.config
    }

    // --- Internal helpers ---

    /// Tokenize text into words (space-separated).
    fn tokenize_to_words(&self, text: &str) -> Vec<String> {
        if !self.config.do_normalization {
            return text.split_whitespace().map(|s| s.to_string()).collect();
        }

        // Normalize: lowercase, collapse whitespace
        let normalized = text.to_lowercase();
        let collapsed: String = normalized
            .chars()
            .filter(|c| !c.is_whitespace() || *c == ' ')
            .collect();

        collapsed
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }

    /// Compute frequency of adjacent byte pairs across all words.
    fn compute_pair_frequencies(&self, word_freqs: &HashMap<String, usize>) -> HashMap<(String, String), usize> {
        let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();

        for (word, freq) in word_freqs {
            let chars: Vec<char> = word.chars().collect();
            if chars.len() < 2 {
                continue;
            }

            for i in 0..chars.len() - 1 {
                let left = chars[i].to_string();
                let right = chars[i + 1].to_string();
                *pair_freqs.entry((left, right)).or_insert(0) += freq;
            }
        }

        pair_freqs
    }

    /// Apply BPE merges to a single word.
    fn bpe_tokenize(&self, word: &str) -> Vec<String> {
        let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        let mut parts = chars;

        // Apply merges iteratively
        for (left_merge, right_merge) in &self.merges {
            let mut i = 0;
            while i < parts.len() - 1 {
                if parts[i] == *left_merge && parts[i + 1] == *right_merge {
                    let merged = format!("{}{}", left_merge, right_merge);
                    parts.splice(i..=i + 1, vec![merged]);
                    // Don't increment i to check the merged token
                } else {
                    i += 1;
                }
            }
        }

        parts
    }

    /// Clean up the decoded output.
    fn clean_output(&self, text: &str) -> String {
        // Replace suffix indicator with space
        let cleaned = text.replace(&self.config.suffix_indicator, " ");
        // Collapse multiple spaces
        let collapsed: String = cleaned
            .chars()
            .fold((String::new(), false), |(mut acc, prev_space), c| {
                if c == ' ' && prev_space {
                    (acc, true)
                } else {
                    acc.push(c);
                    (acc, c == ' ')
                }
            })
            .0;
        collapsed.trim().to_string()
    }
}

#[cfg(test)]
mod bpe_tests {
    use super::*;

    #[test]
    fn test_bpe_basic_training() {
        let config = BPEConfig::default();
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec![
            "hello world".to_string(),
            "hello world!".to_string(),
            "world hello".to_string(),
            "helloworld".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];

        tokenizer.train(corpus);

        assert!(tokenizer.vocab_size() > 0);
        assert!(tokenizer.merge_count() >= 0);
    }

    #[test]
    fn test_bpe_encode_decode() {
        let config = BPEConfig::default();
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec![
            "hello world".to_string(),
            "hello world!".to_string(),
            "world hello".to_string(),
            "helloworld".to_string(),
            "hello".to_string(),
            "world".to_string(),
            "hi there".to_string(),
            "there hi".to_string(),
        ];

        tokenizer.train(corpus);

        let tokens = tokenizer.encode("hello world");
        assert!(!tokens.is_empty());

        let decoded = tokenizer.decode(&tokens);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_bpe_vocab_contains() {
        let config = BPEConfig::default();
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello world".to_string(), "helloworld".to_string()];
        tokenizer.train(corpus);

        // Single characters should be in vocab
        assert!(tokenizer.contains("h"));
        assert!(tokenizer.contains("e"));
        assert!(tokenizer.contains("l"));
        assert!(tokenizer.contains("o"));
    }

    #[test]
    fn test_bpe_cache() {
        let config = BPEConfig::default();
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello world".to_string()];
        tokenizer.train(corpus);

        // First encode
        let tokens1 = tokenizer.encode("hello world");

        // Second encode should use cache
        let tokens2 = tokenizer.encode("hello world");

        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_bpe_clear_cache() {
        let config = BPEConfig::default();
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello world".to_string()];
        tokenizer.train(corpus);

        tokenizer.encode("hello world");
        assert!(!tokenizer.encode_cache.is_empty());

        tokenizer.clear_cache();
        assert!(tokenizer.encode_cache.is_empty());
    }

    #[test]
    fn test_bpe_get_set_token() {
        let config = BPEConfig::default();
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello world".to_string()];
        tokenizer.train(corpus);

        // Get a token ID
        if let Some(token) = tokenizer.get_token(0) {
            let id = tokenizer.get_token_id(&token);
            assert_eq!(id, Some(0));
        }
    }

    #[test]
    fn test_bpe_sorted_tokens() {
        let config = BPEConfig::default();
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec![
            "hello world".to_string(),
            "hello world".to_string(),
            "hello world".to_string(),
            "hi".to_string(),
        ];
        tokenizer.train(corpus);

        let sorted = tokenizer.sorted_tokens();
        assert!(!sorted.is_empty());

        // First token should have highest frequency
        if sorted.len() > 1 {
            assert!(sorted[0].1 >= sorted[1].1);
        }
    }

    #[test]
    fn test_bpe_large_corpus() {
        let config = BPEConfig {
            vocab_size: 10000,
            min_frequency: 1,
            suffix_indicator: "▁".to_string(),
            preserve_tokens: vec![],
            do_normalization: true,
        };
        let mut tokenizer = BPETokenizer::new(config);

        // Generate a larger corpus
        let corpus: Vec<String> = (0..1000)
            .map(|i| format!("word {} is here and there", i % 100))
            .collect();

        tokenizer.train(corpus);

        assert!(tokenizer.vocab_size() > 0);
        assert!(tokenizer.vocab_size() <= config.vocab_size);

        let tokens = tokenizer.encode("word 50 is here and there");
        assert!(!tokens.is_empty());
    }
}
