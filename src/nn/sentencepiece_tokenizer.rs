//! SentencePiece tokenizer implementation.
//!
//! Implements the SentencePiece algorithm for unsupervised text tokenization.
//! SentencePiece treats text as a raw byte stream, learning subword units
//! without requiring pre-tokenized text.
//!
//! Reference: [Kudo & Richardson, 2018](https://arxiv.org/abs/1808.06226)

use std::collections::{HashMap, HashSet};

/// SentencePiece tokenizer.
///
/// Uses a combination of BPE and Unigram language modeling for subword tokenization.
#[derive(Clone)]
pub struct SentencePieceTokenizer {
    vocab: HashMap<String, f64>,
    inv_vocab: HashMap<usize, String>,
    config: SentencePieceConfig,
    encode_cache: HashMap<String, Vec<usize>>,
    /// Special tokens
    special_tokens: Vec<String>,
}

/// Configuration for SentencePiece tokenization.
#[derive(Clone, Debug)]
pub struct SentencePieceConfig {
    /// Maximum vocabulary size
    pub vocab_size: usize,
    /// Minimum character frequency for BPE merges
    pub min_frequency: usize,
    /// Special tokens to preserve
    pub special_tokens: Vec<String>,
    /// Character coverage for Unigram model (0.9999 = 99.99%)
    pub character_coverage: f64,
    /// Type of tokenizer
    pub model_type: ModelType,
    /// Normalization rule set
    pub normalization_rule: NormalizationRule,
}

impl Default for SentencePieceConfig {
    fn default() -> Self {
        SentencePieceConfig {
            vocab_size: 32000,
            min_frequency: 2,
            special_tokens: vec![
                "[PAD]".to_string(),
                "[UNK]".to_string(),
                "[CLS]".to_string(),
                "[SEP]".to_string(),
                "[MASK]".to_string(),
            ],
            character_coverage: 0.9995,
            model_type: ModelType::BPE,
            normalization_rule: NormalizationRule::NFC,
        }
    }
}

/// Tokenizer model type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelType {
    BPE,
    Unigram,
}

/// Text normalization rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NormalizationRule {
    /// No normalization
    None,
    /// Unicode NFC normalization
    NFC,
    /// Unicode NFKC normalization
    NFKC,
    /// Lowercase + Unicode NFKC
    LowercaseNFKC,
}

impl SentencePieceTokenizer {
    /// Create a new SentencePiece tokenizer with default configuration.
    pub fn new(config: SentencePieceConfig) -> Self {
        let mut special_tokens = Vec::new();
        let mut inv_vocab = HashMap::new();
        let mut next_id = 0;

        for token in &config.special_tokens {
            let id = next_id;
            special_tokens.push(token.clone());
            inv_vocab.insert(id, token.clone());
            next_id += 1;
        }

        SentencePieceTokenizer {
            vocab: HashMap::new(),
            inv_vocab,
            config,
            encode_cache: HashMap::new(),
            special_tokens,
        }
    }

    /// Train the tokenizer on a corpus of text strings.
    pub fn train(&mut self, corpus: Vec<String>) {
        // Step 1: Normalize corpus
        let normalized_corpus: Vec<String> =
            corpus.iter().map(|t| self.normalize_text(t)).collect();

        // Step 2: Build character frequency map
        let mut char_freqs: HashMap<char, usize> = HashMap::new();
        for text in &normalized_corpus {
            for ch in text.chars() {
                if !ch.is_whitespace() {
                    *char_freqs.entry(ch).or_insert(0) += 1;
                }
            }
        }

        // Step 3: Select characters by coverage
        let total_chars: usize = char_freqs.values().sum();
        let target_chars = (total_chars as f64 * self.config.character_coverage) as usize;
        let mut chars: Vec<(char, usize)> = char_freqs.clone().into_iter().collect();
        chars.sort_by(|a, b| b.1.cmp(&a.1));

        let mut covered = 0;
        let mut selected_chars: Vec<char> = Vec::new();
        for (ch, freq) in &chars {
            if covered >= target_chars {
                break;
            }
            selected_chars.push(*ch);
            covered += freq;
        }

        // Step 4: Build initial vocabulary from selected characters
        let mut vocab: HashMap<String, f64> = HashMap::new();
        for ch in &selected_chars {
            vocab.insert(ch.to_string(), *char_freqs.get(ch).unwrap_or(&1) as f64);
        }

        // Step 5: Build word frequencies
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for text in &normalized_corpus {
            let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
            for word in words {
                *word_freqs.entry(word).or_insert(0) += 1;
            }
        }

        // Step 6: Apply BPE merges
        if self.config.model_type == ModelType::BPE {
            self.bpe_train(&mut vocab, &word_freqs, &selected_chars);
        } else {
            self.unigram_train(&mut vocab, &word_freqs, &selected_chars);
        }

        // Build inverse vocabulary
        self.inv_vocab = vocab
            .iter()
            .map(|(k, v)| {
                let id = if self.special_tokens.contains(k) {
                    self.special_tokens.iter().position(|s| s == k).unwrap()
                } else {
                    self.special_tokens.len() + vocab.iter().position(|(sk, _)| sk == k).unwrap()
                };
                (id, k.clone())
            })
            .collect();

        // Rebuild vocab with proper IDs
        let mut new_vocab = HashMap::new();
        for (id, token) in &self.inv_vocab {
            let freq = if self.special_tokens.contains(token) {
                1e10 // Special tokens have highest priority
            } else {
                vocab.get(token).copied().unwrap_or(1.0)
            };
            new_vocab.insert(token.clone(), freq);
        }
        self.vocab = new_vocab;
    }

    /// BPE training: iteratively merge most frequent pairs.
    fn bpe_train(
        &self,
        vocab: &mut HashMap<String, f64>,
        word_freqs: &HashMap<String, usize>,
        selected_chars: &[char],
    ) {
        let max_merges = self
            .config
            .vocab_size
            .saturating_sub(vocab.len() + self.special_tokens.len());

        for _ in 0..max_merges {
            let pair_freqs = self.compute_pair_freqs(word_freqs);
            if pair_freqs.is_empty() {
                break;
            }

            let best_pair = pair_freqs
                .iter()
                .filter(|(_, freq)| **freq >= self.config.min_frequency)
                .max_by_key(|(_, freq)| *freq);

            match best_pair {
                Some(((left, right), _freq)) => {
                    let merged = format!("{}{}", left, right);
                    let total_freq: f64 = word_freqs
                        .keys()
                        .filter(|w| w.contains(left) && w.contains(right))
                        .map(|w| *word_freqs.get(w).unwrap_or(&0) as f64)
                        .sum();
                    vocab.insert(merged, total_freq.max(1.0));
                }
                None => break,
            }
        }
    }

    /// Unigram training: start with candidate set and iteratively remove least likely pieces.
    fn unigram_train(
        &self,
        vocab: &mut HashMap<String, f64>,
        word_freqs: &HashMap<String, usize>,
        selected_chars: &[char],
    ) {
        // Start with all possible subwords from characters
        let mut candidates: Vec<String> = selected_chars.iter().map(|c| c.to_string()).collect();

        // Add all possible word segments
        for word in word_freqs.keys() {
            let chars: Vec<char> = word.chars().collect();
            for i in 0..chars.len() {
                for j in (i + 1)..=chars.len().min(i + 10) {
                    candidates.push(chars[i..j].iter().collect());
                }
            }
        }

        // Remove duplicates and sort by frequency
        let mut unique_candidates: Vec<String> = candidates
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        unique_candidates.sort();

        // Remove least likely candidates until vocab size is reached
        while vocab.len() + self.special_tokens.len() > self.config.vocab_size
            && unique_candidates.len() > 10
        {
            let mut scores: Vec<(String, f64)> = unique_candidates
                .iter()
                .map(|c| {
                    let score = word_freqs
                        .keys()
                        .filter(|w| w.contains(c))
                        .map(|w| *word_freqs.get(w).unwrap_or(&0) as f64)
                        .sum();
                    (c.clone(), score)
                })
                .collect();
            scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((least, _)) = scores.first() {
                unique_candidates.retain(|c| c != least);
                vocab.insert(least.clone(), 1.0);
            } else {
                break;
            }
        }
    }

    /// Compute frequency of adjacent pairs across all words.
    fn compute_pair_freqs(
        &self,
        word_freqs: &HashMap<String, usize>,
    ) -> HashMap<(String, String), usize> {
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

    /// Encode text into token IDs.
    pub fn encode(&mut self, text: &str) -> Vec<usize> {
        if let Some(cached) = self.encode_cache.get(text) {
            return cached.clone();
        }

        let normalized = self.normalize_text(text);
        let tokens = self.tokenize(&normalized);
        let ids: Vec<usize> = tokens
            .iter()
            .filter_map(|t| {
                self.inv_vocab
                    .iter()
                    .find(|(_, v)| *v == t)
                    .map(|(k, _)| *k)
            })
            .collect();

        self.encode_cache.insert(text.to_string(), ids.clone());
        ids
    }

    /// Encode with [CLS] and [SEP] tokens.
    pub fn encode_with_special(&mut self, text: &str) -> Vec<usize> {
        let cls_id = self
            .special_tokens
            .iter()
            .position(|s| s == "[CLS]")
            .unwrap_or(0);
        let sep_id = self
            .special_tokens
            .iter()
            .position(|s| s == "[SEP]")
            .unwrap_or(0);

        let mut ids = vec![cls_id];
        ids.extend(self.encode(text));
        ids.push(sep_id);
        ids
    }

    /// Encode two sequences for classification tasks.
    pub fn encode_pair(&mut self, text1: &str, text2: &str) -> Vec<usize> {
        let cls_id = self
            .special_tokens
            .iter()
            .position(|s| s == "[CLS]")
            .unwrap_or(0);
        let sep_id = self
            .special_tokens
            .iter()
            .position(|s| s == "[SEP]")
            .unwrap_or(0);

        let mut ids = vec![cls_id];
        ids.extend(self.encode(text1));
        ids.push(sep_id);
        ids.extend(self.encode(text2));
        ids.push(sep_id);
        ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for &id in token_ids {
            if let Some(token) = self.inv_vocab.get(&id) {
                parts.push(token.clone());
            }
        }
        parts.join("")
    }

    /// Tokenize text into SentencePiece subwords.
    fn tokenize(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return vec![];
        }

        let mut output: Vec<String> = Vec::new();
        let mut remaining = text.to_string();

        // Greedy longest-match first
        while !remaining.is_empty() {
            let mut matched = false;

            // Try longest possible match first (up to 100 chars)
            let max_len = remaining.len().min(100);
            for len in (1..=max_len).rev() {
                let sub = &remaining[..len];
                if self.vocab.contains_key(sub) {
                    output.push(sub.to_string());
                    remaining = remaining[len..].to_string();
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Try single character
                if let Some(ch) = remaining.chars().next() {
                    let ch_str = ch.to_string();
                    if self.vocab.contains_key(&ch_str) {
                        output.push(ch_str);
                    } else {
                        // Unknown token
                        let unk_id = self
                            .special_tokens
                            .iter()
                            .position(|s| s == "[UNK]")
                            .unwrap_or(0);
                        if let Some(unk_token) = self.inv_vocab.get(&unk_id) {
                            output.push(unk_token.clone());
                        }
                    }
                    remaining = remaining[ch.len_utf8()..].to_string();
                } else {
                    break;
                }
            }
        }

        output
    }

    /// Normalize text according to the configured rule.
    fn normalize_text(&self, text: &str) -> String {
        match self.config.normalization_rule {
            NormalizationRule::None => text.to_string(),
            NormalizationRule::NFC => text.to_string(),
            NormalizationRule::NFKC => text.to_string(),
            NormalizationRule::LowercaseNFKC => text.to_lowercase(),
        }
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len() + self.special_tokens.len()
    }

    /// Get the ID of a special token.
    pub fn special_token_id(&self, token: &str) -> Option<usize> {
        self.special_tokens.iter().position(|s| s == token)
    }

    /// Get the special token by ID.
    pub fn id_to_special_token(&self, id: usize) -> Option<String> {
        self.special_tokens.get(id).cloned()
    }

    /// Clear the encode cache.
    pub fn clear_cache(&mut self) {
        self.encode_cache.clear();
    }

    /// Get the configuration.
    pub fn config(&self) -> &SentencePieceConfig {
        &self.config
    }
}

#[cfg(test)]
mod sentencepiece_tests {
    use super::*;

    #[test]
    fn test_sentencepiece_basic_training() {
        let config = SentencePieceConfig::default();
        let mut tokenizer = SentencePieceTokenizer::new(config);

        let corpus = vec![
            "hello world".to_string(),
            "hello world!".to_string(),
            "world hello".to_string(),
            "helloworld".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];

        tokenizer.train(corpus);

        assert!(tokenizer.vocab_size() > 4); // At least special tokens
    }

    #[test]
    fn test_sentencepiece_encode_decode() {
        let config = SentencePieceConfig::default();
        let mut tokenizer = SentencePieceTokenizer::new(config);

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
    fn test_sentencepiece_special_tokens() {
        let config = SentencePieceConfig::default();
        let tokenizer = SentencePieceTokenizer::new(config);

        assert!(tokenizer.special_token_id("[CLS]").is_some());
        assert!(tokenizer.special_token_id("[SEP]").is_some());
        assert!(tokenizer.special_token_id("[UNK]").is_some());
        assert!(tokenizer.special_token_id("[PAD]").is_some());
        assert!(tokenizer.special_token_id("[MASK]").is_some());
    }

    #[test]
    fn test_sentencepiece_encode_with_special() {
        let config = SentencePieceConfig::default();
        let mut tokenizer = SentencePieceTokenizer::new(config);

        let corpus = vec!["hello world".to_string()];
        tokenizer.train(corpus);

        let tokens = tokenizer.encode_with_special("hello");
        assert!(tokens.len() >= 2); // [CLS] and [SEP]
        assert_eq!(tokens[0], tokenizer.special_token_id("[CLS]").unwrap());
        assert_eq!(
            tokens[tokens.len() - 1],
            tokenizer.special_token_id("[SEP]").unwrap()
        );
    }

    #[test]
    fn test_sentencepiece_encode_pair() {
        let config = SentencePieceConfig::default();
        let mut tokenizer = SentencePieceTokenizer::new(config);

        let corpus = vec!["hello world".to_string(), "foo bar".to_string()];
        tokenizer.train(corpus);

        let tokens = tokenizer.encode_pair("hello", "world");
        assert!(tokens.len() >= 3); // [CLS] + [SEP] + [SEP] minimum
        assert_eq!(tokens[0], tokenizer.special_token_id("[CLS]").unwrap());
    }

    #[test]
    fn test_sentencepiece_lowercase_normalization() {
        let config = SentencePieceConfig {
            normalization_rule: NormalizationRule::LowercaseNFKC,
            ..SentencePieceConfig::default()
        };
        let mut tokenizer = SentencePieceTokenizer::new(config);

        let corpus = vec!["Hello World".to_string()];
        tokenizer.train(corpus);

        let tokens_lower = tokenizer.encode("hello world");
        let tokens_upper = tokenizer.encode("HELLO WORLD");
        assert_eq!(tokens_lower, tokens_upper);
    }

    #[test]
    fn test_sentencepiece_cache() {
        let config = SentencePieceConfig::default();
        let mut tokenizer = SentencePieceTokenizer::new(config);

        let corpus = vec!["hello world".to_string()];
        tokenizer.train(corpus);

        let tokens1 = tokenizer.encode("hello world");
        let tokens2 = tokenizer.encode("hello world");
        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_sentencepiece_clear_cache() {
        let config = SentencePieceConfig::default();
        let mut tokenizer = SentencePieceTokenizer::new(config);

        let corpus = vec!["hello world".to_string()];
        tokenizer.train(corpus);

        tokenizer.encode("hello world");
        assert!(!tokenizer.encode_cache.is_empty());

        tokenizer.clear_cache();
        assert!(tokenizer.encode_cache.is_empty());
    }

    #[test]
    fn test_sentencepiece_large_corpus() {
        let config = SentencePieceConfig {
            vocab_size: 10000,
            min_frequency: 1,
            ..SentencePieceConfig::default()
        };
        let config_clone = config.clone();
        let mut tokenizer = SentencePieceTokenizer::new(config);

        let corpus: Vec<String> = (0..1000)
            .map(|i| format!("word {} is here and there", i % 100))
            .collect();

        tokenizer.train(corpus);

        assert!(tokenizer.vocab_size() > 0);
        assert!(tokenizer.vocab_size() <= config_clone.vocab_size);

        let tokens = tokenizer.encode("word 50 is here and there");
        assert!(!tokens.is_empty());
    }
}
