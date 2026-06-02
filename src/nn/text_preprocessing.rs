//! Text preprocessing pipelines.
//!
//! Provides common text normalization, cleaning, and transformation operations
//! used in NLP pipelines: lowercasing, punctuation handling, whitespace normalization,
//! character filtering, and more.

use std::collections::HashMap;

/// Text normalization configuration.
#[derive(Clone, Debug)]
pub struct TextNormalizeConfig {
    /// Whether to lowercase text
    pub lowercase: bool,
    /// Whether to remove punctuation
    pub remove_punctuation: bool,
    /// Whether to collapse whitespace
    pub collapse_whitespace: bool,
    /// Whether to remove digits
    pub remove_digits: bool,
    /// Whether to remove extra whitespace
    pub remove_extra_spaces: bool,
    /// Whether to strip leading/trailing whitespace
    pub strip: bool,
    /// Custom character replacements
    pub replacements: HashMap<char, String>,
    /// Characters to preserve (if remove_punctuation is true)
    pub preserve_chars: Vec<char>,
}

impl Default for TextNormalizeConfig {
    fn default() -> Self {
        TextNormalizeConfig {
            lowercase: true,
            remove_punctuation: false,
            collapse_whitespace: true,
            remove_digits: false,
            remove_extra_spaces: true,
            strip: true,
            replacements: HashMap::new(),
            preserve_chars: vec!['.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}', '"', '\'', '\n', '\t'],
        }
    }
}

/// Text normalizer.
pub struct TextNormalizer {
    config: TextNormalizeConfig,
}

impl TextNormalizer {
    /// Create a new text normalizer.
    pub fn new(config: TextNormalizeConfig) -> Self {
        TextNormalizer { config }
    }

    /// Normalize text according to configuration.
    pub fn normalize(&self, text: &str) -> String {
        let mut result = text.to_string();

        if self.config.strip {
            result = result.trim().to_string();
        }

        if self.config.lowercase {
            result = result.to_lowercase();
        }

        // Apply custom replacements
        for (from, to) in &self.config.replacements {
            result = result.replace(*from, to.as_str());
        }

        if self.config.remove_punctuation {
            result = result
                .chars()
                .filter(|c| {
                    self.config.preserve_chars.contains(c)
                        || !self.is_punctuation(*c)
                })
                .collect();
        }

        if self.config.remove_digits {
            result = result
                .chars()
                .filter(|c| !c.is_ascii_digit())
                .collect();
        }

        if self.config.collapse_whitespace {
            result = self.collapse_whitespace_str(&result);
        }

        if self.config.remove_extra_spaces {
            result = result
                .split_whitespace()
                .collect::<Vec<&str>>()
                .join(" ");
        }

        if self.config.strip {
            result = result.trim().to_string();
        }

        result
    }

    /// Check if a character is punctuation.
    fn is_punctuation(&self, c: char) -> bool {
        c.is_ascii_punctuation()
            || matches!(c, '\u{0021}'..='\u{002F}' | '\u{003A}'..='\u{0040}' | '\u{005B}'..='\u{0060}' | '\u{007B}'..='\u{007E}' | '\u{2000}'..='\u{206F}' | '\u{2E00}'..='\u{2E7F}' | '\u{3000}'..='\u{303F}')
    }

    /// Collapse multiple whitespace characters into single space.
    fn collapse_whitespace_str(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut prev_was_space = false;

        for c in text.chars() {
            if c.is_whitespace() {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            } else {
                result.push(c);
                prev_was_space = false;
            }
        }

        result
    }

    /// Normalize a batch of texts.
    pub fn normalize_batch(&self, texts: &[String]) -> Vec<String> {
        texts.iter().map(|t| self.normalize(t)).collect()
    }
}

/// Text cleaning pipeline.
pub struct TextCleaner {
    operations: Vec<TextOperation>,
}

/// Text cleaning operation.
#[derive(Clone)]
pub enum TextOperation {
    Lowercase,
    RemovePunctuation,
    CollapseWhitespace,
    RemoveDigits,
    Strip,
    RemoveExtraSpaces,
    CustomReplace { from: String, to: String },
    RegexReplace { pattern: String, replacement: String },
    Custom(Box<dyn Fn(&str) -> String + Send + Sync>),
}

impl TextCleaner {
    /// Create a new text cleaner.
    pub fn new() -> Self {
        TextCleaner {
            operations: Vec::new(),
        }
    }

    /// Add lowercase operation.
    pub fn with_lowercase(mut self) -> Self {
        self.operations.push(TextOperation::Lowercase);
        self
    }

    /// Add remove punctuation operation.
    pub fn with_remove_punctuation(mut self) -> Self {
        self.operations.push(TextOperation::RemovePunctuation);
        self
    }

    /// Add collapse whitespace operation.
    pub fn with_collapse_whitespace(mut self) -> Self {
        self.operations.push(TextOperation::CollapseWhitespace);
        self
    }

    /// Add remove digits operation.
    pub fn with_remove_digits(mut self) -> Self {
        self.operations.push(TextOperation::RemoveDigits);
        self
    }

    /// Add strip operation.
    pub fn with_strip(mut self) -> Self {
        self.operations.push(TextOperation::Strip);
        self
    }

    /// Add remove extra spaces operation.
    pub fn with_remove_extra_spaces(mut self) -> Self {
        self.operations.push(TextOperation::RemoveExtraSpaces);
        self
    }

    /// Add custom replace operation.
    pub fn with_custom_replace(mut self, from: &str, to: &str) -> Self {
        self.operations.push(TextOperation::CustomReplace {
            from: from.to_string(),
            to: to.to_string(),
        });
        self
    }

    /// Add custom function operation.
    pub fn with_custom<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        self.operations.push(TextOperation::Custom(Box::new(f)));
        self
    }

    /// Apply all cleaning operations to text.
    pub fn clean(&self, text: &str) -> String {
        let mut result = text.to_string();

        for op in &self.operations {
            result = match op {
                TextOperation::Lowercase => result.to_lowercase(),
                TextOperation::RemovePunctuation => result
                    .chars()
                    .filter(|c| !c.is_ascii_punctuation() && !c.is_punctuation())
                    .collect(),
                TextOperation::CollapseWhitespace => {
                    let mut out = String::with_capacity(result.len());
                    let mut prev_space = false;
                    for c in result.chars() {
                        if c.is_whitespace() {
                            if !prev_space {
                                out.push(' ');
                                prev_space = true;
                            }
                        } else {
                            out.push(c);
                            prev_space = false;
                        }
                    }
                    out
                }
                TextOperation::RemoveDigits => result
                    .chars()
                    .filter(|c| !c.is_ascii_digit())
                    .collect(),
                TextOperation::Strip => result.trim().to_string(),
                TextOperation::RemoveExtraSpaces => result
                    .split_whitespace()
                    .collect::<Vec<&str>>()
                    .join(" "),
                TextOperation::CustomReplace { from, to } => result.replace(from, to),
                TextOperation::RegexReplace { .. } => {
                    // Simplified: just store the pattern for now
                    log::warn!("RegexReplace not yet implemented, skipping");
                    result
                }
                TextOperation::Custom(f) => f(&result),
            };
        }

        result
    }

    /// Clean a batch of texts.
    pub fn clean_batch(&self, texts: &[String]) -> Vec<String> {
        texts.iter().map(|t| self.clean(t)).collect()
    }
}

/// Common text cleaning pipelines.
impl TextCleaner {
    /// Standard NLP cleaning pipeline: lowercase, remove punctuation, collapse whitespace.
    pub fn standard_nlp() -> Self {
        TextCleaner::new()
            .with_lowercase()
            .with_remove_punctuation()
            .with_collapse_whitespace()
            .with_strip()
    }

    /// Aggressive cleaning: remove everything except alphanumeric.
    pub fn aggressive() -> Self {
        TextCleaner::new()
            .with_lowercase()
            .with_remove_punctuation()
            .with_remove_digits()
            .with_collapse_whitespace()
            .with_strip()
    }

    /// Preserve punctuation but clean whitespace.
    pub fn preserve_punctuation() -> Self {
        TextCleaner::new()
            .with_lowercase()
            .with_collapse_whitespace()
            .with_strip()
    }
}

/// Text preprocessing pipeline combining tokenization, normalization, and cleaning.
pub struct TextPreprocessor {
    cleaner: TextCleaner,
    normalizer: Option<TextNormalizer>,
    max_length: Option<usize>,
    pad_token: String,
    pad_id: usize,
}

impl TextPreprocessor {
    /// Create a new text preprocessor.
    pub fn new() -> Self {
        TextPreprocessor {
            cleaner: TextCleaner::standard_nlp(),
            normalizer: None,
            max_length: None,
            pad_token: "[PAD]".to_string(),
            pad_id: 0,
        }
    }

    /// Set the text cleaner.
    pub fn with_cleaner(mut self, cleaner: TextCleaner) -> Self {
        self.cleaner = cleaner;
        self
    }

    /// Set the text normalizer.
    pub fn with_normalizer(mut self, normalizer: TextNormalizer) -> Self {
        self.normalizer = Some(normalizer);
        self
    }

    /// Set maximum sequence length.
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Set padding token and ID.
    pub fn with_padding(mut self, token: &str, id: usize) -> Self {
        self.pad_token = token.to_string();
        self.pad_id = id;
        self
    }

    /// Preprocess a single text.
    pub fn preprocess(&self, text: &str) -> String {
        let cleaned = self.cleaner.clean(text);
        if let Some(normalizer) = &self.normalizer {
            normalizer.normalize(&cleaned)
        } else {
            cleaned
        }
    }

    /// Preprocess a batch of texts.
    pub fn preprocess_batch(&self, texts: &[String]) -> Vec<String> {
        texts.iter().map(|t| self.preprocess(t)).collect()
    }

    /// Get the padding ID.
    pub fn pad_id(&self) -> usize {
        self.pad_id
    }

    /// Get the maximum length.
    pub fn max_length(&self) -> Option<usize> {
        self.max_length
    }
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod text_preprocessing_tests {
    use super::*;

    #[test]
    fn test_text_normalizer_lowercase() {
        let config = TextNormalizeConfig {
            lowercase: true,
            ..TextNormalizeConfig::default()
        };
        let normalizer = TextNormalizer::new(config);
        let result = normalizer.normalize("Hello World");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_text_normalizer_remove_punctuation() {
        let config = TextNormalizeConfig {
            remove_punctuation: true,
            ..TextNormalizeConfig::default()
        };
        let normalizer = TextNormalizer::new(config);
        let result = normalizer.normalize("Hello, World!");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_text_normalizer_collapse_whitespace() {
        let config = TextNormalizeConfig {
            collapse_whitespace: true,
            ..TextNormalizeConfig::default()
        };
        let normalizer = TextNormalizer::new(config);
        let result = normalizer.normalize("Hello   World");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_text_cleaner_standard() {
        let cleaner = TextCleaner::standard_nlp();
        let result = cleaner.clean("Hello, World!  How are you?");
        assert_eq!(result, "hello world how are you");
    }

    #[test]
    fn test_text_cleaner_aggressive() {
        let cleaner = TextCleaner::aggressive();
        let result = cleaner.clean("Hello World 123!");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_text_cleaner_preserve_punctuation() {
        let cleaner = TextCleaner::preserve_punctuation();
        let result = cleaner.clean("Hello, World!  How are you?");
        assert_eq!(result, "hello, world! how are you?");
    }

    #[test]
    fn test_text_cleaner_custom_replace() {
        let cleaner = TextCleaner::new().with_custom_replace("http://", "");
        let result = cleaner.clean("Visit http://example.com for more");
        assert_eq!(result, "visit example.com for more");
    }

    #[test]
    fn test_text_preprocessor() {
        let preprocessor = TextPreprocessor::new();
        let result = preprocessor.preprocess("Hello, World!");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_text_preprocessor_batch() {
        let preprocessor = TextPreprocessor::new();
        let texts = vec!["Hello World".to_string(), "Goodbye World".to_string()];
        let results = preprocessor.preprocess_batch(&texts);
        assert_eq!(results[0], "hello world");
        assert_eq!(results[1], "goodbye world");
    }

    #[test]
    fn test_text_normalizer_remove_digits() {
        let config = TextNormalizeConfig {
            remove_digits: true,
            ..TextNormalizeConfig::default()
        };
        let normalizer = TextNormalizer::new(config);
        let result = normalizer.normalize("Hello World 123");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_text_normalizer_custom_replacements() {
        let mut replacements = HashMap::new();
        replacements.insert('\'', "");
        let config = TextNormalizeConfig {
            replacements,
            ..TextNormalizeConfig::default()
        };
        let normalizer = TextNormalizer::new(config);
        let result = normalizer.normalize("Hello World's");
        assert_eq!(result, "hello worlds");
    }

    #[test]
    fn test_text_cleaner_custom_function() {
        let cleaner = TextCleaner::new().with_custom(|text| {
            text.replace("foo", "bar")
        });
        let result = cleaner.clean("foo bar foo");
        assert_eq!(result, "bar bar bar");
    }
}
