//! Streaming data loader for processing large datasets without loading everything into memory.
//!
//! Provides utilities for efficiently loading and processing data in streaming fashion,
//! useful for datasets larger than available RAM.

use std::fs::File;
use std::io::{BufRead, BufReader, Result as IoResult};
use std::path::Path;

/// Configuration for streaming data loading.
#[derive(Clone, Debug)]
pub struct StreamingDataLoaderConfig {
    /// Path to the data file
    pub file_path: String,
    /// Number of samples to batch before yielding
    pub batch_size: usize,
    /// Buffer size for reading (in bytes)
    pub buffer_size: usize,
    /// Number of samples to skip at the beginning
    pub skip_rows: usize,
    /// Maximum number of samples to load (-1 for no limit)
    pub max_samples: i64,
}

impl StreamingDataLoaderConfig {
    /// Create a new configuration with defaults.
    pub fn new(file_path: impl Into<String>) -> Self {
        StreamingDataLoaderConfig {
            file_path: file_path.into(),
            batch_size: 32,
            buffer_size: 8192,
            skip_rows: 0,
            max_samples: -1,
        }
    }

    /// Set batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set buffer size.
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Set rows to skip.
    pub fn with_skip_rows(mut self, skip_rows: usize) -> Self {
        self.skip_rows = skip_rows;
        self
    }

    /// Set maximum samples to load.
    pub fn with_max_samples(mut self, max_samples: i64) -> Self {
        self.max_samples = max_samples;
        self
    }
}

/// Streaming data loader for reading large files line-by-line.
pub struct StreamingDataLoader {
    reader: BufReader<File>,
    config: StreamingDataLoaderConfig,
    current_row: usize,
    total_read: usize,
}

impl StreamingDataLoader {
    /// Create a new streaming data loader.
    pub fn new(config: StreamingDataLoaderConfig) -> IoResult<Self> {
        let file = File::open(&config.file_path)?;
        let reader = BufReader::with_capacity(config.buffer_size, file);

        Ok(StreamingDataLoader {
            reader,
            config,
            current_row: 0,
            total_read: 0,
        })
    }

    /// Get the next batch of lines.
    pub fn next_batch(&mut self) -> IoResult<Option<Vec<String>>> {
        let mut batch = Vec::new();

        // Skip initial rows if needed
        while self.current_row < self.config.skip_rows {
            let mut line = String::new();
            let bytes_read = self.reader.read_line(&mut line)?;
            if bytes_read == 0 {
                return Ok(None);
            }
            self.current_row += 1;
        }

        // Check max samples limit
        let remaining = if self.config.max_samples < 0 {
            usize::MAX
        } else {
            ((self.config.max_samples as usize) - self.total_read).max(0)
        };

        if remaining == 0 {
            return Ok(None);
        }

        // Read batch
        let batch_size = self.config.batch_size.min(remaining);

        for _ in 0..batch_size {
            let mut line = String::new();
            let bytes_read = self.reader.read_line(&mut line)?;

            if bytes_read == 0 {
                break;
            }

            // Remove trailing newline
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }

            batch.push(line);
            self.total_read += 1;
        }

        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }

    /// Get the current row number.
    pub fn current_row(&self) -> usize {
        self.current_row
    }

    /// Get total samples read so far.
    pub fn total_read(&self) -> usize {
        self.total_read
    }
}

/// Iterator adapter for streaming data loader.
pub struct StreamingDataIter {
    loader: StreamingDataLoader,
    exhausted: bool,
}

impl StreamingDataIter {
    /// Create a new iterator from a loader.
    pub fn new(loader: StreamingDataLoader) -> Self {
        StreamingDataIter {
            loader,
            exhausted: false,
        }
    }
}

impl Iterator for StreamingDataIter {
    type Item = Vec<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        match self.loader.next_batch() {
            Ok(Some(batch)) => Some(batch),
            Ok(None) => {
                self.exhausted = true;
                None
            }
            Err(e) => {
                log::error!("Error reading batch: {}", e);
                self.exhausted = true;
                None
            }
        }
    }
}

/// CSV parser for streaming data.
pub struct StreamingCSVParser {
    delimiter: char,
    skip_header: bool,
}

impl StreamingCSVParser {
    /// Create a new CSV parser.
    pub fn new() -> Self {
        StreamingCSVParser {
            delimiter: ',',
            skip_header: false,
        }
    }

    /// Set delimiter character.
    pub fn with_delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set whether to skip header row.
    pub fn with_skip_header(mut self, skip_header: bool) -> Self {
        self.skip_header = skip_header;
        self
    }

    /// Parse a line into fields.
    pub fn parse_line(&self, line: &str) -> Vec<String> {
        line.split(self.delimiter)
            .map(|field| field.trim().to_string())
            .collect()
    }

    /// Parse CSV file in streaming fashion.
    pub fn parse_streaming(
        &self,
        file_path: impl AsRef<Path>,
        batch_size: usize,
    ) -> IoResult<StreamingCSVIter> {
        let config = StreamingDataLoaderConfig::new(file_path.as_ref().to_string_lossy().to_string())
            .with_batch_size(batch_size)
            .with_skip_rows(if self.skip_header { 1 } else { 0 });

        let loader = StreamingDataLoader::new(config)?;
        let delimiter = self.delimiter;

        Ok(StreamingCSVIter {
            loader: StreamingDataIter::new(loader),
            delimiter,
        })
    }
}

impl Default for StreamingCSVParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over CSV batches (as parsed fields).
pub struct StreamingCSVIter {
    loader: StreamingDataIter,
    delimiter: char,
}

impl Iterator for StreamingCSVIter {
    type Item = Vec<Vec<String>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.loader.next().map(|batch| {
            batch
                .into_iter()
                .map(|line| {
                    line.split(self.delimiter)
                        .map(|field| field.trim().to_string())
                        .collect()
                })
                .collect()
        })
    }
}

/// JSON lines (JSONL) streaming parser.
pub struct StreamingJSONLParser;

impl StreamingJSONLParser {
    /// Parse JSONL file in streaming fashion.
    pub fn parse_streaming(
        file_path: impl AsRef<Path>,
        batch_size: usize,
    ) -> IoResult<StreamingJSONLIter> {
        let config = StreamingDataLoaderConfig::new(file_path.as_ref().to_string_lossy().to_string())
            .with_batch_size(batch_size);

        let loader = StreamingDataLoader::new(config)?;

        Ok(StreamingJSONLIter {
            loader: StreamingDataIter::new(loader),
        })
    }
}

/// Iterator over JSONL batches (as unparsed JSON strings).
pub struct StreamingJSONLIter {
    loader: StreamingDataIter,
}

impl Iterator for StreamingJSONLIter {
    type Item = Vec<String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.loader.next()
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[test]
    fn test_streaming_data_loader_config() {
        let config = StreamingDataLoaderConfig::new("test.txt")
            .with_batch_size(64)
            .with_skip_rows(1)
            .with_max_samples(1000);

        assert_eq!(config.batch_size, 64);
        assert_eq!(config.skip_rows, 1);
        assert_eq!(config.max_samples, 1000);
    }

    #[test]
    fn test_streaming_csv_parser() {
        let parser = StreamingCSVParser::new()
            .with_delimiter(',')
            .with_skip_header(true);

        let line = "field1,field2,field3";
        let fields = parser.parse_line(line);

        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "field1");
        assert_eq!(fields[1], "field2");
        assert_eq!(fields[2], "field3");
    }

    #[test]
    fn test_streaming_csv_parser_with_spaces() {
        let parser = StreamingCSVParser::new();

        let line = "field1 , field2 , field3";
        let fields = parser.parse_line(line);

        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "field1");
        assert_eq!(fields[1], "field2");
        assert_eq!(fields[2], "field3");
    }

    #[test]
    fn test_streaming_csv_parser_custom_delimiter() {
        let parser = StreamingCSVParser::new().with_delimiter('|');

        let line = "field1 | field2 | field3";
        let fields = parser.parse_line(line);

        assert_eq!(fields.len(), 3);
    }
}
