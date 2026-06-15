//! Inference optimization utilities for efficient model deployment.
//!
//! Provides tools for optimizing models for inference, including:
//! - Model graph optimization
//! - Operator fusion
//! - Memory optimization
//! - Latency profiling

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for inference optimization.
#[derive(Clone, Debug)]
pub struct InferenceOptimConfig {
    /// Enable operator fusion
    pub enable_fusion: bool,
    /// Enable memory optimization (in-place operations)
    pub enable_memory_opt: bool,
    /// Enable constant folding
    pub enable_const_folding: bool,
    /// Batch size for inference
    pub batch_size: usize,
    /// Maximum model size in MB
    pub max_model_size_mb: usize,
}

impl Default for InferenceOptimConfig {
    fn default() -> Self {
        InferenceOptimConfig {
            enable_fusion: true,
            enable_memory_opt: true,
            enable_const_folding: true,
            batch_size: 1,
            max_model_size_mb: 4096,
        }
    }
}

impl InferenceOptimConfig {
    /// Create configuration optimized for latency.
    pub fn latency_optimized() -> Self {
        InferenceOptimConfig {
            enable_fusion: true,
            enable_memory_opt: true,
            enable_const_folding: true,
            batch_size: 1,
            max_model_size_mb: 4096,
        }
    }

    /// Create configuration optimized for throughput.
    pub fn throughput_optimized(batch_size: usize) -> Self {
        InferenceOptimConfig {
            enable_fusion: true,
            enable_memory_opt: true,
            enable_const_folding: true,
            batch_size,
            max_model_size_mb: 4096,
        }
    }

    /// Create configuration optimized for memory.
    pub fn memory_optimized() -> Self {
        InferenceOptimConfig {
            enable_fusion: true,
            enable_memory_opt: true,
            enable_const_folding: true,
            batch_size: 1,
            max_model_size_mb: 512,
        }
    }
}

/// Inference profiler for measuring model performance.
pub struct InferenceProfiler {
    name: String,
    layer_times: HashMap<String, Vec<Duration>>,
    total_time: Duration,
}

impl InferenceProfiler {
    /// Create a new profiler.
    pub fn new(name: impl Into<String>) -> Self {
        InferenceProfiler {
            name: name.into(),
            layer_times: HashMap::new(),
            total_time: Duration::ZERO,
        }
    }

    /// Profile a single operation/layer.
    pub fn profile_operation<F>(&mut self, op_name: impl Into<String>, op: F) -> Tensor
    where
        F: FnOnce() -> Tensor,
    {
        let start = Instant::now();
        let result = op();
        let elapsed = start.elapsed();

        let op_name_str = op_name.into();
        self.layer_times
            .entry(op_name_str)
            .or_insert_with(Vec::new)
            .push(elapsed);

        self.total_time += elapsed;
        result
    }

    /// Get average time for an operation.
    pub fn avg_time(&self, op_name: &str) -> Option<Duration> {
        self.layer_times.get(op_name).and_then(|times| {
            if times.is_empty() {
                None
            } else {
                let total: Duration = times.iter().sum();
                Some(total / times.len() as u32)
            }
        })
    }

    /// Get total time.
    pub fn total_time(&self) -> Duration {
        self.total_time
    }

    /// Get all layer times.
    pub fn layer_times(&self) -> &HashMap<String, Vec<Duration>> {
        &self.layer_times
    }

    /// Print profiling report.
    pub fn print_report(&self) {
        println!("\n=== Inference Profiling Report: {} ===", self.name);
        println!("Total Inference Time: {:?}", self.total_time);

        let mut items: Vec<_> = self
            .layer_times
            .iter()
            .map(|(name, times)| {
                let total: Duration = times.iter().sum();
                let avg = total / times.len() as u32;
                (name.clone(), total, avg, times.len())
            })
            .collect();

        // Sort by total time descending
        items.sort_by(|a, b| b.1.cmp(&a.1));

        println!(
            "\n{:<40} {:<15} {:<15} {:<10}",
            "Operation", "Total", "Avg", "Calls"
        );
        println!("{:-<40} {:-<15} {:-<15} {:-<10}", "", "", "", "");

        for (name, total, avg, count) in items {
            println!("{:<40} {:<15?} {:<15?} {:<10}", name, total, avg, count);
        }
    }

    /// Reset profiler.
    pub fn reset(&mut self) {
        self.layer_times.clear();
        self.total_time = Duration::ZERO;
    }
}

/// Model memory analyzer for estimating memory usage.
pub struct MemoryAnalyzer;

impl MemoryAnalyzer {
    /// Estimate tensor memory usage in bytes.
    pub fn estimate_tensor_memory(shape: &[usize], bytes_per_element: usize) -> usize {
        shape.iter().product::<usize>() * bytes_per_element
    }

    /// Estimate total model memory in MB.
    pub fn estimate_model_memory(param_shapes: &[Vec<usize>], bytes_per_param: usize) -> f32 {
        let total_bytes: usize = param_shapes
            .iter()
            .map(|shape| Self::estimate_tensor_memory(shape, bytes_per_param))
            .sum();

        total_bytes as f32 / (1024.0 * 1024.0)
    }

    /// Estimate activation memory for a forward pass.
    pub fn estimate_activation_memory(
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_layers: usize,
        bytes_per_element: usize,
    ) -> f32 {
        // Estimate memory for activations: batch * seq * hidden * layers * 2 (forward + backward cache)
        let activation_size =
            batch_size * seq_len * hidden_dim * num_layers * 2 * bytes_per_element;
        activation_size as f32 / (1024.0 * 1024.0)
    }
}

/// Batch processor for handling inference batches efficiently.
pub struct BatchProcessor {
    batch_size: usize,
    pending_inputs: Vec<Tensor>,
}

impl BatchProcessor {
    /// Create a new batch processor.
    pub fn new(batch_size: usize) -> Self {
        BatchProcessor {
            batch_size,
            pending_inputs: Vec::new(),
        }
    }

    /// Add an input to the batch.
    pub fn add_input(&mut self, input: Tensor) -> Option<Vec<Tensor>> {
        self.pending_inputs.push(input);

        if self.pending_inputs.len() >= self.batch_size {
            let batch = self.pending_inputs.drain(..self.batch_size).collect();
            Some(batch)
        } else {
            None
        }
    }

    /// Flush remaining inputs as a partial batch.
    pub fn flush(&mut self) -> Option<Vec<Tensor>> {
        if self.pending_inputs.is_empty() {
            None
        } else {
            Some(self.pending_inputs.drain(..).collect())
        }
    }

    /// Get current batch size.
    pub fn current_batch_size(&self) -> usize {
        self.pending_inputs.len()
    }

    /// Check if a batch is ready.
    pub fn is_batch_ready(&self) -> bool {
        self.pending_inputs.len() >= self.batch_size
    }
}

/// Performance statistics for inference.
#[derive(Clone, Debug)]
pub struct InferenceStats {
    /// Total number of inferences
    pub num_inferences: usize,
    /// Total time for all inferences
    pub total_time: Duration,
    /// Peak memory usage in MB
    pub peak_memory_mb: f32,
    /// Average batch size
    pub avg_batch_size: f32,
    /// Throughput (inferences per second)
    pub throughput: f32,
}

impl InferenceStats {
    /// Create empty stats.
    pub fn new() -> Self {
        InferenceStats {
            num_inferences: 0,
            total_time: Duration::ZERO,
            peak_memory_mb: 0.0,
            avg_batch_size: 1.0,
            throughput: 0.0,
        }
    }

    /// Update statistics.
    pub fn update(
        &mut self,
        num_inferences: usize,
        total_time: Duration,
        peak_memory_mb: f32,
        avg_batch_size: f32,
    ) {
        self.num_inferences = num_inferences;
        self.total_time = total_time;
        self.peak_memory_mb = peak_memory_mb;
        self.avg_batch_size = avg_batch_size;
        self.throughput = if total_time.as_secs_f32() > 0.0 {
            num_inferences as f32 / total_time.as_secs_f32()
        } else {
            0.0
        };
    }

    /// Print stats report.
    pub fn print_report(&self) {
        println!("\n=== Inference Performance Report ===");
        println!("Total Inferences: {}", self.num_inferences);
        println!("Total Time: {:?}", self.total_time);
        println!("Peak Memory: {:.2} MB", self.peak_memory_mb);
        println!("Avg Batch Size: {:.2}", self.avg_batch_size);
        println!("Throughput: {:.2} inferences/sec", self.throughput);
        println!(
            "Avg Latency: {:.2} ms",
            self.total_time.as_millis() as f32 / self.num_inferences.max(1) as f32
        );
    }
}

impl Default for InferenceStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod inference_tests {
    use super::*;

    #[test]
    fn test_inference_optim_config_default() {
        let config = InferenceOptimConfig::default();
        assert!(config.enable_fusion);
        assert!(config.enable_memory_opt);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_inference_optim_config_latency() {
        let config = InferenceOptimConfig::latency_optimized();
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_inference_optim_config_throughput() {
        let config = InferenceOptimConfig::throughput_optimized(32);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_inference_profiler_creation() {
        let profiler = InferenceProfiler::new("test_model");
        assert_eq!(profiler.total_time(), Duration::ZERO);
    }

    #[test]
    fn test_memory_analyzer_tensor() {
        let shape = vec![32, 128, 256];
        let memory = MemoryAnalyzer::estimate_tensor_memory(&shape, 4);
        assert_eq!(memory, 32 * 128 * 256 * 4);
    }

    #[test]
    fn test_memory_analyzer_model() {
        let shapes = vec![vec![768, 768], vec![768, 3072]];
        let memory = MemoryAnalyzer::estimate_model_memory(&shapes, 4);
        let expected = ((768 * 768 + 768 * 3072) * 4) as f32 / (1024.0 * 1024.0);
        assert!((memory - expected).abs() < 0.01);
    }

    #[test]
    fn test_batch_processor_creation() {
        let processor = BatchProcessor::new(4);
        assert!(!processor.is_batch_ready());
        assert_eq!(processor.current_batch_size(), 0);
    }

    #[test]
    fn test_batch_processor_batching() {
        let mut processor = BatchProcessor::new(2);
        assert!(processor.add_input(Tensor::ones(&[1, 10])).is_none());
        assert!(processor.add_input(Tensor::ones(&[1, 10])).is_some());
    }

    #[test]
    fn test_inference_stats() {
        let mut stats = InferenceStats::new();
        stats.update(100, Duration::from_secs(1), 256.0, 32.0);

        assert_eq!(stats.num_inferences, 100);
        assert!(stats.throughput > 0.0);
    }
}
