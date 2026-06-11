//! Monitoring and Observability for Tensor Engine
//!
//! This module provides comprehensive monitoring capabilities including:
//! - Training metrics logging
//! - Inference latency monitoring  
//! - Memory usage tracking
//! - Error rate monitoring
//! - Custom metrics collection

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Training metrics collected during model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Current epoch number
    pub epoch: usize,
    /// Total number of epochs planned
    pub total_epochs: usize,
    /// Current step within the epoch
    pub step: usize,
    /// Training loss value
    pub train_loss: f32,
    /// Validation loss value (if available)
    pub val_loss: Option<f32>,
    /// Learning rate
    pub learning_rate: f32,
    /// Gradient norm for stability monitoring
    pub gradient_norm: Option<f32>,
    /// Time elapsed since training started
    pub elapsed_time: Duration,
    /// Time per step in milliseconds
    pub time_per_step_ms: f64,
    /// Samples processed per second
    pub samples_per_second: f64,
}

impl TrainingMetrics {
    /// Create a new TrainingMetrics instance
    pub fn new(
        epoch: usize,
        total_epochs: usize,
        step: usize,
        train_loss: f32,
        learning_rate: f32,
        elapsed_time: Duration,
    ) -> Self {
        let time_per_step_ms = if step > 0 {
            elapsed_time.as_millis() as f64 / step as f64
        } else {
            0.0
        };

        Self {
            epoch,
            total_epochs,
            step,
            train_loss,
            val_loss: None,
            learning_rate,
            gradient_norm: None,
            elapsed_time,
            time_per_step_ms,
            samples_per_second: 0.0,
        }
    }

    /// Update validation loss if available
    pub fn with_val_loss(mut self, val_loss: f32) -> Self {
        self.val_loss = Some(val_loss);
        self
    }

    /// Update gradient norm for stability monitoring
    pub fn with_gradient_norm(mut self, gradient_norm: f32) -> Self {
        self.gradient_norm = Some(gradient_norm);
        self
    }

    /// Update samples per second metric
    pub fn with_samples_per_second(mut self, samples_per_second: f64) -> Self {
        self.samples_per_second = samples_per_second;
        self
    }
}

/// Inference performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    /// Request ID for tracing
    pub request_id: String,
    /// Model identifier
    pub model_id: String,
    /// Input sequence length
    pub input_length: usize,
    /// Output sequence length
    pub output_length: usize,
    /// Total inference time in milliseconds
    pub total_time_ms: f64,
    /// Time to first token in milliseconds
    pub ttft_ms: f64,
    /// Tokens per second
    pub tokens_per_second: f64,
    /// Memory used during inference (bytes)
    pub memory_used_bytes: usize,
    /// Peak memory usage during inference (bytes)
    pub peak_memory_bytes: usize,
}

impl InferenceMetrics {
    /// Create a new InferenceMetrics instance with request timing
    pub fn new(
        request_id: String,
        model_id: String,
        input_length: usize,
        output_length: usize,
        total_time_ms: f64,
        ttft_ms: f64,
    ) -> Self {
        let tokens_per_second = if total_time_ms > 0.0 {
            (output_length as f64 / total_time_ms) * 1000.0
        } else {
            0.0
        };

        Self {
            request_id,
            model_id,
            input_length,
            output_length,
            total_time_ms,
            ttft_ms,
            tokens_per_second,
            memory_used_bytes: 0,
            peak_memory_bytes: 0,
        }
    }

    /// Update memory usage metrics
    pub fn with_memory(mut self, used_bytes: usize, peak_bytes: usize) -> Self {
        self.memory_used_bytes = used_bytes;
        self.peak_memory_bytes = peak_bytes;
        self
    }
}

/// Memory usage tracker for monitoring system resources
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    /// Current memory usage in bytes
    current_usage: Arc<RwLock<usize>>,
    /// Peak memory usage in bytes
    peak_usage: Arc<RwLock<usize>>,
    /// Total allocated memory by tensors (bytes)
    tensor_allocations: Arc<RwLock<HashMap<String, usize>>>,
}

impl MemoryTracker {
    /// Create a new MemoryTracker instance
    pub fn new() -> Self {
        Self {
            current_usage: Arc::new(RwLock::new(0)),
            peak_usage: Arc::new(RwLock::new(0)),
            tensor_allocations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record memory allocation for a specific tensor
    pub fn record_allocation(&self, name: &str, bytes: usize) {
        let mut usage = self.current_usage.write().unwrap();
        *usage += bytes;

        let mut peak = self.peak_usage.write().unwrap();
        if *usage > *peak {
            *peak = *usage;
        }

        let mut allocations = self.tensor_allocations.write().unwrap();
        allocations.insert(name.to_string(), bytes);
    }

    /// Record memory deallocation for a specific tensor
    pub fn record_deallocation(&self, name: &str) {
        let mut usage = self.current_usage.write().unwrap();
        let mut allocations = self.tensor_allocations.write().unwrap();

        if let Some(bytes) = allocations.remove(name) {
            *usage -= bytes;
        }
    }

    /// Get current memory usage in bytes
    pub fn get_current_usage(&self) -> usize {
        *self.current_usage.read().unwrap()
    }

    /// Get peak memory usage in bytes
    pub fn get_peak_usage(&self) -> usize {
        *self.peak_usage.read().unwrap()
    }

    /// Get all tensor allocations
    pub fn get_tensor_allocations(&self) -> HashMap<String, usize> {
        self.tensor_allocations.read().unwrap().clone()
    }

    /// Reset peak usage tracker (useful for benchmarking)
    pub fn reset_peak(&self) {
        *self.peak_usage.write().unwrap() = 0;
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Error rate monitor for tracking system reliability
#[derive(Debug, Clone)]
pub struct ErrorMonitor {
    /// Total number of operations tracked
    total_operations: Arc<RwLock<usize>>,
    /// Number of failed operations
    failed_operations: Arc<RwLock<usize>>,
    /// Error types and their counts
    error_types: Arc<RwLock<HashMap<String, usize>>>,
    /// Time window for rate calculation (in seconds)
    #[allow(dead_code)]
    window_size_secs: u64,
}

impl ErrorMonitor {
    /// Create a new ErrorMonitor instance
    pub fn new(window_size_secs: u64) -> Self {
        Self {
            total_operations: Arc::new(RwLock::new(0)),
            failed_operations: Arc::new(RwLock::new(0)),
            error_types: Arc::new(RwLock::new(HashMap::new())),
            window_size_secs,
        }
    }

    /// Record a successful operation
    pub fn record_success(&self) {
        let mut total = self.total_operations.write().unwrap();
        *total += 1;
    }

    /// Record a failed operation with error type
    pub fn record_failure(&self, error_type: &str) {
        let mut total = self.total_operations.write().unwrap();
        *total += 1;

        let mut failed = self.failed_operations.write().unwrap();
        *failed += 1;

        let mut error_types = self.error_types.write().unwrap();
        *error_types.entry(error_type.to_string()).or_insert(0) += 1;
    }

    /// Get current error rate as a percentage (0-100)
    pub fn get_error_rate(&self) -> f64 {
        let total = *self.total_operations.read().unwrap();
        if total == 0 {
            return 0.0;
        }

        let failed = *self.failed_operations.read().unwrap();
        (failed as f64 / total as f64) * 100.0
    }

    /// Get error type distribution
    pub fn get_error_distribution(&self) -> HashMap<String, usize> {
        self.error_types.read().unwrap().clone()
    }

    /// Get overall statistics
    pub fn get_stats(&self) -> (usize, usize, f64) {
        let total = *self.total_operations.read().unwrap();
        let failed = *self.failed_operations.read().unwrap();
        let error_rate = self.get_error_rate();
        (total, failed, error_rate)
    }
}

impl Default for ErrorMonitor {
    fn default() -> Self {
        Self::new(60) // 1 minute window by default
    }
}

/// Custom metrics collector for application-specific monitoring
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    /// Named metric values
    metrics: Arc<RwLock<HashMap<String, f64>>>,
    /// Counter metrics (monotonically increasing)
    counters: Arc<RwLock<HashMap<String, u64>>>,
}

impl MetricsCollector {
    /// Create a new MetricsCollector instance
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set a gauge metric (can go up or down)
    pub fn set_gauge(&self, name: &str, value: f64) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.insert(name.to_string(), value);
    }

    /// Increment a counter metric
    pub fn increment_counter(&self, name: &str, amount: u64) {
        let mut counters = self.counters.write().unwrap();
        *counters.entry(name.to_string()).or_insert(0) += amount;
    }

    /// Get all gauge metrics
    pub fn get_all_gauges(&self) -> HashMap<String, f64> {
        self.metrics.read().unwrap().clone()
    }

    /// Get all counter values
    pub fn get_all_counters(&self) -> HashMap<String, u64> {
        self.counters.read().unwrap().clone()
    }

    /// Export metrics as JSON-serializable format
    pub fn export_metrics(&self) -> serde_json::Value {
        let gauges = self.get_all_gauges();
        let counters = self.get_all_counters();

        serde_json::json!({
            "gauges": gauges,
            "counters": counters,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Structured logger for production-grade logging
#[derive(Debug, Clone)]
pub struct StructuredLogger {
    /// Log level filter
    min_level: log::LevelFilter,
}

impl StructuredLogger {
    /// Create a new StructuredLogger instance
    pub fn new(min_level: log::LevelFilter) -> Self {
        Self { min_level }
    }

    /// Initialize the logger with the given level filter
    pub fn init(&self) -> Result<(), log::SetLoggerError> {
        log::set_max_level(self.min_level);
        log::set_boxed_logger(Box::new(StructuredLoggerImpl {
            min_level: self.min_level,
        }))
    }

    /// Log a training metric event
    pub fn log_training_metric(&self, metrics: &TrainingMetrics) {
        if log::log_enabled!(log::Level::Info) {
            let json = serde_json::json!({
                "event": "training_metric",
                "epoch": metrics.epoch,
                "step": metrics.step,
                "loss": metrics.train_loss,
                "learning_rate": metrics.learning_rate,
                "time_per_step_ms": metrics.time_per_step_ms,
            });
            log::info!("{}", json);
        }
    }

    /// Log an inference event
    pub fn log_inference(&self, metrics: &InferenceMetrics) {
        if log::log_enabled!(log::Level::Info) {
            let json = serde_json::json!({
                "event": "inference",
                "model_id": metrics.model_id,
                "input_length": metrics.input_length,
                "output_length": metrics.output_length,
                "total_time_ms": metrics.total_time_ms,
                "tokens_per_second": metrics.tokens_per_second,
            });
            log::info!("{}", json);
        }
    }

    /// Log a memory usage event
    pub fn log_memory_usage(&self, tracker: &MemoryTracker) {
        if log::log_enabled!(log::Level::Warn) && tracker.get_current_usage() > 1024 * 1024 * 1024 {
            // Warn if > 1GB
            let json = serde_json::json!({
                "event": "memory_warning",
                "current_bytes": tracker.get_current_usage(),
                "peak_bytes": tracker.get_peak_usage(),
            });
            log::warn!("{}", json);
        }
    }

    /// Log an error event with context
    pub fn log_error(&self, error_type: &str, message: &str) {
        if log::log_enabled!(log::Level::Error) {
            let json = serde_json::json!({
                "event": "error",
                "type": error_type,
                "message": message,
            });
            log::error!("{}", json);
        }
    }
}

struct StructuredLoggerImpl {
    min_level: log::LevelFilter,
}

impl log::Log for StructuredLoggerImpl {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= self.min_level
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            println!("{} - {}", record.level(), record.args());
        }
    }

    fn flush(&self) {
        // Flush stdout to ensure all log messages are written
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}

/// Export all monitoring modules for easy access
pub mod training_metrics {
    pub use super::{StructuredLogger, TrainingMetrics};
}

pub mod inference_monitoring {
    pub use super::{InferenceMetrics, StructuredLogger};
}

pub mod memory_tracking {
    pub use super::MemoryTracker;
}

pub mod error_monitoring {
    pub use super::ErrorMonitor;
}

pub mod custom_metrics {
    pub use super::MetricsCollector;
}
