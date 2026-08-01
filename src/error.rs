//! Error types and validation module for Tensor Engine
//!
//! This module provides structured error handling to replace panics throughout the codebase,
//! enabling better error recovery, debugging, and production robustness.

/// Comprehensive error types for Tensor Engine operations
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    /// Shape mismatch errors
    #[error("Shape mismatch in {operation}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        operation: String,
    },

    /// Memory allocation errors
    #[error("Out of memory: requested {requested_bytes} bytes, only {available_bytes} available")]
    OutOfMemory {
        requested_bytes: usize,
        available_bytes: usize,
    },

    /// Device errors
    #[error("Device {device_id} error: {message}")]
    DeviceError { device_id: i32, message: String },

    /// Backend-specific errors
    #[error("Backend {backend_name} error: {message}")]
    BackendError {
        backend_name: String,
        message: String,
    },

    /// Computation errors
    #[error("Computation error in {operation}: {details}")]
    ComputationError { operation: String, details: String },

    /// I/O errors
    #[error("I/O error at {path}: {details}")]
    IoError { path: String, details: String },

    /// Validation errors
    #[error("Validation error: field={field}, value={value}, constraint={constraint}")]
    ValidationError {
        field: String,
        value: String,
        constraint: String,
    },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Generic error with message
    #[error("{message}")]
    Generic { message: String },
}

/// Result type alias for operations that can fail
pub type TensorResult<T> = Result<T, TensorError>;

impl From<ndarray::ShapeError> for TensorError {
    fn from(err: ndarray::ShapeError) -> Self {
        TensorError::ShapeMismatch {
            expected: vec![],
            actual: vec![],
            operation: format!("shape error: {err}"),
        }
    }
}

impl From<std::io::Error> for TensorError {
    fn from(err: std::io::Error) -> Self {
        TensorError::IoError {
            path: String::new(),
            details: err.to_string(),
        }
    }
}

impl From<std::num::ParseIntError> for TensorError {
    fn from(err: std::num::ParseIntError) -> Self {
        TensorError::ConfigError {
            message: err.to_string(),
        }
    }
}

impl From<std::num::ParseFloatError> for TensorError {
    fn from(err: std::num::ParseFloatError) -> Self {
        TensorError::ConfigError {
            message: err.to_string(),
        }
    }
}

impl From<String> for TensorError {
    fn from(msg: String) -> Self {
        TensorError::Generic { message: msg }
    }
}

impl From<&str> for TensorError {
    fn from(msg: &str) -> Self {
        TensorError::Generic {
            message: msg.to_string(),
        }
    }
}

// Error handling and validation utilities for Tensor Engine
pub mod shape_validation {
    use super::{TensorError, TensorResult};
    use crate::tensor::Tensor;

    /// Validate shapes for binary operations
    pub fn validate_binary_shapes(a: &Tensor, b: &Tensor, operation: &str) -> TensorResult<()> {
        let a_shape = a.lock().storage.shape().to_vec();
        let b_shape = b.lock().storage.shape().to_vec();

        if a_shape != b_shape {
            Err(TensorError::ShapeMismatch {
                expected: a_shape.to_vec(),
                actual: b_shape.to_vec(),
                operation: operation.to_string(),
            })
        } else {
            Ok(())
        }
    }

    /// Validate shapes for matrix multiplication
    pub fn validate_matmul_shapes(a: &Tensor, b: &Tensor) -> TensorResult<(usize, usize, usize)> {
        let a_shape = a.lock().storage.shape().to_vec();
        let b_shape = b.lock().storage.shape().to_vec();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![2, 2],
                actual: vec![a_shape.len(), b_shape.len()],
                operation: "matrix multiplication".to_string(),
            });
        }

        let (m, k) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);

        if k != k2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![2, 2],
                actual: vec![2, 2],
                operation: "matrix multiplication".to_string(),
            });
        }

        Ok((m, k, n))
    }

    /// Validate broadcasting capabilities
    pub fn can_broadcast(from: &[usize], to: &[usize]) -> bool {
        // Check if shapes are compatible for broadcasting
        if from.len() > to.len() {
            // Can broadcast if later dimensions match or are 1
            from.iter()
                .rev()
                .zip(to.iter().rev())
                .all(|(&f, &t)| f == 1 || f == t)
        } else {
            // Can broadcast if earlier dimensions match
            from.iter().zip(to.iter()).all(|(f, t)| f == t)
        }
    }
}

/// Memory management utilities
pub mod memory {
    use std::sync::{Mutex, OnceLock};

    /// Track memory usage statistics
    #[derive(Debug, Default, Clone)]
    pub struct MemoryStats {
        pub total_allocated: usize,
        pub peak_usage: usize,
        pub allocation_count: usize,
        pub deallocation_count: usize,
    }

    static MEMORY_STATS: OnceLock<Mutex<MemoryStats>> = OnceLock::new();

    /// Initialize memory tracking
    pub fn init_memory_tracking() {
        let _ = MEMORY_STATS.set(Mutex::new(MemoryStats::default()));
    }

    /// Record allocation
    pub fn record_allocation(bytes: usize) {
        if let Some(mutex) = MEMORY_STATS.get() {
            if let Ok(mut stats) = mutex.lock() {
                stats.total_allocated += bytes;
                stats.allocation_count += 1;
                stats.peak_usage = stats.peak_usage.max(stats.total_allocated);
            }
        }
    }

    /// Record deallocation
    pub fn record_deallocation(_bytes: usize) {
        if let Some(mutex) = MEMORY_STATS.get() {
            if let Ok(mut stats) = mutex.lock() {
                stats.deallocation_count += 1;
            }
        }
    }

    /// Get current memory statistics
    pub fn get_memory_stats() -> MemoryStats {
        if let Some(mutex) = MEMORY_STATS.get() {
            if let Ok(stats) = mutex.lock() {
                return stats.clone();
            }
        }
        MemoryStats::default()
    }
}
