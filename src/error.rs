//! Error types and validation module for Tensor Engine
//!
//! This module provides structured error handling to replace panics throughout the codebase,
//! enabling better error recovery, debugging, and production robustness.

/// Comprehensive error types for Tensor Engine operations
#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    /// Shape mismatch errors
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        operation: String,
    },

    /// Memory allocation errors
    OutOfMemory {
        requested_bytes: usize,
        available_bytes: usize,
    },

    /// Device errors
    DeviceError { device_id: i32, message: String },

    /// Backend-specific errors
    BackendError {
        backend_name: String,
        message: String,
    },

    /// Computation errors
    ComputationError { operation: String, details: String },

    /// I/O errors
    IoError { path: String, details: String },

    /// Validation errors
    ValidationError {
        field: String,
        value: String,
        constraint: String,
    },

    /// Generic error with message
    Generic { message: String },
}

impl std::error::Error for TensorError {}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch {
                expected,
                actual,
                operation,
            } => {
                write!(
                    f,
                    "Shape mismatch in {}: expected {:?}, got {:?}",
                    operation, expected, actual
                )
            }
            TensorError::OutOfMemory {
                requested_bytes,
                available_bytes,
            } => {
                write!(
                    f,
                    "Out of memory: requested {} bytes, only {} available",
                    requested_bytes, available_bytes
                )
            }
            TensorError::DeviceError { device_id, message } => {
                write!(f, "Device {} error: {}", device_id, message)
            }
            TensorError::BackendError {
                backend_name,
                message,
            } => {
                write!(f, "Backend {} error: {}", backend_name, message)
            }
            TensorError::ComputationError { operation, details } => {
                write!(f, "Computation error in {}: {}", operation, details)
            }
            TensorError::IoError { path, details } => {
                write!(f, "I/O error at {}: {}", path, details)
            }
            TensorError::ValidationError {
                field,
                value,
                constraint,
            } => {
                write!(
                    f,
                    "Validation error: field={}, value={}, constraint={}",
                    field, value, constraint
                )
            }
            TensorError::Generic { message } => {
                write!(f, "{}", message)
            }
        }
    }
}

/// Result type alias for operations that can fail
pub type TensorResult<T> = Result<T, TensorError>;

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
