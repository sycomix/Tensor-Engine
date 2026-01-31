//! Asynchronous Tensor Operations
//!
//! This module provides non-blocking tensor operations that can run concurrently
//! with other work. Operations return futures that resolve to tensor results.
//!
//! # Features
//!
//! - **Non-blocking operations**: Tensor computations run in a thread pool
//! - **Computation overlap**: Multiple operations can run concurrently
//! - **Transfer overlap**: Data transfers can overlap with computation (GPU)
//!
//! # Example
//!
//! ```rust,ignore
//! use tensor_engine::async_ops::{AsyncContext, AsyncTensor};
//! use tensor_engine::tensor::Tensor;
//!
//! #[tokio::main]
//! async fn main() {
//!     let ctx = AsyncContext::new();
//!     
//!     let a = Tensor::randn(&[512, 512]);
//!     let b = Tensor::randn(&[512, 512]);
//!     
//!     // Non-blocking matmul
//!     let result = ctx.matmul_async(&a, &b).await;
//! }
//! ```

use crate::tensor::Tensor;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

#[allow(unused_imports)]
use ndarray::ArrayD;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

/// Error type for async tensor operations.
#[derive(Debug, Clone)]
pub enum AsyncError {
    /// The operation was cancelled before completion.
    Cancelled,
    /// An error occurred during the operation.
    OperationFailed(String),
    /// The async context was shut down.
    ContextShutdown,
    /// The result channel was dropped.
    ChannelClosed,
}

impl std::fmt::Display for AsyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsyncError::Cancelled => write!(f, "Operation was cancelled"),
            AsyncError::OperationFailed(msg) => write!(f, "Operation failed: {}", msg),
            AsyncError::ContextShutdown => write!(f, "Async context was shut down"),
            AsyncError::ChannelClosed => write!(f, "Result channel was closed"),
        }
    }
}

impl std::error::Error for AsyncError {}

/// Result type for async tensor operations.
pub type AsyncResult<T> = Result<T, AsyncError>;

/// Statistics for async operation monitoring.
#[derive(Debug, Default)]
pub struct AsyncStatistics {
    /// Total operations submitted.
    pub total_submitted: AtomicUsize,
    /// Operations currently in flight.
    pub in_flight: AtomicUsize,
    /// Operations completed successfully.
    pub completed: AtomicUsize,
    /// Operations that failed.
    pub failed: AtomicUsize,
}

impl AsyncStatistics {
    /// Get the number of operations in flight.
    pub fn get_in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Get the success rate (0.0 to 1.0).
    pub fn success_rate(&self) -> f64 {
        let completed = self.completed.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        let total = completed + failed;
        if total == 0 {
            return 1.0;
        }
        completed as f64 / total as f64
    }
}

/// Configuration for the async execution context.
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    /// Maximum number of concurrent operations.
    pub max_concurrent_ops: usize,
    /// Whether to enable operation statistics.
    pub enable_statistics: bool,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            max_concurrent_ops: num_cpus(),
            enable_statistics: false,
        }
    }
}

impl AsyncConfig {
    /// Create configuration with statistics enabled.
    pub fn with_statistics(mut self) -> Self {
        self.enable_statistics = true;
        self
    }

    /// Set maximum concurrent operations.
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent_ops = max;
        self
    }
}

/// Get the number of CPUs available.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

/// Inner state for the async context.
struct AsyncContextInner {
    /// Configuration.
    #[allow(dead_code)]
    config: AsyncConfig,
    /// Statistics (optional).
    statistics: Option<AsyncStatistics>,
    /// Semaphore for limiting concurrent operations.
    semaphore: Arc<tokio::sync::Semaphore>,
}

/// Execution context for async tensor operations.
///
/// The context manages a thread pool and limits the number of concurrent
/// operations to prevent resource exhaustion.
#[derive(Clone)]
pub struct AsyncContext {
    inner: Arc<AsyncContextInner>,
}

impl AsyncContext {
    /// Create a new async context with default configuration.
    pub fn new() -> Self {
        Self::with_config(AsyncConfig::default())
    }

    /// Create a new async context with the given configuration.
    pub fn with_config(config: AsyncConfig) -> Self {
        log::info!(
            "Creating AsyncContext with max {} concurrent operations",
            config.max_concurrent_ops
        );

        let statistics = if config.enable_statistics {
            Some(AsyncStatistics::default())
        } else {
            None
        };

        let semaphore = Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_ops));

        Self {
            inner: Arc::new(AsyncContextInner {
                config,
                statistics,
                semaphore,
            }),
        }
    }

    /// Get statistics if enabled.
    pub fn statistics(&self) -> Option<&AsyncStatistics> {
        self.inner.statistics.as_ref()
    }

    /// Perform async matrix multiplication.
    ///
    /// Returns a future that resolves to the result tensor.
    pub fn matmul_async(&self, a: &Tensor, b: &Tensor) -> TensorFuture {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let _inner = Arc::clone(&self.inner);

        self.spawn_operation(move || {
            // Perform matmul using existing Tensor API
            a_clone.matmul(&b_clone)
        })
    }

    /// Perform async element-wise addition.
    pub fn add_async(&self, a: &Tensor, b: &Tensor) -> TensorFuture {
        let a_clone = a.clone();
        let b_clone = b.clone();

        self.spawn_operation(move || a_clone.add(&b_clone))
    }

    /// Perform async element-wise multiplication.
    pub fn mul_async(&self, a: &Tensor, b: &Tensor) -> TensorFuture {
        let a_clone = a.clone();
        let b_clone = b.clone();

        self.spawn_operation(move || a_clone.mul(&b_clone))
    }

    /// Perform async element-wise subtraction.
    pub fn sub_async(&self, a: &Tensor, b: &Tensor) -> TensorFuture {
        let a_clone = a.clone();
        let b_clone = b.clone();

        self.spawn_operation(move || a_clone.sub(&b_clone))
    }

    /// Perform async element-wise division.
    pub fn div_async(&self, a: &Tensor, b: &Tensor) -> TensorFuture {
        let a_clone = a.clone();
        let b_clone = b.clone();

        self.spawn_operation(move || a_clone.div(&b_clone))
    }

    /// Perform async ReLU activation.
    pub fn relu_async(&self, t: &Tensor) -> TensorFuture {
        let t_clone = t.clone();

        self.spawn_operation(move || t_clone.relu())
    }

    /// Perform async sigmoid activation.
    pub fn sigmoid_async(&self, t: &Tensor) -> TensorFuture {
        let t_clone = t.clone();

        self.spawn_operation(move || t_clone.sigmoid())
    }

    /// Perform async softmax along the last axis.
    pub fn softmax_async(&self, t: &Tensor) -> TensorFuture {
        let t_clone = t.clone();
        // Default to last axis (-1)
        let axis = t_clone.lock().storage.shape().len().saturating_sub(1);
        self.spawn_operation(move || t_clone.softmax(axis))
    }

    /// Perform async sum reduction.
    pub fn sum_async(&self, t: &Tensor) -> TensorFuture {
        let t_clone = t.clone();

        self.spawn_operation(move || t_clone.sum())
    }

    /// Perform async mean reduction.
    pub fn mean_async(&self, t: &Tensor) -> TensorFuture {
        let t_clone = t.clone();

        self.spawn_operation(move || t_clone.mean())
    }

    /// Perform async layer normalization.
    ///
    /// Uses identity gamma (1s) and zero beta for simplicity.
    pub fn layer_norm_async(&self, t: &Tensor, axis: usize, eps: f32) -> TensorFuture {
        let t_clone = t.clone();
        let shape = t_clone.lock().storage.shape();
        let norm_size = shape.get(axis).copied().unwrap_or(1);
        // Create identity gamma and zero beta
        let gamma = Tensor::ones(&[norm_size]);
        let beta = Tensor::zeros(&[norm_size]);

        self.spawn_operation(move || t_clone.layer_norm(axis, eps, &gamma, &beta))
    }

    /// Spawn an async operation with concurrency limiting.
    fn spawn_operation<F>(&self, op: F) -> TensorFuture
    where
        F: FnOnce() -> Tensor + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        let (tx, rx) = oneshot::channel();

        // Track statistics
        if let Some(ref stats) = inner.statistics {
            stats.total_submitted.fetch_add(1, Ordering::Relaxed);
            stats.in_flight.fetch_add(1, Ordering::Relaxed);
        }

        let stats_clone = inner.statistics.as_ref().map(|_| Arc::clone(&inner));

        let handle = tokio::spawn(async move {
            // Acquire semaphore permit to limit concurrency
            let _permit = inner.semaphore.acquire().await;

            // Run the operation in a blocking task to not block the async runtime
            let result = tokio::task::spawn_blocking(op).await;

            // Update statistics
            if let Some(ref inner) = stats_clone {
                if let Some(ref stats) = inner.statistics {
                    stats.in_flight.fetch_sub(1, Ordering::Relaxed);
                    match &result {
                        Ok(_) => {
                            stats.completed.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            stats.failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }

            // Send result
            let tensor_result = result.map_err(|e| AsyncError::OperationFailed(e.to_string()));
            let _ = tx.send(tensor_result);
        });

        TensorFuture {
            receiver: Some(rx),
            handle: Some(handle),
        }
    }

    /// Execute multiple operations concurrently and collect results.
    pub async fn join_all(&self, futures: Vec<TensorFuture>) -> Vec<AsyncResult<Tensor>> {
        let mut results = Vec::with_capacity(futures.len());
        for future in futures {
            results.push(future.await);
        }
        results
    }
}

impl Default for AsyncContext {
    fn default() -> Self {
        Self::new()
    }
}

/// A future that resolves to a tensor result.
///
/// This is the return type of all async tensor operations.
pub struct TensorFuture {
    receiver: Option<oneshot::Receiver<AsyncResult<Tensor>>>,
    handle: Option<JoinHandle<()>>,
}

impl TensorFuture {
    /// Cancel the pending operation.
    ///
    /// After cancellation, awaiting this future will return `AsyncError::Cancelled`.
    pub fn cancel(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
        self.receiver = None;
    }

    /// Check if the operation is still pending.
    pub fn is_pending(&self) -> bool {
        self.receiver.is_some()
    }
}

impl Future for TensorFuture {
    type Output = AsyncResult<Tensor>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.receiver.as_mut() {
            None => Poll::Ready(Err(AsyncError::Cancelled)),
            Some(rx) => {
                let pinned = Pin::new(rx);
                match pinned.poll(cx) {
                    Poll::Pending => Poll::Pending,
                    Poll::Ready(Ok(result)) => {
                        self.receiver = None;
                        Poll::Ready(result)
                    }
                    Poll::Ready(Err(_)) => {
                        self.receiver = None;
                        Poll::Ready(Err(AsyncError::ChannelClosed))
                    }
                }
            }
        }
    }
}

/// Extension trait for async operations on Tensor.
pub trait AsyncTensorExt {
    /// Perform async matrix multiplication.
    fn matmul_async(&self, other: &Self, ctx: &AsyncContext) -> TensorFuture;

    /// Perform async element-wise addition.
    fn add_async(&self, other: &Self, ctx: &AsyncContext) -> TensorFuture;

    /// Perform async element-wise multiplication.
    fn mul_async(&self, other: &Self, ctx: &AsyncContext) -> TensorFuture;

    /// Perform async ReLU activation.
    fn relu_async(&self, ctx: &AsyncContext) -> TensorFuture;

    /// Perform async softmax.
    fn softmax_async(&self, ctx: &AsyncContext) -> TensorFuture;
}

impl AsyncTensorExt for Tensor {
    fn matmul_async(&self, other: &Self, ctx: &AsyncContext) -> TensorFuture {
        ctx.matmul_async(self, other)
    }

    fn add_async(&self, other: &Self, ctx: &AsyncContext) -> TensorFuture {
        ctx.add_async(self, other)
    }

    fn mul_async(&self, other: &Self, ctx: &AsyncContext) -> TensorFuture {
        ctx.mul_async(self, other)
    }

    fn relu_async(&self, ctx: &AsyncContext) -> TensorFuture {
        ctx.relu_async(self)
    }

    fn softmax_async(&self, ctx: &AsyncContext) -> TensorFuture {
        ctx.softmax_async(self)
    }
}

/// Batch async operations for efficient parallel execution.
pub struct AsyncBatch {
    ctx: AsyncContext,
    operations: Vec<TensorFuture>,
}

impl AsyncBatch {
    /// Create a new batch with the given context.
    pub fn new(ctx: AsyncContext) -> Self {
        Self {
            ctx,
            operations: Vec::new(),
        }
    }

    /// Add a matmul operation to the batch.
    pub fn matmul(mut self, a: &Tensor, b: &Tensor) -> Self {
        self.operations.push(self.ctx.matmul_async(a, b));
        self
    }

    /// Add an add operation to the batch.
    pub fn add(mut self, a: &Tensor, b: &Tensor) -> Self {
        self.operations.push(self.ctx.add_async(a, b));
        self
    }

    /// Add a relu operation to the batch.
    pub fn relu(mut self, t: &Tensor) -> Self {
        self.operations.push(self.ctx.relu_async(t));
        self
    }

    /// Execute all operations and collect results.
    pub async fn execute(self) -> Vec<AsyncResult<Tensor>> {
        self.ctx.join_all(self.operations).await
    }

    /// Get the number of pending operations.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    fn create_test_tensor(shape: &[usize], val: f32) -> Tensor {
        let data = ArrayD::from_elem(IxDyn(shape), val);
        Tensor::new(data, false)
    }

    #[tokio::test]
    async fn test_async_matmul() {
        let ctx = AsyncContext::new();

        let a = create_test_tensor(&[2, 3], 1.0);
        let b = create_test_tensor(&[3, 2], 2.0);

        let result = ctx.matmul_async(&a, &b).await;
        assert!(result.is_ok());

        let tensor = result.expect("Matmul should succeed");
        let shape = tensor.lock().storage.shape();
        assert_eq!(shape, vec![2, 2]);
    }

    #[tokio::test]
    async fn test_async_add() {
        let ctx = AsyncContext::new();

        let a = create_test_tensor(&[2, 2], 1.0);
        let b = create_test_tensor(&[2, 2], 2.0);

        let result = ctx.add_async(&a, &b).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_relu() {
        let ctx = AsyncContext::new();

        let t = create_test_tensor(&[2, 2], -1.0);

        let result = ctx.relu_async(&t).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        let ctx = AsyncContext::with_config(
            AsyncConfig::default()
                .with_statistics()
                .with_max_concurrent(4),
        );

        let tensors: Vec<_> = (0..10)
            .map(|i| create_test_tensor(&[32, 32], i as f32))
            .collect();

        let mut futures = Vec::new();
        for i in 0..9 {
            futures.push(ctx.add_async(&tensors[i], &tensors[i + 1]));
        }

        let results = ctx.join_all(futures).await;
        assert_eq!(results.len(), 9);
        assert!(results.iter().all(|r| r.is_ok()));

        if let Some(stats) = ctx.statistics() {
            assert_eq!(stats.completed.load(Ordering::Relaxed), 9);
        }
    }

    #[tokio::test]
    async fn test_tensor_ext_trait() {
        let ctx = AsyncContext::new();

        let a = create_test_tensor(&[2, 3], 1.0);
        let b = create_test_tensor(&[3, 4], 1.0);

        // Use the extension trait
        let result = a.matmul_async(&b, &ctx).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_batch() {
        let ctx = AsyncContext::new();

        let a = create_test_tensor(&[2, 2], 1.0);
        let b = create_test_tensor(&[2, 2], 2.0);

        let batch = AsyncBatch::new(ctx.clone()).add(&a, &b).relu(&a);

        assert_eq!(batch.len(), 2);

        let results = batch.execute().await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn test_cancel_operation() {
        let ctx = AsyncContext::new();

        let a = create_test_tensor(&[100, 100], 1.0);
        let b = create_test_tensor(&[100, 100], 1.0);

        let mut future = ctx.matmul_async(&a, &b);
        future.cancel();

        let result = future.await;
        assert!(matches!(result, Err(AsyncError::Cancelled)));
    }
}
