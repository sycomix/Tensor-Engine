//! Memory Pool for Tensor Allocations
//!
//! This module provides an arena-based memory pool to reduce allocation overhead
//! for tensor operations. Instead of allocating/deallocating memory for each tensor,
//! the pool maintains a set of reusable buffers organized by size class.
//!
//! # Features
//!
//! - **Size classes**: Power-of-2 allocation sizes for efficient bucketing
//! - **Thread-safe**: Uses `RwLock` for concurrent access
//! - **RAII**: `PooledBuffer` automatically returns memory to the pool on drop
//! - **Statistics**: Optional allocation tracking for debugging
//!
//! # Example
//!
//! ```rust
//! use tensor_engine::memory_pool::{TensorPool, PoolConfig};
//!
//! let pool = TensorPool::new(PoolConfig::default());
//! let buffer = pool.allocate(1024); // Gets a 1KB buffer
//! // Use buffer contents
//! drop(buffer); // Returns to pool for reuse
//! ```

use std::alloc::{alloc, dealloc, Layout};
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Configuration for the tensor memory pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum total memory the pool can hold (in bytes).
    /// When exceeded, oldest buffers are deallocated instead of cached.
    pub max_pool_size_bytes: usize,

    /// Number of buffers to pre-allocate per size class during initialization.
    /// Set to 0 to disable pre-allocation.
    pub initial_capacity_per_size: usize,

    /// Enable statistics tracking (allocation counts, hit rates, etc.).
    /// Slight performance overhead when enabled.
    pub enable_statistics: bool,

    /// Alignment for all allocations (must be power of 2).
    /// 64 bytes is optimal for SIMD operations.
    pub alignment: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size_bytes: 1024 * 1024 * 1024, // 1 GB default
            initial_capacity_per_size: 0,
            enable_statistics: false,
            alignment: 64, // Cache-line aligned for performance
        }
    }
}

impl PoolConfig {
    /// Create a configuration with a specific maximum pool size.
    pub fn with_max_size(max_bytes: usize) -> Self {
        Self {
            max_pool_size_bytes: max_bytes,
            ..Default::default()
        }
    }

    /// Enable statistics tracking.
    pub fn with_statistics(mut self) -> Self {
        self.enable_statistics = true;
        self
    }

    /// Set pre-allocation count per size class.
    pub fn with_preallocate(mut self, count: usize) -> Self {
        self.initial_capacity_per_size = count;
        self
    }
}

/// Statistics for pool usage monitoring.
#[derive(Debug, Default)]
pub struct PoolStatistics {
    /// Total number of allocation requests.
    pub total_allocations: AtomicU64,
    /// Allocations served from the pool (cache hits).
    pub pool_hits: AtomicU64,
    /// Allocations that required new memory (cache misses).
    pub pool_misses: AtomicU64,
    /// Total number of deallocations (returns to pool).
    pub total_deallocations: AtomicU64,
    /// Current memory held by the pool (in bytes).
    pub current_pool_bytes: AtomicUsize,
    /// Peak memory held by the pool (in bytes).
    pub peak_pool_bytes: AtomicUsize,
}

impl PoolStatistics {
    /// Calculate the pool hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_allocations.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let hits = self.pool_hits.load(Ordering::Relaxed);
        hits as f64 / total as f64
    }

    /// Get a human-readable summary of statistics.
    pub fn summary(&self) -> String {
        format!(
            "Pool Statistics:\n  Total allocations: {}\n  Pool hits: {} ({:.1}%)\n  Pool misses: {}\n  Current pool size: {} bytes\n  Peak pool size: {} bytes",
            self.total_allocations.load(Ordering::Relaxed),
            self.pool_hits.load(Ordering::Relaxed),
            self.hit_rate() * 100.0,
            self.pool_misses.load(Ordering::Relaxed),
            self.current_pool_bytes.load(Ordering::Relaxed),
            self.peak_pool_bytes.load(Ordering::Relaxed)
        )
    }

    fn record_allocation(&self, hit: bool, size: usize) {
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        if hit {
            self.pool_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.pool_misses.fetch_add(1, Ordering::Relaxed);
        }
        // Update current size (we're removing from pool on hit)
        if hit {
            self.current_pool_bytes.fetch_sub(size, Ordering::Relaxed);
        }
    }

    fn record_deallocation(&self, size: usize) {
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        let new_size = self.current_pool_bytes.fetch_add(size, Ordering::Relaxed) + size;
        // Update peak if needed
        let mut peak = self.peak_pool_bytes.load(Ordering::Relaxed);
        while new_size > peak {
            match self.peak_pool_bytes.compare_exchange_weak(
                peak,
                new_size,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => peak = current,
            }
        }
    }
}

/// Size class index for pool bucketing.
/// Each class corresponds to a power-of-2 size.
const NUM_SIZE_CLASSES: usize = 15;

/// Minimum size class (1 KB = 2^10 bytes).
const MIN_SIZE_CLASS_BITS: usize = 10;

/// Maximum size for pooled allocations (16 MB = 2^24 bytes).
/// Larger allocations go directly to the system allocator.
const MAX_POOLED_SIZE: usize = 1 << 24; // 16 MB

/// Get the size class index for a given byte count.
/// Returns None for sizes that exceed MAX_POOLED_SIZE.
fn size_class_for_bytes(bytes: usize) -> Option<usize> {
    if bytes <= 1 {
        return Some(0);
    }
    if bytes > MAX_POOLED_SIZE {
        return None;
    }
    // Round up to next power of 2
    let bits = (bytes - 1).ilog2() as usize + 1;
    if bits <= MIN_SIZE_CLASS_BITS {
        Some(0)
    } else {
        let class = bits - MIN_SIZE_CLASS_BITS;
        if class >= NUM_SIZE_CLASSES {
            None
        } else {
            Some(class)
        }
    }
}

/// Get the allocation size for a size class.
fn size_for_class(class: usize) -> usize {
    1 << (MIN_SIZE_CLASS_BITS + class)
}

/// Raw buffer stored in the pool.
struct RawBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
}

// Safety: RawBuffer owns its memory and access is synchronized by RwLock
unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        // Safety: ptr was allocated with this layout
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// A pooled memory buffer that returns to the pool on drop.
///
/// This is an RAII wrapper that ensures memory is properly recycled.
pub struct PooledBuffer {
    ptr: NonNull<u8>,
    size: usize,
    capacity: usize,
    layout: Layout,
    pool: Arc<TensorPoolInner>,
    size_class: Option<usize>,
}

// Safety: PooledBuffer owns its memory slice exclusively
unsafe impl Send for PooledBuffer {}
unsafe impl Sync for PooledBuffer {}

impl PooledBuffer {
    /// Get a raw pointer to the buffer data.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get a mutable raw pointer to the buffer data.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the requested size of the buffer.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Get the actual capacity of the underlying allocation.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get a slice view of the buffer.
    ///
    /// # Safety
    /// Caller must ensure the buffer contains valid data up to `len()`.
    pub unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr.as_ptr(), self.size)
    }

    /// Get a mutable slice view of the buffer.
    ///
    /// # Safety
    /// Caller must ensure the buffer is properly initialized.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size)
    }

    /// Get a typed slice view of the buffer.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - The buffer contains valid, initialized data
    /// - Data is properly aligned for type T
    /// - `len() / size_of::<T>()` elements are valid
    pub unsafe fn as_typed_slice<T>(&self) -> &[T] {
        let count = self.size / size_of::<T>();
        std::slice::from_raw_parts(self.ptr.as_ptr() as *const T, count)
    }

    /// Get a mutable typed slice view of the buffer.
    ///
    /// # Safety
    /// See `as_typed_slice`.
    pub unsafe fn as_typed_slice_mut<T>(&mut self) -> &mut [T] {
        let count = self.size / size_of::<T>();
        std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count)
    }

    /// Zero-initialize the buffer contents.
    pub fn zero(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.capacity);
        }
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Return buffer to pool instead of deallocating
        self.pool
            .return_buffer(self.ptr, self.layout, self.size_class);
    }
}

/// Thread-safe inner pool state.
struct TensorPoolInner {
    /// Free lists for each size class.
    free_lists: [RwLock<VecDeque<RawBuffer>>; NUM_SIZE_CLASSES],
    /// Pool configuration.
    config: PoolConfig,
    /// Pool statistics (optional).
    statistics: Option<PoolStatistics>,
}

impl TensorPoolInner {
    fn new(config: PoolConfig) -> Self {
        // Initialize empty free lists
        let free_lists = std::array::from_fn(|_| RwLock::new(VecDeque::new()));

        let statistics = if config.enable_statistics {
            Some(PoolStatistics::default())
        } else {
            None
        };

        let pool = Self {
            free_lists,
            config,
            statistics,
        };

        // Pre-allocate if configured
        if pool.config.initial_capacity_per_size > 0 {
            log::info!(
                "Pre-allocating {} buffers per size class",
                pool.config.initial_capacity_per_size
            );
            for class in 0..NUM_SIZE_CLASSES {
                let size = size_for_class(class);
                let layout =
                    Layout::from_size_align(size, pool.config.alignment).expect("Invalid layout");
                let mut list = pool.free_lists[class].write().expect("Lock poisoned");
                for _ in 0..pool.config.initial_capacity_per_size {
                    // Allocate and add to free list
                    let ptr = unsafe { alloc(layout) };
                    if let Some(ptr) = NonNull::new(ptr) {
                        list.push_back(RawBuffer { ptr, layout });
                    }
                }
            }
        }

        pool
    }

    fn allocate(self: &Arc<Self>, size: usize) -> PooledBuffer {
        let size_class = size_class_for_bytes(size);
        let capacity = size_class.map(size_for_class).unwrap_or(size);
        let layout = Layout::from_size_align(capacity, self.config.alignment)
            .expect("Invalid allocation layout");

        // Try to get from pool first
        let (ptr, hit) = if let Some(class) = size_class {
            let mut list = self.free_lists[class].write().expect("Lock poisoned");
            if let Some(buffer) = list.pop_front() {
                // Recycle existing buffer
                let ptr = buffer.ptr;
                std::mem::forget(buffer); // Don't drop, we're reusing the memory
                (ptr, true)
            } else {
                // Allocate new
                let raw_ptr = unsafe { alloc(layout) };
                let ptr = NonNull::new(raw_ptr).expect("Memory allocation failed");
                (ptr, false)
            }
        } else {
            // Direct allocation for oversized requests
            let raw_ptr = unsafe { alloc(layout) };
            let ptr = NonNull::new(raw_ptr).expect("Memory allocation failed");
            (ptr, false)
        };

        // Record statistics
        if let Some(ref stats) = self.statistics {
            stats.record_allocation(hit, capacity);
        }

        PooledBuffer {
            ptr,
            size,
            capacity,
            layout,
            pool: Arc::clone(self),
            size_class,
        }
    }

    fn return_buffer(&self, ptr: NonNull<u8>, layout: Layout, size_class: Option<usize>) {
        let capacity = layout.size();

        // Check if we should cache or deallocate
        let current_pool_size = self
            .statistics
            .as_ref()
            .map(|s| s.current_pool_bytes.load(Ordering::Relaxed))
            .unwrap_or(0);

        let should_cache = size_class.is_some()
            && (current_pool_size + capacity) <= self.config.max_pool_size_bytes;

        if should_cache {
            if let Some(class) = size_class {
                // Return to pool for reuse
                let mut list = self.free_lists[class].write().expect("Lock poisoned");
                list.push_back(RawBuffer { ptr, layout });

                // Record statistics
                if let Some(ref stats) = self.statistics {
                    stats.record_deallocation(capacity);
                }
            }
        } else {
            // Pool is full or oversized allocation - deallocate directly
            unsafe {
                dealloc(ptr.as_ptr(), layout);
            }
        }
    }
}

/// Thread-safe memory pool for tensor allocations.
///
/// The pool maintains separate free lists for different size classes,
/// allowing efficient reuse of similarly-sized allocations.
#[derive(Clone)]
pub struct TensorPool {
    inner: Arc<TensorPoolInner>,
}

impl TensorPool {
    /// Create a new memory pool with the given configuration.
    pub fn new(config: PoolConfig) -> Self {
        log::info!(
            "Creating TensorPool with max size {} bytes",
            config.max_pool_size_bytes
        );
        Self {
            inner: Arc::new(TensorPoolInner::new(config)),
        }
    }

    /// Create a pool with default settings.
    pub fn with_defaults() -> Self {
        Self::new(PoolConfig::default())
    }

    /// Allocate a buffer of at least the specified size.
    ///
    /// The actual capacity may be larger due to size class rounding.
    /// The buffer contents are uninitialized.
    pub fn allocate(&self, size: usize) -> PooledBuffer {
        self.inner.allocate(size)
    }

    /// Allocate a zero-initialized buffer.
    pub fn allocate_zeroed(&self, size: usize) -> PooledBuffer {
        let mut buffer = self.allocate(size);
        buffer.zero();
        buffer
    }

    /// Allocate a buffer sized for N elements of type T.
    pub fn allocate_typed<T>(&self, count: usize) -> PooledBuffer {
        let size = count * size_of::<T>();
        self.allocate(size)
    }

    /// Allocate a zero-initialized buffer sized for N elements of type T.
    pub fn allocate_typed_zeroed<T>(&self, count: usize) -> PooledBuffer {
        let size = count * size_of::<T>();
        self.allocate_zeroed(size)
    }

    /// Get pool statistics if enabled.
    pub fn statistics(&self) -> Option<&PoolStatistics> {
        self.inner.statistics.as_ref()
    }

    /// Clear all cached buffers to reclaim memory.
    ///
    /// Outstanding `PooledBuffer` handles remain valid.
    pub fn clear(&self) {
        log::info!("Clearing TensorPool caches");
        for class in 0..NUM_SIZE_CLASSES {
            let mut list = self.inner.free_lists[class].write().expect("Lock poisoned");
            list.clear(); // RawBuffer::drop deallocates each buffer
        }
        // Reset statistics pool size counter
        if let Some(ref stats) = self.inner.statistics {
            stats.current_pool_bytes.store(0, Ordering::Relaxed);
        }
    }

    /// Trim the pool to a target size by removing oldest cached buffers.
    pub fn trim_to(&self, target_bytes: usize) {
        let current = self
            .inner
            .statistics
            .as_ref()
            .map(|s| s.current_pool_bytes.load(Ordering::Relaxed))
            .unwrap_or(0);

        if current <= target_bytes {
            return;
        }

        log::info!(
            "Trimming TensorPool from {} to {} bytes",
            current,
            target_bytes
        );

        let mut to_remove = current - target_bytes;

        // Remove from largest size classes first (most memory impact)
        for class in (0..NUM_SIZE_CLASSES).rev() {
            if to_remove == 0 {
                break;
            }
            let size = size_for_class(class);
            let mut list = self.inner.free_lists[class].write().expect("Lock poisoned");
            while to_remove > 0 && !list.is_empty() {
                let _ = list.pop_front(); // Drops RawBuffer, freeing memory
                to_remove = to_remove.saturating_sub(size);
                if let Some(ref stats) = self.inner.statistics {
                    stats.current_pool_bytes.fetch_sub(size, Ordering::Relaxed);
                }
            }
        }
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// Global pool for convenient access
static GLOBAL_POOL: std::sync::OnceLock<TensorPool> = std::sync::OnceLock::new();

/// Get a reference to the global tensor pool.
///
/// The global pool is lazily initialized with default settings on first access.
pub fn global_pool() -> &'static TensorPool {
    GLOBAL_POOL.get_or_init(TensorPool::with_defaults)
}

/// Initialize the global pool with custom configuration.
///
/// This must be called before any calls to `global_pool()`.
/// Returns `Err` if the pool was already initialized.
pub fn init_global_pool(config: PoolConfig) -> Result<(), &'static str> {
    GLOBAL_POOL
        .set(TensorPool::new(config))
        .map_err(|_| "Global pool already initialized")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_calculation() {
        // Empty and tiny allocations go to class 0
        assert_eq!(size_class_for_bytes(0), Some(0));
        assert_eq!(size_class_for_bytes(1), Some(0));
        assert_eq!(size_class_for_bytes(1023), Some(0));
        assert_eq!(size_class_for_bytes(1024), Some(0));

        // 1KB-4KB -> class 1
        assert_eq!(size_class_for_bytes(1025), Some(1));
        assert_eq!(size_class_for_bytes(2048), Some(1));
        assert_eq!(size_class_for_bytes(4096), Some(2));

        // Large allocations
        assert_eq!(size_class_for_bytes(16 * 1024 * 1024), Some(14)); // 16MB exactly
        assert_eq!(size_class_for_bytes(16 * 1024 * 1024 + 1), None); // > 16MB
    }

    #[test]
    fn test_basic_allocation() {
        let pool = TensorPool::new(PoolConfig::default().with_statistics());

        // Allocate and use
        let mut buf = pool.allocate(1000);
        assert!(buf.capacity() >= 1000);
        buf.zero();

        // Drop should return to pool
        drop(buf);

        // Next allocation should hit pool
        let buf2 = pool.allocate(500);
        assert!(buf2.capacity() >= 500);

        let stats = pool.statistics().expect("Statistics enabled");
        assert!(stats.total_allocations.load(Ordering::Relaxed) >= 2);
    }

    #[test]
    fn test_pool_hit() {
        let pool = TensorPool::new(PoolConfig::default().with_statistics());

        // First allocation - miss
        let buf1 = pool.allocate(1024);
        drop(buf1);

        // Second allocation - should hit
        let _buf2 = pool.allocate(1024);

        let stats = pool.statistics().expect("Statistics enabled");
        assert_eq!(stats.pool_hits.load(Ordering::Relaxed), 1);
        assert_eq!(stats.pool_misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_typed_allocation() {
        let pool = TensorPool::with_defaults();

        let buf = pool.allocate_typed::<f32>(256);
        assert!(buf.capacity() >= 256 * 4);

        let buf_zeroed = pool.allocate_typed_zeroed::<f32>(256);
        unsafe {
            let slice = buf_zeroed.as_typed_slice::<f32>();
            assert!(slice.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn test_clear_and_trim() {
        let pool = TensorPool::new(PoolConfig::default().with_statistics());

        // Allocate and return several buffers
        for _ in 0..10 {
            let buf = pool.allocate(4096);
            drop(buf);
        }

        let stats = pool.statistics().expect("Statistics enabled");
        let before = stats.current_pool_bytes.load(Ordering::Relaxed);
        assert!(before > 0);

        // Clear should free all cached buffers
        pool.clear();

        let after = stats.current_pool_bytes.load(Ordering::Relaxed);
        assert_eq!(after, 0);
    }

    #[test]
    fn test_oversized_allocation() {
        let pool = TensorPool::with_defaults();

        // Allocation larger than MAX_POOLED_SIZE
        let buf = pool.allocate(32 * 1024 * 1024); // 32 MB
        assert!(buf.capacity() >= 32 * 1024 * 1024);
        // This won't be cached, goes directly to system allocator
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let pool = Arc::new(TensorPool::new(PoolConfig::default().with_statistics()));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let pool_clone = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let mut buf = pool_clone.allocate(1024 * 4);
                    buf.zero();
                    // Small work
                    thread::yield_now();
                    drop(buf);
                }
            }));
        }

        for h in handles {
            h.join().expect("Thread panicked");
        }

        let stats = pool.statistics().expect("Statistics enabled");
        assert_eq!(stats.total_allocations.load(Ordering::Relaxed), 800);
    }
}
