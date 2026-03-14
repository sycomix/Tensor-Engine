//! GPU Memory Management for Tensor Engine
//!
//! This module provides efficient GPU memory allocation with pooling,
//! caching, and management specifically optimized for WGPU-based backends.

use ndarray::{ArrayD, IxDyn};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Configuration for GPU memory pool management.
#[derive(Debug, Clone)]
pub struct GpuMemoryConfig {
    /// Maximum total GPU memory the pool can manage (in bytes).
    pub max_gpu_memory_bytes: usize,

    /// Number of buffers to pre-allocate per size class during initialization.
    pub initial_capacity_per_size: usize,

    /// Enable statistics tracking for monitoring GPU memory usage.
    pub enable_statistics: bool,

    /// Minimum allocation size (in bytes) for GPU allocations.
    min_allocation_size: usize,

    /// Maximum allocation size before falling back to direct allocation.
    max_pooled_size: usize,
}

impl Default for GpuMemoryConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB default
            initial_capacity_per_size: 4,
            enable_statistics: false,
            min_allocation_size: 64,
            max_pooled_size: 16 * 1024 * 1024, // 16 MB
        }
    }
}

impl GpuMemoryConfig {
    /// Create a configuration with a specific maximum GPU memory size.
    pub fn with_max_memory(max_bytes: usize) -> Self {
        Self {
            max_gpu_memory_bytes: max_bytes,
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

/// Statistics for GPU memory pool monitoring.
#[derive(Debug, Default)]
pub struct GpuMemoryStatistics {
    /// Total number of allocation requests.
    pub total_allocations: AtomicU64,
    /// Allocations served from the pool (cache hits).
    pub pool_hits: AtomicU64,
    /// Allocations that required new GPU memory (cache misses).
    pub pool_misses: AtomicU64,
    /// Total number of deallocations (returns to pool).
    pub total_deallocations: AtomicU64,
    /// Current GPU memory held by the pool (in bytes).
    pub current_gpu_bytes: AtomicUsize,
    /// Peak GPU memory used by the pool (in bytes).
    pub peak_gpu_bytes: AtomicUsize,
}

impl GpuMemoryStatistics {
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
            "GPU Memory Statistics:\n  Total allocations: {}\n  Pool hits: {} ({:.1}%)\n  Pool misses: {}\n  Current GPU memory: {:.2} MB\n  Peak GPU memory: {:.2} MB",
            self.total_allocations.load(Ordering::Relaxed),
            self.pool_hits.load(Ordering::Relaxed),
            self.hit_rate() * 100.0,
            self.pool_misses.load(Ordering::Relaxed),
            self.current_gpu_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
            self.peak_gpu_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
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
            self.current_gpu_bytes.fetch_sub(size, Ordering::Relaxed);
        }
    }

    fn record_deallocation(&self, size: usize) {
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        
        let new_size = self.current_gpu_bytes.fetch_add(size, Ordering::Relaxed) + size;
        
        // Update peak if needed
        let mut peak = self.peak_gpu_bytes.load(Ordering::Relaxed);
        while new_size > peak {
            match self.peak_gpu_bytes.compare_exchange_weak(
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

/// Size class index for GPU memory bucketing.
const NUM_SIZE_CLASSES: usize = 15;

/// Minimum size class (64 bytes).
const MIN_SIZE_CLASS_BITS: usize = 6; // 2^6 = 64

/// Maximum size for pooled allocations (16 MB = 2^24 bytes).
const MAX_POOLED_SIZE: usize = 1 << 24; // 16 MB

/// Get the size class index for a given byte count.
fn gpu_size_class_for_bytes(bytes: usize) -> Option<usize> {
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
fn gpu_size_for_class(class: usize) -> usize {
    1 << (MIN_SIZE_CLASS_BITS + class)
}

/// GPU memory buffer handle with RAII semantics.
pub struct GpuBuffer {
    /// WGPU buffer handle
    pub buffer: wgpu::Buffer,
    
    /// Buffer size in bytes
    pub size: usize,
    
    /// Size class for pooling purposes
    pub size_class: Option<usize>,
    
    /// Reference to the memory pool for recycling
    pub pool: Arc<GpuMemoryPoolInner>,
}

// Safety: GpuBuffer owns its buffer handle and is thread-safe via Arc
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // Return buffer to pool for reuse instead of immediate deallocation
        if let Some(class) = self.size_class {
            self.pool.free_buffer(self.buffer.clone(), class);
        } else {
            // For oversized allocations, we might want to actually drop them
            log::warn!("Dropping oversized GPU buffer without pooling");
        }
    }
}

impl GpuBuffer {
    /// Create a new GPU buffer handle.
    pub fn new(
        buffer: wgpu::Buffer,
        size: usize,
        pool: Arc<GpuMemoryPoolInner>,
    ) -> Self {
        let size_class = gpu_size_class_for_bytes(size);
        
        log::debug!("Created GpuBuffer: size={}, class={}", size, size_class);
        
        Self {
            buffer,
            size,
            size_class,
            pool,
        }
    }

    /// Get the buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Map the buffer for reading (async operation).
    pub async fn map_read(&self) -> Result<wgpu::BufferSlice, wgpu::MapError> {
        let slice = self.buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        
        slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        // Poll for completion (blocking - not ideal but works for demo)
        self.pool.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(_)) = receiver.try_recv() {
            Ok(slice)
        } else {
            Err(wgpu::MapError::Failed)
        }
    }

    /// Write data to the buffer.
    pub fn write(&self, data: &[u8]) {
        self.pool.queue.write_buffer(
            &self.buffer,
            0,
            data,
        );
    }

    /// Read data from the buffer synchronously (blocking).
    pub fn read_sync(&self) -> Vec<u8> {
        let mut result = vec![0u8; self.size];
        
        let slice = self.buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        
        slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        // Poll for completion
        self.pool.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(_)) = receiver.try_recv() {
            slice.slice(..self.size as u64).get_mapped_range().clone_into(&mut result[..]);
        } else {
            log::error!("Failed to read from GPU buffer");
        }

        self.buffer.unmap();
        
        result
    }

    /// Create a zero-initialized buffer.
    pub fn zeroed(pool: &GpuMemoryPool, size: usize) -> Self {
        let layout = wgpu::util::BufferInitDescriptor {
            label: Some("Zeroed GPU Buffer"),
            contents: &[0u8; 0], // Will be resized by the allocator
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        };

        let buffer = pool.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Zeroed GPU Buffer"),
            size: size as u64,
            usage: layout.usage,
            mapped_at_creation: false,
        });

        Self::new(buffer, size, Arc::clone(&pool.inner))
    }
}

/// Inner state for the GPU memory pool.
struct GpuMemoryPoolInner {
    /// WGPU device reference
    pub device: wgpu::Device,
    
    /// WGPU queue for command submission
    pub queue: wgpu::Queue,
    
    /// Free buffers organized by size class
    free_lists: [RwLock<VecDeque<wgpu::Buffer>>; NUM_SIZE_CLASSES],
    
    /// Statistics tracking
    statistics: Option<GpuMemoryStatistics>,
    
    /// Maximum total memory the pool can hold (in bytes)
    max_pool_size_bytes: usize,
}

/// GPU Memory Pool for efficient buffer allocation and reuse.
pub struct GpuMemoryPool {
    inner: Arc<GpuMemoryPoolInner>,
    config: GpuMemoryConfig,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool with default configuration.
    pub fn new(config: GpuMemoryConfig) -> Result<Self, String> {
        let device = wgpu::Device; // Placeholder - would need actual device from WGPU backend
        let queue = wgpu::Queue;   // Placeholder
        
        log::info!("Creating GPU Memory Pool with max {} bytes", config.max_gpu_memory_bytes);

        let inner = Arc::new(GpuMemoryPoolInner {
            device,
            queue,
            free_lists: std::array::from_fn(|_| RwLock::new(VecDeque::new())),
            statistics: if config.enable_statistics {
                Some(GpuMemoryStatistics::default())
            } else {
                None
            },
            max_pool_size_bytes: config.max_gpu_memory_bytes,
        });

        Ok(Self { inner, config })
    }

    /// Allocate a buffer of the specified size.
    pub fn allocate(&self, size: usize) -> GpuBuffer {
        let stats = self.inner.statistics.as_ref();
        
        // Check if we can serve from pool (cache hit)
        if let Some(class) = gpu_size_class_for_bytes(size) {
            let mut free_list = self.inner.free_lists[class].write().expect("Lock poisoned");
            
            if !free_list.is_empty() && stats.map_or(true, |s| {
                s.current_gpu_bytes.load(Ordering::Relaxed) + size <= self.config.max_gpu_memory_bytes
            }) {
                let buffer = free_list.pop_front().expect("List not empty after check");
                
                if let Some(stats) = &self.inner.statistics {
                    stats.record_allocation(true, size);
                }

                log::debug!("GPU allocation HIT: size={}, class={}", size, class);
                return GpuBuffer::new(buffer, size, Arc::clone(&self.inner));
            }
        }

        // Cache miss - allocate new buffer
        let layout = wgpu::util::BufferInitDescriptor {
            label: Some("GPU Buffer Allocation"),
            contents: &[0u8; 0],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        };

        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Buffer Allocation"),
            size: size as u64,
            usage: layout.usage,
            mapped_at_creation: false,
        });

        if let Some(stats) = &self.inner.statistics {
            stats.record_allocation(false, size);
            
            // Update current pool size
            let new_size = stats.current_gpu_bytes.fetch_add(size, Ordering::Relaxed) + size;
            
            // Check if we exceeded max memory and need to trim
            if new_size > self.config.max_gpu_memory_bytes {
                log::warn!("GPU memory usage exceeds limit: {} bytes", new_size);
                self.trim_to(self.config.max_gpu_memory_bytes / 2); // Trim to half capacity
            }
        }

        log::debug!("GPU allocation MISS: size={}, class={:?}", size, gpu_size_class_for_bytes(size));
        
        GpuBuffer::new(buffer, size, Arc::clone(&self.inner))
    }

    /// Allocate a buffer sized for N elements of type T.
    pub fn allocate_typed<T>(&self, count: usize) -> GpuBuffer {
        let size = count * std::mem::size_of::<T>();
        self.allocate(size)
    }

    /// Allocate a zero-initialized buffer sized for N elements of type T.
    pub fn allocate_typed_zeroed<T>(&self, count: usize) -> GpuBuffer {
        let size = count * std::mem::size_of::<T>();
        
        let layout = wgpu::util::BufferInitDescriptor {
            label: Some("Zeroed GPU Buffer Typed"),
            contents: bytemuck::cast_slice(&vec![0.0f32; count]), // Zero initialization
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        };

        let buffer = self.inner.device.create_buffer_init(&layout);
        
        GpuBuffer::new(buffer, size, Arc::clone(&self.inner))
    }

    /// Get pool statistics if enabled.
    pub fn statistics(&self) -> Option<&GpuMemoryStatistics> {
        self.inner.statistics.as_ref()
    }

    /// Clear all cached buffers to reclaim memory.
    pub fn clear(&self) {
        log::info!("Clearing GPU Memory Pool caches");
        
        for class in 0..NUM_SIZE_CLASSES {
            let mut list = self.inner.free_lists[class].write().expect("Lock poisoned");
            
            // Drop all buffers to free memory
            while let Some(buffer) = list.pop_front() {
                drop(buffer);
            }
        }

        if let Some(ref stats) = self.inner.statistics {
            stats.current_gpu_bytes.store(0, Ordering::Relaxed);
        }
    }

    /// Trim the pool to a target size by removing oldest cached buffers.
    pub fn trim_to(&self, target_bytes: usize) {
        if let Some(stats) = &self.inner.statistics {
            let current = stats.current_gpu_bytes.load(Ordering::Relaxed);
            
            if current <= target_bytes {
                return;
            }

            log::info!(
                "Trimming GPU Memory Pool from {:.2} MB to {:.2} MB",
                current as f64 / (1024.0 * 1024.0),
                target_bytes as f64 / (1024.0 * 1024.0)
            );

            let mut to_remove = current - target_bytes;

            // Remove from largest size classes first (most memory impact)
            for class in (0..NUM_SIZE_CLASSES).rev() {
                if to_remove == 0 {
                    break;
                }
                
                let size = gpu_size_for_class(class);
                let mut list = self.inner.free_lists[class].write().expect("Lock poisoned");
                
                while to_remove > 0 && !list.is_empty() {
                    let _ = list.pop_front(); // Drops buffer, freeing memory
                    to_remove = to_remove.saturating_sub(size);
                    
                    stats.current_gpu_bytes.fetch_sub(size, Ordering::Relaxed);
                }
            }
        }
    }

    /// Free a buffer back to the pool.
    fn free_buffer(&self, buffer: wgpu::Buffer, class: usize) {
        if let Some(stats) = &self.inner.statistics {
            stats.record_deallocation(gpu_size_for_class(class));
        }

        log::debug!("Freeing GPU buffer to pool: size={}, class={}", gpu_size_for_class(class), class);

        // Check if we have room in the pool before caching
        let mut free_list = self.inner.free_lists[class].write().expect("Lock poisoned");
        
        // Limit cache size per class (e.g., max 10 buffers per class)
        if free_list.len() < 10 {
            free_list.push_back(buffer);
        } else {
            log::debug!("GPU buffer pool full for class {}, dropping buffer", class);
            drop(buffer); // Actually deallocate instead of caching
        }
    }

    /// Synchronize GPU operations.
    pub fn synchronize(&self) {
        self.inner.device.poll(wgpu::Maintain::Wait);
    }
}

impl Drop for GpuMemoryPool {
    fn drop(&mut self) {
        log::info!("Dropping GPU Memory Pool");
        
        // Clear all cached buffers to ensure proper cleanup
        self.clear();
        
        if let Some(stats) = &self.inner.statistics {
            log::info!("{}", stats.summary());
        }
    }
}

impl Default for GpuMemoryPool {
    fn default() -> Self {
        Self::new(GpuMemoryConfig::default()).expect("Failed to create GPU memory pool")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_size_class_calculation() {
        assert_eq!(gpu_size_class_for_bytes(0), Some(0));
        assert_eq!(gpu_size_class_for_bytes(1), Some(0));
        assert_eq!(gpu_size_class_for_bytes(63), Some(0));
        assert_eq!(gpu_size_class_for_bytes(64), Some(0));

        assert_eq!(gpu_size_class_for_bytes(65), Some(1));
        assert_eq!(gpu_size_class_for_bytes(128), Some(1));
        assert_eq!(gpu_size_class_for_bytes(256), Some(2));

        assert_eq!(gpu_size_class_for_bytes(16 * 1024 * 1024), Some(14)); // 16MB exactly
        assert_eq!(gpu_size_class_for_bytes(16 * 1024 * 1024 + 1), None); // > 16MB
    }

    #[test]
    fn test_basic_allocation() {
        let config = GpuMemoryConfig::default().with_statistics();
        
        // Note: This test would need a real WGPU device to work properly
        // For now, we just verify the configuration is valid
        assert_eq!(config.max_gpu_memory_bytes, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_statistics() {
        let config = GpuMemoryConfig::default().with_statistics();
        
        // Verify statistics are enabled
        assert!(config.enable_statistics);
    }
}
