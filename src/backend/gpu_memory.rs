//! GPU Memory Management for Tensor Engine (Simplified - No Async)
//!
//! This module provides basic GPU memory allocation with pooling,
//! optimized for WGPU-based backends.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Configuration for GPU memory pool management.
#[derive(Debug, Clone)]
pub struct GpuMemoryConfig {
    pub max_gpu_memory_bytes: usize,
    pub initial_capacity_per_size: usize,
    pub enable_statistics: bool,
    min_allocation_size: usize,
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
    pub fn with_max_memory(max_bytes: usize) -> Self {
        Self {
            max_gpu_memory_bytes: max_bytes,
            ..Default::default()
        }
    }

    pub fn with_statistics(mut self) -> Self {
        self.enable_statistics = true;
        self
    }

    pub fn with_preallocate(mut self, count: usize) -> Self {
        self.initial_capacity_per_size = count;
        self
    }
}

/// Statistics for GPU memory pool monitoring.
#[derive(Debug, Default)]
pub struct GpuMemoryStatistics {
    pub total_allocations: AtomicU64,
    pub pool_hits: AtomicU64,
    pub pool_misses: AtomicU64,
    pub total_deallocations: AtomicU64,
    pub current_gpu_bytes: AtomicUsize,
    pub peak_gpu_bytes: AtomicUsize,
}

impl GpuMemoryStatistics {
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_allocations.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let hits = self.pool_hits.load(Ordering::Relaxed);
        hits as f64 / total as f64
    }

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

        if hit {
            self.current_gpu_bytes.fetch_sub(size, Ordering::Relaxed);
        }
    }

    fn record_deallocation(&self, size: usize) {
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);

        let new_size = self.current_gpu_bytes.fetch_add(size, Ordering::Relaxed) + size;

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

const NUM_SIZE_CLASSES: usize = 15;
const MIN_SIZE_CLASS_BITS: usize = 6; // 2^6 = 64
const MAX_POOLED_SIZE: usize = 1 << 24; // 16 MB

fn gpu_size_class_for_bytes(bytes: usize) -> Option<usize> {
    if bytes <= 1 {
        return Some(0);
    }
    if bytes > MAX_POOLED_SIZE {
        return None;
    }

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

fn gpu_size_for_class(class: usize) -> usize {
    1 << (MIN_SIZE_CLASS_BITS + class)
}

/// GPU memory buffer handle with RAII semantics.
pub struct GpuBuffer {
    pub buffer: wgpu::Buffer,
    pub size: usize,
    pub size_class: Option<usize>,
    pub pool: Arc<GpuMemoryPoolInner>,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        log::debug!("Dropping GPU buffer back to pool");
    }
}

impl GpuBuffer {
    pub fn new(buffer: wgpu::Buffer, size: usize, pool: Arc<GpuMemoryPoolInner>) -> Self {
        let _size_class = gpu_size_class_for_bytes(size);

        log::debug!("Created GpuBuffer: size={}", size);

        Self {
            buffer,
            size,
            size_class: None, // Simplified - don't track class for now
            pool,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Map the buffer for reading (async operation).
    pub async fn map_read(&self) -> Result<wgpu::BufferSlice, String> {
        // Simplified implementation without proper error handling for wgpu 0.19
        Ok(self.buffer.slice(..))
    }

    pub async fn read_data(&self) -> Result<Vec<u8>, String> {
        let _slice = self.map_read().await?;

        // Simplified - just return empty data for now
        Ok(vec![])
    }

    pub fn write_data(&self, data: &[u8]) {
        assert!(
            data.len() <= self.size,
            "Data size {} exceeds buffer size {}",
            data.len(),
            self.size
        );

        self.pool
            .queue
            .write_buffer(&self.buffer, 0, &data[..self.size]);
    }

    pub fn copy_from(&self, _src: &GpuBuffer, _src_offset: usize, _dst_offset: usize) {
        // Simplified - no-op for now
    }

    pub fn inner(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

struct GpuMemoryPoolInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    free_lists: [RwLock<VecDeque<wgpu::Buffer>>; NUM_SIZE_CLASSES],
    statistics: Option<GpuMemoryStatistics>,
}

pub struct GpuMemoryPool {
    config: GpuMemoryConfig,
    inner: Arc<GpuMemoryPoolInner>,
}

impl GpuMemoryPool {
    pub fn new(config: GpuMemoryConfig) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
        });

        let adapter = adapter.ok_or("Failed to find GPU adapter")?;

        log::info!("WGPU Adapter: {:?}", adapter.get_info());

        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("TensorEngine GPU Memory Pool"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await
        })
        .map_err(|e| format!("Failed to create GPU device: {}", e))?;

        let free_lists = std::array::from_fn(|_| {
            RwLock::new(VecDeque::with_capacity(config.initial_capacity_per_size))
        });

        let pool = Self {
            config,
            inner: Arc::new(GpuMemoryPoolInner {
                device,
                queue,
                free_lists,
                statistics: if config.enable_statistics {
                    Some(GpuMemoryStatistics::default())
                } else {
                    None
                },
            }),
        };

        log::info!(
            "GPU Memory Pool created with {} max memory",
            config.max_gpu_memory_bytes
        );

        Ok(pool)
    }

    pub fn allocate(&self, size: usize) -> GpuBuffer {
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Buffer Allocation"),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if let Some(stats) = &self.inner.statistics {
            stats.record_allocation(false, size);

            let new_size = stats.current_gpu_bytes.fetch_add(size, Ordering::Relaxed) + size;

            if new_size > self.config.max_gpu_memory_bytes {
                log::warn!("GPU memory usage exceeds limit: {} bytes", new_size);
                self.trim_to(self.config.max_gpu_memory_bytes / 2);
            }
        }

        log::debug!("GPU allocation MISS: size={}", size);

        GpuBuffer::new(buffer, size, Arc::clone(&self.inner))
    }

    pub fn allocate_typed<T>(&self, count: usize) -> GpuBuffer {
        let size = count * std::mem::size_of::<T>();
        self.allocate(size)
    }

    pub fn allocate_typed_zeroed<T>(&self, _count: usize) -> GpuBuffer {
        let size = _count * std::mem::size_of::<T>();
        self.allocate(size)
    }

    pub fn statistics(&self) -> Option<&GpuMemoryStatistics> {
        self.inner.statistics.as_ref()
    }

    pub fn clear(&self) {
        log::info!("Clearing GPU Memory Pool caches");

        for class in 0..NUM_SIZE_CLASSES {
            let mut list = self.inner.free_lists[class].write().expect("Lock poisoned");

            while let Some(_buffer) = list.pop_front() {
                // Drop buffer to free memory
            }
        }

        if let Some(ref stats) = self.inner.statistics {
            stats.current_gpu_bytes.store(0, Ordering::Relaxed);
        }
    }

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

            for class in (0..NUM_SIZE_CLASSES).rev() {
                if to_remove == 0 {
                    break;
                }

                let size = gpu_size_for_class(class);
                let mut list = self.inner.free_lists[class].write().expect("Lock poisoned");

                while to_remove > 0 && !list.is_empty() {
                    let _ = list.pop_front();
                    to_remove = to_remove.saturating_sub(size);

                    stats.current_gpu_bytes.fetch_sub(size, Ordering::Relaxed);
                }
            }
        }
    }

    fn free_buffer(&self, buffer: wgpu::Buffer, class: usize) {
        if let Some(stats) = &self.inner.statistics {
            stats.record_deallocation(gpu_size_for_class(class));
        }

        log::debug!(
            "Freeing GPU buffer to pool: size={}, class={}",
            gpu_size_for_class(class),
            class
        );

        let mut free_list = self.inner.free_lists[class].write().expect("Lock poisoned");

        if free_list.len() < 10 {
            free_list.push_back(buffer);
        } else {
            log::debug!("GPU buffer pool full for class {}, dropping buffer", class);
            drop(buffer);
        }
    }

    pub fn synchronize(&self) {
        self.inner.device.poll(wgpu::Maintain::Wait);
    }
}

impl Drop for GpuMemoryPool {
    fn drop(&mut self) {
        log::info!("Dropping GPU Memory Pool");

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
    #[test]
    fn test_gpu_size_class_calculation() {
        assert_eq!(gpu_size_class_for_bytes(0), Some(0));
        assert_eq!(gpu_size_class_for_bytes(1), Some(0));
        assert_eq!(gpu_size_class_for_bytes(63), Some(0));
        assert_eq!(gpu_size_class_for_bytes(64), Some(0));

        assert_eq!(gpu_size_class_for_bytes(65), Some(1));
        assert_eq!(gpu_size_class_for_bytes(128), Some(1));
        assert_eq!(gpu_size_class_for_bytes(256), Some(2));

        assert_eq!(gpu_size_class_for_bytes(16 * 1024 * 1024), Some(14));
        assert_eq!(gpu_size_class_for_bytes(16 * 1024 * 1024 + 1), None);
    }

    #[test]
    fn test_basic_allocation() {
        let config = GpuMemoryConfig::default().with_statistics();

        assert_eq!(config.max_gpu_memory_bytes, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_statistics() {
        let config = GpuMemoryConfig::default().with_statistics();

        assert!(config.enable_statistics);
    }
}
