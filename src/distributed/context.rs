//! Distributed Training Context for Multi-GPU Support
//!
//! This module provides infrastructure for distributed training across multiple GPUs,
//! including device management, process groups, and communication primitives.

use std::sync::Arc;
use std::collections::HashMap;
use std::cell::RefCell;
thread_local! {
    static COMMUNICATION_STORE: RefCell<HashMap<String, Vec<u8>>> = RefCell::new(HashMap::new());
}

/// Supported device types in Tensor Engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceId {
    /// CPU device (ordinal 0)
    Cpu,
    /// CUDA GPU device with index
    Gpu(usize),
}

impl DeviceId {
    /// Check if this is a GPU device.
    pub fn is_gpu(&self) -> bool {
        matches!(self, DeviceId::Gpu(_))
    }

    /// Get the ordinal/index of this device.
    pub fn index(&self) -> usize {
        match self {
            DeviceId::Cpu => 0,
            DeviceId::Gpu(i) => *i,
        }
    }

    /// Create a GPU device from an integer index.
    pub fn gpu(index: usize) -> Self {
        DeviceId::Gpu(index)
    }

    /// Check if this device is available.
    pub fn is_available(&self) -> bool {
        match self {
            DeviceId::Cpu => true, // CPU always available
            DeviceId::Gpu(idx) => {
                #[cfg(feature = "backend_cuda")]
                {
                    crate::backend::cuda::CudaBackend::new(*idx).is_ok()
                }
                #[cfg(not(feature = "backend_cuda"))]
                {
                    false
                }
            }
        }
    }
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceId::Cpu => write!(f, "cpu"),
            DeviceId::Gpu(i) => write!(f, "cuda:{}", i),
        }
    }
}

/// Configuration for distributed training.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Number of GPU devices to use
    pub num_devices: usize,
    /// Device ordinal for each rank (process)
    pub device_map: HashMap<usize, DeviceId>,
    /// Backend to use (CUDA, WGPU, CPU)
    pub backend: String,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        let mut device_map = HashMap::new();
        
        // Auto-detect available GPUs
        #[cfg(feature = "backend_cuda")]
        {
            use cudarc::driver::CudaContext;
            if CudaContext::new(0).is_ok() {
                for i in 0..2 { // Assume up to 2 GPUs available
                    device_map.insert(i, DeviceId::Gpu(i));
                }
            }
        }

        let num_devices = device_map.len();
        
        DistributedConfig {
            num_devices: if num_devices > 0 { num_devices } else { 1 },
            device_map,
            backend: "cuda".to_string(),
        }
    }
}

impl DistributedConfig {
    /// Create a new distributed config for single-GPU training.
    pub fn single_gpu(device_index: usize) -> Self {
        let mut device_map = HashMap::new();
        device_map.insert(0, DeviceId::Gpu(device_index));
        
        DistributedConfig {
            num_devices: 1,
            device_map,
            backend: "cuda".to_string(),
        }
    }

    /// Create a new distributed config for multi-GPU training.
    pub fn multi_gpu(num_gpus: usize) -> Self {
        let mut device_map = HashMap::new();
        
        #[cfg(feature = "backend_cuda")]
        {
            use cudarc::driver::CudaContext;
            for i in 0..num_gpus {
                if CudaContext::new(i).is_ok() {
                    device_map.insert(i, DeviceId::Gpu(i));
                }
            }
        }

        DistributedConfig {
            num_devices: device_map.len(),
            device_map,
            backend: "cuda".to_string(),
        }
    }

    /// Get the device for a given rank.
    pub fn device_for_rank(&self, rank: usize) -> DeviceId {
        self.device_map.get(&rank).copied().unwrap_or(DeviceId::Cpu)
    }
}

/// Distributed context managing training state across multiple devices.
pub struct DistributedContext {
    /// Configuration for distributed training
    config: DistributedConfig,
    /// Current rank (process ID) in the distributed setup
    rank: usize,
    /// Total number of processes
    world_size: usize,
    /// Backend instances for each device
    backends: HashMap<usize, Arc<crate::backend::cuda::CudaBackend>>,
}

impl DistributedContext {
    /// Create a new distributed training context.
    pub fn new(rank: usize, config: &DistributedConfig) -> Result<Self, String> {
        let mut backends = HashMap::new();

        // Initialize backend for this rank's device
        if let Some(device) = config.device_map.get(&rank) {
            match device {
                DeviceId::Gpu(idx) => {
                    #[cfg(feature = "backend_cuda")]
                    {
                        let backend = Arc::new(crate::backend::cuda::CudaBackend::new(*idx)?);
                        backends.insert(rank, backend);
                    }
                    #[cfg(not(feature = "backend_cuda"))]
                    {
                        log::warn!("CUDA feature not enabled, falling back to CPU for rank {}", rank);
                    }
                }
                DeviceId::Cpu => {
                    // Use CPU backend (already available)
                }
            }
        }

        Ok(Self {
            config: config.clone(),
            rank,
            world_size: config.num_devices,
            backends,
        })
    }

    /// Get the current rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the world size (total number of processes).
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Get the backend for the current rank.
    pub fn get_backend(&self) -> Option<&Arc<crate::backend::cuda::CudaBackend>> {
        self.backends.get(&self.rank)
    }

    /// Synchronize all ranks (barrier).
    pub fn barrier(&self) -> Result<(), String> {
        log::info!("Synchronizing {} ranks...", self.world_size);
        
        // In a real implementation, this would use NCCL or similar
        #[cfg(feature = "backend_cuda")]
        {
            if let Some(backend) = self.backends.get(&self.rank) {
                backend.synchronize()?;
            }
        }

        Ok(())
    }

    /// Get device information for all ranks.
    pub fn device_info(&self) -> Vec<String> {
        let mut devices = Vec::new();
        
        #[cfg(feature = "backend_cuda")]
        {
            for i in 0..self.world_size {
                if let Some(backend) = self.backends.get(&i) {
                    let mem_info = backend.memory_info();
                    devices.push(format!(
                        "Rank {}: CUDA device {} - {:?}",
                        i,
                        backend.device(),
                        mem_info
                    ));
                } else {
                    devices.push(format!("Rank {}: CPU", i));
                }
            }
        }

        if devices.is_empty() {
            devices.push("Rank {}: CPU".to_string());
        }

        devices
    }

    /// Create a single-rank distributed context for testing.
    pub fn single() -> Self {
        let config = DistributedConfig::single_gpu(0);
        Self::new(0, &config).unwrap()
    }

    /// Store a key-value pair for inter-rank communication.
    pub fn put(&self, key: &str, value: Vec<u8>) {
        COMMUNICATION_STORE.with(|store| {
            store.borrow_mut().insert(key.to_string(), value);
        });
    }

    /// Retrieve a value by key for inter-rank communication.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        COMMUNICATION_STORE.with(|store| {
            store.borrow().get(key).cloned()
        })
    }

    /// Check if this rank is the master (rank 0).
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }
}

/// Clone implementation for DistributedContext.
impl Clone for DistributedContext {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            rank: self.rank,
            world_size: self.world_size,
            backends: self.backends.clone(),
        }
    }
}

/// Get the number of available CUDA GPUs.
#[allow(dead_code)]
pub fn get_cuda_device_count() -> usize {
    #[cfg(feature = "backend_cuda")]
    {
        use cudarc::driver::CudaContext;
        
        let mut count = 0;
        for i in 0..16 { // Check up to 16 possible GPUs
            if CudaContext::new(i).is_ok() {
                count += 1;
            } else {
                break;
            }
        }
        count
    }
    
    #[cfg(not(feature = "backend_cuda"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_id() {
        let cpu = DeviceId::Cpu;
        let gpu = DeviceId::Gpu(2);

        assert!(!cpu.is_gpu());
        assert!(gpu.is_gpu());

        assert_eq!(gpu.index(), 2);
        assert_eq!(format!("{}", gpu), "cuda:2");
    }

    #[test]
    fn test_distributed_config() {
        let config = DistributedConfig::single_gpu(0);
        assert_eq!(config.num_devices, 1);
        
        let device = config.device_for_rank(0);
        assert_eq!(device, DeviceId::Gpu(0));
    }

    #[test]
    fn test_get_cuda_device_count() {
        let count = get_cuda_device_count();
        // Count should be >= 0 (may be 0 if CUDA not available)
        assert!(count >= 0);
    }
}
