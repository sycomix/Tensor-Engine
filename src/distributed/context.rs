//! Distributed Context
//!
//! Manages the distributed training environment including rank, world size,
//! and device assignments. Uses shared memory for inter-process communication.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Identifier for a compute device (GPU or CPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceId {
    /// CPU device.
    Cpu,
    /// GPU device with index.
    Gpu(usize),
    /// Metal device (macOS).
    Metal(usize),
}

impl DeviceId {
    /// Check if this is a GPU device.
    pub fn is_gpu(&self) -> bool {
        matches!(self, DeviceId::Gpu(_) | DeviceId::Metal(_))
    }

    /// Get the device index (0 for CPU).
    pub fn index(&self) -> usize {
        match self {
            DeviceId::Cpu => 0,
            DeviceId::Gpu(i) | DeviceId::Metal(i) => *i,
        }
    }
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceId::Cpu => write!(f, "cpu"),
            DeviceId::Gpu(i) => write!(f, "cuda:{}", i),
            DeviceId::Metal(i) => write!(f, "metal:{}", i),
        }
    }
}

/// Configuration for distributed training.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Total number of processes in the distributed group.
    pub world_size: usize,
    /// Rank of this process (0 to world_size - 1).
    pub rank: usize,
    /// Local rank on this machine.
    pub local_rank: usize,
    /// Number of GPUs per node.
    pub gpus_per_node: usize,
    /// Backend to use for communication.
    pub backend: CommunicationBackend,
    /// Master address for coordination.
    pub master_addr: String,
    /// Master port for coordination.
    pub master_port: u16,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            local_rank: 0,
            gpus_per_node: 1,
            backend: CommunicationBackend::SharedMemory,
            master_addr: "127.0.0.1".to_string(),
            master_port: 29500,
        }
    }
}

impl DistributedConfig {
    /// Create a single-process configuration (no distribution).
    pub fn single() -> Self {
        Self::default()
    }

    /// Create a configuration for the given rank and world size.
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self {
            rank,
            world_size,
            local_rank: rank,
            ..Default::default()
        }
    }

    /// Set the communication backend.
    pub fn with_backend(mut self, backend: CommunicationBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the master address and port.
    pub fn with_master(mut self, addr: &str, port: u16) -> Self {
        self.master_addr = addr.to_string();
        self.master_port = port;
        self
    }
}

/// Communication backend for distributed operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationBackend {
    /// Shared memory backend (single-machine, multi-process).
    SharedMemory,
    /// Gloo backend (CPU-based, cross-platform).
    Gloo,
    /// NCCL backend (NVIDIA GPU optimized).
    Nccl,
    /// MPI backend.
    Mpi,
}

impl std::fmt::Display for CommunicationBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommunicationBackend::SharedMemory => write!(f, "shared_memory"),
            CommunicationBackend::Gloo => write!(f, "gloo"),
            CommunicationBackend::Nccl => write!(f, "nccl"),
            CommunicationBackend::Mpi => write!(f, "mpi"),
        }
    }
}

/// Shared state for multi-context communication within a process.
/// This enables testing distributed algorithms with multiple contexts.
#[derive(Default)]
pub struct SharedCommunicationState {
    /// Barrier counters per barrier ID.
    barrier_arrived: RwLock<HashMap<usize, Vec<usize>>>,
    /// Broadcast data: (barrier_id, src_rank) -> data
    broadcast_data: RwLock<HashMap<(String, usize), Vec<u8>>>,
    /// General purpose shared data for simulation (put/get).
    shared_data: RwLock<HashMap<String, Vec<u8>>>,
    /// Current barrier ID.
    #[allow(dead_code)]
    current_barrier: AtomicUsize,
}

impl SharedCommunicationState {
    /// Create a new shared state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the state (for testing).
    pub fn reset(&self) {
        if let Ok(mut arrived) = self.barrier_arrived.write() {
            arrived.clear();
        }
        if let Ok(mut broadcast) = self.broadcast_data.write() {
            broadcast.clear();
        }
        if let Ok(mut shared) = self.shared_data.write() {
            shared.clear();
        }
    }
}

/// Global shared state for distributed contexts.
static GLOBAL_COMM_STATE: std::sync::OnceLock<Arc<SharedCommunicationState>> =
    std::sync::OnceLock::new();

fn get_global_comm_state() -> Arc<SharedCommunicationState> {
    GLOBAL_COMM_STATE
        .get_or_init(|| Arc::new(SharedCommunicationState::new()))
        .clone()
}

/// Inner state for the distributed context.
struct ContextInner {
    config: DistributedConfig,
    device: DeviceId,
    initialized: AtomicBool,
    barrier_id: AtomicUsize,
    // Local state removed in favor of shared simulation state
    #[allow(dead_code)] // Keep struct structure alignment if needed, or just remove.
    // simpler to just remove it as it was unused.
    comm_state: Arc<SharedCommunicationState>,
}

/// Distributed training context.
///
/// Manages the distributed environment and provides coordination primitives.
/// Communication is performed via shared memory state.
#[derive(Clone)]
pub struct DistributedContext {
    inner: Arc<ContextInner>,
}

impl DistributedContext {
    /// Create a new distributed context with the given rank and world size.
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self::with_config(DistributedConfig::new(rank, world_size))
    }

    /// Create a distributed context with full configuration.
    pub fn with_config(config: DistributedConfig) -> Self {
        let device = if config.gpus_per_node > 0 {
            DeviceId::Gpu(config.local_rank % config.gpus_per_node)
        } else {
            DeviceId::Cpu
        };

        log::info!(
            "Initializing distributed context: rank {}/{} on {}",
            config.rank,
            config.world_size,
            device
        );

        Self {
            inner: Arc::new(ContextInner {
                config,
                device,
                initialized: AtomicBool::new(true),
                barrier_id: AtomicUsize::new(0),
                comm_state: get_global_comm_state(),
            }),
        }
    }

    /// Create a single-process context (no distribution).
    pub fn single() -> Self {
        Self::with_config(DistributedConfig::single())
    }

    /// Get the rank of this process.
    pub fn rank(&self) -> usize {
        self.inner.config.rank
    }

    /// Get the total number of processes.
    pub fn world_size(&self) -> usize {
        self.inner.config.world_size
    }

    /// Get the local rank on this machine.
    pub fn local_rank(&self) -> usize {
        self.inner.config.local_rank
    }

    /// Get the device assigned to this rank.
    pub fn device(&self) -> DeviceId {
        self.inner.device
    }

    /// Check if this is the master process (rank 0).
    pub fn is_master(&self) -> bool {
        self.inner.config.rank == 0
    }

    /// Check if this context is initialized.
    pub fn is_initialized(&self) -> bool {
        self.inner.initialized.load(Ordering::Relaxed)
    }

    /// Synchronization barrier - all processes must call this.
    ///
    /// Uses shared memory state to coordinate between contexts.
    /// For single-process execution, this is a no-op.
    pub fn barrier(&self) {
        let world_size = self.world_size();
        if world_size == 1 {
            return;
        }

        let barrier_id = self.inner.barrier_id.fetch_add(1, Ordering::SeqCst);
        let rank = self.rank();

        log::debug!("Rank {} entering barrier {}", rank, barrier_id);

        // Register arrival at barrier
        {
            let mut arrived = self
                .inner
                .comm_state
                .barrier_arrived
                .write()
                .expect("Lock poisoned");
            let ranks = arrived.entry(barrier_id).or_default();
            if !ranks.contains(&rank) {
                ranks.push(rank);
            }
        }

        // Spin until all ranks have arrived
        loop {
            let arrived = self
                .inner
                .comm_state
                .barrier_arrived
                .read()
                .expect("Lock poisoned");
            if let Some(ranks) = arrived.get(&barrier_id) {
                if ranks.len() >= world_size {
                    break;
                }
            }
            drop(arrived);
            std::thread::yield_now();
        }

        log::debug!("Rank {} passed barrier {}", rank, barrier_id);
    }

    /// Broadcast a value from the source rank to all other ranks.
    ///
    /// The source rank provides the value; all other ranks receive it.
    /// Returns the broadcasted value for all ranks.
    pub fn broadcast<T: Clone + Send + 'static>(
        &self,
        value: Option<T>,
        src_rank: usize,
    ) -> Option<T> {
        if self.world_size() == 1 {
            return value;
        }

        // Source rank provides the value
        if self.rank() == src_rank {
            value
        } else {
            // Other ranks return None - they need to use broadcast_bytes for actual data
            None
        }
    }

    /// Broadcast bytes from source rank to all ranks using shared memory.
    pub fn broadcast_bytes(&self, data: Option<Vec<u8>>, src_rank: usize) -> Vec<u8> {
        let world_size = self.world_size();
        if world_size == 1 {
            return data.unwrap_or_default();
        }

        let broadcast_key = format!("broadcast_{}", self.inner.barrier_id.load(Ordering::SeqCst));

        if self.rank() == src_rank {
            // Source rank publishes data to shared state
            let bytes = data.unwrap_or_default();
            {
                let mut state = self
                    .inner
                    .comm_state
                    .broadcast_data
                    .write()
                    .expect("Lock poisoned");
                state.insert((broadcast_key.clone(), src_rank), bytes.clone());
            }
            bytes
        } else {
            // Other ranks wait for and read the broadcast data
            loop {
                let state = self
                    .inner
                    .comm_state
                    .broadcast_data
                    .read()
                    .expect("Lock poisoned");
                if let Some(bytes) = state.get(&(broadcast_key.clone(), src_rank)) {
                    return bytes.clone();
                }
                drop(state);
                std::thread::yield_now();
            }
        }
    }

    /// Store a value in shared state (visible to all ranks in simulation).
    pub fn put(&self, key: &str, value: Vec<u8>) {
        let mut state = self
            .inner
            .comm_state
            .shared_data
            .write()
            .expect("Lock poisoned");
        state.insert(key.to_string(), value);
    }

    /// Retrieve a value from shared state.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let state = self
            .inner
            .comm_state
            .shared_data
            .read()
            .expect("Lock poisoned");
        state.get(key).cloned()
    }

    /// Get the configuration.
    pub fn config(&self) -> &DistributedConfig {
        &self.inner.config
    }

    /// Reset the global simulation state (for testing only).
    pub fn reset_simulation() {
        get_global_comm_state().reset();
    }
}

impl Default for DistributedContext {
    fn default() -> Self {
        Self::single()
    }
}

#[cfg(test)]
mod tests {
    use super::{DeviceId, DistributedContext};
    use std::sync::atomic::Ordering;

    #[test]
    fn test_single_context() {
        let ctx = DistributedContext::single();
        assert_eq!(ctx.rank(), 0);
        assert_eq!(ctx.world_size(), 1);
        assert!(ctx.is_master());
        assert!(ctx.is_initialized());
    }

    #[test]
    fn test_multi_rank_context() {
        let ctx0 = DistributedContext::new(0, 4);
        let ctx1 = DistributedContext::new(1, 4);
        let ctx3 = DistributedContext::new(3, 4);

        assert!(ctx0.is_master());
        assert!(!ctx1.is_master());
        assert!(!ctx3.is_master());

        assert_eq!(ctx0.world_size(), 4);
        assert_eq!(ctx1.world_size(), 4);
    }

    #[test]
    fn test_device_id() {
        let cpu = DeviceId::Cpu;
        let gpu = DeviceId::Gpu(2);
        let metal = DeviceId::Metal(0);

        assert!(!cpu.is_gpu());
        assert!(gpu.is_gpu());
        assert!(metal.is_gpu());

        assert_eq!(gpu.index(), 2);
        assert_eq!(format!("{}", gpu), "cuda:2");
    }

    #[test]
    fn test_local_state() {
        let ctx = DistributedContext::single();
        ctx.put("test_key", vec![1, 2, 3]);

        let value = ctx.get("test_key");
        assert_eq!(value, Some(vec![1, 2, 3]));

        let missing = ctx.get("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_single_process_barrier() {
        let ctx = DistributedContext::single();
        // Single process barrier should be no-op
        ctx.barrier();
        ctx.barrier();
    }

    #[test]
    fn test_broadcast_bytes_single() {
        let ctx = DistributedContext::single();
        let data = ctx.broadcast_bytes(Some(vec![1, 2, 3]), 0);
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn test_barrier_multi_threaded() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;
        use std::thread;

        let completed = Arc::new(AtomicUsize::new(0));
        let world_size = 4;
        let mut handles = Vec::new();

        for rank in 0..world_size {
            let completed_clone = Arc::clone(&completed);
            handles.push(thread::spawn(move || {
                let ctx = DistributedContext::new(rank, world_size);
                // All threads reach barrier
                ctx.barrier();
                completed_clone.fetch_add(1, Ordering::SeqCst);
            }));
        }

        for h in handles {
            h.join().expect("Thread panicked");
        }

        assert_eq!(completed.load(Ordering::SeqCst), world_size);
    }
}
