//! Distributed Training Infrastructure
//!
//! This module provides utilities for multi-GPU distributed training,
//! including data parallelism, gradient synchronization, and checkpointing.
//!
//! # Features
//!
//! - **Data Parallel**: Automatically shard batches across devices
//! - **Gradient Sync**: All-reduce operations for gradient aggregation
//! - **Checkpointing**: Robust distributed checkpoint save/load
//!
//! # Example
//!
//! ```rust,ignore
//! use tensor_engine::distributed::{DistributedContext, DataParallel};
//!
//! let ctx = DistributedContext::new(0, 4); // rank 0 of 4
//! let model = DataParallel::new(my_model, &ctx);
//! ```

mod all_reduce;
mod checkpoint;
mod context;
mod data_parallel;

pub use all_reduce::{AllReduce, ReduceOp};
pub use checkpoint::{CheckpointConfig, DistributedCheckpoint};
pub use context::{DeviceId, DistributedConfig, DistributedContext};
pub use data_parallel::{DataParallel, ShardedBatch};
