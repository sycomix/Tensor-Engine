#![recursion_limit = "4096"]

//! This crate provides a tensor library with automatic differentiation.

#[cfg(feature = "rocket")]
#[macro_use]
extern crate rocket;

pub mod amp;

pub mod autograd;
pub mod backend;
pub mod config;
pub mod dtype;
pub mod error;
pub mod generation;
pub mod io;
pub mod labels;
#[path = "nn/mod.rs"]
pub mod nn;
#[cfg(feature = "safe_tensors")]
pub use io::safetensors_loader::apply_kronos_bytes_to_module_bytes;
#[cfg(feature = "safe_tensors")]
pub use io::safetensors_loader::load_safetensors_from_bytes;
pub mod compat_blas;
pub mod lr_scheduler;
pub mod memory_pool;
pub mod ops;
pub mod optim;
pub mod quantization;
#[cfg(feature = "server")]
pub mod server;
pub mod tensor;
pub mod tokenizer;

#[cfg(feature = "hf_compat")]
pub mod hf_compat;

pub mod tensor_utils;

#[cfg(feature = "metrics")]
pub mod monitoring;

// References module contains experimental/reference implementations
// It may require nightly features if enabled
#[cfg(feature = "compat")]
pub mod compat;

#[cfg(feature = "hf_compat")]
pub mod hf_bridge;

#[cfg(feature = "async_ops")]
pub mod async_ops;

#[cfg(feature = "distributed")]
pub mod distributed;

#[cfg(feature = "python_bindings")]
pub mod python_bindings;
