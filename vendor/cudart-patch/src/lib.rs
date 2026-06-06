#![allow(non_upper_case_globals)]

extern crate static_assertions;

pub use runtime::{
  CudaError,
  CudaResult,
  CudaDevice,
  CudaStream,
  CudaEvent,
  CudaEventStatus,
  CudaMemcpyKind,
  cuda_alloc_host,
  cuda_alloc_device,
  cuda_free_device,
  cuda_memset,
  cuda_memset_async,
  cuda_memcpy,
  cuda_memcpy_async,
  cuda_memcpy_2d_async,
  cuda_memcpy_peer_async,
};

pub mod ffi;
pub mod runtime;

#[cfg(feature = "cuda_6_5")]  const_assert_eq!(cuda_api_version; ffi::driver::__CUDA_API_VERSION,  6050);
#[cfg(feature = "cuda_6_5")]  const_assert_eq!(cuda_version;     ffi::driver::CUDA_VERSION,        6050);

#[cfg(feature = "cuda_7_0")]  const_assert_eq!(cuda_api_version; ffi::driver::__CUDA_API_VERSION,  7000);
#[cfg(feature = "cuda_7_0")]  const_assert_eq!(cuda_version;     ffi::driver::CUDA_VERSION,        7000);

#[cfg(feature = "cuda_7_5")]  const_assert_eq!(cuda_api_version; ffi::driver::__CUDA_API_VERSION,  7050);
#[cfg(feature = "cuda_7_5")]  const_assert_eq!(cuda_version;     ffi::driver::CUDA_VERSION,        7050);

#[cfg(feature = "cuda_8_0")]  const_assert_eq!(cuda_api_version; ffi::driver::__CUDA_API_VERSION,  8000);
#[cfg(feature = "cuda_8_0")]  const_assert_eq!(cuda_version;     ffi::driver::CUDA_VERSION,        8000);

#[cfg(feature = "cuda_9_0")]  const_assert_eq!(cuda_api_version; ffi::driver::__CUDA_API_VERSION,  9000);
#[cfg(feature = "cuda_9_0")]  const_assert_eq!(cuda_version;     ffi::driver::CUDA_VERSION,        9000);

#[cfg(feature = "cuda_9_1")]  const_assert_eq!(cuda_api_version; ffi::driver::__CUDA_API_VERSION,  9010);
#[cfg(feature = "cuda_9_1")]  const_assert_eq!(cuda_version;     ffi::driver::CUDA_VERSION,        9010);

#[cfg(feature = "cuda_9_2")]  const_assert_eq!(cuda_api_version; ffi::driver::__CUDA_API_VERSION,  9020);
#[cfg(feature = "cuda_9_2")]  const_assert_eq!(cuda_version;     ffi::driver::CUDA_VERSION,        9020);

#[cfg(feature = "cuda_10_0")] const_assert_eq!(cuda_api_version; ffi::driver::__CUDA_API_VERSION, 10000);
#[cfg(feature = "cuda_10_0")] const_assert_eq!(cuda_version;     ffi::driver::CUDA_VERSION,       10000);
