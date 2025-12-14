# Backend plan for tensor_engine

This document outlines a plan to support backend abstraction for CPU and GPU (wgpu/cudarc) and how to structure kernels.

Goals:

- Provide a minimal CPU backend that uses ndarray / matrixmultiply / OpenBLAS. Keep current code path as default.
- Introduce a Backend trait describing ops that can be implemented for specific kernels (matmul, conv, softmax,
  layernorm, etc.).
- Implement a `cpu` backend that uses existing ops as a reference.
- Implement a `wgpu` backend first for WebGPU support and memory-safe GPU paths.
- Implement a `cudarc` backend for CUDA accelerated kernels; optional feature.

Suggested API sketch:

- trait Backend {
- fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor;
- fn conv2d(&self, input: &Tensor, weight: &Tensor) -> Tensor;
- // other operations
  -}

- Provide a global runtime or per-thread selected backend via `BackendRegistry` or a context object passed into ops.
- For now, implement the CPU backend and plugin-style modules that register with the runtime at startup.

CI & testing:

- Add CI matrix for `with_tch`, `with_tokenizers`, and `safe_tensors` as already added.
- Add test suites that run cpu-only and feature gated tests.
- Add benchmark comparisons between CPU and GPU backends using `benches/`.

Notes:

- Provide a simple CPU to GPU conversion path for Tensor storage — i.e., a `Tensor::to_device(device)` API that returns
  a Tensor stored in a backend-specific memory region.
- Use `wgpu` and `cudarc` features behind optional Cargo features.
