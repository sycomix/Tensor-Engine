# Backend Trait Migration Plan

This document outlines a migration plan to implement the `Backend` trait (defined
in `src/backend.rs`) and to migrate expensive ops in `src/ops.rs` to use backend
implementations.

Motivation

- Allow optimized kernels for various backends (CPU, OpenBLAS, CUDA, wgpu).
- Centralize backend selection and detection logic.
- Keep existing ops as a fallback while enabling optimized paths.

High-level steps

1. Define trait surface in `src/backend.rs` (already present). Add higher-level
   hooks as needed (e.g., `matmul`, `conv2d`, `softmax`) with consistent signatures.
2. Implement `CpuBackend` in `src/backend.rs` which delegates to existing ops but
   provides optimized versions where available (e.g., OpenBLAS).
3. For ops that can be accelerated (e.g., MatMul, Conv), provide `backend.matmul`
   call site in `src/ops.rs` and return `Option<ArrayD<f32>>` when backend
   can compute a `Some(result)`.
4. Move optimized CPU/OpenBLAS detection into `backend` implementations, not
   spread across `ops.rs`.
5. Add unit tests to ensure consistent forward/backward results across backends.

Notes & Considerations

- Maintain existing `ops` implementations as a fallback to ensure correctness.
- Begin migration with a single op (MatMul) and expand progressively.
- Add configuration and auto-detection for backends and a debug logging flag
  to force a backend.
- Ensure thread-safety for backends — `Arc<dyn Backend>`, and safe locking.

Tracking

- Create a GitHub issue titled "Migrate ops to Backend trait" and link here.
- Add `docs/backend_migration_plan.md` entry to the project docs and track
  progress via PRs.

If you want, I can start implementing the MatMul backend migration as the next
step with tests and a small benchmark to validate results.