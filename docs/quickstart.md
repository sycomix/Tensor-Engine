# Quickstart — Tensor Engine

This doc helps contributors run and test the project locally.

Prerequisites

- Rust toolchain (stable/1.70+ recommended)
- Python 3.11 (if building Python bindings) and maturin (optional)
- On Windows: Visual Studio C++ toolchain
- Optional: OpenBLAS binaries (set OPENBLAS_DIR or use `scripts/setup_dev_repo.ps1`)

Build & Run

- Build library
    - `cargo build` (default features)
    - `cargo build --features "openblas,python_bindings"` (add features as needed)

- Run tests
    - `cargo test`
    - `cargo test --features "openblas,with_tch"` (if you have those toolchains installed)

- Python bindings (dev)
    - Install `maturin` in your environment, e.g. `pip install maturin`
    - Build & install dev wheel (editable): `maturin develop --release --bindings pyo3 --features python_bindings`
    - Or build a release wheel: `maturin build --release --bindings pyo3 --features python_bindings`
    - Run the Python smoke test (in the venv where the package is installed):
      ```bash
      python tests/python_smoke_test.py
      ```

Examples

- Rust examples:
    - `cargo run --example blas_check --features "openblas"` (Unix only)
    - `cargo run --example sample_diffusion` (if features/inputs satisfied)
- Python examples:
    - `python examples/linear_regression.py`
    - `python examples/load_model.py` (requires Python bindings and tokenizers if using them)

Notes

- When building on Windows with `openblas`, run `scripts/setup_dev_repo.ps1` to configure `OPENBLAS_DIR` and add DLLs to
  PATH.
- Many tests and examples require optional features (OpenBLAS, Python bindings, tch); run `cargo test --all-features`
  only in CI planned environments.
- Optional: `--features with_tch` requires a prebuilt libtorch or installed Python torch wheel; on Windows, follow CI
  approach to download shared libtorch and set `LIBTORCH` and PATH accordingly.

If you want a minimal quickstart for a new dev environment, I can add a `scripts/setup_dev_repo.ps1` walkthrough and a
`docs/index.md` linking to this quickstart.