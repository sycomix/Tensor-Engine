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


### BLAS / OpenBLAS runtime notes
- By default, the Rust extension will attempt to use BLAS (via the `openblas` feature) for fast matrix multiplies. At runtime the Python wheel needs to be able to resolve BLAS symbols (e.g., `cblas_sgemm`).

- If you have a system OpenBLAS available, install it (Linux example):
  ```bash
  # Debian/Ubuntu
  sudo apt update && sudo apt install -y libopenblas-dev libopenblas-base
  ```
  Then rebuild the wheel enabling the `openblas` feature so the extension links to the system BLAS:
  ```bash
  OPENBLAS_DIR=/usr/ make || true
  maturin develop --release --bindings pyo3 --features "python_bindings openblas"
  ```

- If you do not have a system BLAS available, two options exist:
  - Quick workaround (no install required): preload a small included stub that implements `cblas_sgemm` (naive, correct but slow):
    ```bash
    LD_PRELOAD=/path/to/Tensor-Engine/libcblas_stub.so python your_script.py
    ```
  - Build OpenBLAS from source (recommended if you want performance):
    ```bash
    # Example build (Linux)
    sudo apt install -y build-essential gfortran git
    git clone https://github.com/xianyi/OpenBLAS.git
    cd OpenBLAS
    make DYNAMIC_ARCH=1 NO_SHARED=0 USE_OPENMP=0
    sudo make install
    # then rebuild wheel with --features openblas
    maturin develop --release --bindings pyo3 --features "python_bindings openblas"
    ```

- Note: The repo includes a minimal, compiled fallback stub and a Rust fallback to avoid hard failures; they are functional but unoptimized. Prefer installing a proper OpenBLAS for real workloads.  


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