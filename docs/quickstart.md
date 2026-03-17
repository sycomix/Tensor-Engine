# Quickstart — Tensor Engine

This doc helps contributors run and test the project locally.

## Prerequisites

- Rust toolchain (stable/1.75+ recommended)
- Python 3.10+ (if building Python bindings) and maturin ≥ 1.11
- On Windows: Visual Studio C++ toolchain (MSVC)
- Optional: OpenBLAS binaries (set `OPENBLAS_DIR` or use `scripts/setup_dev_repo.ps1`)

## Build & Run

**Build library**

```bash
cargo build                                           # default features
cargo build --features "openblas,python_bindings"     # add features as needed
```

**Run tests**

```bash
cargo test
cargo test --features "openblas"
```

**Python wheel (release)**

```bash
pip install maturin
# Windows — set OPENBLAS path first:
# $env:OPENBLAS_DIR = "path\to\OpenBLAS-0.3.30-x64-64"
maturin build --release \
  --features "python_bindings,safe_tensors,with_tokenizers,vision,audio,multi_precision,async_ops,openblas,quantized,distributed" \
  --out dist
pip install dist/tensor_engine-0.3.1-*.whl
```

**Python wheel (editable / dev)**

```bash
maturin develop --release --features "python_bindings,safe_tensors,with_tokenizers,openblas"
```

**Smoke test**

```python
import tensor_engine as te
t = te.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
print(t.shape())   # [2, 2]
print(t.relu().get_data())
```


## BLAS / OpenBLAS runtime notes

By default, the Rust extension uses BLAS via the `openblas` feature for fast matrix multiplies.

**Linux (Debian/Ubuntu)**

```bash
sudo apt update && sudo apt install -y libopenblas-dev libopenblas-base
maturin develop --release --features "python_bindings,openblas"
```

**Windows**

Download OpenBLAS binaries (e.g. OpenBLAS-0.3.30-x64-64) and set `OPENBLAS_DIR`:

```powershell
$env:OPENBLAS_DIR = "C:\path\to\OpenBLAS-0.3.30-x64-64"
maturin build --release --features "python_bindings,openblas" --out dist
```

Copy the `OpenBLAS/bin` directory to your `PATH` or alongside the wheel so `openblas.dll` can be found at runtime.

If no system BLAS is available the library falls back to a Rust pure-Rust matrix multiply—correct but slower.

## Examples

**Rust**

```bash
cargo run --example blas_check --features "openblas"
cargo run --example sample_diffusion
```

**Python**

```bash
python examples/linear_regression.py
python examples/transformer_demo.py
python examples/chat_llama.py /path/to/model.safetensors
```

## Notes

- On Windows with `openblas`, run `scripts/setup_dev_repo.ps1` to configure `OPENBLAS_DIR` and PATH automatically.
- Many tests require optional features; use `cargo test --features "openblas,safe_tensors"` for the most common subset.
- The `with_tch` feature requires a prebuilt libtorch; set `LIBTORCH` and add it to PATH before building.
- Minimum supported Python: **3.10** (CPython). PyPy is not tested.