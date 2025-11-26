# tensor_engine

A fast, Rust-based tensor library with automatic differentiation and Python bindings, designed for building and training neural networks.

## Quickstart

### Installation

First, install the dependencies:

```bash
pip install maturin
```

Then, build and install the package:

```bash
maturin develop
```

### Basic Usage

```python
import tensor_engine as te
import numpy as np

# Create tensors
a = te.Tensor([1.0, 2.0, 3.0], [3])
b = te.Tensor([4.0, 5.0, 6.0], [3])

# Perform operations
c = a + b
print(c)  # [5.0, 7.0, 9.0]

# Automatic differentiation
a = te.Tensor([1.0, 2.0], [2])
b = te.Tensor([3.0, 4.0], [2])
c = a * b
c.backward()

print(a.grad)  # [3.0, 4.0]
print(b.grad)  # [1.0, 2.0]
```

### Training a Simple Model

```python
import tensor_engine as te
import numpy as np

# Generate toy data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(float)

# Convert to tensors
X_tensor = te.Tensor(X.flatten(), X.shape)
y_tensor = te.Tensor(y, [100])

# Simple linear model
model = te.Linear(2, 1)

# Training loop
for epoch in range(100):
    # Forward pass
    pred = model.forward(X_tensor)
    loss = ((pred - y_tensor) ** 2).sum()

    # Backward pass
    loss.backward()

    # Update parameters
    for param in model.parameters():
        param -= param.grad * 0.01
        param.zero_grad()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data[0]}")
```

### Using Optimizers

```python
import tensor_engine as te
import numpy as np

# Data
X = te.Tensor(np.random.randn(10, 5).flatten(), [10, 5])
y = te.Tensor(np.random.randn(10, 1).flatten(), [10, 1])

# Model
model = te.Linear(5, 1)

# Optimizer
optimizer = te.SGD(0.01, 0.9)

for epoch in range(50):
    pred = model.forward(X)
    loss = ((pred - y) ** 2).mean()
    loss.backward()

    optimizer.step(model.parameters())
    optimizer.zero_grad(model.parameters())

    print(f"Epoch {epoch}, Loss: {loss.data[0]}")
```

### Neural Network with Sequential

```python
import tensor_engine as te

# Define a simple NN
model = te.Sequential([
    te.Linear(10, 5),
    te.ReLU(),  # Assuming ReLU is exposed
    te.Linear(5, 1)
])

# Note: Sequential and ReLU need to be implemented in Python wrapper
```

> The `nn` module implementation lives under `src/nn/mod.rs` with layer submodules like `src/nn/flatten.rs` (Flatten layer).

## Features

- Automatic differentiation
- Neural network primitives (Linear, Sequential)
- Neural network primitives (Linear, Sequential, Conv2D, Dropout, MaxPool, Flatten)

### CrossEntropy with logits example

```python
import tensor_engine as te
# logits shape (N, C)
logits = te.Tensor([1.0, 2.0, -1.0, 0.1, 0.2, 0.3], [2, 3])
targets = te.Tensor([1.0, 2.0], [2])  # label indices: 1, 2
loss = logits.cross_entropy_with_logits(targets)
loss.backward()
```

If you already have probabilities, you can call `te.Tensor.softmax()` or compute log_softmax using `log_softmax()`.

### SoftmaxCrossEntropy combined op (efficient)

```python
import tensor_engine as te
logits = te.Tensor([1.0, 2.0, -1.0, 0.1, 0.2, 0.3], [2, 3])
targets = te.Tensor([1.0, 2.0], [2])  # label indices: 1, 2
loss = logits.softmax_cross_entropy_with_logits(targets)
loss.backward()
```

### NLLLoss example (expects log-probs and integer labels)

```python
import tensor_engine as te
logits = te.Tensor([1.0, 2.0, -1.0], [1, 3])
log_probs = logits.log_softmax(1)
targets = te.Tensor([1.0], [1])
loss = te.NLLLoss().forward(log_probs, targets)
loss.backward()
```

- Optimizers (SGD, Adam)
- Python bindings
- High performance (Rust backend)
- Basic loss functions (MSE, CrossEntropy, CrossEntropy with logits, NLLLoss)

## Development

To run tests:

```bash
cargo test
```

To run benchmarks:

```bash
cargo bench
```

To run a smaller set of benches (useful for CI or faster local validation), set the `CI_BENCH` environment variable so benches run fewer sizes and shorter measurement times:

On Linux/macOS:

```bash
CI_BENCH=1 cargo bench --bench matmul_bench --features openblas
```

On Windows (PowerShell):

```pwsh
$env:CI_BENCH = 1
cargo bench --bench matmul_bench --features openblas
```

To build Python extension:

```bash
maturin develop --release
```

## Optional: Enable OpenBLAS for CBLAS-backed matmul

By default, `tensor_engine` uses a Rust-optimized kernel for matrix multiplication (`matrixmultiply`). If you want to use a native OpenBLAS (CBLAS) implementation for matmul, enable the `openblas` feature and ensure your system has a BLAS library available and accessible at build time.

On Windows, you can use the prebuilt `OpenBLAS-0.3.30-x64-64` included in the repository by setting the `OPENBLAS_DIR` environment variable to that path and enabling the feature:

```pwsh
$env:OPENBLAS_DIR = "$(Resolve-Path .\OpenBLAS-0.3.30-x64-64)"
cargo build --features openblas
```

Important: runtime DLL discovery on Windows

- After building with `--features openblas` you must ensure that the OpenBLAS DLL is discoverable at program run time. On Windows, set `PATH` to include `OpenBLAS-0.3.30-x64-64\bin` — otherwise the loader will fail with a missing DLL error when running the tests or Python module.

Extra debugging notes:

- We've added an `examples/blas_check.rs` program which calls `cblas_sgemm` with fixed inputs and prints dimensions and results so you can validate whether your OpenBLAS distribution expects row-major or column-major layout. Run it with:

```pwsh
cargo run --example blas_check --features openblas
```

- `MatMul` will prefer a CBLAS implementation when `--features openblas` is enabled. If the BLAS result does not match the NumPy/`ndarray` dot result (or if the library prints an SGEMM parameter error), `tensor_engine` will currently fall back to the pure-Rust `ndarray` dot product to ensure correctness and avoid panics.

If you'd like to help improve CBLAS detection and support, run `examples/blas_check.rs` and check whether your BLAS prefers `RowMajor` or `ColMajor` and the LDA/LDB/ldc parameters used; report your findings in an issue to help us implement an automatic detection for more stable BLAS usage.

Example (PowerShell):

```pwsh
$env:OPENBLAS_DIR = "$(Resolve-Path .\OpenBLAS-0.3.30-x64-64)"
$env:PATH = "$env:OPENBLAS_DIR\bin;$env:PATH"
cargo test --workspace --features openblas -- --nocapture
```

## CI & performance regression detection

This repository now includes a reduced set of performance benches that run on CI (Ubuntu) to spot regressions. We store a small baseline in `ci/baseline` and run a compare script that fails the CI job when the selected matmul bench (default: `matmul_50x50`) regresses more than a configurable threshold (currently 15%).

To run an offline compare with a baseline (for local validation):

```bash
python3 ci/compare_criterion_reports.py ci/baseline/matmul_50x50_estimates.json target/criterion/matmul/matmul_50x50/new/estimates.json matmul_50x50 15
```

To update the baseline, replace the file in `ci/baseline` after validating improved performance and creating a PR that updates the baseline artifact.

## Nightly full benches

We also run the full bench suite nightly (Ubuntu) via the `Scheduled Benchmarks` workflow to collect artifacts and observe performance trends over time. Nightly runs save Criterion outputs in the workflow artifacts. Maintainers can download these artifacts to analyze longer-term regressions or improvements.

## Windows OpenBLAS fallback and build strategy

The CI attempts to use the `OpenBLAS-0.3.30-x64-64` prebuilt binaries included in the repository on Windows for speed. If that attempts fails (linker error or incompatible import libs), the Windows job falls back to building OpenBLAS from source using the `blas-src` crate (enabled via the `openblas` feature), to maximize cross-platform feasibility. This provides two paths for CI to obtain OpenBLAS functionality:

- Prebuilt binaries: fast startup using `OPENBLAS_DIR` and `libopenblas.lib` import library
- Build-from-source via `blas-src`: slower but more robust across toolchains

## Automatic baseline update workflow (via GitHub Actions)

We provide a manual action named `Update Bench Baseline` which can be used to run the full matmul bench suite and automatically create a PR to update baseline JSON files in `ci/baseline` if a measurable improvement is detected. This helps maintain baseline transparency and keeps baseline updates under human review.

It is **manual** and intended to be used by maintainers to accept improvements. Once you run the action (Actions -> Update Bench Baseline -> Run), the action runs the bench, compares the `matmul_50x50` new result to the baseline, and opens a PR if an improvement is found.

Recommended helper script: `scripts/setup_dev_repo.ps1` (Windows)

- A small convenience script `scripts/setup_dev_repo.ps1` is included to set `OPENBLAS_DIR` and add the OpenBLAS bin to your PATH for the session or persist as user env vars if you prefer. To use it:

```pwsh
# from repo root
.\scripts\setup_dev_repo.ps1 -PersistUserEnvironment $false
# then run tests
cargo test --workspace --features openblas -- --nocapture
```

On Linux/macOS, install a system OpenBLAS and then build with the feature flag:

```bash
sudo apt-get install libopenblas-dev
cargo build --features openblas
```

If linking fails on your platform, you can continue using the default Rust backend (no `openblas` feature) which avoids external dependencies.
