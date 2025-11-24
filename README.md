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

## Features

- Automatic differentiation
- Neural network primitives (Linear, Sequential)
- Neural network primitives (Linear, Sequential, Conv2D, Dropout, MaxPool)

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
- Basic loss functions (MSE, CrossEntropy placeholder - CrossEntropy expects probabilities)

## Development

To run tests:

```bash
cargo test
```

To run benchmarks:

```bash
cargo bench
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

On Linux/macOS, install a system OpenBLAS and then build with the feature flag:

```bash
sudo apt-get install libopenblas-dev
cargo build --features openblas
```

If linking fails on your platform, you can continue using the default Rust backend (no `openblas` feature) which avoids external dependencies.
