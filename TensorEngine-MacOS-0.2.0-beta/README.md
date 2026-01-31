# Tensor Engine v0.2.0 Beta

High-performance tensor computation library with GPU acceleration for macOS.

## Installation

Double-click `install.sh` or run in Terminal:
```bash
./install.sh
```

## Features

- **Metal GPU Acceleration** - Native Apple Silicon support
- **Distributed Training** - Multi-GPU data parallelism
- **Async Operations** - Non-blocking tensor computations
- **SafeTensors** - Safe model serialization
- **FP16/BF16** - Half-precision support
- **Vision** - Image processing utilities
- **Quantization** - Model compression

## Quick Start

```python
import tensor_engine
from tensor_engine import Tensor

# Create tensors
a = Tensor.randn([512, 512])
b = Tensor.randn([512, 512])

# Matrix multiplication
c = a.matmul(b)
```

## Requirements

- macOS 11.0+ (Apple Silicon / ARM64)
- Python 3.9+

## Contents

- `install.sh` - Installer script
- `tensor_engine-0.2.0-*.whl` - Python package
- `libtensor_engine.dylib` - Native library

---
© 2026 Tensor Engine Team
