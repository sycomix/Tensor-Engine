# Tensor Engine Linux Installation Package (Full Features)

## Overview

This package contains the Tensor Engine library and Python bindings for Linux systems with **all features enabled** for maximum functionality and performance.

## Version

- **Tensor Engine**: 0.2.0 (Full Features)
- **Target Platform**: Linux x86_64
- **Python Version**: 3.11
- **Build Type**: Release (optimized)
- **Features**: All available features enabled

## Enabled Features

### Core Features
- **OpenBLAS**: Optimized BLAS library for matrix operations
- **Multi-precision**: f16 and bf16 data type support
- **SafeTensors**: Efficient tensor serialization format
- **Python Bindings**: Full PyO3 integration

### ML/AI Features
- **Tokenizers**: HuggingFace tokenizers integration
- **PyTorch Compatibility**: tch bindings for PyTorch interoperability
- **Audio Processing**: hound + rubato for audio I/O and resampling
- **Vision/Image**: image crate for computer vision tasks

### Performance Features
- **Async Operations**: tokio async runtime support
- **Parallel I/O**: rayon for data parallelism
- **Memory Optimization**: Advanced memory management

## Package Contents

```
Tensor-Engine-Linux-0.2.0/
├── install.sh                           # Installation script
├── libtensor_engine.so                 # Shared library (10.6MB, all features)
├── tensor_engine-0.2.0-cp311-cp311-manylinux_2_34_x86_64.whl  # Python wheel (3.6MB)
└── README.md                            # This file
```

## System Requirements

- **Operating System**: Linux (x86_64)
- **Python**: 3.8 or higher (tested with 3.11)
- **Memory**: At least 1GB RAM (recommended for full features)
- **Disk Space**: 100MB for installation
- **Optional**: OpenBLAS system package for best performance

## Installation

### Option 1: System-wide Installation (Recommended)

```bash
sudo ./install.sh
```

This installs Tensor Engine to `/opt/tensor-engine` and makes it available system-wide.

### Option 2: User Installation

```bash
./install.sh --user
```

This installs the Python wheel for the current user only.

### Option 3: Custom Directory

```bash
sudo ./install.sh --dir /custom/path/tensor-engine
```

This installs Tensor Engine to a custom directory.

## Installation Options

| Option | Description |
|--------|-------------|
| `-d, --dir DIR` | Specify custom installation directory |
| `-u, --user` | Install Python wheel for current user only |
| `-h, --help` | Show help message |

## Post-Installation Setup

After installation, source the environment variables:

```bash
source /opt/tensor-engine/env.sh
```

Or add this line to your `~/.bashrc` or `~/.zshrc`:

```bash
source /opt/tensor-engine/env.sh
```

## Verification

Test the installation:

```bash
python3 -c "import tensor_engine; print('Tensor Engine is ready!')"
```

Test specific features:

```bash
# Basic tensor operations
python3 -c "import tensor_engine as te; x = te.Tensor([1, 2, 3], [3]); print(x.get_data())"

# Multi-precision support
python3 -c "import tensor_engine as te; x = te.Tensor.randn([2, 2], dtype='f16'); print('f16 tensor created')"

# SafeTensors support
python3 -c "import tensor_engine as te; print('SafeTensors support available')"

# Audio processing
python3 -c "import tensor_engine as te; print('Audio processing features available')"

# Vision/image processing
python3 -c "import tensor_engine as te; print('Vision processing features available')"
```

## Usage Examples

### Basic Tensor Operations

```python
import tensor_engine as te

# Create tensors
x = te.Tensor([1.0, 2.0, 3.0], [3])
y = te.Tensor([4.0, 5.0, 6.0], [3])

# Basic operations with OpenBLAS acceleration
z = x + y
print(z.get_data())  # [5.0, 7.0, 9.0]

# Matrix multiplication (optimized with OpenBLAS)
a = te.Tensor([[1.0, 2.0], [3.0, 4.0]], [2, 2])
b = te.Tensor([[5.0, 6.0], [7.0, 8.0]], [2, 2])
c = a.matmul(b)
print(c.get_data())  # [[19.0, 22.0], [43.0, 50.0]]
```

### Multi-precision Operations

```python
import tensor_engine as te

# Create f16 tensors (half precision)
x_f16 = te.Tensor.randn([1000, 1000], dtype='f16')
y_f16 = te.Tensor.randn([1000, 1000], dtype='f16')

# Operations maintain precision
z_f16 = x_f16.matmul(y_f16)
print(f"f16 result shape: {z_f16.shape}")

# Create bf16 tensors (bfloat16)
x_bf16 = te.Tensor.randn([512, 512], dtype='bf16')
print(f"bf16 tensor created: {x_bf16.shape}")
```

### SafeTensors Integration

```python
import tensor_engine as te

# Create a model
model = te.Linear(768, 768)

# Save to SafeTensors format
state_dict = model.parameters()
te.save_safetensors(state_dict, "model.safetensors")

# Load from SafeTensors format
loaded_state = te.load_safetensors("model.safetensors")
print(f"Loaded {len(loaded_state)} parameters from SafeTensors")
```

### Tokenizers Integration

```python
import tensor_engine as te

# Use built-in tokenizer support
if hasattr(te, 'Tokenizer'):
    tokenizer = te.Tokenizer.from_pretrained("gpt2")
    text = "Hello, Tensor Engine!"
    tokens = tokenizer.encode(text)
    print(f"Tokenized: {tokens}")
    
    # Decode back to text
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
```

### Audio Processing

```python
import tensor_engine as te
import numpy as np

# Process audio data (if audio files available)
if hasattr(te, 'load_audio'):
    # Load audio file
    audio_data, sample_rate = te.load_audio("example.wav")
    print(f"Audio shape: {audio_data.shape}, Sample rate: {sample_rate}")
    
    # Resample audio
    resampled = te.resample_audio(audio_data, sample_rate, 16000)
    print(f"Resampled audio shape: {resampled.shape}")
```

### Vision/Image Processing

```python
import tensor_engine as te

# Process images (if image files available)
if hasattr(te, 'load_image'):
    # Load image
    image = te.load_image("example.jpg")
    print(f"Image shape: {image.shape}")
    
    # Convert to tensor
    image_tensor = te.image_to_tensor(image)
    print(f"Image tensor shape: {image_tensor.shape}")
```

### Async Operations

```python
import tensor_engine as te
import asyncio

async def async_example():
    # Create tensors
    x = te.Tensor.randn([1000, 1000])
    y = te.Tensor.randn([1000, 1000])
    
    # Async matrix multiplication
    result = await te.async_matmul(x, y)
    print(f"Async result shape: {result.shape}")

# Run async operation
asyncio.run(async_example())
```

### Neural Network Example

```python
import tensor_engine as te

# Create a simple neural network
class SimpleNet(te.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = te.Linear(784, 256)
        self.linear2 = te.Linear(256, 10)
        self.relu = te.nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Create and use the network
net = SimpleNet()
x = te.Tensor.randn([32, 784])  # Batch of 32 samples
output = net(x)
print(f"Network output shape: {output.shape}")
```

## Features

### High Performance
- **OpenBLAS Integration**: Optimized linear algebra operations
- **Multi-threading**: Parallel processing with rayon
- **Memory Efficiency**: Smart memory management

### Multi-Precision Support
- **f32**: Standard 32-bit floating point
- **f16**: Half precision for memory efficiency
- **bf16**: Bfloat16 for ML training

### Data Formats
- **SafeTensors**: Fast and safe tensor serialization
- **PyTorch Compatibility**: Interoperability with PyTorch models
- **Custom Formats**: Flexible data loading

### Modalities
- **Text**: Tokenizer integration and text processing
- **Audio**: Audio loading, processing, and resampling
- **Vision**: Image loading and preprocessing

### Async & Parallel
- **Async Operations**: Non-blocking computations
- **Parallel I/O**: Concurrent data processing
- **Scalable Architecture**: Built for performance

## Troubleshooting

### Python Import Error

If you get an import error, ensure:

1. The Python wheel is installed correctly
2. You're using the correct Python version (3.8+)
3. The library path is set correctly

```bash
# Check installation
python3 -m pip show tensor_engine

# Reinstall if necessary
python3 -m pip uninstall tensor_engine
python3 -m pip install tensor_engine-0.2.0-cp311-cp311-manylinux_2_34_x86_64.whl
```

### Library Not Found Error

If you encounter library loading errors:

1. Source the environment script:
   ```bash
   source /opt/tensor-engine/env.sh
   ```

2. Check if the library exists:
   ```bash
   ls -la /opt/tensor-engine/lib/libtensor_engine.so
   ```

3. Verify library cache:
   ```bash
   sudo ldconfig -v | grep tensor_engine
   ```

### Performance Issues

For best performance:

1. Ensure OpenBLAS is properly configured:
   ```bash
   export OPENBLAS_NUM_THREADS=4
   ```

2. Check that all features are enabled:
   ```bash
   python3 -c "import tensor_engine; print('Features loaded successfully')"
   ```

### Memory Issues

With all features enabled, the library uses more memory:

1. Monitor memory usage during operations
2. Use appropriate data types (f16 for large models)
3. Clean up unused tensors

## Development

For development purposes, you can also build from source:

```bash
# Clone the repository
git clone https://github.com/your-org/tensor-engine.git
cd tensor-engine

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build with all features
cargo build --release --features "python_bindings,openblas,multi_precision,safe_tensors,with_tokenizers,with_tch,audio,vision,dtype_f16,dtype_bf16,async_ops,parallel_io"

# Build Python wheel
python3 -m maturin build --release --features "python_bindings,openblas,multi_precision,safe_tensors,with_tokenizers,with_tch,audio,vision,dtype_f16,dtype_bf16,async_ops,parallel_io"
```

## Uninstallation

To uninstall Tensor Engine:

```bash
sudo /opt/tensor-engine/uninstall.sh
```

Or if you installed to a custom directory:

```bash
sudo /custom/path/tensor-engine/uninstall.sh
```

## Support

For issues, questions, or contributions:

- **Documentation**: See the project documentation
- **Issues**: Report bugs via the issue tracker
- **Community**: Join discussions in the community forum

## License

This software is released under the MIT License. See the LICENSE file for details.

## Acknowledgments

Tensor Engine is built with:
- Rust for performance and memory safety
- PyO3 for Python bindings
- OpenBLAS for optimized linear algebra
- ndarray for numerical computing
- tokenizers for text processing
- image crate for vision tasks
- tokio for async operations
- rayon for parallel processing
- And many other excellent open-source libraries