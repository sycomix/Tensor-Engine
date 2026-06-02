# Non-Linear Out-of-Order Bias (NL-OOB)

Tensor Engine supports **NL-OOB**, a mechanism for injecting distance-based inductive biases into Transformer attention
without relying on sequence order. This is particularly useful for modeling 1D sequences with long-range dependencies (
Power Law) or 3D geometries (Molecules).

## Concept

Standard Transformers use positional encodings (APE/RoPE) or linear biases (ALiBi) that assume a fixed sequence. NL-OOB
generalizes strictly relative attention by applying a non-linear decay function $\phi(d)$ to a distance matrix $D_{ij}$.

$$ Attention_{ij} \propto \exp(q_i k_j^T - \lambda_h \cdot \phi(D_{ij})) $$

Where:

- $\phi(d) = \log(1+d)$ (Type A: scale-free/power-law) or $d^2$ (Type B: Gaussian).
- $\lambda_h$ is a learnable, head-specific slope initialized geometrically.

## API Usage

### Rust

```rust
use tensor_engine::nn::{TransformerBlock, BiasFunction};

let block = TransformerBlock::new_with_nl_oob(
    d_model, 
    d_ff, 
    num_heads, 
    BiasFunction::Logarithmic, // or Gaussian
    8.0 // max scale
);
```

### Python

```python
import tensor_engine as te

# Initialize with NL-OOB config
block = te.TransformerBlock(
    d_model=64,
    d_ff=128,
    num_heads=4,
    nl_oob_config="logarithmic", # or "gaussian"
    nl_oob_max_scale=8.0
)

# Forward pass with explicit distance matrix
# dist_tensor shape: [Batch, Seq, Seq] or [Seq, Seq]
output = block.forward_with_distance(x, dist_tensor)
```

## Example: Protein Stability Transformer

A complete end-to-end example is available in `examples/NL-OOB/`. It demonstrates:

1. **Data Loading**: Parsing Parquet files for protein sequences.
2. **Architecture**: Using `logarithmic` bias on 1D sequence distance to model folding stability.
3. **Training**: Custom training loop with `MSELoss`.
4. **Serving**: HTTP server serving predictions with robust weight loading.

See [examples/NL-OOB/README.md](../examples/NL-OOB/README.md) for details.
