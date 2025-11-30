#!/usr/bin/env python3
"""
Transformer demo using the `tensor_engine` Python bindings. Use after `maturin develop`.
"""
import numpy as np
import tensor_engine as te

def demo():
    # Create a small batch of sequences
    batch = 2
    seq = 4
    d_model = 8
    x = te.Tensor(list((np.arange(batch * seq * d_model, dtype=np.float32) * 0.01).flatten()), [batch, seq, d_model])
    tb = te.TransformerBlock(d_model, d_model * 2, num_heads=2)
    out = tb.forward(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

if __name__ == "__main__":
    demo()
