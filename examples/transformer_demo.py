#!/usr/bin/env python3
"""Transformer demo using the `tensor_engine` Python bindings.

This example requires installing the Python extension (maturin develop --release).
"""
from __future__ import annotations
# pylint: disable=import-error, missing-function-docstring, line-too-long
import numpy as np
try:
    import tensor_engine as te  # type: ignore
except ImportError:  # pragma: no cover
    te = None  # type: ignore

def demo() -> None:
    if te is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")

    # Create a small batch of sequences
    batch = 2
    seq = 4
    d_model = 8
    x_arr = (np.arange(batch * seq * d_model, dtype=np.float32) * 0.01).flatten()
    x = te.Tensor(list(x_arr), [batch, seq, d_model])
    tb = te.TransformerBlock(d_model, d_model * 2, num_heads=2)
    out = tb.forward(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # NL-OOB example: create with a 'logarithmic' bias function and max_scale 2.0
    tb_oob = te.TransformerBlock(
        d_model, d_model * 2, num_heads=2, nl_oob_config="logarithmic", nl_oob_max_scale=2.0
    )
    # Create a distance matrix (seq x seq) representing absolute token distance
    dist_arr = np.zeros((seq, seq), dtype=np.float32)
    for i in range(seq):
        for j in range(seq):
            dist_arr[i, j] = abs(i - j)
    dist_t = te.Tensor(list(dist_arr.flatten()), [seq, seq])
    out2 = tb_oob.forward_with_distance(x, dist_t)
    print("NL-OOB Output shape:", out2.shape)

if __name__ == "__main__":
    demo()
