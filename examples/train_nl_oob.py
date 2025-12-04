#!/usr/bin/env python3
"""Train a tiny TransformerBlock with NL-OOB and show slopes updating during training.

This demo relies on the Python bindings for `tensor_engine`. If not installed,
the script will raise at runtime — use `maturin develop --release` to install.
"""
from __future__ import annotations
# pylint: disable=import-error, missing-function-docstring, line-too-long
# pylint: disable=import-error

import numpy as np
try:
    import tensor_engine as te  # type: ignore
except ImportError:  # pragma: no cover
    # For IDE/static analysis we tolerate missing package; runtime requires installation.
    te = None  # type: ignore


def main() -> None:
    """Run a few training steps for a toy TransformerBlock and print slope stats."""
    batch = 4
    seq = 8
    d_model = 16
    d_ff = 32
    num_heads = 2

    # toy random data
    if te is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")

    rng = np.random.default_rng(42)
    x_arr = rng.normal(size=(batch * seq * d_model)).astype(np.float32)
    x = te.Tensor(list(x_arr), [batch, seq, d_model])
    y_arr = rng.normal(size=(batch * seq * d_model)).astype(np.float32)
    y_target = te.Tensor(list(y_arr), [batch, seq, d_model])

    # distance matrix per batch as absolute token distance
    dist = np.abs(np.subtract.outer(np.arange(seq), np.arange(seq))).astype(np.float32)
    # Use same distance for all batch entries (shape [seq, seq])
    dist_t = te.Tensor(list(dist.flatten()), [seq, seq])

    tb = te.TransformerBlock(
        d_model,
        d_ff,
        num_heads,
        nl_oob_config="logarithmic",
        nl_oob_max_scale=2.0,
    )
    opt = te.Adam(1e-3, 0.9, 0.999, 1e-8)

    # find slopes parameter
    named = tb.named_parameters("tb")
    slopes_idx = None
    for i, (k, _) in enumerate(named):
        if "nl_oob.slopes" in k:
            slopes_idx = i
            break
    assert slopes_idx is not None, "NL-OOB slopes not found"

    def get_slopes():
        return tb.named_parameters("tb")[slopes_idx][1]

    print("Initial slopes:\n", get_slopes())
    loss_fn = te.MSELoss()

    for step in range(20):
        opt.zero_grad(tb.parameters())
        out = tb.forward_with_distance(x, dist_t)
        loss = loss_fn.forward(out, y_target)
        loss.backward()
        opt.step(tb.parameters())
        if (step + 1) % 5 == 0:
            print(f"Step {step+1}, loss={loss}")
            print("Slopes:\n", get_slopes())

    print("Final slopes:\n", get_slopes())

if __name__ == "__main__":
    main()
