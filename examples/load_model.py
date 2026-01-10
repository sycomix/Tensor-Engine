#!/usr/bin/env python3
"""Load a SafeTensors file and apply a state dict to a TransformerBlock using Python bindings.

The script expects the `tensor_engine` Python package to be available (e.g.
installed via `maturin develop --release`). It is tolerant of missing imports
for IDE linting by using a fallback import with a helpful error if the package
is not available at runtime.
"""
from __future__ import annotations

import argparse
import logging
import sys

try:
    import tensor_engine as te  # type: ignore
except ImportError:  # pragma: no cover
    # For IDE/static analysis we tolerate missing package; runtime requires installation.
    te = None  # type: ignore


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    """CLI entrypoint: applies a SafeTensors state dict to a TransformerBlock.
    """
    # allow automated harness to skip if no args
    if len(sys.argv) <= 1:
        print('No safetensors path provided; skipping load_model example')
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("safetensors_path")
    parser.add_argument("--transpose", action="store_true")
    args = parser.parse_args()

    if te is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")

    with open(args.safetensors_path, "rb") as f:
        data = f.read()

    # Create a module that matches the model state (dimensions must match saved weights)
    d_model = 128
    d_ff = 512
    num_heads = 8
    tb = te.TransformerBlock(
        d_model,
        d_ff,
        num_heads,
        nl_oob_config="logarithmic",
        nl_oob_max_scale=2.0,
    )

    # Apply state dict directly into the module (root prefix is optional and depends on naming)
    te.py_load_safetensors_into_module(data, args.transpose, tb, "mha")
    logger.info("Loaded state into TransformerBlock")

    # Print slopes if present
    named = tb.named_parameters("mha")
    for k, t in named:
        if "nl_oob.slopes" in k:
            logger.info("Found slopes: %s %s", k, t)


if __name__ == "__main__":
    main()
