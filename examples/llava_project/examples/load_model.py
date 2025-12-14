#!/usr/bin/env python3
"""Load a SafeTensors file and apply a state dict to a TransformerBlock using Python bindings.
"""
from __future__ import annotations

import argparse
import logging

try:
    import tensor_engine as te  # type: ignore
except ImportError:  # pragma: no cover
    te = None  # type: ignore


def main() -> None:
    """CLI to load a SafeTensors file and apply it to a TransformerBlock using Python bindings."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("safetensors_path")
    parser.add_argument("--transpose", action="store_true")
    args = parser.parse_args()

    if te is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")

    try:
        with open(args.safetensors_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        logger.error("SafeTensors path not found: %s", args.safetensors_path)
        raise
    except (OSError, UnicodeDecodeError) as e:
        logger.exception("Failed to read SafeTensors file: %s", args.safetensors_path)
        raise

    d_model = 128
    d_ff = 512
    num_heads = 8
    tb = te.TransformerBlock(d_model, d_ff, num_heads, nl_oob_config="logarithmic", nl_oob_max_scale=2.0)

    try:
        te.py_load_safetensors_into_module(data, args.transpose, tb, "mha")
        logger.info("Loaded state into TransformerBlock")
    except (RuntimeError, TypeError, ValueError) as e:
        logger.exception("Failed to apply SafeTensors into TransformerBlock: %s", e)
        raise

    named = tb.named_parameters("mha")
    for k, t in named:
        if "nl_oob.slopes" in k:
            logger.info("Found slopes: %s %s", k, t)


if __name__ == "__main__":
    main()
