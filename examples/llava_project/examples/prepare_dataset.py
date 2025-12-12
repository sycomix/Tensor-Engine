#!/usr/bin/env python3
"""
Prepare a small synthetic dataset for toy LLAVA-like training.
"""
from __future__ import annotations
import argparse
import logging
import json
import os
import numpy as np


def generate_dataset(
    out: str = "examples/data/synthetic_llava.jsonl",
    count: int = 32,
    height: int = 32,
    width: int = 32,
    channels: int = 3,
    tokenizer: str | None = None,
) -> None:
    """Generate a synthetic dataset and optionally tokenizes using a Hugging Face tokenizer.

    Args:
        out: output jsonl path
        count: number of examples
        height, width, channels: image dims
        tokenizer: Optional HF tokenizer name or local path to tokenizer
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    os.makedirs(os.path.dirname(out), exist_ok=True)

    hf_tokenizer = None
    if tokenizer is not None:
        try:
            from transformers import AutoTokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        except (ImportError, OSError, ValueError) as e:
            logger.warning("transformers tokenizer not available or failed to load; falling back to whitespace tokenizer: %s", e)

    with open(out, "w", encoding="utf-8") as fh:
        for i in range(count):
            img = np.random.rand(height, width, channels).astype(np.float32)
            input_text = f"Describe the image {i}"
            target_text = f"A synthetic image number {i}."
            payload = {
                "image": img.flatten().tolist(),
                "height": height,
                "width": width,
                "channels": channels,
                "input_text": input_text,
                "target_text": target_text,
            }
            if hf_tokenizer is not None:
                try:
                    input_ids = hf_tokenizer.encode(input_text, truncation=True)
                    target_ids = hf_tokenizer.encode(target_text, truncation=True)
                    payload["input_ids"] = [int(x) for x in input_ids]
                    payload["target_ids"] = [int(x) for x in target_ids]
                except (TypeError, ValueError) as e:
                    logger.exception("Tokenizer failed to encode sample %s; writing without ids: %s", i, e)
            fh.write(json.dumps(payload) + "\n")

    logger.info("Saved %d synthetic examples to %s", count, out)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/data/synthetic_llava.jsonl")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    generate_dataset(args.out, args.count, args.height, args.width, args.channels, args.tokenizer)


if __name__ == "__main__":
    main()
