#!/usr/bin/env python3
"""
Prepare a small synthetic dataset for toy LLAVA-like training.

This script creates a JSONL file where each line contains a JSON object:
{
    "image": [0.0, 0.0, 0.0],  # flattened float32 values in HxWxC order (length = height*width*channels)
  "height": H,
  "width": W,
  "channels": C,
  "input_text": "prompt",
  "target_text": "answer"
}

It tries to use the HF tokenizer (python `transformers` or `tokenizers`) if available; otherwise it falls back to a simple whitespace tokenizer.
"""
from __future__ import annotations
import argparse
import logging
import json
import importlib
import random
from pathlib import Path
from typing import Any


def simple_tokenizer(text: str, vocab: dict[str, int]) -> list[int]:
    """Tokenize text by whitespace and build a simple integer vocabulary."""
    tokens = []
    for w in text.strip().split():
        if w not in vocab:
            vocab[w] = len(vocab)
        tokens.append(vocab[w])
    return tokens


def main() -> None:
    """Generate a tiny synthetic multimodal dataset and write it as JSONL."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/data/synthetic_llava.jsonl")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Optional tokenizer (HF pretrained name) to use, e.g. "
            "'bert-base-uncased'"
        ),
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # No tokenizer used for dataset generation — vocabulary will be built in training script
    if args.count <= 0:
        raise SystemExit("--count must be > 0")

    # If a HF tokenizer is requested, try to import and use it.
    # We avoid a hard import so this script remains runnable without transformers.
    hf_tokenizer: Any | None = None
    if args.tokenizer is not None:
        try:
            transformers_mod = importlib.import_module("transformers")
            auto_tokenizer = getattr(transformers_mod, "AutoTokenizer")
            hf_tokenizer = auto_tokenizer.from_pretrained(args.tokenizer)
        except (
            ImportError,
            ModuleNotFoundError,
            AttributeError,
            OSError,
            RuntimeError,
            ValueError,
        ) as err:
            logger.warning(
                "Tokenizer not available or failed to load; falling back to whitespace tokenizer: %s",
                err,
            )

    with open(out_path, "w", encoding="utf-8") as fh:
        rng = random.Random(42)
        for i in range(args.count):
            # random image
            n = args.height * args.width * args.channels
            img_flat = [float(rng.random()) for _ in range(n)]
            # input/prompt: short synthetic instruction
            input_text = f"Describe the image {i}"
            # target text: brief caption
            target_text = f"A synthetic image number {i}."
            # save flattened as list (small, so acceptable)
            payload = {
                "image": img_flat,
                "height": args.height,
                "width": args.width,
                "channels": args.channels,
                "input_text": input_text,
                "target_text": target_text,
            }
            if hf_tokenizer is not None:
                # Add token id arrays to payload
                payload["input_ids"] = hf_tokenizer.encode(input_text, truncation=True)
                payload["target_ids"] = hf_tokenizer.encode(target_text, truncation=True)
            fh.write(json.dumps(payload) + "\n")

    logger.info("Saved %d synthetic examples to %s", args.count, out_path)


if __name__ == "__main__":
    main()
