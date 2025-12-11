#!/usr/bin/env python3
"""
Prepare a small synthetic dataset for toy LLAVA-like training.

This script creates a JSONL file where each line contains a JSON object:
{
  "image": [...],  # flattened float32 values in HxWxC order
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
import os
import numpy as np


def simple_tokenizer(text: str, vocab: dict):
    tokens = []
    for w in text.strip().split():
        if w not in vocab:
            vocab[w] = len(vocab)
        tokens.append(vocab[w])
    return tokens


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/data/synthetic_llava.jsonl")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer (HF pretrained name) to use, e.g. 'bert-base-uncased'")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # No tokenizer used for dataset generation — vocabulary will be built in training script
    if args.count <= 0:
        raise SystemExit("--count must be > 0")

    # If a HF tokenizer is requested, try to import and use it
    hf_tokenizer = None
    if args.tokenizer is not None:
        try:
            from transformers import AutoTokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            logger.warning("transformers tokenizer not available or failed to load; falling back to simple whitespace tokenizer: %s", e)

    with open(args.out, "w", encoding="utf-8") as fh:
        for i in range(args.count):
            # random image
            img = np.random.rand(args.height, args.width, args.channels).astype(np.float32)
            # input/prompt: short synthetic instruction
            input_text = f"Describe the image {i}"
            # target text: brief caption
            target_text = f"A synthetic image number {i}."
            # save flattened as list (small, so acceptable)
            payload = {
                "image": img.flatten().tolist(),
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

    logger.info(f"Saved {args.count} synthetic examples to {args.out}")


if __name__ == "__main__":
    main()
