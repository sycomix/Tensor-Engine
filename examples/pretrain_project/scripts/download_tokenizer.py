#!/usr/bin/env python3
"""Download and save a Hugging Face tokenizer locally.

Usage:
  python scripts/download_tokenizer.py --name bert-base-uncased --out examples/tokenizer

This uses `transformers` to fetch a tokenizer and writes `tokenizer.json` into the output directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="HF tokenizer model name or local path")
    parser.add_argument("--out", required=True, help="Directory to write tokenizer files")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as e:
        logger.error("transformers is required to download tokenizers: %s", e)
        raise

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading tokenizer '%s'", args.name)
    try:
        tok = AutoTokenizer.from_pretrained(args.name)
    except (OSError, ValueError) as e:
        logger.exception("Failed to download tokenizer '%s': %s", args.name, e)
        raise

    logger.info("Saving tokenizer to %s", str(outdir))
    try:
        tok.save_pretrained(str(outdir))
    except OSError as e:
        logger.exception("Failed to save tokenizer to %s: %s", str(outdir), e)
        raise

    logger.info("Saved tokenizer. Expected tokenizer.json at: %s", str(outdir / "tokenizer.json"))


if __name__ == "__main__":
    main()
