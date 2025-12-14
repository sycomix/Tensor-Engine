#!/usr/bin/env python3
"""Write a small sample text corpus for smoke pretraining.

This is intentionally tiny and is meant only to verify the full pretraining loop.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

SAMPLE_LINES = [
    "Tensor Engine is a tiny tensor library with autograd.",
    "This sample corpus exists to smoke test the pretraining loop.",
    "The quick brown fox jumps over the lazy dog.",
    "Hello world. This is a language model pretraining example.",
    "Small datasets will overfit quickly; use a real corpus for meaningful training.",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample_corpus.txt", help="Output text file path")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(SAMPLE_LINES) + "\n", encoding="utf-8")
    logging.getLogger(__name__).info("Wrote sample corpus to %s", str(out_path))


if __name__ == "__main__":
    main()
