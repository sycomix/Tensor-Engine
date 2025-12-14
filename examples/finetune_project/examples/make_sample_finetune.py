#!/usr/bin/env python3
"""Write a small fine-tuning text dataset.

Each line is treated as training text.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

SAMPLE_LINES = [
    "Instruction: Explain overfitting. Response: Overfitting is when a model memorizes training data and generalizes poorly.",
    "Instruction: Define gradient descent. Response: An optimization method that updates parameters to reduce loss.",
    "Instruction: What is a tokenizer? Response: A tool that converts text into token ids.",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample_finetune.txt", help="Output text file path")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(SAMPLE_LINES) + "\n", encoding="utf-8")
    logging.getLogger(__name__).info("Wrote sample finetune dataset to %s", str(out_path))


if __name__ == "__main__":
    main()
