#!/usr/bin/env python3
"""Write a tiny supervised fine-tuning (SFT) text dataset.

Each line is treated as training text. For real SFT, use an instruction-response format
and a tokenizer that matches your target distribution.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


SAMPLE_LINES = [
    "Instruction: Say hello.\nResponse: Hello!",
    "Instruction: Count to three.\nResponse: One, two, three.",
    "Instruction: What is Tensor Engine?\nResponse: A tensor library with autograd.",
    "Instruction: Summarize: the cat sat on the mat.\nResponse: A cat is sitting on a mat.",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample_sft.txt", help="Output text file path")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(SAMPLE_LINES) + "\n", encoding="utf-8")
    logging.getLogger(__name__).info("Wrote sample SFT dataset to %s", str(out_path))


if __name__ == "__main__":
    main()
