#!/usr/bin/env python3
"""
Download and save a Hugging Face tokenizer locally into `examples/tokenizer`.

Usage:
    python scripts/download_tokenizer.py --name bert-base-uncased --out examples/tokenizer
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

def main():
    """Download and save a Hugging Face tokenizer locally."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='HF tokenizer model name or path')
    parser.add_argument('--out', default='examples/tokenizer', help='Local path to save tokenizer')
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        logger.error('transformers package is required to download tokenizer: %s', e)
        raise

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info('Downloading tokenizer %s to %s', args.name, str(outdir))
    try:
        tok = AutoTokenizer.from_pretrained(args.name)
    except (OSError, ValueError) as e:
        logger.exception('Failed to download tokenizer %s: %s', args.name, e)
        raise
    try:
        tok.save_pretrained(str(outdir))
    except OSError as e:
        logger.exception('Failed to save tokenizer to %s: %s', outdir, e)
        raise
    logger.info('Saved tokenizer to %s', str(outdir))

if __name__ == '__main__':
    main()
