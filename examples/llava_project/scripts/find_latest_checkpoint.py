#!/usr/bin/env python3
"""
Find the most recent checkpoint file (preferring partial checkpoints) and print the path.

Usage:
    python scripts/find_latest_checkpoint.py --dir examples/models --ext .ckpt.safetensors

This script prints the path to stdout, or exits with code 1 if no checkpoint is found.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


def find_latest_checkpoint(directory: str, ext: str = '.ckpt.safetensors', prefer_partial: bool = True) -> Optional[
    Path]:
    logger = logging.getLogger(__name__)
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning('Directory does not exist: %s', directory)
        return None

    # Find partial checkpoints (*.partial.*.<ext>) and standard checkpoints (*.ckpt.safetensors)
    partials = list(dir_path.glob(f'*.partial.*{ext}'))
    # Standard ckpt pattern (avoid selecting partial files)
    std_ckpts = [p for p in dir_path.glob(f'*{ext}') if '.partial.' not in p.name]

    def latest(files):
        if not files:
            return None
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0]

    if prefer_partial:
        choice = latest(partials) or latest(std_ckpts)
    else:
        choice = latest(std_ckpts) or latest(partials)

    if choice is None:
        logger.info('No checkpoint files found in %s (ext=%s)', directory, ext)
    else:
        logger.info('Found checkpoint: %s', choice)

    return choice


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='examples/models', help='Directory to search for checkpoints')
    parser.add_argument('--ext', default='.ckpt.safetensors', help='Checkpoint file extension to search for')
    parser.add_argument('--prefer-partial', action='store_true', help='Prefer partial checkpoints when selecting')
    args = parser.parse_args()

    res = find_latest_checkpoint(args.dir, args.ext, args.prefer_partial)
    if res is None:
        raise SystemExit(1)
    sys.stdout.write(str(res) + '\n')


if __name__ == '__main__':
    main()
