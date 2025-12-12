#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/resume_train_from_latest.sh [--dir examples/models] [--ext .ckpt.safetensors] [extra args]
DIR=examples/models
EXT=.ckpt.safetensors
PREFER_PARTIAL=1

if [ "$1" = "--dir" ]; then
  DIR="$2"; shift 2
fi
if [ "$1" = "--ext" ]; then
  EXT="$2"; shift 2
fi

# Find latest checkpoint using the helper script
CKPT=$(python scripts/find_latest_checkpoint.py --dir "$DIR" --ext "$EXT" --prefer-partial)
if [ -z "$CKPT" ]; then
  echo "No checkpoint found in $DIR with ext $EXT" >&2
  exit 1
fi

echo "Resuming training using checkpoint: $CKPT"
python -m examples.train_llava --resume --checkpoint "$CKPT" "$@"
