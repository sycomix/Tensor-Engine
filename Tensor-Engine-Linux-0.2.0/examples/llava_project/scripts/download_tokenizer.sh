#!/usr/bin/env bash
set -euo pipefail
NAME=${1:-bert-base-uncased}
OUT=${2:-examples/tokenizer}

python scripts/download_tokenizer.py --name "$NAME" --out "$OUT"
