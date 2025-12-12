#!/usr/bin/env bash
set -euo pipefail
EPOCHS=${1:-3}
BATCH=${2:-4}
CHECKPOINT_INTERVAL=${3:-1}
shift 3 || true
EXTRA_ARGS="$@"

. venv/bin/activate
python -m examples.train_llava --epochs ${EPOCHS} --batch ${BATCH} --checkpoint-interval ${CHECKPOINT_INTERVAL} ${EXTRA_ARGS}
