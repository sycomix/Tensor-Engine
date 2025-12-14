#!/usr/bin/env bash
set -euo pipefail

FEATURES=${1:-python_bindings,with_tokenizers}

echo "Setting up finetune_project venv and building tensor_engine ($FEATURES)"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install -r ./requirements.txt

pushd ../.. >/dev/null
  ./examples/finetune_project/venv/bin/python -m maturin develop --release --features "$FEATURES"
popd >/dev/null

echo "Done. Activate with 'source venv/bin/activate' and run: python -m examples.train_finetune_lm --help"
