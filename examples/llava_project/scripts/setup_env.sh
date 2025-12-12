#!/usr/bin/env bash
set -euo pipefail

action=${1:-full}

python -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ "$action" = "full" ] || [ "$action" = "build" ]; then
  mkdir -p third_party
  pushd third_party
  if [ ! -d Tensor-Engine ]; then
    git clone https://github.com/sycomix/Tensor-Engine.git
  fi
  pushd Tensor-Engine
  pip install maturin
  python -m maturin develop --release
  popd
  popd
fi

echo "Setup done. Activate the environment with 'source venv/bin/activate' and run: python -m examples.train_llava"
