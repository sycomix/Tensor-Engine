#!/usr/bin/env bash
# Build the docs locally (Unix)
set -euo pipefail

echo "Building Tensor Engine docs..."
python -m pip install --upgrade pip setuptools || true
python -m pip install mkdocs mkdocs-material pymdown-extensions -q

mkdocs build --clean -d site

echo "Docs built into ./site/"
