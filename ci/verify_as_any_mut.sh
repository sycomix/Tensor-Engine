#!/usr/bin/env bash
set -euo pipefail

# Run the as_any_mut verification script
python3 scripts/verify_as_any_mut.py

echo "as_any_mut verification succeeded"