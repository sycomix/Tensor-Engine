# Zero-Torch Policy

Tensor Engine is an independent tensor and machine-learning runtime. The project
must not install, import, link, execute, or optionally enable Torch, PyTorch,
libtorch, or `tch`.

This rule applies to runtime code, optional features, tests, benchmarks,
examples, evaluation tools, setup scripts, development dependencies, CI, and
fixture or numerical-reference generation.

## Permitted compatibility boundary

Tensor Engine may parse an existing legacy `.pt` or TorchScript checkpoint using
its own Rust implementation. The parser must not load or link an external Torch
runtime. Serialized wire-format strings are permitted only inside the legacy
parser modules.

SafeTensors is the canonical format for new checkpoints and test fixtures.

## Independent correctness references

Correctness tests should use analytical results, finite differences, NumPy,
documented golden vectors, or cross-backend comparison against Tensor Engine's
CPU implementation. Golden vectors must not require Torch to regenerate.

Run `python scripts/check_no_torch.py` to audit the repository.

