# Canonical Runtime

Tensor Engine has one canonical execution stack:

- `src/tensor.rs`: Tensor identity, storage ownership, shape, dtype, and device state
- `src/ops.rs`: differentiable operation definitions
- `src/autograd.rs`: graph traversal, gradient accumulation, and gradient utilities
- `src/backend/`: device implementations selected by the core dispatcher
- `src/nn/`: modules whose parameters and outputs use the canonical Tensor
- `src/python_bindings.rs`: Python exposure of the same Rust Tensor and modules

New public APIs and features must use this stack.

## No alternate runtime

The former `src/compat/` inference stack and its `compat` feature have been
removed. Tensor, model, tokenizer, loading, and serving features must use the
canonical modules above. Architecture checks reject reintroducing the removed
runtime.

## Runtime contract

The canonical Tensor owns or shares storage through its existing synchronized
handle. Operations must:

- validate shape, rank, dtype, and device inputs before execution;
- return typed errors for recoverable user input failures;
- record all information required by backward during forward;
- reduce broadcast gradients to the original operand shapes;
- preserve dtype and device unless an API explicitly documents conversion;
- never silently move data between devices;
- use the CPU implementation as the numerical correctness reference;
- report unsupported backend operations instead of silently changing semantics.

Modules must return every trainable parameter from `parameters()` and
`named_parameters()`, and checkpoint keys must be deterministic.

## Canonical server registry

`engine serve --model-registry <dir>` loads one model per child directory.
Each model directory contains `config.json`, `model.safetensors`, and
`tokenizer.json`. The config accepts Hugging Face Llama dimension names and the
equivalent Tensor Engine aliases. Registry entries are loaded in sorted order,
validated before allocation, and rejected when required canonical parameters
are missing.

## Migration sequence

1. Expand canonical model-format coverage in `src/io/`.
2. Add end-to-end server fixtures for registry loading and OpenAI requests.
3. Continue closing Torch API parity gaps on the canonical runtime.

Run `python scripts/check_runtime_boundaries.py` to verify the isolation rules.
