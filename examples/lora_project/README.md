# LoRA Adapter Project (head adapter)

This folder is a self-contained Python project that demonstrates **training a LoRA-style low-rank adapter** on top of an
existing Tensor Engine text model.

Because Tensor Engine’s `TransformerBlock` is a Rust module, this example applies the low-rank update to the **vocab
head** (logits projection). It still follows the core LoRA idea:

- freeze the base model weights
- train only a small low-rank module (`A` and `B`) and add it as a residual update

## Prerequisites

You need a **base checkpoint** produced by `examples/pretrain_project`:

- `runs/pretrain_ckpt.npz`
- `runs/pretrain_ckpt.npz.config.json`

## Quick start (Windows / PowerShell)

1) Open PowerShell **in this folder** (`examples/lora_project`).
2) Set up venv and build `tensor_engine`:

- `./scripts/setup_env.ps1`

3) Create a tiny fine-tuning dataset:

- `python -m examples.make_sample_sft`

4) Train the adapter:

- `python -m examples.train_lora_adapter \
    --base runs/pretrain_ckpt.npz \
    --text data/sample_sft.txt \
    --tokenizer-json examples/tokenizer/tokenizer.json \
    --save runs/lora_adapter.npz`

## Outputs

- `runs/lora_adapter.npz` — LoRA adapter tensors
- `runs/lora_adapter.npz.config.json` — adapter metadata (rank/alpha + base checkpoint reference)

## Notes

- The adapter is applied at logits time: `logits = base_logits + scale * (B(A(hidden)))`.
- This keeps the example simple and fully functional with today’s Python bindings.
