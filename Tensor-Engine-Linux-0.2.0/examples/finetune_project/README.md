# Fine-tuning Project (existing checkpoints)

This folder is a self-contained Python project that demonstrates **fine-tuning an existing Tensor Engine text model**.

It loads a base `.npz` checkpoint produced by `examples/pretrain_project` and continues training on a new dataset.

## Quick start (Windows / PowerShell)

1) Open PowerShell **in this folder** (`examples/finetune_project`).
2) Create venv + install deps + build `tensor_engine`:

- `./scripts/setup_env.ps1`

3) Create a tiny fine-tuning dataset:

- `python -m examples.make_sample_finetune`

4) Fine-tune from a base checkpoint:

- `python -m examples.train_finetune_lm \
    --base runs/pretrain_ckpt.npz \
    --text data/sample_finetune.txt \
    --tokenizer-json examples/tokenizer/tokenizer.json \
    --save runs/finetuned_ckpt.npz`

## Notes

- This project fine-tunes **all** parameters by default.
- If you want parameter-efficient tuning instead, use `examples/lora_project`.
