# Text Pretraining Project (from scratch)

This folder is a self-contained Python project that **pretrains a small causal language model from scratch** using only
**Tensor Engine** + a text dataset.

It uses:

- `tensor_engine.Tokenizer` (Rust-side HF tokenizer loader; requires Tensor Engine built with `with_tokenizers`)
- `tensor_engine.TransformerBlock` (LLaMA-style causal blocks)
- `tensor_engine.Linear` for the vocab head
- `Tensor.embedding_lookup` for token/position embeddings

## Quick start (Windows / PowerShell)

1) Open PowerShell **in this folder** (`examples/pretrain_project`).
2) Create venv + install deps + build `tensor_engine`:

- Default features:
    - `python_bindings,with_tokenizers`
- Optional speed-up:
    - add `openblas`

Run:

- `./scripts/setup_env.ps1`

3) Download a tokenizer (writes `tokenizer.json` into the output dir):

- `python scripts/download_tokenizer.py --name bert-base-uncased --out examples/tokenizer`

4) Create a tiny sample dataset (or bring your own text file):

- `python -m examples.make_sample_corpus`

5) Pretrain:

-
`python -m examples.train_pretrain_lm --text data/sample_corpus.txt --tokenizer-json examples/tokenizer/tokenizer.json --save runs/pretrain_ckpt.npz`

## Dataset format

A plain UTF-8 text file. Each line is treated as a training string.

For real pretraining, provide a large text corpus and increase `--steps`, `--batch`, `--seq-len`, `--d-model`, and
`--depth`.

## Outputs

The training script saves:

- `*.npz` checkpoint containing model tensors (embedding, positional embedding, transformer blocks, head)
- `*.json` sidecar config next to the checkpoint

These outputs are consumed by:

- `examples/finetune_project` (full fine-tuning)
- `examples/lora_project` (LoRA-style adapter tuning on the vocab head)

## Notes

- Token ids are passed as float tensors (Tensor Engine’s current Python API uses `f32` tensors).
- The Transformer blocks are created in **LLaMA-style causal** mode (`llama_style=True`, `use_rope=True`).
- If you built Tensor Engine without `with_tokenizers`, `tensor_engine.Tokenizer` will not be available.
