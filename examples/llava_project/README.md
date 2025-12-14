# LLAVA Training Project

This folder is a self-contained Python project that trains and runs a LLaVA-style multimodal model using the `tensor_engine` Python bindings.

The workflow is based on **real images** referenced by a simple **manifest TSV** and uses the Rust-side image loader (`tensor_engine.ImageTextDataLoader`) plus an on-disk Hugging Face tokenizer file (`tokenizer.json`) loaded by `tensor_engine.Tokenizer`.

## Quick start (Windows / PowerShell)

1. Ensure Rust (via rustup), Python 3.9+ and Visual Studio Build Tools are installed.
2. Open PowerShell in the repo root.
3. Run `./scripts\setup_env.ps1` (creates venv, installs deps, builds `tensor_engine`).
4. Prepare a manifest and train.

## Prepare a manifest from a dataset

Manifest format (TSV):

```
/abs/path/to/image.jpg\tCaption text
```

Create one from common dataset metadata formats:

```
python scripts/prepare_manifest.py --metadata <metadata.json|metadata.jsonl|metadata.csv|metadata.tsv> --images-root <dir> --output examples/data/manifest.txt
```

The script supports:

- COCO JSON (`images` + `annotations`)
- JSONL with `image_path` + `caption` keys (configurable)
- CSV/TSV with `image_path` + `caption` columns (configurable)

### Smoke manifest (bundled sample image)

If you just want to sanity-check the pipeline without downloading a dataset, this project includes a tiny `sample_image.ppm` and a helper that writes a one-line manifest with an **absolute** path:

```
python -m examples.make_sample_manifest
```

This writes `examples/data/sample_manifest.txt`.

## Tokenizer (tokenizer.json)

Download a tokenizer locally (this writes `tokenizer.json` under the output directory):

```
python scripts/download_tokenizer.py --name bert-base-uncased --out examples/tokenizer
```

Then pass the JSON file (or the directory containing it) to training/generation via `--tokenizer-json`.

## Train

Recommended config (usable on CPU, non-trivial size):

```
python -m examples.train_llava \
  --manifest examples/data/manifest.txt \
  --tokenizer-json examples/tokenizer \
  --config examples/llava_model_config_base.json \
  --image-w 224 --image-h 224 \
  --augment --shuffle \
  --epochs 1 --batch 2 \
  --save examples/models/llava_model.safetensors
```

Smoke run using the bundled sample image (no external dataset):

```
python -m examples.make_sample_manifest
python -m examples.train_llava \
  --manifest examples/data/sample_manifest.txt \
  --tokenizer-json examples/tokenizer \
  --config examples/llava_model_config_base.json \
  --epochs 1 --batch 1 \
  --save examples/models/llava_sample.safetensors
```

`train_llava.py` writes a sidecar file next to the model:

- `examples/models/llava_model.config.json`

That config is used by `generate_llava.py` to reconstruct the exact architecture.

### Resume / checkpoints

- Use `--resume` to load from a checkpoint.
- Use `--checkpoint` to control the checkpoint path (defaults to `<save>.ckpt.safetensors`).
- Use `--checkpoint-interval N` to save every N epochs (0 disables periodic saving).

## Generate

Run greedy decoding on a real image:

```
python -m examples.generate_llava \
  --image /abs/path/to/some.jpg \
  --model_path examples/models/llava_model.safetensors \
  --tokenizer-json examples/tokenizer \
  --prompt "Describe the image."
```

Smoke generate using the bundled sample image:

```
python -m examples.generate_llava \
  --image examples/data/sample_image.ppm \
  --model_path examples/models/llava_sample.safetensors \
  --tokenizer-json examples/tokenizer
```

## Synthetic mode (explicit)

For a fast smoke run without any dataset, `train_llava.py` can still generate and train on a synthetic JSONL dataset, but it is intentionally behind an explicit flag:

```
python -m examples.train_llava --synthetic --epochs 1 --batch 2
```
