# LLAVA Training Project

This folder is a self-contained Python project that trains and runs a LLaVA-style multimodal model using the `tensor_engine` Python bindings.

The workflow is based on **real images** referenced by a simple **manifest TSV** and uses the Rust-side image loader (`tensor_engine.ImageTextDataLoader`) plus an on-disk Hugging Face tokenizer file (`tokenizer.json`) loaded by `tensor_engine.Tokenizer`.

## Quick start (Windows / PowerShell)

1. Ensure Rust (via rustup), Python 3.9+ and Visual Studio Build Tools are installed.
2. Open PowerShell in the repo root.
3. Run `./scripts\setup_env.ps1` (creates venv, installs deps, builds `tensor_engine`).
4. Prepare a manifest and train.

## Docker (optional, reproducible Linux environment)

This project includes a Docker build for a reproducible, Linux-based training/runtime environment.

Important notes:

- The Docker build compiles the **local** Tensor-Engine repo into the image using `maturin develop`.
  That means the **Docker build context must be the repo root** (so the Rust crate is available).
- The manifest format used here contains **absolute paths**. When training inside Docker, the paths in
  your manifest must be valid **inside the container** (e.g., `/data/images/0001.jpg`), not Windows paths.
- The image’s default entrypoint runs a short **synthetic smoke train**. For real training, override the
  command as shown below.

### Build the image

From the repo root:

```
docker build -t tensor-engine-llava -f examples/llava_project/Dockerfile .
```

### Run the default smoke job

This runs the image entrypoint (synthetic data, 1 epoch):

```
docker run --rm -it tensor-engine-llava
```

### Run training on a real dataset (mounted)

Mount images + an output folder. The easiest workflow is:

1) mount your images at a known container path (e.g., `/data/images`)
2) generate a manifest inside the container so it contains container-absolute paths
3) download a tokenizer inside the container to a mounted directory
4) run `train_llava`

Example (replace the host paths with your own):

```
docker run --rm -it \
  -v /path/on/host/images:/data/images:ro \
  -v /path/on/host/out:/out \
  tensor-engine-llava bash -lc "\
    . venv/bin/activate && \
    python scripts/download_tokenizer.py --name bert-base-uncased --out /out/tokenizer && \
    python scripts/prepare_manifest.py --metadata /out/metadata.json --images-root /data/images --output /out/manifest.txt && \
    python -m examples.train_llava \
      --manifest /out/manifest.txt \
      --tokenizer-json /out/tokenizer \
      --config examples/llava_model_config_base.json \
      --image-w 224 --image-h 224 \
      --augment --shuffle \
      --epochs 1 --batch 2 \
      --save /out/llava_model.safetensors\
  "
```

If you already have a manifest file, ensure it uses container paths (for example `/data/images/000000000001.jpg`) and mount it:

```
docker run --rm -it \
  -v /path/on/host/images:/data/images:ro \
  -v /path/on/host/manifest.txt:/data/manifest.txt:ro \
  -v /path/on/host/tokenizer:/data/tokenizer:ro \
  -v /path/on/host/out:/out \
  tensor-engine-llava bash -lc "\
    . venv/bin/activate && \
    python -m examples.train_llava \
      --manifest /data/manifest.txt \
      --tokenizer-json /data/tokenizer \
      --config examples/llava_model_config_base.json \
      --epochs 1 --batch 2 \
      --save /out/llava_model.safetensors\
  "
```

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
