# LLAVA Training Project Framework

This repository contains a minimal framework to reproduce the `llava` example from the `Tensor-Engine` project as a local project dependency. It provides scripts for setting up the environment, cloning and building Tensor-Engine (Python bindings), and running the small toy training example that generates a tiny multimodal model.

The example uses a synthetic dataset and the toy ``MultimodalLLM`` in the `Tensor-Engine` Python bindings.

Quick steps (Windows / PowerShell):

1. Ensure Rust (via rustup), Python 3.9+ and Visual Studio Build Tools are installed.
2. Open PowerShell in the project root.
3. ./scripts\setup_env.ps1 - will create a venv, install Python dependencies, clone and build the Tensor-Engine binding.
4. Run training: `python -m examples.train_llava --epochs 5 --batch 4`

Linux (Bash):

1. Ensure Rust (via rustup) and Python 3.9+ are installed.
2. ./scripts/setup_env.sh
3. Run training: `python -m examples.train_llava --epochs 5 --batch 4`

Docker (recommended for reproducible builds):

Build the image:

```bash
# from project root (ensure you build from the repo root so the local Tensor-Engine sources are included in the build context)
# Build with local repository mounted as build context so Docker uses the correct sources and features
docker build -f examples/llava_project/Dockerfile -t llava-train:local .
```

Note: If you see errors like "module 'tensor_engine' has no attribute 'VisionTransformer'" when running examples inside the container, rebuild the wheel with Python bindings and the vision feature enabled. The Dockerfile builds Tensor-Engine by running maturin; ensure the `maturin` invocation includes the features to expose the multimodal/vision bindings:

```bash
# Example: enable python bindings and vision feature when building locally
python3 -m maturin develop --release --manifest-path D:\Tensor-Engine\Cargo.toml --features "python_bindings,vision"
```

Run training inside the container:

```bash
docker run --rm -it llava-train:local python -m examples.train_llava --epochs 5
```

Notes:
- Tensor-Engine is built from source. The `setup_env` scripts will clone the `sycomix/Tensor-Engine` repo into the local `third_party` directory and run `maturin develop --release` to install the Python bindings.
- The training uses a toy dataset and a tiny model so it runs quickly for development and CI validation.
- If you already have `tensor_engine` installed (for example from an editable local install), you can skip the build step.

- Optionally you can use a Hugging Face tokenizer to tokenize the synthetic dataset and to decode outputs. Use the `scripts/download_tokenizer.py` script or the `scripts/download_tokenizer.sh` / `scripts/download_tokenizer.ps1` wrappers to download a tokenizer locally into `examples/tokenizer`.
	- Example (Windows PowerShell):
		```powershell
		.\venv\Scripts\Activate.ps1
		python scripts/download_tokenizer.py --name bert-base-uncased --out examples/tokenizer
		```
	- Example (Bash):
		```bash
		source venv/bin/activate
		./scripts/download_tokenizer.sh bert-base-uncased examples/tokenizer
		```
	- When running `train_llava.py` you can provide the tokenizer with `--tokenizer examples/tokenizer` and the dataset will be generated with token ids.
	- Additional dataset and checkpoint flags supported by `train_llava.py`:
		- `--force`: Force dataset regeneration even if the data file exists.
		- `--tokenized-data <path>`: Write tokenized dataset to a separate file and preserve the original data file; if provided, this tokenized file will be used for training.
		- `--resume`: Attempt to resume model weights from a checkpoint file if present.
		- `--checkpoint <path>`: Path to checkpoint file (defaults to save path with `.ckpt.safetensors`).
		- `--checkpoint-interval N`: Save a checkpoint every N epochs (default 1). Set to 0 to disable periodic checkpoints.

	- Example to regenerate and write tokenized data separately, then run training and resume from checkpoint:
		```powershell
		# Generate tokenized dataset to a separate file and train
		.\venv\Scripts\Activate.ps1
		python -m examples.train_llava --force --tokenizer examples/tokenizer --tokenized-data examples/data/synthetic_llava_tokenized.jsonl --epochs 5

		# Resume training from checkpoint
		python -m examples.train_llava --resume --checkpoint examples/models/llava_model.ckpt.safetensors
		```

Tokenization & checkpoint workflow examples (Windows PowerShell)
```
powershell
.\venv\Scripts\Activate.ps1

# 1) Download an HF tokenizer locally (optional but recommended when using --tokenizer)
python scripts/download_tokenizer.py --name bert-base-uncased --out examples/tokenizer

# 2) Regenerate both the original (un-tokenized) dataset and a separate tokenized dataset
#    (keeps original in `--data` and writes tokenized to `--tokenized-data`)
python -m examples.train_llava --force --tokenizer examples/tokenizer --tokenized-data examples/data/synthetic_llava_tokenized.jsonl --data examples/data/synthetic_llava.jsonl --epochs 0

# 3) Run training using the tokenized data and save model checkpoints every epoch
python -m examples.train_llava --tokenized-data examples/data/synthetic_llava_tokenized.jsonl --save examples/models/llava_model.safetensors --checkpoint-interval 1 --epochs 5

# 4) Resume training from the latest checkpoint (default checkpoint path is <save>.ckpt.safetensors)
python -m examples.train_llava --resume --checkpoint examples/models/llava_model.ckpt.safetensors --epochs 3
```

Tokenization & checkpoint workflow examples (Linux / macOS - Bash)
```
bash
source venv/bin/activate

# 1) Download an HF tokenizer (optional)
./scripts/download_tokenizer.sh bert-base-uncased examples/tokenizer

# 2) Regenerate both the original and tokenized dataset files
python -m examples.train_llava --force --tokenizer examples/tokenizer --tokenized-data examples/data/synthetic_llava_tokenized.jsonl --data examples/data/synthetic_llava.jsonl --epochs 0

# 3) Run training using the tokenized data and save model checkpoints every epoch
python -m examples.train_llava --tokenized-data examples/data/synthetic_llava_tokenized.jsonl --save examples/models/llava_model.safetensors --checkpoint-interval 1 --epochs 5

# 4) Resume training from checkpoint
python -m examples.train_llava --resume --checkpoint examples/models/llava_model.ckpt.safetensors --epochs 3
```

Automatic partial-checkpoint resume helper
-----------------------------------------

If a training process fails, `train_llava.py` will attempt to save a partial checkpoint in the same directory as the save path. For example, if you set `--save examples/models/llava_model.safetensors`, a partial checkpoint will be written with a filename pattern like `llava_model.ckpt.partial.1700000000.safetensors`.

You can automatically find the most recent checkpoint and resume training using the helper scripts:

Windows PowerShell:
```
# Find latest and resume
.\venv\Scripts\Activate.ps1
python scripts/find_latest_checkpoint.py --dir examples/models --ext .ckpt.safetensors
python -m examples.train_llava --resume --checkpoint $(python scripts/find_latest_checkpoint.py --dir examples/models --ext .ckpt.safetensors) --epochs 3
```

Linux (Bash):
```
# Find latest and resume
source venv/bin/activate
CKPT=$(python scripts/find_latest_checkpoint.py --dir examples/models --ext .ckpt.safetensors)
python -m examples.train_llava --resume --checkpoint "$CKPT" --epochs 3
```

Convenience wrapper scripts are included:
- `scripts/resume_train_from_latest.sh` — Bash wrapper to run training resumed from the latest checkpoint.
- `scripts/resume_train_from_latest.ps1` — PowerShell wrapper.

Makefile targets to resume training
---------------------------------
You can also use `make` for convenience to resume from the latest checkpoint:

Linux / macOS (Bash):
```
# Default: uses examples/models and .ckpt.safetensors
make resume_latest ARGS="--epochs 3"

# Custom path, pass through arguments
make resume_latest CKPT_DIR=examples/models CKPT_EXT=.ckpt.safetensors ARGS="--epochs 3 --batch 2"
```

Windows (PowerShell):
```
# Use the PowerShell-specific Make target which calls the PS1 wrapper
make resume_latest_win ARGS="--epochs 3"
```

Makefile training helpers
------------------------
You can use `make` to run training, including a PowerShell specific target for Windows that calls `scripts/run_train.ps1`.

Linux / macOS (Bash):
```
make train EPOCHS=5 BATCH=4 CHECKPOINT_INTERVAL=1 ARGS="--tokenizer examples/tokenizer --save examples/models/llava_model.safetensors"
```

Windows (PowerShell):
```
make train_win EPOCHS=5 BATCH=4 CHECKPOINT_INTERVAL=1 ARGS="--tokenizer examples/tokenizer --save examples/models/llava_model.safetensors"
```

Custom checkpoint path example
------------------------------
By default, the checkpoint path used for resume/periodic saving is derived from your `--save` argument by appending `.ckpt.safetensors` to the base name. Example:

```
python -m examples.train_llava --save examples/models/my_model.safetensors
# Implicit checkpoint path is examples/models/my_model.ckpt.safetensors
```

If you want to override the checkpoint path, provide `--checkpoint` explicitly:

```
# Set both save and checkpoint to custom locations (example paths)
python -m examples.train_llava --save examples/models/my_model.safetensors --checkpoint /tmp/checkpoints/model_a.ckpt.safetensors --epochs 3
```

This allows you to control where the framework writes checkpoints and how resume lookups are performed.

Full model vs toy model
-----------------------
The training script defaults to a toy configuration for fast local runs. If you'd like to build a larger `full` model, use the `--full-model` flag which will load `examples/llava_model_config.json` and use its values for d_model, d_ff, num_heads, depth, patch_size, and vocab_size.

Use a custom model config JSON with `--config`. The file should contain the following fields:
```
{
	"d_model": 32,
	"d_ff": 128,
	"num_heads": 4,
	"depth": 2,
	"patch_size": 8,
	"vocab_size": 256,
	"max_len": 512
}
```

Example - run with full model config:
```
python -m examples.train_llava --full-model --tokenized-data examples/data/synthetic_llava_tokenized.jsonl --epochs 1

# or use a custom config
python -m examples.train_llava --config examples/llava_model_config.json --epochs 1
```


More advanced setups (accelerators, larger datasets, distributed training) are out-of-scope for this minimal demo framework.
