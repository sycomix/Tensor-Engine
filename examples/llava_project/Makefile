# Minimal Makefile for building dependencies and running training

ENV=venv
PY=python
PIP=$(ENV)/Scripts/pip
ACTIVATE_PS=./$(ENV)/Scripts/Activate.ps1
CKPT_DIR=examples/models
CKPT_EXT=.ckpt.safetensors
ARGS=

.PHONY: setup build_tengine train clean resume_latest resume_latest_win

setup:
	python -m venv $(ENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

build_tengine:
	# Clone Tensor-Engine and build Python bindings (maturin). The script will handle idempotence.
	./scripts/setup_env.ps1 build

EPOCHS ?= 3
BATCH ?= 4
CHECKPOINT_INTERVAL ?= 1

train:
	$(PY) -m examples.train_llava --epochs $(EPOCHS) --batch $(BATCH) --checkpoint-interval $(CHECKPOINT_INTERVAL) $(ARGS)

resume_latest:
	# find and resume from latest checkpoint (Bash)
	./scripts/resume_train_from_latest.sh --dir $(CKPT_DIR) --ext $(CKPT_EXT) $(ARGS)

resume_latest_win:
	# find and resume from latest checkpoint (PowerShell)
	powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/resume_train_from_latest.ps1 -Dir $(CKPT_DIR) -Ext $(CKPT_EXT) -- $(ARGS)

train_win:
	# Runs training using PowerShell runner (Windows). Use ARGS to pass extra args (e.g., --tokenizer, --save)
	powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/run_train.ps1 -Epochs $(EPOCHS) -Batch $(BATCH) -CheckpointInterval $(CHECKPOINT_INTERVAL) -- $(ARGS)

clean:
	rm -rf $(ENV) build dist *.egg-info
	rm -rf third_party/Tensor-Engine
