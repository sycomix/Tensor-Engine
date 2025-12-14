# Console Guide Scripts

This folder contains small convenience scripts to help developers set up, build, and run examples for the Tensor Engine project.

- `te-guide.ps1`: PowerShell TUI for Windows (run with pwsh)
- `te-guide.sh`: Bash TUI for Linux/macOS

They are interactive wizards to help with the following tasks:
- Setup OpenBLAS (Windows)
- Build with Cargo features
- Install Python dev wheel with maturity (maturin)
- Run Python smoke tests
- Run example scripts
- Run example scripts (including the new minimal LLaVA examples: `examples/prepare_dataset.py`, `examples/train_llava.py`, `examples/generate_llava.py`)

Example: Prepare dataset, train, and generate
```pwsh
python examples/prepare_dataset.py --out examples/data/synthetic_llava.jsonl --count 16 --tokenizer bert-base-uncased
maturin develop --release --bindings pyo3 --features "python_bindings"
python examples/train_llava.py --epochs 1 --batch 2 --save examples/models/llava_model.safetensors
python examples/generate_llava.py --prompt "Describe the image 0" --model_path examples/models/llava_model.safetensors --steps 4
```

Usage (PowerShell):
```pwsh
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\te-guide.ps1
```

Usage (Linux / macOS shell):
```sh
bash scripts/te-guide.sh
```

Notes:
- The scripts automate common commands and present human-friendly options. They are not a replacement for CI workflow scripts and assume the user has developer toolchains installed.
- `te-guide.sh` is implemented with POSIX shell constructs (Bash recommended).
Additional features:
- Preflight checks: both scripts include a preflight check to validate the presence of `cargo`, `python`, `pip`, and `maturin`, and a quick check for `OPENBLAS_DIR`.
- Show next steps: a menu option to display `next.md` helps new contributors follow the proposed roadmap directly from the wizard.
- Optional guided install: if `maturin` is not installed, both scripts will prompt the user to optionally install it using `pip` (`pip install --user maturin`) or `python -m pip install --user maturin`).
- If the `maturin` binary is not available in the user's PATH after installation, both scripts will fall back to running `python -m maturin develop`.
- If the binary is installed in a user script folder (e.g. `~/.local/bin` on Unix or `%APPDATA%\Python\Scripts` on Windows) but not present in PATH, the scripts will offer to temporarily add it to the current shell session PATH to enable direct `maturin` invocation.

New developer utilities
- `scripts/add_as_any_mut.py`: helper script to add missing `as_any_mut` methods to `impl Module for` blocks (useful when mass-updating module impls). Use with care and run the verifier after changes.
- `scripts/verify_as_any_mut.py`: verification script to ensure all `impl Module` blocks include `as_any_mut` and that no `impl Operation` block contains the method; suitable to run in CI.
- `ci/verify_as_any_mut.sh` / `ci/verify_as_any_mut.ps1`: small cross-platform wrappers useful for CI integration. (See `tests/as_any_mut_verification.rs` which runs the verifier as part of `cargo test` when Python is available.)

Security note: the scripts will only attempt to install `maturin` upon explicit user consent. Review commands and environment before allowing automatic installs and consider installing in a virtual environment if preferred.

