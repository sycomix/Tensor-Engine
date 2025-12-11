#!/usr/bin/env bash
# Console guide wizard for Tensor Engine (Linux/macOS shell)
# Usage: bash scripts/te-guide.sh

set -euo pipefail

function pause() { read -rp "Press Enter to continue..."; }

function show_menu() {
    echo "Tensor Engine Console Guide"
    echo "================================="
    echo "1) Build (cargo)"
    echo "2) Python bindings & dev wheel (maturin)"
    echo "3) Run Python smoke test"
    echo "4) Run example script"
    echo "5) Prepare LLaVA dataset (examples/prepare_dataset.py)"
    echo "6) Train LLaVA minimal example (examples/train_llava.py)"
    echo "7) Generate from LLaVA minimal model (examples/generate_llava.py)"
    echo "8) Quick commands"
    echo "9) Preflight checks"
    echo "10) Show next.md (walkthrough)"
    echo "q) Quit"
}

function run_cmd() {
    echo "Running: $*"
    eval "$@"
}

function guide_build() {
    echo "Enter features to pass to cargo (e.g. 'openblas,python_bindings') or leave blank:"
    read -r features
    if [ -z "$features" ]; then
        run_cmd cargo build
    else
        run_cmd "cargo build --features '$features'"
    fi
}

function guide_python() {
    echo "Install maturin if missing and develop wheel. Enter features (e.g. 'python_bindings,with_tokenizers'):"
    read -r features
    if ! command -v maturin >/dev/null 2>&1; then
        echo "maturin not found. Do you want to install it via pip? [Y/n]"
        read -r yn
        if [ -z "$yn" ] || [[ "$yn" =~ ^[Yy]$ ]]; then
            if command -v pip >/dev/null 2>&1; then
                echo "Installing maturin via pip..."
                pip install --user maturin
            elif command -v python >/dev/null 2>&1; then
                echo "pip not found, trying python -m pip install..."
                python -m pip install --user maturin || true
            else
                echo "pip/python not found; cannot install maturin automatically." >&2
            fi
        else
            echo "Skipping maturin install."
        fi
    fi
    if [ -z "$features" ]; then
        CMD="maturin develop --release --bindings pyo3"
    else
        CMD="maturin develop --release --bindings pyo3 --features \"$features\""
    fi
    # attempt the direct maturin; if not found, fallback to `python -m maturin`
    if command -v maturin >/dev/null 2>&1; then
        run_cmd "$CMD"
    elif command -v python >/dev/null 2>&1; then
        echo "maturin not found in PATH; running via 'python -m maturin'"
        run_cmd "python -m maturin ${CMD#maturin }"
        # If maturin is still not available as a direct command, ask to temporarily add Python user scripts to PATH (e.g., ~/.local/bin)
        if ! command -v maturin >/dev/null 2>&1; then
            PY_SCRIPTS=$(python -c 'import site, os, sys; base = site.USER_BASE; print(os.path.join(base, "bin"))' 2>/dev/null || true)
            if [ -n "$PY_SCRIPTS" ] && [ -d "$PY_SCRIPTS" ]; then
                echo "It looks like maturin may be installed in: $PY_SCRIPTS"
                read -rp "Temporarily add '$PY_SCRIPTS' to PATH for this session? [y/N] " yn
                if [ -n "$yn" ] && [[ "$yn" =~ ^[Yy] ]]; then
                    export PATH="$PY_SCRIPTS:$PATH"
                    echo "Temporarily added $PY_SCRIPTS to PATH. Re-running maturin..."
                    if command -v maturin >/dev/null 2>&1; then
                        run_cmd "$CMD"
                    else
                        echo "maturin still not found; running via 'python -m maturin' fallback";
                        run_cmd "python -m maturin ${CMD#maturin }"
                    fi
                fi
            fi
        fi
    else
        echo "Cannot find maturin or python to run fallback. Please add maturin to PATH or use 'python -m maturin'" >&2
    fi
}

function guide_prepare() {
    echo "Preparing synthetic LLaVA dataset..."
    run_cmd "python examples/prepare_dataset.py"
}

function guide_train() {
    echo "Training minimal LLaVA example (1 epoch)..."
    run_cmd "python examples/train_llava.py --epochs 1 --batch 2"
}

function guide_generate() {
    echo "Generating from LLaVA minimal model..."
    run_cmd "python examples/generate_llava.py --prompt 'Describe the image 0' --steps 4"
}

function run_smoke_test() {
    python tests/python_smoke_test.py || true
}

function run_example() {
    echo "Enter example file (examples/load_model.py or examples/linear_regression.py or examples/train_nl_oob.py)"
    read -r example
    if [ -n "$example" ]; then
        run_cmd "python $example"
    fi
}

function quick_commands() {
    echo "Quick commands:";
    echo "  cargo build --features 'openblas,python_bindings'";
    echo "  maturin develop --release --features 'python_bindings'";
    echo "  cargo test --features 'openblas'";
}

while true; do
    show_menu
    read -r -p "Choice: " choice
    case $choice in
        1) guide_build; pause;
        2) guide_python; pause;
        3) run_smoke_test; pause;
        4) run_example; pause;
        5) guide_prepare; pause;
        6) guide_train; pause;
        7) guide_generate; pause;
        8) quick_commands; pause;
        9) preflight_checks; pause;
        10) show_next_steps; pause;
        q|Q) echo "Goodbye"; exit 0;;
        *) echo "Invalid choice."; pause;;
    esac
done

    function preflight_checks() {
        echo "Performing preflight checks..."
        local ok=0
        if command -v cargo >/dev/null 2>&1; then echo "cargo: present"; else echo "cargo: not found"; ok=1; fi
        if command -v python >/dev/null 2>&1; then echo "python: present"; else echo "python: not found"; ok=1; fi
        if command -v pip >/dev/null 2>&1; then echo "pip: present"; else echo "pip: not found"; fi
        if command -v maturin >/dev/null 2>&1; then echo "maturin: present"; else echo "maturin: not found"; fi
        if [ -z "${OPENBLAS_DIR:-}" ]; then echo "OPENBLAS_DIR: not set"; else echo "OPENBLAS_DIR: set -> $OPENBLAS_DIR"; fi
        if [ $ok -ne 0 ]; then echo "Some required commands are missing. See the README or the tool suggestions."; fi
    }

    function show_next_steps() {
        local path="$(dirname "$0")/../next.md"
        if [ -f "$path" ]; then
            if command -v less >/dev/null 2>&1; then
                less "$path"
            else
                cat "$path"
            fi
        else
            echo "next.md not found at $path"
        fi
    }
