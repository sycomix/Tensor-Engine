param(
    [Parameter(Position = 0)]
    [string]$Features = 'python_bindings,with_tokenizers'
)

$ErrorActionPreference = 'Stop'

Write-Host "Setting up pretrain_project venv and building tensor_engine ($Features)"

# Create virtual environment
if (-not (Test-Path -Path .\venv)) {
    python -m venv venv
}

& .\venv\Scripts\python -m pip install --upgrade pip
& .\venv\Scripts\pip install -r .\requirements.txt

# Build the local repo's Python bindings into this venv.
# Repo root is two levels up from examples/pretrain_project.
Push-Location ..\..
try {
    Write-Host "Building tensor_engine via maturin develop --release --features '$Features'"
    & .\examples\pretrain_project\venv\Scripts\python -m maturin develop --release --features "$Features"
} finally {
    Pop-Location
}

Write-Host "Done. Activate with .\venv\Scripts\Activate.ps1 and run: python -m examples.train_pretrain_lm --help"
