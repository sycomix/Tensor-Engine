param(
    [Parameter(Position = 0)]
    [string]$Features = 'python_bindings,with_tokenizers'
)

$ErrorActionPreference = 'Stop'

Write-Host "Setting up finetune_project venv and building tensor_engine ($Features)"

if (-not (Test-Path -Path .\venv)) {
    python -m venv venv
}

& .\venv\Scripts\python -m pip install --upgrade pip
& .\venv\Scripts\pip install -r .\requirements.txt

Push-Location ..\..
try {
    Write-Host "Building tensor_engine via maturin develop --release --features '$Features'"
    & .\examples\finetune_project\venv\Scripts\python -m maturin develop --release --features "$Features"
} finally {
    Pop-Location
}

Write-Host "Done. Activate with .\venv\Scripts\Activate.ps1 and run: python -m examples.train_finetune_lm --help"
