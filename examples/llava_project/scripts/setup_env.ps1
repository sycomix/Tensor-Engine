param(
    [Parameter(Position = 0)]
    [string]$Action = 'full'
)

# PowerShell setup script for Windows
Write-Host "Running setup: $Action"

$ErrorActionPreference = 'Stop'

# Create virtual environment
if (-not (Test-Path -Path .\venv)) {
    python -m venv venv
}

& .\venv\Scripts\python -m pip install --upgrade pip
& .\venv\Scripts\pip install -r ..\requirements.txt

# Clone and build Tensor-Engine if requested
if ($Action -eq 'full' -or $Action -eq 'build') {
    if (-not (Test-Path -Path .\..\third_party)) {
        New-Item -ItemType Directory -Path .\..\third_party | Out-Null
    }
    Push-Location .\..\third_party
    if (-not (Test-Path -Path .\Tensor-Engine)) {
        git clone https://github.com/sycomix/Tensor-Engine.git
    }
    Push-Location .\Tensor-Engine
    # Ensure maturin/rust toolchain installed. The user should already have Rust toolchain on Windows.
    & ..\..\venv\Scripts\pip install maturin
    Write-Host "Building tensor_engine Python package via maturin develop (this can take a few minutes)."
    & ..\..\venv\Scripts\python -m maturin develop --release
    Pop-Location
    Pop-Location
}

Write-Host "Setup done. Activate the environment with .\venv\Scripts\Activate.ps1 and run: python -m examples.train_llava"
