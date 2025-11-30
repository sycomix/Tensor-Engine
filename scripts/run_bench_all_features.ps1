# Usage: .\run_bench_all_features.ps1 [-SetupOpenBlas]
param(
    [switch]$SetupOpenBlas
)

$ErrorActionPreference = 'Stop'

if ($SetupOpenBlas) {
    Write-Host "Setting up OpenBLAS (running scripts/setup_dev_repo.ps1)"
    # Ensure script exists
    if (Test-Path "./scripts/setup_dev_repo.ps1") {
        & .\scripts\setup_dev_repo.ps1
    } else {
        Write-Warning "No scripts/setup_dev_repo.ps1 found; skipping OpenBLAS automatic setup."
    }
}

# Set OpenBLAS dir if present in repo
$openblas_dir = Join-Path $PWD "OpenBLAS-0.3.30-x64-64"
if (Test-Path $openblas_dir) {
    $env:OPENBLAS_DIR = $openblas_dir
    Write-Host "Using OPENBLAS_DIR=$env:OPENBLAS_DIR"
} else {
    Write-Warning "OpenBLAS directory not found at $openblas_dir. If you want BLAS-backed benches, set OPENBLAS_DIR before running the script."
}

# The explicit features to enable that are comparable to --all-features but avoid python_bindings
$feature_set = "openblas,multi_precision,dtype_f16,dtype_bf16,dtype_f8"
Write-Host "Running benches with features: $feature_set"
cmd /c "cargo bench --features $feature_set --no-default-features"

# Next, try running with py_abi3 (no python required at build time) for python bindings bench coverage
Write-Host "Running benches with also py_abi3 (PYO3_NO_PYTHON=1)"
$env:PYO3_NO_PYTHON = "1"
$feature_set_abi3 = "openblas,multi_precision,dtype_f16,dtype_bf16,dtype_f8,py_abi3"
cmd /c "cargo bench --features $feature_set_abi3 --no-default-features"

Write-Host "Done running benches with both feature sets."

# (Optional) Run with --all-features explicitly (this may require platform-specific dev libs; run if you know your environment supports it)
Write-Host "If you want to try --all-features directly, run: cargo bench --all-features"
