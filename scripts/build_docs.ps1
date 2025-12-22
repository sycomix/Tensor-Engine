# Build the docs locally (Windows PowerShell)
# Usage: .\scripts\build_docs.ps1 [-Clean]
param(
    [switch]$Clean
)

Write-Host "Building Tensor Engine docs..."

# Ensure pipenv/venv optional - use pip
Write-Host "Installing doc dependencies (mkdocs, mkdocs-material) if missing..."
& python -m pip install --upgrade pip setuptools
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip upgrade failed, continuing"
}
& python -m pip install mkdocs mkdocs-material pymdown-extensions -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip install of mkdocs failed; ensure network or install them manually"
}

$mkdocsCmd = "python -m mkdocs build --clean -d site"
if ($Clean) { $mkdocsCmd = "$mkdocsCmd --clean" }

Write-Host "Running: $mkdocsCmd"
& python -m mkdocs build --clean -d site

if ($LASTEXITCODE -ne 0) {
    Write-Error "mkdocs build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Docs built into ./site/"
