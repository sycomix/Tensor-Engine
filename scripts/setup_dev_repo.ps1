Param(
    [switch]$PersistUserEnvironment = $false
)

# Setup script for OpenBLAS and environment preparation on Windows
try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
    $repoRoot = Resolve-Path (Join-Path $scriptDir "..")
    $openblasDefault = Join-Path $repoRoot "OpenBLAS-0.3.30-x64-64"
    if (-not (Test-Path $openblasDefault)) {
        Write-Warning "Default OpenBLAS path not found: $openblasDefault"
        Write-Host "Please set the 'OPENBLAS_DIR' environment variable manually or install OpenBLAS and rerun this script."
        exit 1
    }
    $openblasResolved = Resolve-Path $openblasDefault
    $openblasPath = $openblasResolved.Path

    # Set session env var for this PowerShell session
    $env:OPENBLAS_DIR = $openblasPath
    # Prepend bin path to PATH for this session
    $openblasBin = Join-Path $openblasPath 'bin'
    if (-not (Test-Path $openblasBin)) {
        Write-Warning "OpenBLAS bin directory not found: $openblasBin"
    }
    else {
        $env:PATH = "$openblasBin;$env:PATH"
    }

    Write-Host "OPENBLAS_DIR set to: $env:OPENBLAS_DIR"
    Write-Host "Added to PATH (session): $openblasBin"

    if ($PersistUserEnvironment) {
        # Persist user environment variables (requires new shells to pick up)
        [Environment]::SetEnvironmentVariable('OPENBLAS_DIR', $openblasPath, 'User')
        $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
        if ($userPath -notlike "*$openblasBin*") {
            $newUserPath = "$openblasBin;$userPath"
            [Environment]::SetEnvironmentVariable('PATH', $newUserPath, 'User')
            Write-Host "Persisted $openblasBin to user's PATH (requires reopening your shell)."
        }
        else {
            Write-Host "OpenBLAS bin already present in user's PATH." 
        }
    }

    # Quick runtime check for DLL
    $dllPath = Join-Path $openblasBin 'libopenblas.dll'
    if (Test-Path $dllPath) {
        Write-Host "Found OpenBLAS DLL: $dllPath"
    }
    else {
        Write-Warning "Could not find libopenblas.dll. Build may succeed but run may fail unless the DLL is installed on PATH."
    }
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}

Write-Host "OpenBLAS dev setup complete."
