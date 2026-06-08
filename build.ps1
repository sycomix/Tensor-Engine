param(
    [string]$Target = "",
    [switch]$DebugBuild = $false,
    [switch]$Menu = $false
)

$ReleaseFlag = if (-not $DebugBuild) { "--release" } else { "" }

# All features except with_tch (libtorch DLLs not on PATH)
# cffi is included and works — the VS 2026 warning is cosmetic
$ENGINE_FEATURES = @(
    "compat","safe_tensors","with_tokenizers","multi_precision",
    "openblas","opencl","backend_cuda","cffi",
    "async_ops","server","vision","audio","backend_wgpu",
    "grpc_server","metrics","audio_fft",
    "dtype_f16","dtype_bf16","hf_compat","python_bindings",
    "native_tokenizer","quantized","dtype_f8",
    "parallel_io","distributed","py_abi3"
) -join ","

function Build-Engine {
    param([switch]$Full)
    if ($Full) {
        Write-Host "Building engine (all features)..." -ForegroundColor Green
        Write-Host "  Note: needs libtorch.dll (install from pytorch.org)" -ForegroundColor Yellow
        cargo build $ReleaseFlag --bin engine --all-features
    } else {
        Write-Host "Building engine..." -ForegroundColor Green
        Write-Host "  All features except with_tch (libtorch DLLs)" -ForegroundColor DarkGray
        cargo build $ReleaseFlag --bin engine --features $ENGINE_FEATURES
    }
}

function Build-Lib {
    Write-Host "Building lib (all features)..." -ForegroundColor Green
    Write-Host "  Note: --lib flag keeps engine.exe untouched" -ForegroundColor Yellow
    cargo build $ReleaseFlag --lib --all-features
}

function Build-Rllama {
    Write-Host "Building rllama..." -ForegroundColor Green
    cargo build $ReleaseFlag --bin rllama --features $ENGINE_FEATURES
}

function Show-Menu {
    Clear-Host
    Write-Host "Tensor Engine Build Menu" -ForegroundColor Cyan
    $mode = if ($ReleaseFlag) { "release" } else { "debug" }
    Write-Host "Mode: $mode`n" -ForegroundColor Yellow
    Write-Host "1) engine     (all feats except with_tch - works)"
    Write-Host "2) engine-full (everything - needs libtorch DLL)"
    Write-Host "3) lib        (all features, for Python)"
    Write-Host "4) rllama"
    Write-Host "5) lib + engine"
    Write-Host "q) quit"
    Write-Host ""
    $choice = Read-Host "Choice"
    return $choice
}

function Show-Result {
    if ($global:lastExitCode -eq 0) { Write-Host "`nOK" -ForegroundColor Green }
    else { Write-Host "`nFAILED" -ForegroundColor Red }
    Write-Host ""; pause
}

if ($Menu -or (-not $Target)) {
    while ($true) {
        switch (Show-Menu) {
            "1" { Build-Engine; Show-Result }
            "2" { Build-Engine -Full; Show-Result }
            "3" { Build-Lib; Show-Result }
            "4" { Build-Rllama; Show-Result }
            "5" { Build-Lib; Build-Engine; Show-Result }
            "q" { exit 0 }
        }
    }
    exit
}

switch ($Target.ToLower()) {
    "engine" { Build-Engine }
    "engine-full" { Build-Engine -Full }
    "rllama" { Build-Rllama }
    "lib" { Build-Lib }
    "all" { Build-Lib; Build-Engine }
    default {
        Write-Host "Usage: .\build.ps1 -Target <engine|engine-full|rllama|lib|all> [-DebugBuild] [-Menu]"
        Write-Host ""
        Write-Host "  engine       - all feats except with_tch (runs now)"
        Write-Host "  engine-full  - all features (needs libtorch on PATH)"
        Write-Host "  lib          - all features for Python bindings"
        Write-Host "  rllama       - rllama.exe"
        Write-Host "  all          - lib + engine"
    }
}
