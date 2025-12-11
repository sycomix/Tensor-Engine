<#
.SYNOPSIS
  Console guide wizard for Tensor Engine (PowerShell)

DESCRIPTION
  Interactive step-by-step guide to set up the workspace, build features, install Python wheel,
  run smoke tests and examples. Designed for Windows PowerShell (pwsh.exe).

USAGE
  Open PowerShell and run:
    pwsh scripts\te-guide.ps1
#>

# Basic helper: read single keypress
function Get-KeyPress() {
    $key = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    return $key.Character
}

function Write-Menu {
    param([string[]]$options)
    for ($i = 0; $i -lt $options.Length; $i++) {
        Write-Host "[$($i+1)] $($options[$i])"
    }
    # Removed static numbered entries - the caller will provide these as options
    Write-Host "[Q] Quit"
}

function Invoke-CommandLine {
    param([string]$cmd)
    Write-Host "Running: $cmd" -ForegroundColor Cyan
    try {
        Invoke-Expression $cmd
    } catch {
        Write-Host "Command failed: $_" -ForegroundColor Red
    }
}

function Test-CommandExists {
    param([string]$exe)
    $found = Get-Command $exe -ErrorAction SilentlyContinue
    if ($null -eq $found) { return $false } else { return $true }
}

function Test-Preflight {
    Write-Host "Performing preflight checks..." -ForegroundColor Cyan
    $haveCargo = Test-CommandExists -exe cargo
    $havePython = Test-CommandExists -exe python
    $havePip = Test-CommandExists -exe pip
    $haveMaturin = Test-CommandExists -exe maturin
    $openblasSet = $false
    if ($env:OPENBLAS_DIR) { $openblasSet = $true }

    Write-Host "Tool  | Present" -ForegroundColor Gray
    Write-Host "cargo  : $haveCargo"
    Write-Host "python : $havePython"
    Write-Host "pip    : $havePip"
    Write-Host "maturin: $haveMaturin" 
    Write-Host "OPENBLAS_DIR env: $openblasSet"

    if (-not $haveCargo) { Write-Host "cargo not found. Please install Rust toolchain: https://rustup.rs" -ForegroundColor Yellow }
    if (-not $havePython) { Write-Host "python not found. Please install Python 3.11+" -ForegroundColor Yellow }
    if (-not $havePip -and $havePython) { Write-Host "pip not found. Ensure pip is available for Python." -ForegroundColor Yellow }
    if (-not $haveMaturin -and $havePip) { Write-Host "maturin not found. You can install it with: pip install maturin" -ForegroundColor Yellow }
    if (-not $openblasSet) { Write-Host "OPENBLAS_DIR not set; on Windows, run scripts\setup_dev_repo.ps1 or set OPENBLAS_DIR manually if you plan to use openblas feature." -ForegroundColor Yellow }

    Write-Host "Preflight checks done." -ForegroundColor Green
}

function Get-OpenBLASInfo {
    Write-Host "OpenBLAS setup for Windows" -ForegroundColor Yellow
    Write-Host "The repo includes a prebuilt OpenBLAS under OpenBLAS-0.3.30-x64-64." -ForegroundColor Green
    Write-Host "You can configure OPENBLAS_DIR and PATH for the current session with the following script:" -ForegroundColor Green
    Write-Host "  scripts\setup_dev_repo.ps1 -PersistUserEnvironment" -ForegroundColor Gray
    if ((Get-Host).UI.RawUI.KeyAvailable) { Read-Host -Prompt 'Press Enter to continue' } else { Get-KeyPress | Out-Null }
}

function Get-PythonUserScriptsPath {
    if (-not (Test-CommandExists -exe python)) { return $null }
    try {
        $pyCmd = 'import site, os, sys; base = site.USER_BASE; folder = "Scripts" if sys.platform.startswith("win") else "bin"; print(os.path.join(base, folder))'
        $out = & python -c $pyCmd 2>$null
        if ($out) { return $out.Trim() } else { return $null }
    } catch {
        return $null
    }
}

function Start-Build {
    param([string]$features)
    $farg = $features
    if ([string]::IsNullOrWhiteSpace($farg)) {
        $farg = ""
    } else {
        # ensure features is wrapped in single quotes for safe shell parsing
        $farg = " --features '$farg'"
    }
    Write-Host "Building with features:$features" -ForegroundColor Yellow
    Invoke-CommandLine "cargo build$farg"
}

function Start-PythonBuild {
    $featuresInput = Read-Host -Prompt "Enter desired features (comma separated), e.g. 'python_bindings,openblas,with_tokenizers,safe_tensors'"
    # sanitize spaces and leave comma-separated features as-is
    $featuresInput = $featuresInput -replace '\s+',''
    if ([string]::IsNullOrWhiteSpace($featuresInput)) {
        $cmd = "maturin develop --release --bindings pyo3"
    } else {
        $cmd = "maturin develop --release --bindings pyo3 --features '$featuresInput'"
    }
    Write-Host "Installing Python dev wheel (maturin) -> $cmd" -ForegroundColor Green
    # If maturin not present, prompt the user to install it
    if (-not (Test-CommandExists -exe maturin)) {
        $install = Read-Host -Prompt "maturin not found. Install now using pip? (Y/n)"
        if (($install -eq '') -or ($install -match '^[Yy]$')) {
            if (Test-CommandExists -exe pip) {
                Invoke-CommandLine "pip install maturin"
            } elseif (Test-CommandExists -exe python) {
                Invoke-CommandLine "python -m pip install maturin"
            } else {
                Write-Host "pip/python not found; cannot install maturin automatically." -ForegroundColor Yellow
            }
        } else {
            Write-Host "Skipping maturin install. You can still run maturin commands if installed manually." -ForegroundColor Yellow
        }
    }
    # Try direct runtimes; if the 'maturin' binary isn't found, fall back to python -m maturin
    try {
        Invoke-CommandLine $cmd
    } catch {
        Write-Host "Direct 'maturin' invocation failed. Attempting fallback: python -m maturin" -ForegroundColor Yellow
        $pyCmd = $cmd -replace '^maturin', 'python -m maturin'
        Invoke-CommandLine $pyCmd
    }

    # If maturin binary wasn't found (installed to user site packages but not on PATH), offer to add Python user scripts path to PATH for this session and re-run
    if (-not (Test-CommandExists -exe maturin)) {
        $scriptsPath = Get-PythonUserScriptsPath
        if ($scriptsPath -and (Test-Path $scriptsPath)) {
            $choice = Read-Host -Prompt "The 'maturin' executable wasn't found on PATH. Temporarily add '$scriptsPath' to PATH for this session? (y/N)"
            if ($choice -match '^[Yy]') {
                $env:PATH = "$scriptsPath;$env:PATH"
                Write-Host "Temporarily added '$scriptsPath' to PATH for this session. Re-running build..." -ForegroundColor Green
                try { Invoke-CommandLine $cmd } catch { Write-Host "Re-run failed; trying fallback: python -m maturin" -ForegroundColor Yellow; $pyCmd = $cmd -replace '^maturin', 'python -m maturin'; Invoke-CommandLine $pyCmd }
            }
        }
    }
}

function Start-SmokeTests {
    Write-Host "Running smoke test: tests/python_smoke_test.py" -ForegroundColor Yellow
    Invoke-CommandLine "python tests/python_smoke_test.py"
}

function Start-Examples {
    Write-Host "Examples available in examples/" -ForegroundColor Yellow
    Write-Host "Select an example to run (enter exact command)." -ForegroundColor Green
    Write-Host "Options: examples/load_model.py, examples/linear_regression.py, examples/train_nl_oob.py" -ForegroundColor Gray
    $ex = Read-Host -Prompt "Example to run (file), or 'none'"
    if ($ex -and $ex -ne "none") {
        $cmd = "python $ex"
        Invoke-CommandLine $cmd
    }
}

function Start-Guide {
    Write-Host "Welcome to the Tensor Engine Console Guide" -ForegroundColor Magenta
    while ($true) {
        Write-Menu -options @(
            "OpenBLAS setup (Windows)",
            "Build (Cargo)",
            "Python bindings & dev wheel (maturin)",
            "Run Python smoke test",
            "Run example script",
            "Prepare LLaVA dataset (examples/prepare_dataset.py)",
            "Train LLaVA minimal example (examples/train_llava.py)",
            "Generate from LLaVA minimal model (examples/generate_llava.py)",
            "Preflight checks",
            "Show Next Steps (next.md)",
            "Show quick commands"
        )
        $choice = Read-Host -Prompt "Enter choice number"
        switch ($choice.ToLower()) {
            "1" { Get-OpenBLASInfo }
            "2" {
                $f = Read-Host -Prompt "Enter features (comma separated) or leave blank"
                Start-Build $f
            }
            "3" { Start-PythonBuild }
            "4" { Start-SmokeTests }
            "5" { Start-Examples }
            "6" { Invoke-CommandLine "python examples/prepare_dataset.py" }
            "7" { Invoke-CommandLine "python examples/train_llava.py --epochs 1 --batch 2" }
            "8" { Invoke-CommandLine "python examples/generate_llava.py --prompt 'Describe the image 0' --steps 4" }
            "9" { Test-Preflight }
            "10" { Get-NextSteps }
            "11" {
                Write-Host "Quick commands:" -ForegroundColor Green
                Write-Host "  cargo build --features 'openblas,python_bindings'"
                Write-Host "  maturin develop --release --features 'python_bindings'"
                Write-Host "  cargo test --features 'openblas'"
            }
            { $_ -match 'q|quit|exit' } { break }
            default { Write-Host "Unknown option" -ForegroundColor Red }
        }
    }
    Write-Host "Exiting guide. Happy hacking!" -ForegroundColor Cyan
}

# Show 'next.md' content into the console
function Get-NextSteps {
    $path = Join-Path $PSScriptRoot "..\next.md"
    if (Test-Path $path) {
        Write-Host "Opening next.md (press Enter to scroll page by page)..." -ForegroundColor Green
        $content = Get-Content $path
        $pageSize = 20
        for ($i=0; $i -lt $content.Count; $i += $pageSize) {
            $block = $content[$i..([math]::Min($i + $pageSize - 1, $content.Count - 1))]
            $block -join "`n" | Write-Host
            if ($i + $pageSize -lt $content.Count) { Read-Host -Prompt "--More-- Press Enter to continue or 'q' to quit" | Out-Null; }
        }
    } else {
        Write-Host "no next.md found at: $path" -ForegroundColor Yellow
    }
}

# Entry
Test-Preflight
Start-Guide
