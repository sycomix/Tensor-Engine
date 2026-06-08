# Unity-Tensor-Engine Setup Script (PowerShell)
# Sets up the Tensor-Engine Rust ML framework for Unity integration
# Place this in your Documents folder and run: .\Unity-TE-setup.ps1

$ErrorActionPreference = "Stop"

# Colors for output
$RED    = [ConsoleColor]::Red
$GREEN  = [ConsoleColor]::Green
$YELLOW = [ConsoleColor]::Yellow
$BLUE   = [ConsoleColor]::Blue
$WHITE  = [ConsoleColor]::White

# Directories
$SCRIPT_DIR = $PSScriptRoot
$TE_DIR     = Join-Path $SCRIPT_DIR "Tensor-Engine"

# If TE_DIR doesn't exist but we're already in a git repo, use current dir
if (-not (Test-Path (Join-Path $TE_DIR ".git"))) {
    if (Test-Path (Join-Path $SCRIPT_DIR ".git")) {
        $TE_DIR = $SCRIPT_DIR
    }
}

function Write-Section($text) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor $BLUE
    Write-Host "   $text" -ForegroundColor $BLUE
    Write-Host "================================================" -ForegroundColor $BLUE
    Write-Host ""
}

function Write-Step($text) {
    Write-Host "Step $text..." -ForegroundColor $YELLOW
    Write-Host ""
}

function Write-Ok($text) {
    Write-Host "[OK] $text" -ForegroundColor $GREEN
}

function Write-Fail($text) {
    Write-Host "[FAIL] $text" -ForegroundColor $RED
}

function Write-Warn($text) {
    Write-Host "[WARN] $text" -ForegroundColor $YELLOW
}

function Write-Info($text) {
    Write-Host "[INFO] $text" -ForegroundColor $Blue
}

# Function to check if a command exists
function Test-Command($cmdName, $displayName) {
    try {
        $null = Get-Command $cmdName -ErrorAction Stop
        $path = (Get-Command $cmdName).Source
        Write-Ok "$displayName found: $path"
        return $true
    } catch {
        Write-Fail "$displayName not found. Please install it first."
        return $false
    }
}

# Function to check Python version
function Test-PythonVersion($pythonCmd) {
    try {
        $versionStr = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        $parts = $versionStr -split '\.'
        $major = [int]$parts[0]
        $minor = [int]$parts[1]

        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
            Write-Fail "Python 3.8+ required. Found: $versionStr"
            return $false
        }
        Write-Ok "Python version: $versionStr"
        return $true
    } catch {
        Write-Fail "Could not determine Python version for '$pythonCmd'"
        return $false
    }
}

# Helper: run a command and return its exit code reliably
function Invoke-CommandWithExitCode($command, $commandArgs, $workingDir = $null) {
    $oldDir = $null
    if ($workingDir) {
        $oldDir = Get-Location
        Set-Location $workingDir
    }
    $exitCode = 0
    try {
        $output = & $command @commandArgs 2>&1
        if ($output) {
            $output | ForEach-Object { Write-Host $_ }
        }
        $exitCode = $LASTEXITCODE
    } catch {
        $exitCode = 1
    } finally {
        if ($oldDir) {
            Set-Location $oldDir
        }
    }
    return $exitCode
}

# Main
Write-Section "Unity + Tensor-Engine Setup Script"

# Step 1: Check prerequisites
Write-Step "Checking prerequisites"

$prereqOk = $true
$prereqOk = (Test-Command "cargo" "Rust/Cargo") -and $prereqOk
$prereqOk = (Test-Command "git" "Git") -and $prereqOk

if (-not $prereqOk) {
    Write-Fail "Prerequisites check failed. Please install missing tools and re-run."
    exit 1
}

# Check Unity (optional but recommended)
$unityFound = $false
try {
    $null = Get-Command "unity" -ErrorAction Stop
    $unityFound = $true
} catch { $null }
if (-not $unityFound) {
    try {
        $null = Get-Command "Unity" -ErrorAction Stop
        $unityFound = $true
    } catch { $null }
}
if ($unityFound) {
    Write-Ok "Unity found"
} else {
    Write-Warn "Unity not found in PATH. You'll need to point to your Unity installation manually."
}

# Step 2: Clone Tensor-Engine
Write-Step "Setting up Tensor-Engine"

if (Test-Path (Join-Path $TE_DIR ".git")) {
    Write-Info "Tensor-Engine already cloned. Updating..."
    Set-Location $TE_DIR
    $pullOk = $false
    $exitCode = Invoke-CommandWithExitCode "git" @("pull", "origin", "main") $TE_DIR
    if ($exitCode -eq 0) {
        $pullOk = $true
    } else {
        Write-Warn "git pull origin main failed (exit $exitCode). Trying master..."
        $exitCode = Invoke-CommandWithExitCode "git" @("pull", "origin", "master") $TE_DIR
        if ($exitCode -eq 0) {
            $pullOk = $true
        }
    }
    if (-not $pullOk) {
        Write-Fail "Failed to update Tensor-Engine. Check git output above."
        exit 1
    }
} else {
    Write-Info "Cloning Tensor-Engine..."
    $exitCode = Invoke-CommandWithExitCode "git" @("clone", "https://github.com/sycomix/Tensor-Engine.git", $TE_DIR)
    if ($exitCode -ne 0) {
        Write-Fail "git clone failed with exit code $exitCode."
        exit 1
    }
    Set-Location $TE_DIR
}

# Step 3: Build Rust engine binary and library
Write-Step "Building Rust engine and library"

Set-Location $TE_DIR

if (Test-Path "Cargo.toml") {
    Write-Info "Building Tensor-Engine engine binary (this may take a few minutes)..."
    $exitCode = Invoke-CommandWithExitCode "cargo" @("build", "--release", "--features", "opencl,cffi") $TE_DIR
    if ($exitCode -eq 0) {
        Write-Ok "Rust build successful"
    } else {
        Write-Fail "Rust build failed (exit code $exitCode). Check the output above."
        exit 1
    }
} else {
    Write-Fail "Cargo.toml not found. Is this the Tensor-Engine repo?"
    exit 1
}

# Step 4: Python bridge is no longer required
Write-Step "Python bridge check"
Write-Info "The Python Flask bridge has been replaced."
Write-Info "PythonBridgeService.cs now spawns engine.exe directly and communicates"
Write-Info "via the Rust engine's native HTTP server (OpenAI-compatible API with SSE streaming)."
Write-Info "No Python dependencies are needed for Unity integration."

# Step 5: Verify installation
Write-Step "Verifying installation"

$enginePath = Join-Path $TE_DIR "target\release\engine.exe"
if (Test-Path $enginePath) {
    Write-Ok "engine.exe found at: $enginePath"
} else {
    Write-Fail "engine.exe not found. Build may have failed."
    exit 1
}

$libPath = Join-Path $TE_DIR "target\release\tensor_engine.dll"
if (Test-Path $libPath) {
    Write-Ok "tensor_engine.dll found at: $libPath"
} else {
    Write-Warn "tensor_engine.dll not found (may not have been built)."
}

# Step 6: Unity integration instructions
Write-Section "Setup Complete!"

Write-Host "Next steps:" -ForegroundColor $GREEN
Write-Host ""
Write-Host "1. Copy the Unity package to your Unity project:" -ForegroundColor $YELLOW
$unityAssetsSrc = Join-Path $TE_DIR "Assets\TensorEngine"
$unityPackagesSrc = Join-Path $TE_DIR "Packages\unity-tensor-engine"
$unityAssetsDst = "$HOME\Documents\YourUnityProject\Assets"
$unityPackagesDst = "$HOME\Documents\YourUnityProject\Packages"
Write-Host "   Copy-Item -Recurse -Force `"$unityAssetsSrc`" `"$unityAssetsDst`"" -ForegroundColor $WHITE
Write-Host "   Copy-Item -Recurse -Force `"$unityPackagesSrc`" `"$unityPackagesDst`"" -ForegroundColor $WHITE
Write-Host ""
Write-Host "2. Copy engine.exe next to your Unity project (or set EnginePath in MonoBrain):" -ForegroundColor $YELLOW
Write-Host "   Copy-Item `"$enginePath`" `"$HOME\Documents\YourUnityProject\`"" -ForegroundColor $WHITE
Write-Host ""
Write-Host "3. In Unity, open Package Manager and add the local package" -ForegroundColor $YELLOW
Write-Host ""
Write-Host "4. Create a GameManager GameObject and add the MonoBrain component" -ForegroundColor $YELLOW
Write-Host "   Set ModelPath to your model directory containing config.json + safetensors" -ForegroundColor $WHITE
Write-Host ""
Write-Host "5. Attach NeuralAgent to your NPCs" -ForegroundColor $YELLOW
Write-Host ""
Write-Host "6. Run the scene — MonoBrain auto-starts engine.exe and discovers models" -ForegroundColor $YELLOW
Write-Host ""
Write-Host "For more info, see:" -ForegroundColor $YELLOW
Write-Host "  $TE_DIR\Assets\TensorEngine\Docs\README.md" -ForegroundColor $WHITE
Write-Host "  $TE_DIR\Assets\TensorEngine\Examples\README.md" -ForegroundColor $WHITE
Write-Host ""
Write-Host "================================================" -ForegroundColor $BLUE
