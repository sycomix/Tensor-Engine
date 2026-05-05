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
$PYTHON_BRIDGE_DIR = Join-Path $TE_DIR "Assets\TensorEngine\PythonBridge"

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
function Invoke-CommandWithExitCode($command, $args) {
    $quotedArgs = $args | ForEach-Object { $_ -replace '"', '""' }
    $argString = $quotedArgs -join ' '
    $fullCmd = "cmd /c `"$command $argString`""
    Invoke-Expression $fullCmd
    return $LASTEXITCODE
}

# Main
Write-Section "Unity + Tensor-Engine Setup Script"

# Step 1: Check prerequisites
Write-Step "Checking prerequisites"

$prereqOk = $true
$prereqOk = (Test-Command "python3" "Python3") -and $prereqOk
if ($prereqOk) {
    $prereqOk = (Test-PythonVersion "python3") -and $prereqOk
}
$prereqOk = (Test-Command "pip3" "pip3") -and $prereqOk
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
    $exitCode = Invoke-CommandWithExitCode "git" @("pull", "origin", "main")
    if ($exitCode -eq 0) {
        $pullOk = $true
    } else {
        Write-Warn "git pull origin main failed (exit $exitCode). Trying master..."
        $exitCode = Invoke-CommandWithExitCode "git" @("pull", "origin", "master")
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

# Step 3: Build Rust Python bindings
Write-Step "Building Rust Python bindings"

Set-Location $TE_DIR

if (Test-Path "Cargo.toml") {
    Write-Info "Building Tensor-Engine (this may take a few minutes)..."
    $exitCode = Invoke-CommandWithExitCode "cargo" @("build", "--release")
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

# Step 4: Install Python dependencies
Write-Step "Installing Python dependencies"

$reqPath = Join-Path $PYTHON_BRIDGE_DIR "requirements.txt"
if (Test-Path $reqPath) {
    Write-Info "Installing Python dependencies..."
    $exitCode = Invoke-CommandWithExitCode "python" @("-m", "pip", "install", "-r", $reqPath, "--quiet")
    if ($exitCode -eq 0) {
        Write-Ok "Python dependencies installed"
    } else {
        Write-Fail "Python dependency installation failed (exit code $exitCode)."
        exit 1
    }
} else {
    Write-Warn "requirements.txt not found. Skipping Python deps."
    Write-Info "Manual installation required:"
    Write-Host "  python -m pip install flask numpy torch safetensors transformers" -ForegroundColor $YELLOW
}

# Step 5: Verify installation
Write-Step "Verifying installation"

$verifyScript = @'
import sys
print(f'Python version: {sys.version}')

modules = [
    ('flask', 'Flask web framework'),
    ('numpy', 'NumPy numerical library'),
    ('torch', 'PyTorch ML framework'),
    ('safetensors', 'SafeTensors format'),
    ('transformers', 'HuggingFace transformers'),
]

all_ok = True
for module, desc in modules:
    try:
        __import__(module)
        print(f'  [OK] {desc}')
    except ImportError:
        print(f'  [FAIL] {desc} not installed')
        all_ok = False

if all_ok:
    print()
    print('  [SUCCESS] All Python modules installed!')
else:
    print()
    print('  [WARNING] Some modules are missing.')
    print('  Run: python -m pip install flask numpy torch safetensors transformers')
    sys.exit(1)
'@

$exitCode = Invoke-CommandWithExitCode "python3" @("-c", $verifyScript)
if ($exitCode -ne 0) {
    Write-Fail "Python verification failed."
    exit 1
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
Write-Host "2. In Unity, open Package Manager and add the local package" -ForegroundColor $YELLOW
Write-Host "   (or use 'Add package from disk' in Package Manager)" -ForegroundColor $WHITE
Write-Host ""
Write-Host "3. Create a GameManager GameObject and add the MonoBrain component" -ForegroundColor $YELLOW
Write-Host ""
Write-Host "4. Attach NeuralAgent to your NPCs and configure model paths" -ForegroundColor $YELLOW
Write-Host ""
Write-Host "5. Run the examples:" -ForegroundColor $YELLOW
Write-Host "   - ExampleSceneSetup.cs for a complete demo" -ForegroundColor $WHITE
Write-Host "   - ExampleNPC.cs for basic NPC dialogue" -ForegroundColor $WHITE
Write-Host "   - ExampleProceduralDialogue.cs for multi-agent conversations" -ForegroundColor $WHITE
Write-Host ""
Write-Host "For more info, see:" -ForegroundColor $YELLOW
Write-Host "  $TE_DIR\Assets\TensorEngine\Docs\README.md" -ForegroundColor $WHITE
Write-Host "  $TE_DIR\Assets\TensorEngine\Examples\README.md" -ForegroundColor $WHITE
Write-Host ""
Write-Host "================================================" -ForegroundColor $BLUE
