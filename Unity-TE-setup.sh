#!/usr/bin/env bash
# Unity-Tensor-Engine Setup Script
# Sets up the Tensor-Engine Rust ML framework for Unity integration
# Place this in your Documents folder and run: bash Unity-TE-setup.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_DIR="${SCRIPT_DIR}/Tensor-Engine"
PYTHON_BRIDGE_DIR="${TE_DIR}/Assets/TensorEngine/PythonBridge"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Unity + Tensor-Engine Setup Script${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}[FAIL] $2 not found. Please install it first.${NC}"
        return 1
    fi
    echo -e "${GREEN}[OK] $2 found: $(command -v $1)${NC}"
    return 0
}

# Function to check Python version
check_python_version() {
    local python_cmd="$1"
    local version
    version=$($python_cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    local major=${version%%.*}
    local minor=${version##*.}
    
    if [[ $major -lt 3 ]] || { [[ $major -eq 3 ]] && [[ $minor -lt 8 ]]; }; then
        echo -e "${RED}[FAIL] Python 3.8+ required. Found: $version${NC}"
        return 1
    fi
    echo -e "${GREEN}[OK] Python version: $version${NC}"
    return 0
}

# Step 1: Check prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"
echo ""

check_command "python3" "Python3"
check_command "pip3" "pip3"
check_command "cargo" "Rust/Cargo"
check_command "git" "Git"

# Check Unity (optional but recommended)
if command -v unity &> /dev/null || command -v Unity &> /dev/null; then
    echo -e "${GREEN}[OK] Unity found${NC}"
else
    echo -e "${YELLOW}[WARN] Unity not found in PATH. You'll need to point to your Unity installation manually.${NC}"
fi

echo ""

# Step 2: Clone Tensor-Engine
echo -e "${YELLOW}Step 2: Setting up Tensor-Engine...${NC}"
echo ""

if [[ -d "${TE_DIR}/.git" ]]; then
    echo -e "${YELLOW}[INFO] Tensor-Engine already cloned. Updating...${NC}"
    cd "${TE_DIR}"
    git pull origin main || git pull origin master
else
    echo -e "${BLUE}[INFO] Cloning Tensor-Engine...${NC}"
    git clone https://github.com/sycomix/Tensor-Engine.git "${TE_DIR}"
    cd "${TE_DIR}"
fi

echo ""

# Step 3: Build Rust Python bindings
echo -e "${YELLOW}Step 3: Building Rust Python bindings...${NC}"
echo ""

cd "${TE_DIR}"

if [[ -f "Cargo.toml" ]]; then
    echo -e "${BLUE}[INFO] Building Tensor-Engine (this may take a few minutes)...${NC}"
    cargo build --release 2>&1 | tail -5
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}[OK] Rust build successful${NC}"
    else
        echo -e "${RED}[FAIL] Rust build failed. Check the output above.${NC}"
        exit 1
    fi
else
    echo -e "${RED}[FAIL] Cargo.toml not found. Is this the Tensor-Engine repo?${NC}"
    exit 1
fi

echo ""

# Step 4: Install Python dependencies
echo -e "${YELLOW}Step 4: Installing Python dependencies...${NC}"
echo ""

if [[ -f "${PYTHON_BRIDGE_DIR}/requirements.txt" ]]; then
    echo -e "${BLUE}[INFO] Installing Python dependencies...${NC}"
    pip3 install -r "${PYTHON_BRIDGE_DIR}/requirements.txt" --quiet
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}[OK] Python dependencies installed${NC}"
    else
        echo -e "${RED}[FAIL] Python dependency installation failed.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[WARN] requirements.txt not found. Skipping Python deps.${NC}"
    echo -e "${YELLOW}[INFO] Manual installation required:${NC}"
    echo -e "  pip3 install flask numpy safetensors tokenizers"
fi

echo ""

# Step 5: Verify installation
echo -e "${YELLOW}Step 5: Verifying installation...${NC}"
echo ""

python3 -c "
import sys
print(f'Python version: {sys.version}')

modules = [
    ('flask', 'Flask web framework'),
    ('numpy', 'NumPy numerical library'),
    ('safetensors', 'SafeTensors format'),
    ('tokenizers', 'Hugging Face tokenizers'),
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
    print('  Run: pip3 install flask numpy safetensors tokenizers')
    sys.exit(1)
"

echo ""

# Step 6: Unity integration instructions
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Setup Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo ""
echo "1. Copy the Unity package to your Unity project:"
echo "   cp -r ${TE_DIR}/Assets/TensorEngine /path/to/your-unity-project/Assets/"
echo "   cp -r ${TE_DIR}/Packages/unity-tensor-engine /path/to/your-unity-project/Packages/"
echo ""
echo "2. In Unity, open Package Manager and add the local package"
echo "   (or use 'Add package from disk' in Package Manager)"
echo ""
echo "3. Create a GameManager GameObject and add the MonoBrain component"
echo ""
echo "4. Attach NeuralAgent to your NPCs and configure model paths"
echo ""
echo "5. Run the examples:"
echo "   - ExampleSceneSetup.cs for a complete demo"
echo "   - ExampleNPC.cs for basic NPC dialogue"
echo "   - ExampleProceduralDialogue.cs for multi-agent conversations"
echo ""
echo -e "${YELLOW}For more info, see:${NC}"
echo "  ${TE_DIR}/Assets/TensorEngine/Docs/README.md"
echo "  ${TE_DIR}/Assets/TensorEngine/Examples/README.md"
echo ""
echo -e "${BLUE}================================================${NC}"
