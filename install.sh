#!/bin/bash
# Tensor Engine Build & Install Script
# Version: 0.4.0

set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Tensor Engine v0.3.1 Build & Install               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. Environment Setup
# ------------------------------------------------------------------------------
echo -e "${BLUE}[1/5] Setting up build environment...${NC}"

# Setup temporary directories to avoid permission issues in some environments
export TMPDIR="/tmp/tensor_engine_build_tmp"
export CARGO_TARGET_DIR="/tmp/tensor_engine_build_target"
mkdir -p "$TMPDIR"
mkdir -p "$CARGO_TARGET_DIR"

echo "   Temporary Build Dir: $TMPDIR"
echo "   Cargo Target Dir:    $CARGO_TARGET_DIR"

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Error: Python not found.${NC}"
    exit 1
fi
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "   Detected Python:     $PYTHON_VERSION"

# Check for Maturin
if ! $PYTHON_CMD -m maturin --version &> /dev/null; then
    echo -e "${BLUE}   maturin not found, installing via pip...${NC}"
    $PYTHON_CMD -m pip install maturin
fi

# Suggest pip upgrade if outdated
PIP_VERSION=$($PYTHON_CMD -m pip --version 2>&1 | awk '{print $2}')
echo "   Pip version: $PIP_VERSION"

# 2. Testing
# ------------------------------------------------------------------------------
echo -e "${BLUE}[2/5] Running verification tests...${NC}"

# Update Cargo.lock to resolve patch version mismatches
echo "   Updating dependencies..."
cargo update

# Run specific new feature tests
echo "   Running MoE tests..."
cargo test --test moe_test --release
echo "   Running Embedding tests..."
cargo test --test embedding_test --release

# 3. Building
# ------------------------------------------------------------------------------
echo -e "${BLUE}[3/5] Building release wheel...${NC}"

# Create dist directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/dist"

# Define feature flags
FEATURES="backend_metal,safe_tensors,python_bindings,async_ops,vision,quantized,distributed"

echo "   Features: $FEATURES"
$PYTHON_CMD -m maturin build --release --features "$FEATURES" --out "$SCRIPT_DIR/dist"

# Find the generated wheel
WHEEL_PATH=$(ls -t "$SCRIPT_DIR/dist"/tensor_engine-*.whl | head -n1)

if [ ! -f "$WHEEL_PATH" ]; then
    echo -e "${RED}❌ Error: Wheel build failed or not found.${NC}"
    exit 1
fi
echo "   Built: $WHEEL_PATH"

# 4. Installation
# ------------------------------------------------------------------------------
echo -e "${BLUE}[4/5] Installing package...${NC}"
$PYTHON_CMD -m pip install --force-reinstall "$WHEEL_PATH"

# 5. Verification
# ------------------------------------------------------------------------------
echo -e "${BLUE}[5/5] Verifying installation...${NC}"
if $PYTHON_CMD -c "import tensor_engine; print('✅ Successfully imported tensor_engine')" 2>/dev/null; then
    echo -e "${GREEN}   ✓ Import verification passed${NC}"
else
    echo -e "${RED}   ✗ Import verification failed${NC}"
    echo -e "${RED}❌ Error: Installation verification failed.${NC}"
    exit 1
fi

# Display package info
PACKAGE_INFO=$($PYTHON_CMD -c "import tensor_engine; print(f'Version: {getattr(tensor_engine, \"__version__\", \"unknown\")}')" 2>/dev/null || echo "Version: unknown")
echo "   $PACKAGE_INFO"

echo ""
echo -e "${GREEN}✨ Build and Installation Complete! ✨${NC}"
echo ""
echo "New Features Available:"
echo "  • Mixture of Experts (MoE)"
echo "  • Sparse & Adaptive Embeddings"
echo "  • Metal Acceleration"
echo ""
echo -e "${BLUE}💡 Tip: Consider upgrading pip for better performance: ${PYTHON_CMD} -m pip install --upgrade pip${NC}"
echo ""

