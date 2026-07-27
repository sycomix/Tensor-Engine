#!/bin/bash
# Tensor Engine Build & Install Script
# Version: 1.0.0-beta.1

set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Tensor Engine v1.0.0-beta.1 Build & Install            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. Environment Setup
# ------------------------------------------------------------------------------
echo -e "${BLUE}[1/4] Setting up build environment...${NC}"

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

# 2. Building
# ------------------------------------------------------------------------------
echo -e "${BLUE}[2/4] Building with all features...${NC}"

# Create dist directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/dist"

echo "   Building Tensor Engine release wheel"
$PYTHON_CMD -m maturin build --release --out "$SCRIPT_DIR/dist"

# Find the generated wheel
WHEEL_PATH=$(ls -t "$SCRIPT_DIR/dist"/tensor_engine-*.whl | head -n1)

if [ ! -f "$WHEEL_PATH" ]; then
    echo -e "${RED}❌ Error: Wheel build failed or not found.${NC}"
    exit 1
fi
echo "   Built: $WHEEL_PATH"

# 3. Installation
# ------------------------------------------------------------------------------
echo -e "${BLUE}[3/4] Installing package...${NC}"
$PYTHON_CMD -m pip install --force-reinstall "$WHEEL_PATH"

# 4. Verification
# ------------------------------------------------------------------------------
echo -e "${BLUE}[4/4] Verifying installation...${NC}"
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

