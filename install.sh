#!/bin/bash
# Tensor Engine Beta Install Script
# Version: 0.2.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_NAME="tensor_engine-0.2.0-cp39-cp39-macosx_11_0_arm64.whl"
WHEEL_PATH="$SCRIPT_DIR/dist/$WHEEL_NAME"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Tensor Engine v0.2.0 Beta Installer                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if wheel exists
if [ ! -f "$WHEEL_PATH" ]; then
    echo "❌ Error: Wheel not found at $WHEEL_PATH"
    echo "   Please run 'maturin build --release' first or check the dist/ directory."
    exit 1
fi

# Detect Python
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python not found. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "🐍 Detected Python: $PYTHON_VERSION"

# Check pip
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "❌ Error: pip not found. Please install pip."
    exit 1
fi

# Install the wheel
echo ""
echo "📦 Installing tensor_engine from wheel..."
echo "   $WHEEL_PATH"
echo ""

$PYTHON_CMD -m pip install --upgrade "$WHEEL_PATH"

# Verify installation
echo ""
echo "🔍 Verifying installation..."
$PYTHON_CMD -c "import tensor_engine; print(f'✅ tensor_engine imported successfully')" 2>/dev/null || {
    echo "⚠️  Warning: Could not verify import. The package may need additional dependencies."
}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Installation Complete!                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Usage:"
echo "  import tensor_engine"
echo "  from tensor_engine import Tensor"
echo ""
echo "Features enabled:"
echo "  • Metal GPU acceleration (Apple Silicon)"
echo "  • Distributed training (multi-GPU)"
echo "  • Async operations"
echo "  • SafeTensors support"
echo "  • FP16/BF16 support"
echo "  • Vision/Image processing"
echo "  • Quantization"
echo ""
