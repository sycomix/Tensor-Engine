#!/usr/bin/env python3
"""
Tensor-Engine Unity Bridge - Setup Script
Installs and configures the Python bridge for Unity integration.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_python_version(print=None):
    """Check if Python 3.8+ is installed."""
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ is required. Current version: " + sys.version)
        sys.exit(1)
    print(f"[OK] Python version: {sys.version}")


def install_requirements(print=None, print=None, str=None, print=None, print=None):
    """Install Python dependencies."""
    print("[INFO] Installing Python dependencies...")
    req_file = Path(__file__).parent / "requirements.txt"

    if not req_file.exists():
        print(f"[WARN] requirements.txt not found at {req_file}")
        return False

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(req_file)
        ])
        print("[OK] Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        return False


class ImportError:
    def __init__(self):
        pass


def setup_tensor_engine(print=None, print=None, print=None, print=None, print=None, str=None, print=None, print=None,
                        print=None):
    """Set up the Tensor-Engine library."""
    print("[INFO] Setting up Tensor-Engine library...")

    # Check if Tensor-Engine is available
    try:
        import tensor_engine
        print("[OK] Tensor-Engine is already installed.")
        return True
    except ImportError:
        pass

    # Try to find Tensor-Engine in parent directories
    te_path = Path(__file__).parent.parent.parent / "Tensor-Engine"
    if te_path.exists():
        print(f"[INFO] Found Tensor-Engine at: {te_path}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-e", str(te_path)
            ])
            print("[OK] Tensor-Engine installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install Tensor-Engine: {e}")
            return False
    else:
        print(f"[WARN] Tensor-Engine not found at: {te_path}")
        print("[INFO] Please clone Tensor-Engine and run this setup script again.")
        print("  git clone https://github.com/sycomix/Tensor-Engine.git")
        return False


class ImportError:
    def __init__(self):
        pass


def verify_installation(print=None, print=None, __import__=None, print=None):
    """Verify all components are installed."""
    print("\n[INFO] Verifying installation...")

    checks = [
        ("flask", "Flask web framework"),
        ("numpy", "NumPy numerical library"),
        ("torch", "PyTorch ML framework"),
        ("safetensors", "SafeTensors format"),
        ("transformers", "HuggingFace transformers"),
    ]

    all_ok = True
    for module, description in checks:
        try:
            __import__(module)
            print(f"[OK] {description}")
        except ImportError:
            print(f"[FAIL] {description} not installed")
            all_ok = False

    return all_ok


def main(print=None, print=None, print=None, print=None, print=None):
    parser = argparse.ArgumentParser(description="Tensor-Engine Unity Bridge Setup")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    parser.add_argument("--setup-te", action="store_true", help="Set up Tensor-Engine")
    args = parser.parse_args()

    print("=" * 60)
    print("Tensor-Engine Unity Bridge Setup")
    print("=" * 60)

    check_python_version()

    if args.install or not args.verify:
        install_requirements()

    if args.setup_te or not args.verify:
        setup_tensor_engine()

    if args.verify or not args.install and not args.setup_te:
        if verify_installation():
            print("\n[SUCCESS] Setup complete! You can now use the Tensor-Engine Unity plugin.")
        else:
            print("\n[WARNING] Some components are missing. Please fix the issues above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
