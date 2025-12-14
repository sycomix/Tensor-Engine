"""Smoke test for the train_llava.py example.

This test runs the example for 1 epoch with a small batch to verify the Python binding logic.
"""
import subprocess
import sys

try:
    subprocess.run([sys.executable, "examples/train_llava.py", "--epochs", "1", "--batch", "2"], check=True)
    print("train_llava example executed successfully")
except subprocess.CalledProcessError as e:
    print(
        "train_llava example failed; ensure tensor_engine Python wheel is installed and maturin was used to install it")
    raise
