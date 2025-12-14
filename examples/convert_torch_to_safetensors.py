#!/usr/bin/env python3
"""Convert a PyTorch state_dict (.pt or .pth) to SafeTensors (.safetensors) format.

Usage: python examples/convert_torch_to_safetensors.py input.pt output.safetensors

This requires `torch` and `safetensors` Python packages:
  pip install torch safetensors
"""
import argparse

import torch
from safetensors.torch import save_file


def convert(input_path: str, output_path: str, transpose_two_dim: bool = True):
    st = torch.load(input_path, map_location="cpu")
    # If a full checkpoint, it may be like {'model': state_dict}
    if isinstance(st, dict) and "state_dict" in st and not isinstance(list(st.keys())[0], str):
        st = st["state_dict"]
    # If nested dicts, try to flatten only top-level fields
    tensors = {}
    for k, v in st.items():
        if hasattr(v, "cpu"):
            t = v.cpu()
            if transpose_two_dim and t.ndim == 2 and k.endswith(".weight"):
                t = t.t()
            tensors[k] = t
    save_file(tensors, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input PyTorch checkpoint (.pt/.pth)")
    parser.add_argument("output", help="Output SafeTensors (.safetensors)")
    parser.add_argument("--no-transpose", dest="transpose", action="store_false",
                        help="Do not transpose 2D weight matrices")
    args = parser.parse_args()
    convert(args.input, args.output, args.transpose)
