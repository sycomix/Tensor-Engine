#!/usr/bin/env python3

import argparse
import sys
from safetensors.tensor_engine import save_file

import tensor_engine as torch


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
    # Skip automated runs when no args provided
    if len(sys.argv) <= 1:
        print('No args provided; skipping convert_torch_to_safetensors example')
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input PyTorch checkpoint (.pt/.pth)")
    parser.add_argument("output", help="Output SafeTensors (.safetensors)")
    parser.add_argument("--no-transpose", dest="transpose", action="store_false",
                        help="Do not transpose 2D weight matrices")
    args = parser.parse_args()
    convert(args.input, args.output, args.transpose)
