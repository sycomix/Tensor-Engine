#!/usr/bin/env python3
"""
Console chat test that loads a local SafeTensors model and runs a simple REPL.

Notes:
- This uses the Python bindings exposed via PyO3 in `src/lib.rs`.
- We load weights with `py_load_safetensors_into_module` and instantiate a TransformerBlock.
- This is a minimal test harness to validate local model loading and an interactive loop.
- Text generation/tokenization are placeholders; adapt to your tokenizer and head as needed.

Usage:
    python examples/chat_safetensors.py <path/to/model.safetensors> [--transpose] [--d_model 512] [--d_ff 2048] [--num_heads 8] [--seq_len 128]

Before running:
    pip install maturin
    maturin develop --release

Example:
    python examples/chat_safetensors.py ./weights.safetensors --transpose --d_model 512 --d_ff 2048 --num_heads 8 --seq_len 128
"""
import argparse
import sys
import numpy as np

try:
    import tensor_engine as te  # PyO3 module name as built by this repo
except ImportError as e:
    print("ERROR: Failed to import `tensor_engine` Python extension. Build with `maturin develop --release`.")
    raise


def build_transformer(d_model: int, d_ff: int, num_heads: int):
    """Construct a minimal TransformerBlock via Python bindings.
    Extend/replace with your higher-level wrapper if exposed.
    """
    # The Python bindings typically expose a TransformerBlock; if your binding uses a different name, adjust here.
    try:
        block = te.TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads)
    except AttributeError:
        print("WARNING: `TransformerBlock` not found in Python bindings; using a Linear layer as a stub.")
        block = te.Linear(d_model, d_model)
    return block


def load_weights_from_safetensors(module, path: str, transpose: bool):
    """Load weights into the module from a SafeTensors file using helper.
    """
    try:
        te.py_load_safetensors_into_module(module, path, transpose)
    except AttributeError:
        # Fallback if helper is under a different name; adjust as needed.
        raise RuntimeError("`py_load_safetensors_into_module` not found in bindings; ensure you're on a recent build.")


def naive_tokenize(text: str, vocab_size: int = 256):
    """Very naive tokenizer: maps bytes to integers (0..255). Returns np.int32 array.
    Replace with your repo's tokenizer when available.
    """
    arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    return arr


def pad_or_trim(arr: np.ndarray, seq_len: int):
    if arr.shape[0] < seq_len:
        pad = np.zeros((seq_len - arr.shape[0],), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)
    else:
        return arr[:seq_len]


def forward_stub(module, token_ids: np.ndarray):
    """Run a forward pass stub. Without a full embedding + LM head, we feed token_ids as float.
    This serves as a basic sanity check the graph executes.
    """
    x = te.Tensor(token_ids.astype(np.float32).tolist(), [token_ids.shape[0]])
    # Pass through the module (TransformerBlock or Linear stub).
    try:
        out = module.forward(x)
    except AttributeError:
        # Some bindings may expose `__call__` instead of forward
        out = module(x)
    # Return a simple statistic as a placeholder for logits.
    data = np.array(out.numpy())
    return float(data.mean()) if data.size > 0 else 0.0


def chat_loop(module, seq_len: int):
    print("Entering chat REPL. Type 'exit' to quit.")
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            break
        if user.lower() in {"exit", "quit"}:
            break
        # Tokenize & pad
        tokens = naive_tokenize(user)
        tokens = pad_or_trim(tokens, seq_len)
        # Forward pass stub
        score = forward_stub(module, tokens)
        # Naive response based on score; replace with real generation.
        if score > 0:
            reply = "Model indicates a positive activation."
        elif score < 0:
            reply = "Model indicates a negative activation."
        else:
            reply = "Model activation neutral."
        print(f"Model> {reply}")


def main():
    p = argparse.ArgumentParser(description="Chat with local SafeTensors model (minimal console REPL)")
    p.add_argument("model", type=str, help="Path to SafeTensors file")
    p.add_argument("--transpose", action="store_true", help="Transpose weights when loading (if needed)")
    p.add_argument("--d_model", type=int, default=512, help="Transformer hidden size")
    p.add_argument("--d_ff", type=int, default=2048, help="Transformer FFN size")
    p.add_argument("--num_heads", type=int, default=8, help="Attention heads")
    p.add_argument("--seq_len", type=int, default=128, help="Sequence length for naive tokens")
    args = p.parse_args()

    module = build_transformer(args.d_model, args.d_ff, args.num_heads)
    load_weights_from_safetensors(module, args.model, args.transpose)

    print("Model loaded.")
    chat_loop(module, args.seq_len)


if __name__ == "__main__":
    main()
