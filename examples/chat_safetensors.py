#!/usr/bin/env python3
"""
Console chat test that loads a local SafeTensors Llama model and runs a simple REPL.

Notes:
- This uses the Python bindings exposed via PyO3 in `src/lib.rs`.
- We load weights with `py_load_safetensors_into_module` and instantiate a Llama model.
- This is a minimal test harness to validate local model loading and an interactive loop.
- Text generation uses proper tokenization from tokenizer.json when available.

Usage:
    python examples/chat_safetensors.py <path/to/model.safetensors> [--transpose] [--vocab_size 128256] [--d_model 3072] [--num_layers 32] [--d_ff 8192] [--num_heads 24] [--kv_heads 24] [--seq_len 128]

Before running:
    pip install maturin
    maturin develop --release

Example:
    python examples/chat_safetensors.py ./Llama-3.2-3B-Instruct/ --transpose --vocab_size 128256 --d_model 3072 --num_layers 32 --d_ff 8192 --num_heads 24 --kv_heads 24 --seq_len 128
"""
import argparse
import numpy as np

try:
    import tensor_engine as te  # PyO3 module name as built by this repo
    from tokenizers import Tokenizer
except ImportError as e:
    print("ERROR: Failed to import required libraries. Install tokenizers.")
    raise


def build_llama(vocab_size: int, d_model: int, num_layers: int, d_ff: int, num_heads: int, kv_heads: int):
    """Construct a Llama model via Python bindings.
    """
    # The Python bindings expose a Llama model.
    model = te.Llama(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, d_ff=d_ff, num_heads=num_heads, kv_heads=kv_heads)
    return model


def load_weights_from_safetensors(module, path: str, transpose: bool):
    """Load weights into the module from SafeTensors files, handling sharded models.
    """
    import os
    import json
    if os.path.isdir(path):
        index_path = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            # Sharded model: load from index.json
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index["weight_map"]
            # Get unique files
            files = set(weight_map.values())
            for file in files:
                file_path = os.path.join(path, file)
                with open(file_path, "rb") as f:
                    data = f.read()
                te.py_load_safetensors_into_module(data, transpose, module, "model")
        else:
            # Single file: find .safetensors file
            safetensors_files = [f for f in os.listdir(path) if f.endswith('.safetensors')]
            if not safetensors_files:
                raise RuntimeError(f"No .safetensors file found in directory {path}")
            if len(safetensors_files) > 1:
                raise RuntimeError(f"Multiple .safetensors files found in {path}: {safetensors_files}. Expected index.json for sharded model.")
            file_path = os.path.join(path, safetensors_files[0])
            with open(file_path, "rb") as f:
                data = f.read()
            te.py_load_safetensors_into_module(data, transpose, module, "model")
    else:
        # Direct file path
        with open(path, "rb") as f:
            data = f.read()
        te.py_load_safetensors_into_module(data, transpose, module, "")
    
    # Load tokenizer if available
    dir_path = os.path.dirname(path) if os.path.isfile(path) else path
    tokenizer_path = os.path.join(dir_path, "tokenizer.json")
    tokenizer = None
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    
    vocab_size = 128256  # Llama 3.2 vocab size
    if tokenizer:
        vocab_size = tokenizer.get_vocab_size()
    
    return vocab_size, tokenizer


def naive_tokenize(text: str, vocab_size: int = 256):
    """Very naive tokenizer: maps bytes to integers (0..vocab_size-1). Returns np.int32 array.
    Replace with your repo's tokenizer when available.
    """
    arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    arr = np.clip(arr, 0, vocab_size - 1)
    return arr


def decode(tokens: list):
    """Decode tokens back to text using naive byte mapping."""
    bytes_arr = np.array(tokens, dtype=np.uint8)
    return bytes_arr.tobytes().decode('utf-8', errors='ignore')


def pad_or_trim(arr: np.ndarray, seq_len: int):
    if arr.shape[0] > seq_len:
        return arr[:seq_len]
    else:
        return arr  # Don't pad, just trim if too long


def forward(module, token_ids: np.ndarray):
    """Run a forward pass through the Llama model.
    """
    # Convert token_ids to tensor
    x = te.Tensor(token_ids.flatten().tolist(), list(token_ids.shape))
    # Pass through the Llama model (handles embedding, layers, norm, lm_head).
    logits = module.forward(x)
    return logits


def generate(module, initial_tokens, max_len=20):
    """Generate continuation tokens using greedy decoding."""
    tokens = list(initial_tokens)
    print(f"Initial tokens: {tokens}")
    for i in range(max_len):
        logits = forward(module, np.array(tokens, dtype=np.int32))
        # Get last token's logits [vocab_size]
        logits_data = logits.get_data()
        logits_np = np.array(logits_data).reshape(logits.shape)
        last_logits = logits_np[-1]
        # Argmax for next token
        next_token = np.argmax(last_logits)
        print(f"Step {i}: next_token = {next_token}, top 5 logits: {np.argsort(last_logits)[-5:][::-1]}, top 5 probs: {last_logits[np.argsort(last_logits)[-5:][::-1]]}")
        tokens.append(int(next_token))
        # Stop if eos (assume <|end_of_text|> or similar, but for naive, stop at 0)
        if next_token == 0 or next_token == 128009:  # Llama EOS token
            break
    return tokens[len(initial_tokens):]  # return generated part


def chat_loop(module, tokenizer, seq_len: int):
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
        if tokenizer:
            tokens = tokenizer.encode(user).ids
        else:
            tokens = naive_tokenize(user, 128256)  # Use Llama vocab size
        tokens = pad_or_trim(np.array(tokens), seq_len)
        # Generate response
        response_tokens = generate(module, tokens, max_len=50)
        if tokenizer:
            reply = tokenizer.decode(response_tokens)
        else:
            reply = decode(response_tokens)
        print(f"Model> {reply}")


def main():
    p = argparse.ArgumentParser(description="Chat with local SafeTensors Llama model (minimal console REPL)")
    p.add_argument("model", type=str, help="Path to SafeTensors file or directory")
    p.add_argument("--transpose", action="store_true", help="Transpose weights when loading (if needed)")
    p.add_argument("--vocab_size", type=int, default=128256, help="Vocabulary size (Llama 3.2 default)")
    p.add_argument("--d_model", type=int, default=3072, help="Model hidden size (Llama 3.2 3B)")
    p.add_argument("--num_layers", type=int, default=32, help="Number of transformer layers")
    p.add_argument("--d_ff", type=int, default=8192, help="Feed-forward hidden size")
    p.add_argument("--num_heads", type=int, default=24, help="Number of attention heads")
    p.add_argument("--kv_heads", type=int, default=24, help="Number of KV attention heads")
    p.add_argument("--seq_len", type=int, default=128, help="Sequence length for tokens")
    args = p.parse_args()

    module = build_llama(args.vocab_size, args.d_model, args.num_layers, args.d_ff, args.num_heads, args.kv_heads)
    vocab_size, tokenizer = load_weights_from_safetensors(module, args.model, args.transpose)

    print(f"Llama model loaded with vocab_size {vocab_size}, d_model {args.d_model}, num_layers {args.num_layers}")
    chat_loop(module, tokenizer, args.seq_len)


if __name__ == "__main__":
    main()
