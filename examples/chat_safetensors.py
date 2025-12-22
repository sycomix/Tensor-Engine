#!/usr/bin/env python3
"""
Console chat test that loads a local SafeTensors Llama model and runs a simple REPL.

Notes:
- This uses the Python bindings exposed via PyO3 in `src/lib.rs`.
- Loads SafeTensors weights with `py_load_safetensors_into_module` into a TransformerBlock.
- Auto-detects config.json and tokenizer.json from the model directory.
- If built with `with_tokenizers` feature, uses te.Tokenizer; otherwise falls back to naive byte tokenizer.
- This is a minimal test harness validating local model loading and an interactive loop.
- Real text generation requires an embedding layer + LM head (not included in this stub).

Usage:
    python examples/chat_safetensors.py <path/to/model.safetensors> [options]

Before running:
    pip install maturin
    maturin build --release --features "python_bindings,safe_tensors,with_tokenizers,openblas,multi_precision"
    pip install target/wheels/tensor_engine-*.whl

Examples:
    # Auto-detect config.json and tokenizer.json in model directory:
    python examples/chat_safetensors.py d:/models/model.safetensors
    
    # Override dimensions manually:
    python examples/chat_safetensors.py ./weights.safetensors --d_model 512 --d_ff 2048 --num_heads 8 --transpose
"""
import argparse
import sys
import json
from pathlib import Path
import logging
import numpy as np

# Configure structured logging for examples
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    import tensor_engine as te  # PyO3 module name as built by this repo
    from tokenizers import Tokenizer
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error("Failed to import `tensor_engine` Python extension. Build with: maturin develop --release --features python_bindings,safe_tensors,with_tokenizers,openblas,multi_precision")
    raise


def try_load_config(model_path: str):
    """Try to load config.json from the same directory as the model.
    Returns dict with d_model, d_ff, num_heads if found, else None.
    """
    model_dir = Path(model_path).parent
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            # Try common config key names
            d_model = config.get("hidden_size") or config.get("d_model") or config.get("n_embd")
            d_ff = config.get("intermediate_size") or config.get("d_ff") or config.get("n_inner")
            num_heads = config.get("num_attention_heads") or config.get("num_heads") or config.get("n_head")
            if d_model and d_ff and num_heads:
                return {"d_model": d_model, "d_ff": d_ff, "num_heads": num_heads}
        except Exception as e:
            logger.warning(f"Failed to parse config.json: {e}")
    return None


def try_find_tokenizer(model_path: str):
    """Try to find tokenizer.json in the same directory as the model.
    Returns path if found, else None.
    """
    model_dir = Path(model_path).parent
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.exists():
        return str(tokenizer_path)
    return None


def build_transformer(d_model: int, d_ff: int, num_heads: int):
    """Construct a minimal TransformerBlock via Python bindings.
    Extend/replace with your higher-level wrapper if exposed.
    """
    # The Python bindings typically expose a TransformerBlock; if your binding uses a different name, adjust here.
    try:
        block = te.TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads)
    except AttributeError:
        logger.warning("`TransformerBlock` not found in Python bindings; using a Linear layer as a stub.")
        block = te.Linear(d_model, d_model)
    return block


def load_weights_from_safetensors(module, path: str, transpose: bool, root: str = ""):
    """Load weights into the module from a SafeTensors file using helper.
    API: py_load_safetensors_into_module(bytes, transpose, module, root)
    """
    # Read file bytes first
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as e:
        raise FileNotFoundError(f"Failed to read SafeTensors file '{path}': {e}")

    # Try to inspect keys if the simple loader is available
    state_dict = None
    if hasattr(te, "py_load_safetensors"):
        try:
            state_dict = te.py_load_safetensors(data, transpose)
            logger.info(f"Loaded SafeTensors archive: {len(state_dict)} tensors")
            # Show a few sample keys and prefix summary
            keys = sorted(list(state_dict.keys()))
            logger.debug(f"Example keys: {keys[:8]}")
            prefixes = set(k.split('.', 2)[0] if '.' in k else k for k in keys)
            logger.debug(f"Top-level prefixes: {sorted(list(prefixes))[:8]}")
        except Exception as e:
            logger.warning(f"py_load_safetensors failed: {e}")
    # Preferred: use rust-level module loader when available
    if hasattr(te, "py_load_safetensors_into_module"):
        # Try applying with the provided root first
        try:
            te.py_load_safetensors_into_module(data, transpose, module, root or None)
            logger.info(f"Applied safetensors to module via rust loader (root='{root}')")
            return
        except Exception as exc:
            logger.warning(f"Rust loader (root='{root}') failed: {exc}")
        # Common case: checkpoints have per-layer keys like model.layers.0.*
        # If we see model.layers.* in the keys, attempt to apply layer 0 to this single-block module
        try:
            if state_dict is not None:
                has_layers = any(k.startswith("model.layers.") for k in state_dict.keys())
                if has_layers:
                    try:
                        te.py_load_safetensors_into_module(data, transpose, module, f"model.layers.0.")
                        logger.info("Applied safetensors to module using root='model.layers.0.'")
                        return
                    except Exception as exc:
                        logger.warning(f"Rust loader with root 'model.layers.0.' failed: {exc}")
        except Exception:
            # keep going to fallback
            pass

    # Last resort: attempt best-effort in-place assignment from state_dict
    if state_dict is None:
        raise RuntimeError("No SafeTensors loader available (neither py_load_safetensors nor py_load_safetensors_into_module are present)")

    assigned = 0
    try:
        named = list(module.named_parameters(""))
    except TypeError:
        named = []
    for name, param in named:
        key = None
        lname = name.lstrip('.')
        # Direct mapping with optional root
        candidates = []
        if root:
            candidates.append(f"{root}{lname}")
        candidates.append(lname)

        # Some common HF -> internal mappings
        if '.mha.linear_q.weight' in name:
            candidates.append(f"model.layers.0.self_attn.q_proj.weight")
        if '.mha.linear_k.weight' in name:
            candidates.append(f"model.layers.0.self_attn.k_proj.weight")
        if '.mha.linear_v.weight' in name:
            candidates.append(f"model.layers.0.self_attn.v_proj.weight")
        if '.mha.linear_o.weight' in name:
            candidates.append(f"model.layers.0.self_attn.o_proj.weight")

        found = False
        for c in candidates:
            if c in state_dict:
                w = state_dict[c]
                try:
                    param.set_data(w.get_data())
                    assigned += 1
                    found = True
                    break
                except Exception as exc:
                        logger.warning(f"Failed to assign param '{name}' from key '{c}': {exc}")
        # Special-case linear1 which may be concatenation of gate+up
        if not found and lname.endswith('linear1.weight'):
            gate_key = f"model.layers.0.mlp.gate_proj.weight"
            up_key = f"model.layers.0.mlp.up_proj.weight"
            if gate_key in state_dict and up_key in state_dict:
                try:
                    gate = state_dict[gate_key].get_data()
                    up = state_dict[up_key].get_data()
                    combined = list(gate) + list(up)
                    param.set_data(combined)
                    assigned += 1
                except Exception as exc:
                    logger.warning(f"Failed to set concatenated linear1.weight: {exc}")

    logger.info(f"Fallback assignment: set {assigned} parameters from state dict (best-effort)")


def naive_tokenize(text: str, vocab_size: int = 256):
    """Very naive tokenizer: maps bytes to integers (0..255). Returns np.int32 array.
    Fallback when te.Tokenizer is not available.
    """
    arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    arr = np.clip(arr, 0, vocab_size - 1)
    return arr


def tokenize_with_te(tokenizer, text: str):
    """Use the te.Tokenizer (HuggingFace tokenizers binding) if available."""
    try:
        ids = tokenizer.encode(text)
        return np.array(ids, dtype=np.int32)
    except Exception as e:
        logger.exception("Tokenizer encode failed; falling back to naive tokenizer")
        return naive_tokenize(text)


def pad_or_trim(arr: np.ndarray, seq_len: int):
    if arr.shape[0] > seq_len:
        return arr[:seq_len]
    else:
        return arr  # Don't pad, just trim if too long


def forward_with_embedding(module, token_ids: np.ndarray, d_model: int, embed_weights: np.ndarray, lm_head: object):
    """End-to-end small pipeline: token ids -> embedding lookup -> TransformerBlock -> LM head logits

    This uses a simple NumPy-based embedding lookup (one-hot matmul) to produce embeddings and a
    `te.Linear` layer as the LM head. It returns diagnostics similar to the previous stub.
    """
    batch_size = 1
    seq_len = token_ids.shape[0]

    # One-hot embedding lookup via NumPy: (seq_len, vocab) @ (vocab, d_model) -> (seq_len, d_model)
    # token_ids expected as 1D int array
    if token_ids.dtype != np.int32 and token_ids.dtype != np.int64:
        token_ids = token_ids.astype(np.int32)
    # Guard against OOB tokens
    vocab = embed_weights.shape[0]
    token_ids = np.clip(token_ids, 0, vocab - 1)

    one_hot = np.zeros((seq_len, vocab), dtype=np.float32)
    one_hot[np.arange(seq_len), token_ids] = 1.0
    emb = one_hot @ embed_weights  # (seq_len, d_model)
    # reshape to (1, seq_len, d_model)
    emb = emb.reshape((batch_size, seq_len, d_model)).astype(np.float32)

    x = te.Tensor(emb.flatten().tolist(), [batch_size, seq_len, d_model])

    # pass through the transformer block
    try:
        out = module.forward(x)
    except AttributeError:
        out = module(x)

    # Take last token hidden state: shape [batch, d_model]
    # out.get_data() returns flat list; use slicing through Tensor methods
    last_hidden = out[:, -1, :]

    # LM head expects input shape [batch, d_model] or [batch, 1, d_model]
    try:
        logits = lm_head.forward(last_hidden)
    except AttributeError:
        # try treating as callable
        logits = lm_head(last_hidden)

    # Convert logits to numpy for diagnostics
    logits_np = np.array(logits.get_data())
    data = logits_np
    shape = logits.shape if hasattr(logits, 'shape') else list(data.shape)
    return {
        "shape": shape,
        "mean": float(data.mean()) if data.size > 0 else 0.0,
        "std": float(data.std()) if data.size > 0 else 0.0,
        "min": float(data.min()) if data.size > 0 else 0.0,
        "max": float(data.max()) if data.size > 0 else 0.0,
        "logits": logits,
    }


def chat_loop(module, seq_len: int, d_model: int, tokenizer=None, embed_weights=None, lm_head=None):
    logger.info("=" * 70)
    logger.info("Interactive Model Diagnostic REPL")
    logger.info("=" * 70)
    logger.info("NOTE: This is a technical stub for testing SafeTensors loading.")
    logger.info("      Real text generation requires:")
    logger.info("      1. Embedding layer (token_ids → embeddings)")
    logger.info("      2. TransformerBlock(s) [✓ loaded from SafeTensors]")
    logger.info("      3. LM head (hidden_states → vocab logits)")
    logger.info("      4. Sampling/decoding loop")
    logger.info("")
    logger.info("Current behavior: Random embeddings → TransformerBlock → Statistics")
    logger.info("Type 'exit' to quit.")
    logger.info("=" * 70)
    
    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("")
            break
        if user.lower() in {"exit", "quit"}:
            break
        
        # Tokenize & pad
        if tokenizer is not None:
            tokens = tokenize_with_te(tokenizer, user)
        else:
            tokens = naive_tokenize(user)
        tokens = pad_or_trim(tokens, seq_len)
        
        logger.info("[Tokenized to %d tokens, padded/trimmed to %d]", len(tokens), seq_len)
        
        # Forward pass (prefer real embedding+lm_head pipeline if available)
        if embed_weights is not None and lm_head is not None:
            stats = forward_with_embedding(module, tokens, d_model, embed_weights, lm_head)
            # logits object included in stats - compute greedy id
            last_logits = stats['logits']
            try:
                last_logits_np = np.array(last_logits.get_data())
                out_id = int(np.argmax(last_logits_np))
            except Exception:
                out_id = 0
            logger.info("Model> Generated token id: %d", out_id)
        else:
            stats = forward_with_embedding(module, tokens, d_model, np.zeros((1, d_model), dtype=np.float32), te.Linear(d_model, 1, True))
            logger.info("Model> Output shape: %s", stats['shape'])
            logger.info("       Activation stats: mean=%.4f, std=%.4f", stats['mean'], stats['std'])
            logger.info("       Range: [%.4f, %.4f]", stats['min'], stats['max'])
            logger.info("")
            logger.info("💡 To get real text: Load a full LM with embedding + lm_head,")
            logger.info("   or use this as a component in a larger generation pipeline.")


def main():
    p = argparse.ArgumentParser(description="Chat with local SafeTensors Llama model (minimal console REPL)")
    p.add_argument("model", type=str, help="Path to SafeTensors file or directory")
    p.add_argument("--transpose", action="store_true", help="Transpose weights when loading (if needed)")
    p.add_argument("--tokenizer", type=str, default=None, help="Path to HuggingFace tokenizer.json (auto-detected if in model dir)")
    p.add_argument("--config", type=str, default=None, help="Path to config.json (auto-detected if in model dir)")
    p.add_argument("--d_model", type=int, default=None, help="Transformer hidden size (overrides config)")
    p.add_argument("--d_ff", type=int, default=None, help="Transformer FFN size (overrides config)")
    p.add_argument("--num_heads", type=int, default=None, help="Attention heads (overrides config)")
    p.add_argument("--seq_len", type=int, default=128, help="Sequence length for naive tokens")
    args = p.parse_args()

    # Try to auto-detect config.json in model directory
    config = None
    if args.config:
        try:
            with open(args.config) as f:
                config = json.load(f)
            logger.info("Loaded config from %s", args.config)
        except Exception as e:
            logger.warning("Failed to load config %s: %s", args.config, e)
    else:
        config = try_load_config(args.model)
        if config:
            logger.info("Auto-detected config.json: d_model=%s, d_ff=%s, num_heads=%s", config['d_model'], config['d_ff'], config['num_heads'])

    # Use config values or CLI args or defaults
    d_model = args.d_model or (config['d_model'] if config else 512)
    d_ff = args.d_ff or (config['d_ff'] if config else 2048)
    num_heads = args.num_heads or (config['num_heads'] if config else 8)

    logger.info("Using model dimensions: d_model=%s, d_ff=%s, num_heads=%s", d_model, d_ff, num_heads)
    
    module = build_transformer(d_model, d_ff, num_heads)
    try:
        load_weights_from_safetensors(module, args.model, args.transpose)
    except FileNotFoundError as e:
        logger.warning("Model file not found (%s) — continuing with randomly initialized module for demo.", e)
    except Exception as e:
        logger.warning("Loading model raised an error (%s) — continuing with randomly initialized module for demo.", e)

    # Setup a tiny embedding + lm_head for simple decoding (vocab is toy-sized)
    vocab_size = 256
    embed_weights = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
    lm_head = te.Linear(d_model, vocab_size, True)

    # Try to auto-detect tokenizer.json
    tokenizer = None
    tokenizer_path = args.tokenizer or try_find_tokenizer(args.model)
    if tokenizer_path:
        try:
            tokenizer = te.Tokenizer.from_file(tokenizer_path)
            logger.info("Tokenizer loaded from %s (vocab_size=%d)", tokenizer_path, tokenizer.vocab_size())
            vocab_size = tokenizer.vocab_size()
        except AttributeError:
            logger.warning("te.Tokenizer not available; rebuild with --features with_tokenizers. Using naive fallback.")
        except Exception as e:
            logger.warning("Failed to load tokenizer: %s. Using naive fallback.", e)

    logger.info("Model loaded.")
    # Start chat loop with the actual embedding and LM head
    chat_loop(module, args.seq_len, d_model, tokenizer=tokenizer, embed_weights=embed_weights, lm_head=lm_head)


if __name__ == "__main__":
    main()
