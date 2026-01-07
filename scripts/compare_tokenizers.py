#!/usr/bin/env python3
"""Compare decoding of a list of token ids between HuggingFace and tensor_engine tokenizers.

Usage: python scripts/compare_tokenizers.py [model_dir]
"""
import sys
from pathlib import Path

ids = [14535, 94599, 42306, 15113, 116667, 8822, 91049, 74038]
model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/Llama-3.2-1B")
print("Model dir:", model_dir)

# Try HuggingFace first
hf_ok = False
try:
    from transformers import AutoTokenizer
    print("HuggingFace transformers available")
    try:
        htok = AutoTokenizer.from_pretrained(str(model_dir))
        print("Loaded HF tokenizer from", model_dir)
        # convert ids to tokens if available
        try:
            tokens = htok.convert_ids_to_tokens(ids)
        except Exception:
            # fallback for older tokenizers
            tokens = [htok._tokenizer.id_to_token(i) for i in ids]
        print("HF per-token strings:")
        for i, t in zip(ids, tokens):
            print(f"  {i}: {t}")
        try:
            dec = htok.decode(ids, skip_special_tokens=False)
        except Exception:
            dec = htok.decode(ids)
        print("HF decoded sequence:")
        print(dec)
        hf_ok = True
    except Exception as exc:
        print("Failed to load HF tokenizer:", exc)
except Exception:
    print("transformers not installed")

# Try tensor_engine tokenizer if available
te_ok = False
try:
    import tensor_engine as te
    print("tensor_engine available")
    # attempt to load tokenizer.json in model_dir
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists() and hasattr(te, "Tokenizer"):
        try:
            tetok = te.Tokenizer.from_file(str(tok_path))
            print("Loaded tensor_engine.Tokenizer from", tok_path)
            try:
                # many tokenizers do not expose per-token string API; fallback to decode single-id lists
                print("tensor_engine per-token decodes:")
                for i in ids:
                    try:
                        s = tetok.decode([int(i)])
                    except Exception:
                        s = tetok.decode([int(i)])
                    print(f"  {i}: {s}")
            except Exception as exc:
                print("Failed per-token decode with tensor_engine tokenizer:", exc)
            try:
                dec = tetok.decode([int(x) for x in ids])
                print("tensor_engine decoded sequence:")
                print(dec)
            except Exception as exc:
                print("tensor_engine decode failed:", exc)
            te_ok = True
        except Exception as exc:
            print("Failed to load tensor_engine.Tokenizer:", exc)
    else:
        print("No tokenizer.json found or tensor_engine.Tokenizer missing")
except Exception as exc:
    print("tensor_engine import failed:", exc)

# Summary suggestion
print("\nSummary:")
print(f"  HF available: {hf_ok}")
print(f"  tensor_engine tokenizer available: {te_ok}")
print("If HF is available, compare HF-per-token strings to tensor_engine's decode to see if tokenization/decoding mismatch explains garbled output.")
