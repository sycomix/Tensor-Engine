#!/usr/bin/env python3
"""Run a deterministic generation demo using the local LLaMA checkpoint.
Usage: python examples/run_demo_chat.py [model.safetensors] [prompt]
"""
import sys
from pathlib import Path
from chat_llama import load_config_json, load_tokenizer, LlamaModel, GenerationConfig, generate_text

MODEL_DEFAULT = Path(__file__).resolve().parents[1] / "Llama-3.2-1B" / "model.safetensors"

if len(sys.argv) > 1:
    model_path = Path(sys.argv[1])
else:
    model_path = MODEL_DEFAULT

prompt = "Hello, how are you?"
if len(sys.argv) > 2:
    prompt = sys.argv[2]

print(f"Model path: {model_path}")
print(f"Prompt: {prompt}")

cfg = load_config_json(model_path)
print(f"Loaded config: hidden_size={cfg.hidden_size}, layers={cfg.num_hidden_layers}")

tokenizer = load_tokenizer(model_path, strict=False)
print(f"Loaded tokenizer (vocab_size={tokenizer.vocab_size()})")

model = LlamaModel(cfg)
print("Initialized model")
model.load_weights(model_path)
print("Loaded weights")

# Greedy decoding via top_k=1 for determinism
gen_cfg = GenerationConfig(max_new_tokens=20, temperature=1.0, top_k=1, top_p=1.0, repetition_penalty=1.0)

# Call the canonical top-level generation function from chat_llama
try:
    out = generate_text(model, tokenizer, prompt, gen_cfg)
    print("\n=== Generated ===")
    print(out)
    print("=== End ===")
except Exception as exc:
    print(f"Generation failed: {exc}")
    raise
