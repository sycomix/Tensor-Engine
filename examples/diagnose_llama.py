#!/usr/bin/env python3
"""Diagnostic script for LLaMA generation and embedding/LM-head checks.

Usage: python examples/diagnose_llama.py examples/Llama-3.2-1B/model.safetensors
"""
from pathlib import Path
import sys
# Ensure repository root is on path so examples package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
from chat_llama import load_config_json, load_tokenizer, LlamaModel, GenerationConfig

if len(sys.argv) < 2:
    print("Usage: diagnose_llama.py <model.safetensors>")
    sys.exit(1)

model_path = Path(sys.argv[1])
print("Model path:", model_path)

# Load config and tokenizer
config = load_config_json(model_path)
print("Config:", config)

tokenizer = load_tokenizer(model_path, strict=False)
print("Tokenizer vocab_size:", tokenizer.vocab_size())

# Init model and load weights
model = LlamaModel(config)
print("Initialized model")
model.load_weights(model_path)
print("Weights loaded")

# Inspect embedding and lm_head shapes
try:
    tok_emb = model.tok_emb
    print("tok_emb.shape:", list(tok_emb.shape))
    sample_id = 1
    tok_row = tok_emb.get_data()[sample_id * config.hidden_size:(sample_id + 1) * config.hidden_size]
    print("tok_emb sample (first 8):", tok_row[:8])
except Exception as e:
    print("tok_emb read failed:", e)

try:
    # lm_head may expose named_parameters; try to find weight
    lm_params = []
    if hasattr(model.lm_head, 'named_parameters'):
        lm_params = list(model.lm_head.named_parameters(''))
    print("lm_params count:", len(lm_params))
    for name, p in lm_params[:3]:
        print("lm param", name, "shape", getattr(p, 'shape', 'unknown'))
except Exception as e:
    print("lm_head read failed:", e)

# Small embedding lookup check
print("Running embedding_lookup sanity test...")
import tensor_engine as te
emb = te.Tensor([0.1 * i for i in range(12)], [3, 4])
ids = te.Tensor([0.0, 2.0], [2])
out = te.Tensor.embedding_lookup(emb, ids)
print("embedding_lookup output shape:", out.shape)
print("embedding_lookup output data (first 8):", out.get_data()[:8])

# Deterministic generation test (top_k=1 -> argmax)
gen_cfg = GenerationConfig(max_new_tokens=8, temperature=1.0, top_k=1, top_p=1.0, repetition_penalty=1.0)
prompt = "Hello, how are you?"
print("Prompt (raw):", prompt)
input_ids = list(tokenizer.encode(prompt))
print("input_ids (raw):", input_ids)

# Also try chat-formatted prompt as used by chat_loop
chat_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
print("Prompt (chat-formatted):", chat_prompt)
chat_ids = list(tokenizer.encode(chat_prompt))
print("chat_ids:", chat_ids)

orig_len = len(input_ids)
for step in range(gen_cfg.max_new_tokens):
    ids_array = np.array(input_ids, dtype=np.int32).reshape(1, len(input_ids))
    ids_tensor = te.Tensor(ids_array.ravel().tolist(), [1, len(input_ids)])
    logits = model.forward(ids_tensor)
    vocab_size = config.vocab_size
    logits_flat = np.array(logits.get_data(), dtype=np.float32)
    try:
        logits_np = logits_flat.reshape((1, len(input_ids), vocab_size))
        last_logits = logits_np[0, -1, :]
    except Exception:
        last_logits = logits_flat[-vocab_size:]

    # Recompute last hidden (run transformer stack manually) and compare to lm_head result
    # embedding lookup
    emb_x = te.Tensor.embedding_lookup(model.tok_emb, ids_tensor)
    # Respect model RoPE configuration: only add absolute pos embeddings when RoPE is disabled
    if not model.config.use_rope and getattr(model, 'pos_emb', None) is not None:
        pos_ids = te.Tensor([float(i) for i in range(len(input_ids))], [len(input_ids)])
        pos_emb = te.Tensor.embedding_lookup(model.pos_emb, pos_ids)
        pos_emb_b = te.Tensor.stack([pos_emb] * 1, axis=0)
        x = te.Tensor([a + b for a, b in zip(emb_x.get_data(), pos_emb_b.get_data())], list(emb_x.shape))
    else:
        x = emb_x
    for layer in model.layers:
        x = layer.forward(x)
    # x shape [1, seq, hidden]; get last hidden
    # flatten and slice
    last_hidden_flat = np.array(x.get_data(), dtype=np.float32)
    last_hidden = last_hidden_flat.reshape((1, len(input_ids), config.hidden_size))[0, -1, :]

    # Get LM head weight param
    weight_param = None
    if hasattr(model.lm_head, 'named_parameters'):
        try:
            lm_params = list(model.lm_head.named_parameters(''))
            for name, p in lm_params:
                if 'weight' in name:
                    weight_param = p
                    break
        except Exception as e:
            logging.debug("Could not retrieve lm_head parameters: %s", e)
    lm_weight_np = None
    if weight_param is not None:
        w_flat = np.array(weight_param.get_data(), dtype=np.float32)
        try:
            lm_weight_np = w_flat.reshape((config.hidden_size, config.vocab_size))
        except Exception:
            lm_weight_np = None

    if lm_weight_np is not None:
        dot_logits = last_hidden @ lm_weight_np
        diff = np.max(np.abs(dot_logits - last_logits))
        argmax_dot = int(np.argmax(dot_logits))
        argmax_logits = int(np.argmax(last_logits))
        print(f"logits vs dot max abs diff={diff:.6f}, argmax_dot={argmax_dot}, argmax_logits={argmax_logits}")
    else:
        print("LM weight not available to compare dot product")

    next_token = int(np.argmax(last_logits))
    input_ids.append(next_token)
    print(f"step {step}: next_token={next_token}")
    # decode generated suffix (current generated portion)
    gen_ids = input_ids[orig_len:]
    try:
        decoded_te = tokenizer.decode(gen_ids)
    except Exception as e:
        logging.debug("Could not decode generated tokens: %s", e)
        decoded_te = '<decode failed>'
    print("decoded_so_far:", decoded_te)

print("Final generated token ids:", input_ids[orig_len:])
print("Final decoded:", tokenizer.decode(input_ids[orig_len:]))
