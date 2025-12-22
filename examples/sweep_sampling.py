#!/usr/bin/env python3
"""Sweep sampling parameters (temperature, top_k, top_p, repetition_penalty).

Usage: python examples/sweep_sampling.py examples/Llama-3.2-1B/model.safetensors

Outputs a ranked list of parameter settings and example generations.
"""
from pathlib import Path
import sys
import numpy as np
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
from chat_llama import load_config_json, load_tokenizer, LlamaModel
import chat_llama as cl
import tensor_engine as te

if len(sys.argv) < 2:
    print("Usage: sweep_sampling.py <model.safetensors>")
    sys.exit(1)

model_path = Path(sys.argv[1])
config = load_config_json(model_path)
print("Model config:", config)

tokenizer = load_tokenizer(model_path, strict=False)
print("Tokenizer vocab size:", tokenizer.vocab_size())

model = LlamaModel(config)
model.load_weights(model_path)
print("Model loaded")

# Prompts to test
prompts = [
    "Hello, how are you?",
    "Tell me a short story about a robot.",
    "What's the capital of France?"
]

# Parameter grid
temps = [0.2, 0.5, 0.7]
top_ks = [10, 50, 100]
top_ps = [0.9, 1.0]
reps = [1.0, 1.1]

results = []

# scoring heuristics
def score_text(token_ids, text):
    # novelty: distinct tokens ratio
    if not token_ids:
        return -1.0
    uniq = len(set(token_ids)) / max(1, len(token_ids))
    # repetition: fraction of tokens equal to previous
    rep = sum(1 for i in range(1, len(token_ids)) if token_ids[i] == token_ids[i-1]) / max(1, len(token_ids)-1)
    # nonprint ratio: fraction of chars outside reasonable set
    printable = sum(1 for c in text if c.isprintable())
    nonprint = 1.0 - (printable / max(1, len(text)))
    # combined score: prefer novelty, penalize repetition and nonprint
    return uniq - 0.5 * rep - nonprint

# generation loop (reuse logic from chat_llama but deterministic reproducible sampling configured)
def generate_with_params(prompt, temp, top_k, top_p, rep_pen, max_new=20):
    toks = list(tokenizer.encode(prompt))
    orig_len = len(toks)
    for step in range(max_new):
        ids_array = np.array(toks, dtype=np.int32).reshape(1, len(toks))
        ids_tensor = te.Tensor(ids_array.ravel().tolist(), [1, len(toks)])
        logits = model.forward(ids_tensor)
        vocab_size = config.vocab_size
        logits_flat = np.array(logits.get_data(), dtype=np.float32)
        try:
            logits_np = logits_flat.reshape((1, len(toks), vocab_size))
            last_logits = logits_np[0, -1, :]
        except Exception:
            last_logits = logits_flat[-vocab_size:]
        next_token = cl.sample_token(last_logits, temp, top_k, top_p, recent_tokens=toks, repetition_penalty=rep_pen)
        toks.append(next_token)
    gen_ids = toks[orig_len:]
    text = tokenizer.decode(gen_ids)
    return gen_ids, text

# run sweep
for temp in temps:
    for top_k in top_ks:
        for top_p in top_ps:
            for rep_pen in reps:
                scores = []
                texts = []
                ids_list = []
                for p in prompts:
                    gen_ids, text = generate_with_params(p, temp, top_k, top_p, rep_pen, max_new=20)
                    sc = score_text(gen_ids, text)
                    scores.append(sc)
                    texts.append(text)
                    ids_list.append(gen_ids)
                avg_score = float(np.mean(scores))
                results.append(((temp, top_k, top_p, rep_pen), avg_score, list(zip(prompts, texts, ids_list))))
                print(f"T={temp} k={top_k} p={top_p} rep={rep_pen} -> score={avg_score:.4f}")

# sort and show top 3
results.sort(key=lambda x: x[1], reverse=True)
print("\nTop 3 parameter settings:")
for i, (params, score, examples) in enumerate(results[:3], 1):
    temp, top_k, top_p, rep_pen = params
    print(f"\n#{i}: temp={temp} top_k={top_k} top_p={top_p} rep_pen={rep_pen} score={score:.4f}")
    for prompt, text, ids in examples:
        print(f"Prompt: {prompt}")
        print(f"Ids: {ids[:12]}... len={len(ids)}")
        print(f"Text: {text}")
        print("---")
