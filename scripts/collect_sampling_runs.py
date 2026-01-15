#!/usr/bin/env python3
"""Collect sampling outputs across seeds and profiles and write JSON results."""
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure examples importable
examples_dir = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0, str(examples_dir))
from chat_llama import load_config_json, load_tokenizer, LlamaModel, GenerationConfig, create_tensor
import numpy as np

MODEL_DIR = Path('examples/Llama-3.2-1B')
MODEL_FILE = next(MODEL_DIR.glob('*.safetensors'))

config = load_config_json(MODEL_DIR)
tokenizer = load_tokenizer(MODEL_DIR, strict=True)
model = LlamaModel(config)
model.load_weights(MODEL_FILE)

prompt = '<|begin_of_text|> Hello'

profiles = {
    'original': GenerationConfig(max_new_tokens=8, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.0),
    'safer': GenerationConfig(max_new_tokens=8, temperature=0.2, top_k=20, top_p=0.8, repetition_penalty=1.0),
    'greedy': GenerationConfig(max_new_tokens=8, temperature=0.01, top_k=1, top_p=1.0, repetition_penalty=1.0),
}

seeds = list(range(30))
results: Dict[str, Dict[str, Dict]] = {}

for pname, cfg in profiles.items():
    results[pname] = {}
    for seed in seeds:
        np.random.seed(seed)
        ids_in = list(tokenizer.encode(prompt))
        generated: List[int] = []
        per_step: List[Dict] = []
        for step in range(cfg.max_new_tokens):
            ids_arr = np.array(ids_in, dtype=np.int32).reshape(1, len(ids_in))
            ids_t = create_tensor(ids_arr.ravel().tolist(), [1, len(ids_in)])
            logits = model.forward(ids_t)
            vocab = model.config.vocab_size
            logits_flat = np.array(logits.get_data(), dtype=np.float32)
            last_logits = logits_flat[-vocab:]

            # scale
            scaled = last_logits / float(cfg.temperature)
            # top-k filter
            if cfg.top_k > 0 and cfg.top_k < vocab:
                candidates = list(np.argpartition(scaled, -cfg.top_k)[-cfg.top_k:])
            else:
                candidates = list(range(vocab))
            # mask candidates
            mask = np.full_like(scaled, -np.inf)
            mask[candidates] = scaled[candidates]
            mask = mask - np.max(mask)
            probs = np.exp(mask)
            ssum = np.sum(probs)
            if ssum <= 0 or not np.isfinite(ssum):
                chosen = int(np.argmax(mask))
                # normalize a safe fallback
                probs = probs / (np.sum(probs) + 1e-12)
                chosen_prob = float(probs[chosen]) if 0 <= chosen < len(probs) else 0.0
            else:
                probs = probs / ssum
                chosen = int(np.random.choice(len(probs), p=probs))
                chosen_prob = float(probs[chosen])

            # compute rank (1-based)
            sorted_idx = np.argsort(-probs)
            rank = int(np.where(sorted_idx == chosen)[0][0]) + 1 if chosen in sorted_idx else -1
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

            try:
                token_str = tokenizer.id_to_token(chosen)
            except Exception:
                try:
                    token_str = tokenizer.decode([chosen])
                except Exception:
                    token_str = str(chosen)

            per_step.append({
                'step': step,
                'chosen': int(chosen),
                'prob': chosen_prob,
                'rank': int(rank),
                'chosen_in_topk': bool(chosen in candidates),
                'entropy': entropy,
                'token_str': token_str,
            })

            generated.append(int(chosen))
            ids_in.append(int(chosen))

        # decode and record
        try:
            decoded = tokenizer.decode(generated)
        except Exception:
            decoded = ""
        # token strings via id_to_token if available
        token_strs = []
        for t in generated:
            try:
                token_strs.append(tokenizer.id_to_token(t))
            except Exception:
                try:
                    token_strs.append(tokenizer.decode([t]))
                except Exception:
                    token_strs.append(str(t))

        results[pname][str(seed)] = {
            'seed': seed,
            'token_ids': generated,
            'token_strs': token_strs,
            'decoded': decoded,
            'per_step': per_step,
        }
        logger.info('Profile=%s seed=%d -> %s', pname, seed, decoded)

# write to file
out_path = Path('scripts/sampling_results.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

logger.info('Wrote sampling results to %s', out_path)
