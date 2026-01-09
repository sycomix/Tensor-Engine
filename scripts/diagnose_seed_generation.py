#!/usr/bin/env python3
"""Run per-step diagnostics for a single profile+seed

Outputs per-step: step, top-k (idx,prob), chosen idx, chosen token (TE), chosen token (HF), entropy
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import logging
import numpy as np
from typing import List

from chat_llama import load_config_json, load_tokenizer, LlamaModel, GenerationConfig, create_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path('examples/Llama-3.2-1B')
MODEL_FILE = next(MODEL_DIR.glob('*.safetensors'))

config = load_config_json(MODEL_DIR)
tokenizer = load_tokenizer(MODEL_DIR, strict=True)
# Try HF tokenizer too
try:
    from transformers import AutoTokenizer as HFAuto
    hf_tok = HFAuto.from_pretrained(str(MODEL_DIR))
except Exception:
    hf_tok = None

model = LlamaModel(config)
model.load_weights(MODEL_FILE)

profiles = {
    'original': GenerationConfig(max_new_tokens=8, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.0),
    'safer': GenerationConfig(max_new_tokens=8, temperature=0.2, top_k=20, top_p=0.8, repetition_penalty=1.0),
    'greedy': GenerationConfig(max_new_tokens=8, temperature=0.01, top_k=1, top_p=1.0, repetition_penalty=1.0),
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--profile', choices=list(profiles.keys()), default='original')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--prompt', type=str, default='<|begin_of_text|> Hello')
args = parser.parse_args()

cfg = profiles[args.profile]
seed = args.seed
prompt = args.prompt

np.random.seed(seed)

input_ids: List[int] = list(tokenizer.encode(prompt))
logger.info('Prompt token ids: %s', input_ids)

chosen_ids = []
for step in range(cfg.max_new_tokens):
    ids_arr = np.array(input_ids, dtype=np.int32).reshape(1, len(input_ids))
    ids_t = create_tensor(ids_arr.ravel().tolist(), [1, len(input_ids)])
    logits = model.forward(ids_t)
    vocab = model.config.vocab_size
    logits_flat = np.array(logits.get_data(), dtype=np.float32)
    last_logits = logits_flat[-vocab:]

    # apply temperature scaling
    scaled = last_logits / float(cfg.temperature)
    # compute softmax probs (stable)
    smax = scaled - np.max(scaled)
    exp = np.exp(smax)
    probs = exp / np.sum(exp)

    # entropy
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-20, 1.0)))

    # top-k list (use min of cfg.top_k and 20 for display)
    K_display = min(20, len(probs))
    topk_idx = np.argsort(probs)[-K_display:][::-1]
    topk = [(int(i), float(probs[i])) for i in topk_idx]

    # perform top_k filter used by sampler
    if cfg.top_k > 0 and cfg.top_k < vocab:
        candidates = list(np.argpartition(scaled, -cfg.top_k)[-cfg.top_k:])
    else:
        candidates = list(range(vocab))

    mask = np.full_like(scaled, -np.inf)
    mask[candidates] = scaled[candidates]
    mask = mask - np.max(mask)
    masked_probs = np.exp(mask)
    ssum = np.sum(masked_probs)
    if ssum <= 0 or not np.isfinite(ssum):
        chosen = int(np.argmax(mask))
    else:
        masked_probs = masked_probs / ssum
        chosen = int(np.random.choice(len(masked_probs), p=masked_probs))

    chosen_ids.append(chosen)
    input_ids.append(chosen)

    # decode chosen with tokenizer wrappers
    try:
        te_tok = tokenizer.id_to_token(chosen) if hasattr(tokenizer, 'id_to_token') else tokenizer.decode([chosen])
    except Exception:
        te_tok = '<decode-fail>'
    if hf_tok is not None:
        try:
            hf_tok_str = hf_tok.convert_ids_to_tokens([chosen])[0]
        except Exception:
            hf_tok_str = '<hf-decode-fail>'
    else:
        hf_tok_str = None

    # print summary for the step
    print('STEP', step)
    print(' entropy=', entropy)
    print(' topk (idx,prob)=', topk[:10])
    print(' chosen=', chosen, ' te_token=', te_tok, ' hf_token=', hf_tok_str)
    print(' chosen_in_cfg.top_k=', chosen in candidates)
    print('-' * 40)

print('Generated ids:', chosen_ids)
if hf_tok is not None:
    try:
        print('TE decode:', tokenizer.decode(chosen_ids))
    except Exception:
        print('TE decode failed')
    try:
        print('HF decode:', hf_tok.decode(chosen_ids))
    except Exception:
        print('HF decode failed')
else:
    try:
        print('TE decode:', tokenizer.decode(chosen_ids))
    except Exception:
        print('TE decode failed')
