from pathlib import Path
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure examples importable
examples_dir = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0, str(examples_dir))
from chat_llama import load_config_json, load_tokenizer, LlamaModel, GenerationConfig

model_dir = Path('examples/Llama-3.2-1B')
model_file = next(model_dir.glob('*.safetensors'))
logger.info('Using model file: %s', model_file)

config = load_config_json(model_dir)
tokenizer = load_tokenizer(model_dir, strict=True)
model = LlamaModel(config)
model.load_weights(model_file)

# Debug generation settings: greedy-like by using top_k=1
import numpy as np

def topk_probs(logits: np.ndarray, k: int = 5):
    """Return top-k indices and their probabilities (stable softmax)."""
    if k <= 0:
        k = 1
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    probs = exp / np.sum(exp)
    topk_idx = np.argsort(probs)[-k:][::-1]
    return topk_idx, probs[topk_idx]

# Print LM head and embedding diagnostics
try:
    emb = getattr(model, 'tok_emb', None)
    if emb is not None:
        edata = np.array(emb.get_data(), dtype=np.float32)
        eshape = list(emb.shape)
        logger.info('Embeddings shape=%s mean=%.6f std=%.6f', eshape, np.mean(edata), np.std(edata))
    # find lm head weight param
    lm_weight = None
    if hasattr(model.lm_head, 'named_parameters'):
        try:
            for name, p in list(cast(Any, model.lm_head).named_parameters('')):
                if 'weight' in name.lstrip('.'):
                    lm_weight = p
                    break
        except Exception:
            pass
    if lm_weight is not None:
        wdata = np.array(lm_weight.get_data(), dtype=np.float32)
        logger.info('LM head weight shape=%s mean=%.6f std=%.6f', list(lm_weight.shape), np.mean(wdata), np.std(wdata))
except Exception as exc:
    logger.warning('LM/embedding diagnostics failed: %s', exc)

# GREEDY run
gen = GenerationConfig(max_new_tokens=8, temperature=0.01, top_k=1, top_p=1.0, repetition_penalty=1.0)
prompt = '<|begin_of_text|> Hello'

# Tokenize prompt
input_ids = list(tokenizer.encode(prompt))
logger.info('Prompt token ids: %s', input_ids)
# show token strings for prompt
try:
    tokens = [tokenizer.id_to_token(t) for t in input_ids]
    logger.info('Prompt tokens: %s', tokens)
except Exception:
    logger.info('Tokenizer does not support id_to_token lookup')

# Run greedy generation step-by-step to capture token ids and top-5 probs
generated = []
orig_len = len(input_ids)
for step in range(gen.max_new_tokens):
    ids_arr = np.array(input_ids, dtype=np.int32).reshape(1, len(input_ids))
    try:
        from chat_llama import create_tensor
        ids_t = create_tensor(ids_arr.ravel().tolist(), [1, len(input_ids)])
        logits = model.forward(ids_t)
        vocab = model.config.vocab_size
        logits_flat = np.array(logits.get_data(), dtype=np.float32)
        last_logits = logits_flat[-vocab:]
    except Exception as exc:
        logger.error('Forward failed: %s', exc)
        break

    top_idx, top_p = topk_probs(last_logits, k=5)
    logger.info('Step %d top-5: %s', step, list(zip(list(top_idx), [float(x) for x in top_p])))

    # greedy via top_k=1
    best = int(top_idx[0])
    generated.append(best)
    input_ids.append(best)
    try:
        tok = tokenizer.id_to_token(best)
    except Exception:
        try:
            tok = tokenizer.decode([best])
        except Exception:
            tok = str(best)
    logger.info('Step %d: token_id=%d token=%s', step, best, tok)

logger.info('GREEDY generated token ids: %s', generated)
try:
    out = tokenizer.decode(generated)
    logger.info('GREEDY decoded generated text: %s', out)
except Exception as exc:
    logger.error('GREEDY decode failed: %s', exc)

# STOCHASTIC run with fixed seed to reproduce earlier sample
import random
seed = 42
np.random.seed(seed)
random.seed(seed)
logger.info('Running STOCHASTIC sampling with seed=%d', seed)

# Original stochastic profile
gen_s = GenerationConfig(max_new_tokens=8, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.0)
input_ids_s = list(tokenizer.encode(prompt))
stoch_generated = []
for step in range(gen_s.max_new_tokens):
    ids_arr = np.array(input_ids_s, dtype=np.int32).reshape(1, len(input_ids_s))
    try:
        ids_t = create_tensor(ids_arr.ravel().tolist(), [1, len(input_ids_s)])
        logits = model.forward(ids_t)
        vocab = model.config.vocab_size
        logits_flat = np.array(logits.get_data(), dtype=np.float32)
        last_logits = logits_flat[-vocab:]
    except Exception as exc:
        logger.error('Forward failed (stochastic): %s', exc)
        break

    # apply temperature
    scaled = last_logits / float(gen_s.temperature)
    top_idx, top_p = topk_probs(scaled, k=10)
    logger.info('STOCH Step %d top-10 (post-temp): %s', step, list(zip(list(top_idx), [float(x) for x in top_p])))

    # top-k filter
    if gen_s.top_k > 0 and gen_s.top_k < vocab:
        candidates = list(np.argpartition(scaled, -gen_s.top_k)[-gen_s.top_k:])
    else:
        candidates = list(range(vocab))

    # convert to probs and sample from masked candidate set
    s_logits = scaled.copy()
    mask = np.full_like(s_logits, -np.inf)
    mask[candidates] = s_logits[candidates]
    mask = mask - np.max(mask)
    probs = np.exp(mask)
    probs_sum = np.sum(probs)
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        chosen = int(np.argmax(mask))
    else:
        probs = probs / probs_sum
        chosen = int(np.random.choice(len(probs), p=probs))

    stoch_generated.append(chosen)
    input_ids_s.append(chosen)
    try:
        tok = tokenizer.id_to_token(chosen)
    except Exception:
        try:
            tok = tokenizer.decode([chosen])
        except Exception:
            tok = str(chosen)
    logger.info('STOCH Step %d: token_id=%d token=%s', step, chosen, tok)

logger.info('STOCH generated token ids: %s', stoch_generated)
try:
    out_s = tokenizer.decode(stoch_generated)
    logger.info('STOCH decoded generated text: %s', out_s)
except Exception as exc:
    logger.error('STOCH decode failed: %s', exc)
    print('STOCH tokens (as ids):', stoch_generated)
    try:
        print('STOCH tokens via id_to_token:')
        print([tokenizer.id_to_token(t) for t in stoch_generated])
    except Exception:
        pass

# If HuggingFace is available, compare its decoding of the generated ids for parity
try:
    from transformers import AutoTokenizer as HFAT
    hf_tok = HFAT.from_pretrained(str(model_file.parent))
    try:
        print('HF decode STOCH:', hf_tok.decode(stoch_generated))
        print('HF tokens STOCH:', hf_tok.convert_ids_to_tokens(stoch_generated))
    except Exception as exc:
        logger.warning('HF decode error for STOCH: %s', exc)
except Exception:
    logger.info('HF tokenizer not available for parity check')

# SAFER stochastic profile (lower temp/top_k)
seed = 42
np.random.seed(seed)
random.seed(seed)
logger.info('Running SAFER stochastic sampling with seed=%d', seed)

gen_safer = GenerationConfig(max_new_tokens=8, temperature=0.2, top_k=20, top_p=0.8, repetition_penalty=1.0)
input_ids_s2 = list(tokenizer.encode(prompt))
stoch_generated_safer = []
for step in range(gen_safer.max_new_tokens):
    ids_arr = np.array(input_ids_s2, dtype=np.int32).reshape(1, len(input_ids_s2))
    try:
        ids_t = create_tensor(ids_arr.ravel().tolist(), [1, len(input_ids_s2)])
        logits = model.forward(ids_t)
        vocab = model.config.vocab_size
        logits_flat = np.array(logits.get_data(), dtype=np.float32)
        last_logits = logits_flat[-vocab:]
    except Exception as exc:
        logger.error('Forward failed (safer stochastic): %s', exc)
        break

    # apply temperature
    scaled = last_logits / float(gen_safer.temperature)
    top_idx, top_p = topk_probs(scaled, k=10)
    logger.info('SAFER Step %d top-10 (post-temp): %s', step, list(zip(list(top_idx), [float(x) for x in top_p])))

    # top-k filter
    if gen_safer.top_k > 0 and gen_safer.top_k < vocab:
        candidates = list(np.argpartition(scaled, -gen_safer.top_k)[-gen_safer.top_k:])
    else:
        candidates = list(range(vocab))

    s_logits = scaled.copy()
    mask = np.full_like(s_logits, -np.inf)
    mask[candidates] = s_logits[candidates]
    mask = mask - np.max(mask)
    probs = np.exp(mask)
    probs_sum = np.sum(probs)
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        chosen = int(np.argmax(mask))
    else:
        probs = probs / probs_sum
        chosen = int(np.random.choice(len(probs), p=probs))

    stoch_generated_safer.append(chosen)
    input_ids_s2.append(chosen)
    try:
        tok = tokenizer.id_to_token(chosen)
    except Exception:
        try:
            tok = tokenizer.decode([chosen])
        except Exception:
            tok = str(chosen)
    logger.info('SAFER Step %d: token_id=%d token=%s', step, chosen, tok)

logger.info('SAFER generated token ids: %s', stoch_generated_safer)
try:
    out_s2 = tokenizer.decode(stoch_generated_safer)
    logger.info('SAFER decoded generated text: %s', out_s2)
except Exception as exc:
    logger.error('SAFER decode failed: %s', exc)
    print('SAFER tokens (as ids):', stoch_generated_safer)
    try:
        print('SAFER tokens via id_to_token:')
        print([tokenizer.id_to_token(t) for t in stoch_generated_safer])
    except Exception:
        pass

