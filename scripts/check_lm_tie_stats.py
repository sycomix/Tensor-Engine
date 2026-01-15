#!/usr/bin/env python3
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from chat_llama import load_config_json, load_tokenizer, LlamaModel

model_dir = Path('examples/Llama-3.2-1B')
model_file = next(model_dir.glob('*.safetensors'))
logger.info('Using model file: %s', model_file)

config = load_config_json(model_dir)
_ = load_tokenizer(model_dir, strict=True)
model = LlamaModel(config)
model.load_weights(model_file)

emb = getattr(model, 'tok_emb', None)
lm_weight = None
if hasattr(model.lm_head, 'named_parameters'):
    try:
        for name, p in list(model.lm_head.named_parameters('')):
            if 'weight' in name.lstrip('.'):
                lm_weight = p
                break
    except Exception:
        pass

if emb is None or lm_weight is None:
    logger.error('Missing embedding or lm weight')
    raise SystemExit(1)

ed = np.array(emb.get_data(), dtype=np.float32)
eshape = list(emb.shape)
if len(eshape) != 2:
    logger.error('Unexpected embedding shape: %s', eshape)
    raise SystemExit(1)
vocab, hidden = eshape
ed = ed.reshape((vocab, hidden))
wd = np.array(lm_weight.get_data(), dtype=np.float32)
# reshape according to reported lm_weight.shape
try:
    wshape = list(lm_weight.shape)
    wd = wd.reshape(tuple(wshape))
except Exception:
    logger.warning('Could not reshape lm weight using its .shape; using flat array')

if wd.shape == (vocab, hidden):
    diff = wd - ed
elif wd.shape == (hidden, vocab):
    diff = wd - ed.T
else:
    logger.error('Shapes not compatible for comparison: emb %s lm %s', ed.shape, wd.shape)
    raise SystemExit(1)

absdiff = np.abs(diff)
for thr in [1e-6, 1e-4, 1e-2, 0.05, 0.1, 0.2, 0.5]:
    frac = np.mean(absdiff > thr)
    logger.info('Fraction of elements with abs diff > %g: %.6f', thr, frac)

mx = np.max(absdiff)
idx = np.unravel_index(np.argmax(absdiff), diff.shape)
logger.info('Max abs diff = %.6e at index %s', mx, idx)
logger.info('Sample values: emb=%f lm=%f diff=%f', ed.flatten()[0], wd.flatten()[0], diff.flatten()[0])
