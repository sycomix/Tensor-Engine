#!/usr/bin/env python3
"""Check LM head tie against token embeddings and print stats."""
from pathlib import Path
import sys
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from chat_llama import load_config_json, load_tokenizer, LlamaModel

model_dir = Path('examples/Llama-3.2-1B')
model_file = next(model_dir.glob('*.safetensors'))
logger.info('Using model file: %s', model_file)

config = load_config_json(model_dir)
# Load tokenizer strict to avoid HF fallback
_ = load_tokenizer(model_dir, strict=True)
model = LlamaModel(config)
model.load_weights(model_file)

# embedding
emb = getattr(model, 'tok_emb', None)
if emb is None:
    logger.error('No token embedding found')
else:
    ed = np.array(emb.get_data(), dtype=np.float32)
    eshape = list(emb.shape)
    logger.info('Embedding shape=%s mean=%.6f std=%.6f', eshape, ed.mean(), ed.std())

# find lm head weight param
lm_weight = None
if hasattr(model.lm_head, 'named_parameters'):
    try:
        for name, p in list(model.lm_head.named_parameters('')):
            if 'weight' in name.lstrip('.'):
                lm_weight = p
                break
    except Exception:
        pass

if lm_weight is None:
    logger.error('No LM head weight found (named_parameters missing)')
else:
    wd = np.array(lm_weight.get_data(), dtype=np.float32)
    wshape = list(lm_weight.shape)
    logger.info('LM weight shape=%s mean=%.6f std=%.6f', wshape, wd.mean(), wd.std())

    # compare
    if emb is not None and len(eshape) == 2:
        vocab_size, hidden_size = eshape
        if wshape == [vocab_size, hidden_size]:
            diff = np.max(np.abs(wd - ed))
            logger.info('LM weight matches embeddings (direct) max_abs_diff=%.6e', diff)
        elif wshape == [hidden_size, vocab_size]:
            diff = np.max(np.abs(wd - ed.T))
            logger.info('LM weight matches embeddings (transposed) max_abs_diff=%.6e', diff)
        else:
            logger.warning('LM weight shape not matching embedding shapes; cannot directly compare')
