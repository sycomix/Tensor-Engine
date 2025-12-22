from pathlib import Path
import sys
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger=logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examples.chat_llama import load_config_json, LlamaModel

model_path = Path("examples/Llama-3.2-1B/model.safetensors")
config = load_config_json(model_path)
model = LlamaModel(config)
model.load_weights(model_path)

# Inspect tokenizer round-trip
from examples.chat_llama import load_tokenizer
try:
    tok = load_tokenizer(model_path)
    sample = "roses are red"
    enc = tok.encode(sample)
    dec = tok.decode(enc)
    logger.info(f"tokenizer roundtrip: sample='{sample}' enc_len={len(enc)} dec='{dec[:200]}'")
except Exception as e:
    logger.error(f"Tokenizer test failed: {e}")

# Inspect token embeddings
try:
    tok = model.tok_emb
    tok_data = np.array(tok.get_data())
    tok_shape = tuple(tok.shape)
    logger.info(f"tok_emb.shape={tok_shape} mean={tok_data.mean():.6f} std={tok_data.std():.6f}")
except Exception as e:
    logger.error(f"tok_emb not available: {e}")

# Inspect LM head weights
try:
    lm_params = []
    if hasattr(model.lm_head,'named_parameters'):
        lm_params = list(model.lm_head.named_parameters(''))
    for name,param in lm_params:
        data = np.array(param.get_data())
        shape = tuple(param.shape) if hasattr(param,'shape') else (data.size,)
        logger.info(f"lm param {name} shape={shape} mean={data.mean():.6f} std={data.std():.6f}")
    # If weight exists, compare with tok_emb
    weight = None
    for name,param in lm_params:
        if name.lstrip('.').endswith('weight'):
            weight = np.array(param.get_data())
            break
    if weight is not None and tok is not None:
        wshape = tuple(param.shape)
        logger.info(f"lm_weight shape {wshape}")
        # try reshape
        if len(tok_shape)==2 and len(wshape)==2:
            vocab,hidden = tok_shape
            if wshape==(hidden,vocab):
                emb = tok_data.reshape(vocab,hidden)
                # compute cos similarity between emb.T flattened and weight flattened
                embT = emb.T.flatten()
                wflat = weight.flatten()
                corr = np.corrcoef(embT, wflat)[0,1]
                logger.info(f"corr between emb.T and lm_weight: {corr:.6f}")
except Exception as e:
    logger.error(f"Error inspecting LM head: {e}")
