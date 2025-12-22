import logging
import tensor_engine as te
from safetensors import safe_open

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

block = te.TransformerBlock(d_model=2048,d_ff=8192,num_heads=32,use_rope=True,llama_style=True)
logger.info('Created block')
with open('d:/models/model.safetensors','rb') as f:
    model_bytes = f.read()
state = te.py_load_safetensors(model_bytes, transpose=False)
logger.info('Loaded state length %d', len(state))

params = list(block.named_parameters(''))
name, param = params[0]
logger.info('Target param: %s shape %s', name, param.shape)
key = 'model.layers.0.self_attn.q_proj.weight'
if key in state:
    w = state[key]
    logger.info('state tensor shape %s %s', w.shape, type(w))
    try:
        # try set_data with tensor
        param.set_data(w)
        logger.info('set_data with tensor succeeded')
    except Exception as e:
        logger.error('set_data with tensor failed: %s', e)
    try:
        # try set_data with list
        param.set_data(w.get_data())
        logger.info('set_data with list succeeded')
    except Exception as e:
        logger.error('set_data with list failed: %s', e)
else:
    logger.warning('key missing')
