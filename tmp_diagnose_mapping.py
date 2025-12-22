from pathlib import Path
import logging
from examples.chat_llama import load_config_json, LlamaModel
import tensor_engine as te

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger=logging.getLogger(__name__)

model_path = Path("examples/Llama-3.2-1B/model.safetensors")
config = load_config_json(model_path)
model = LlamaModel(config)
state = te.py_load_safetensors(open(model_path,'rb').read(), transpose=False)

# gather model named parameters
all_params = []
# lm head
if hasattr(model.lm_head,'named_parameters'):
    try:
        all_params.extend([(f"lm_head::{name}", param) for name,param in model.lm_head.named_parameters('')])
    except Exception:
        pass
# transformer blocks
for i, blk in enumerate(model.layers):
    try:
        named = list(blk.named_parameters(''))
    except Exception:
        named = []
    for name,param in named:
        all_params.append((f"layer_{i}::{name}", param))

# now check mappings
missing = []
mismatched = []
matched = []
for pname,param in all_params:
    group, lname = pname.split('::',1)
    # construct expected keys with appropriate prefixes
    candidates = []
    if group == 'lm_head':
        candidates = ["lm_head.weight","lm_head.bias","output.weight","output.bias","model.lm_head.weight","model.output.weight","model.output.bias"]
    else:
        # layer_{i}:: -> model.layers.{i}.{name}
        layer_idx = int(group.split('_')[1])
        base = f"model.layers.{layer_idx}.{lname.lstrip('.') }"
        # also include older/huggingface names
        alt = base.replace('.mha.linear_q.weight','self_attn.q_proj.weight').replace('.mha.linear_k.weight','self_attn.k_proj.weight').replace('.mha.linear_v.weight','self_attn.v_proj.weight').replace('.mha.linear_o.weight','self_attn.o_proj.weight')
        candidates = [base, alt, lname.lstrip('.')]

    found = False
    for c in candidates:
        if c in state:
            found = True
            # compare shapes if possible
            try:
                s = state[c]
                shape_s = tuple(s.shape)
                try:
                    shape_p = tuple(param.shape)
                except Exception:
                    shape_p = None
                if shape_p and shape_s != shape_p:
                    mismatched.append((pname,c,shape_p,shape_s))
                else:
                    matched.append((pname,c))
            except Exception as e:
                matched.append((pname,c))
            break
    if not found:
        missing.append((pname,candidates))

logger.info(f"Total model params found: {len(all_params)}")
logger.info(f"Matched: {len(matched)}; Mismatched: {len(mismatched)}; Missing: {len(missing)}")
if mismatched:
    logger.info("Mismatched examples (first 20):")
    for m in mismatched[:20]:
        logger.info(str(m))
if missing:
    logger.info("Missing examples (first 20):")
    for m in missing[:20]:
        logger.info(str(m))
