import tensor_engine as te
from pathlib import Path
p = Path('examples/Llama-3.2-1B/model.safetensors')
with p.open('rb') as f:
    b = f.read()
state = te.py_load_safetensors(b, transpose=False)
for k in sorted(state.keys()):
    if 'head' in k or 'lm' in k or 'output' in k:
        print(k)
