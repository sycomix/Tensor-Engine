import tensor_engine as te
from pathlib import Path
p = Path('examples/Llama-3.2-1B/model.safetensors')
with p.open('rb') as f:
    b = f.read()
d = te.py_load_safetensors(b, transpose=False)
print('total:', len(d))
print('\nFirst 80 keys:')
for i,k in enumerate(sorted(d.keys())[:80]):
    print(i, k)
