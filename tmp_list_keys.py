from safetensors import safe_open
f = './examples/Llama-3.2-3B-Instruct/model-00001-of-00002.safetensors'
with safe_open(f, framework='numpy') as s:
    keys = list(s.keys())
print('example keys:', [k for k in keys if 'layers.0' in k][:200])
