import tensor_engine as te
import numpy as np
m = te.Llama(128256,3072,32,8192,24,24)
files = [r"examples/Llama-3.2-3B-Instruct/model-00001-of-00002.safetensors", r"examples/Llama-3.2-3B-Instruct/model-00002-of-00002.safetensors"]
for f in files:
    b = open(f,'rb').read()
    te.py_load_safetensors_into_module(b, True, m, "model")
params = m.named_parameters("")
print('num params', len(params))
for (n, t) in params[:20]:
    arr = np.array(t.get_data()).reshape(t.shape)
    print(n, arr.shape, float(arr.mean()), float(arr.std()), float(arr.max()), float(arr.min()))
# check lm_head
for (n, t) in params:
    if n.endswith('lm_head.weight'):
        arr = np.array(t.get_data()).reshape(t.shape)
        print('lm_head mean/std/max', float(arr.mean()), float(arr.std()), float(arr.max()), float(arr.min()))
