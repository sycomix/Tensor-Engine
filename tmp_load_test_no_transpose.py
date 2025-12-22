import tensor_engine as te
import numpy as np
m = te.Llama(128256,3072,32,8192,24,24)
files = [r"examples/Llama-3.2-3B-Instruct/model-00001-of-00002.safetensors", r"examples/Llama-3.2-3B-Instruct/model-00002-of-00002.safetensors"]
for f in files:
    b = open(f,'rb').read()
    te.py_load_safetensors_into_module(b, False, m, "model")
# sample tokens
tokens = [128000, 1985]
arr = te.Tensor(tokens, [1, len(tokens)])
logits = m.forward(arr)
print('logits shape', logits.shape)
ld = logits.get_data()
log_np = np.array(ld).reshape(logits.shape)
print('last token argmax:', int(np.argmax(log_np[-1])))
print('last token max value:', float(np.max(log_np[-1])))
