import tensor_engine as te
b=open('examples/Llama-3.2-1B/model.safetensors','rb').read()
s=te.py_load_safetensors(b, transpose=False)
print('Total keys:', len(s))
print('\nSample keys in layer 0:')
for k in sorted([k for k in s.keys() if k.startswith('model.layers.0')])[:80]:
    print(k, getattr(s[k],'shape',None))
exp = ['model.layers.0.mha.linear_q.weight','model.layers.0.mha.linear_k.weight','model.layers.0.mha.linear_v.weight','model.layers.0.mha.linear_o.weight','model.layers.0.linear1.weight','model.layers.0.linear2.weight']
print('\nExpected augmented keys present?')
for e in exp:
    print(e, e in s)
