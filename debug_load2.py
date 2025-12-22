import tensor_engine as te

block = te.TransformerBlock(d_model=2048,d_ff=8192,num_heads=32,use_rope=True,llama_style=True)
print('Created block')
with open('d:/models/model.safetensors','rb') as f:
    model_bytes = f.read()
state = te.py_load_safetensors(model_bytes, transpose=False)
print('Loaded state dict length=', len(state))
try:
    block.load_state_dict(state, 'model.layers.0')
    print('load_state_dict returned')
except Exception as e:
    print('load_state_dict raised', e)
