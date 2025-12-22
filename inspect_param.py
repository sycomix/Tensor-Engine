import tensor_engine as te
block = te.TransformerBlock(d_model=2048,d_ff=8192,num_heads=32,use_rope=True,llama_style=True)
params = list(block.named_parameters(''))
name, p = params[0]
print(name, p.shape)
print('methods:', [m for m in dir(p) if not m.startswith('_')])
try:
    d = p.get_data()
    print('get_data len', len(d))
except Exception as e:
    print('get_data error', e)
try:
    lock = p.lock()
    print('lock OK, has storage attr?', hasattr(lock, 'storage'))
    print('requires_grad:', getattr(lock, 'requires_grad', 'no attr'))
except Exception as e:
    print('lock error', e)
