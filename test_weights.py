import tensor_engine as te
import numpy as np
from safetensors import safe_open

block = te.TransformerBlock(
    d_model=2048,
    d_ff=8192,
    num_heads=32,
    use_rope=True,
    llama_style=True
)

print('Testing py_load_safetensors with bfloat16:')
with open('d:\\models\\model.safetensors', 'rb') as f:
    model_bytes = f.read()

state_dict = te.py_load_safetensors(model_bytes, transpose=False)
print(f'  Loaded {len(state_dict)} tensors')

q_weight_tensor = state_dict.get('model.layers.0.self_attn.q_proj.weight')
if q_weight_tensor:
    print(f'  q_proj tensor type: {type(q_weight_tensor)}')
    print(f'  q_proj tensor shape: {q_weight_tensor.shape}')
    has_get_data = hasattr(q_weight_tensor, 'get_data')
    print(f'  Can get_data: {has_get_data}')
    
    print('\nTrying to get data and convert:')
    try:
        data = q_weight_tensor.get_data()
        data_array = np.array(data, dtype=np.float32)
        print(f'  Data array shape: {data_array.shape}, dtype: {data_array.dtype}')
        print(f'  First few values: {data_array.ravel()[:10]}')
    except Exception as e:
        print(f'  Failed: {e}')

