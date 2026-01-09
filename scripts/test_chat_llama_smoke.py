import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('chat_llama', Path(__file__).resolve().parents[1] / 'examples' / 'chat_llama.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
conf = mod.ModelConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=128)
model = mod.LlamaModel(conf)
import numpy as np
ids = np.array([[1,2,3,4]], dtype=np.int32)
ids_t = mod.create_tensor(ids.ravel().tolist(), [1, ids.shape[1]])
logits = model.forward(ids_t)
print('logits shape:', logits.shape)
print('First 10 logits:', logits.get_data()[:10])
