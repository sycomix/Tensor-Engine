import torch, transformers, sys, os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = 'Qwen3-0.6B'
device = 'cpu'
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map=device, trust_remote_code=True)
model.eval()

prompt = 'What is the weather'
enc = tok(prompt, return_tensors='pt')
input_ids = enc['input_ids']
toks = [tok.decode([t]) for t in input_ids[0]]
print('Tokens:', toks)
print('Token IDs:', input_ids[0].tolist())
seq_len = input_ids.shape[1]

layer_q = {}
layer_k = {}
layer_q_normed = {}
layer_k_normed = {}
layer_q_rope = {}
layer_k_rope = {}
layer_attn_scores = {}

def get_attn_hook(layer_id):
    def hook(module, args, kwargs, output):
        # Qwen3Attention forward: (hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, position_embeddings)
        hidden_states = kwargs.get('hidden_states', args[0] if args else None)
        bs, sl, _ = hidden_states.shape
        
        n_heads = module.num_heads
        n_kv_heads = module.num_key_value_heads
        head_dim = module.head_dim
        
        q_raw = module.q_proj(hidden_states).view(bs, sl, n_heads, head_dim)
        k_raw = module.k_proj(hidden_states).view(bs, sl, n_kv_heads, head_dim)
        
        layer_q[layer_id] = q_raw[0].detach().clone()
        layer_k[layer_id] = k_raw[0].detach().clone()
        
        # Apply QK norm (applied to flattened q/k)
        if hasattr(module, 'q_norm') and module.q_norm is not None:
            q_flat = module.q_proj(hidden_states)
            q_normed = module.q_norm(q_flat).view(bs, sl, n_heads, head_dim)
        else:
            q_normed = q_raw
        
        if hasattr(module, 'k_norm') and module.k_norm is not None:
            k_flat = module.k_proj(hidden_states)
            k_normed = module.k_norm(k_flat).view(bs, sl, n_kv_heads, head_dim)
        else:
            k_normed = k_raw
        
        # Expand K for GQA
        if n_heads != n_kv_heads:
            k_normed = k_normed.expand(-1, -1, n_heads // n_kv_heads, -1).reshape(bs, sl, n_heads, head_dim)
        
        layer_q_normed[layer_id] = q_normed[0].detach().clone()
        layer_k_normed[layer_id] = k_normed[0].detach().clone()
        
        # RoPE
        position_ids = torch.arange(sl, dtype=torch.long).unsqueeze(0)
        cos, sin = module.rotary_emb(q_normed, position_ids)
        
        def rotate_half(x):
            x1 = x[..., :x.shape[-1]//2]
            x2 = x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_rope = q_normed * cos + rotate_half(q_normed) * sin
        k_rope = k_normed * cos + rotate_half(k_normed) * sin
        
        layer_q_rope[layer_id] = q_rope[0].detach().clone()
        layer_k_rope[layer_id] = k_rope[0].detach().clone()
        
        # Scores head 0
        q_h0 = q_rope[0, :, 0, :]
        k_h0 = k_rope[0, :, 0, :]
        scores = torch.matmul(q_h0, k_h0.T) / (head_dim ** 0.5)
        mask = torch.full((sl, sl), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        scores_masked = scores + mask
        probs = torch.softmax(scores_masked, dim=-1)
        layer_attn_scores[layer_id] = {
            'raw': scores.detach().clone(),
            'masked': scores_masked.detach().clone(),
            'probs': probs.detach().clone()
        }
    
    return hook

for i, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_hook(get_attn_hook(i), with_kwargs=True)

with torch.no_grad():
    out = model(input_ids, output_hidden_states=True)

print()
print('=== LAYER 0 ===')
print()

print('Q after projection (head 0, first 5 dims):')
for t in range(seq_len):
    vals = layer_q[0][t, 0, :5].tolist()
    print(f'  tok{t}: [{", ".join(f"{v:.6f}" for v in vals)}]')

print()
print('K after projection (head 0, first 5 dims):')
for t in range(seq_len):
    vals = layer_k[0][t, 0, :5].tolist()
    print(f'  tok{t}: [{", ".join(f"{v:.6f}" for v in vals)}]')

print()
print('Q after QK norm (head 0, first 5 dims):')
for t in range(seq_len):
    vals = layer_q_normed[0][t, 0, :5].tolist()
    print(f'  tok{t}: [{", ".join(f"{v:.6f}" for v in vals)}]')

print()
print('K after QK norm (head 0, first 5 dims):')
for t in range(seq_len):
    vals = layer_k_normed[0][t, 0, :5].tolist()
    print(f'  tok{t}: [{", ".join(f"{v:.6f}" for v in vals)}]')

print()
print('Q after RoPE (head 0, first 5 dims):')
for t in range(seq_len):
    vals = layer_q_rope[0][t, 0, :5].tolist()
    print(f'  tok{t}: [{", ".join(f"{v:.6f}" for v in vals)}]')

print()
print('K after RoPE (head 0, first 5 dims):')
for t in range(seq_len):
    vals = layer_k_rope[0][t, 0, :5].tolist()
    print(f'  tok{t}: [{", ".join(f"{v:.6f}" for v in vals)}]')

print()
scores = layer_attn_scores[0]
print('Attention raw scores (head 0):')
for i in range(seq_len):
    print(f'  tok{i}: [{", ".join(f"{v:.4f}" for v in scores["raw"][i].tolist())}]')

print()
print('Attention probs (head 0):')
for i in range(seq_len):
    print(f'  tok{i}: [{", ".join(f"{v:.6f}" for v in scores["probs"][i].tolist())}]')

# Also check if q_norm is per-head or global
print()
print('=== QK Norm weights ===')
for i in range(2):
    layer = model.model.layers[i]
    attn = layer.self_attn
    if hasattr(attn, 'q_norm') and attn.q_norm is not None:
        w = attn.q_norm.weight
        print(f'Layer {i} q_norm shape: {w.shape}, first 5: {w[:5].tolist()}')
    if hasattr(attn, 'k_norm') and attn.k_norm is not None:
        w = attn.k_norm.weight
        print(f'Layer {i} k_norm shape: {w.shape}, first 5: {w[:5].tolist()}')
