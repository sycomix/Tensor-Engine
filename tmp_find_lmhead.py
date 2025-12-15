from safetensors import safe_open
for f in ['examples/Llama-3.2-3B-Instruct/model-00001-of-00002.safetensors','examples/Llama-3.2-3B-Instruct/model-00002-of-00002.safetensors']:
    print('file', f)
    with safe_open(f, framework='numpy') as s:
        print([k for k in s.keys() if 'lm_head' in k or 'embed_tokens' in k or 'lm' in k][:200])
