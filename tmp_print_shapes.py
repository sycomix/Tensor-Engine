from safetensors import safe_open
f = './examples/Llama-3.2-3B-Instruct/model-00001-of-00002.safetensors'
from safetensors import safe_open
with safe_open(f, framework='numpy') as s:
    print('safe_open dir:', [n for n in dir(s) if not n.startswith('_')])
    print('keys:', list(s.keys())[:30])
    # Try getting metadata via s.get_meta if present
    meta = s.metadata()
    print('meta keys sample:', list(meta.keys())[:20])
    for k in list(s.keys())[:30]:
        print('key from keys_list:', repr(k))
        if k in meta:
            m = meta[k]
            print(k, m['shape'], m['dtype'])
        else:
            print('key not found in metadata map')
