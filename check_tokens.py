import json
with open('Qwen2.5-1.5B-Instruct/tokenizer.json', 'r', encoding='utf-8') as f:
    tok = json.load(f)
for t in tok.get('added_tokens', []):
    if 'im_' in t.get('content', ''):
        print(t)
print(f'Total added_tokens: {len(tok.get("added_tokens", []))}')
# Check if normalizer handles byte tokens
model = tok.get('model', {})
vocab = model.get('vocab', {})
# find <|im_start|> in vocab
for k, v in vocab.items():
    if 'im_start' in k or 'im_end' in k:
        print(f'Vocab entry: {k!r} -> {v}')
