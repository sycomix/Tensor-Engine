import json

# Load the tokenizer
with open('Qwen2.5-1.5B-Instruct/tokenizer.json', 'r', encoding='utf-8') as f:
    tok = json.load(f)

vocab = tok['model']['vocab']

# Try to find how the tokenizer handles <|im_start|> and <|im_end|>
# In HuggingFace tokenizers, added_tokens are merged via a pre_tokenizer
# The actual tokenization might split on <|im_start|> boundaries

# Let's check the pre_tokenizer configuration
pre_tokenizer = tok['model'].get('pre_tokenizer', {})
print("Pre-tokenizer:", json.dumps(pre_tokenizer, indent=2)[:500])

# Check the normalizer
normalizer = tok['model'].get('normalizer', {})
print("\nNormalizer:", json.dumps(normalizer, indent=2)[:500])

# Check the decoder
decoder = tok['model'].get('decoder', {})
print("\nDecoder type:", decoder.get('type', 'unknown'))

# Check added_tokens more carefully
added_tokens = tok.get('added_tokens', [])
for at in added_tokens:
    print(f"  id={at['id']}, content={at['content']!r}, special={at.get('special', False)}")
