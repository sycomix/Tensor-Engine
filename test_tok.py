import sys
sys.stdout.reconfigure(encoding='utf-8')
from tokenizers import Tokenizer
tok = Tokenizer.from_file('Qwen3-0.6B/tokenizer.json')
# Test the exact BPE encoding to compare with our Rust output
for text in ['Step-by-step', 'Hello', 'Answer']:
    ids = tok.encode(text, add_special_tokens=False).ids
    tokens = [tok.decode([i]) for i in ids]
    print(f'{repr(text)} -> ids={ids} -> tokens={tokens}')
