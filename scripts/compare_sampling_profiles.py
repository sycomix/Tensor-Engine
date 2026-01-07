#!/usr/bin/env python3
"""Run generate_text with several sampling profiles and print outputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from chat_llama import load_config_json, load_tokenizer, LlamaModel, GenerationConfig, generate_text

model_dir = Path('examples/Llama-3.2-1B')
model_file = next(model_dir.glob('*.safetensors'))
print('Model:', model_file)
config = load_config_json(model_dir)
tokenizer = load_tokenizer(model_dir, strict=True)
model = LlamaModel(config)
model.load_weights(model_file)

prompt = '<|begin_of_text|> Hello'

profiles = {
    'original': GenerationConfig(max_new_tokens=8, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.0),
    'safer': GenerationConfig(max_new_tokens=8, temperature=0.2, top_k=20, top_p=0.8, repetition_penalty=1.0),
    'greedy': GenerationConfig(max_new_tokens=8, temperature=0.01, top_k=1, top_p=1.0, repetition_penalty=1.0),
}

seeds = {'original': 42, 'safer': 123, 'greedy': 7}
for name, cfg in profiles.items():
    seed = seeds.get(name)
    out_raw = generate_text(model, tokenizer, prompt, cfg, postprocess=False, seed=seed)
    out_pp = generate_text(model, tokenizer, prompt, cfg, postprocess=True, seed=seed)
    print('---')
    print('Profile:', name)
    print('Raw :', out_raw)
    print('Post:', out_pp)
