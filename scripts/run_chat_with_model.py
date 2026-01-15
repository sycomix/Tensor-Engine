import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure `examples` is importable
examples_dir = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0, str(examples_dir))
from chat_llama import load_config_json, load_tokenizer, LlamaModel, GenerationConfig, generate_text

import argparse
parser = argparse.ArgumentParser(description='Run chat_llama with a local model (shard or index)')
parser.add_argument('model_dir', nargs='?', default='examples/Llama-3.2-3B-Instruct', help='Path to a model directory')
args = parser.parse_args()

model_dir = Path(args.model_dir)
index_file = model_dir / 'model.safetensors.index.json'
# Prefer loading a single shard (first matching model-*-of-*.safetensors); index file is not directly supported by rust loader
if index_file.exists():
    # pick first shard file (e.g., model-00001-of-00002.safetensors)
    model_file = next(model_dir.glob('model-*-of-*.safetensors'), None)
    if model_file is None:
        model_file = next(model_dir.glob('*.safetensors'), None)
else:
    model_file = next(model_dir.glob('*.safetensors'), None)
if model_file is None:
    logger.error('No .safetensors file found in %s', model_dir)
    sys.exit(1)

logger.info('Using model file: %s (model dir: %s)', model_file, model_dir)
try:
    config = load_config_json(model_dir)
    tokenizer = load_tokenizer(model_dir, strict=True)
except Exception as exc:
    logger.error('Failed to load config/tokenizer: %s', exc)
    raise

model = LlamaModel(config)
try:
    model.load_weights(model_file)
except Exception as exc:
    logger.error('Failed to load weights from %s: %s', model_file, exc)
    raise

# Small generation
gen = GenerationConfig(max_new_tokens=8, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.0)
prompt = '<|begin_of_text|> Hello'
try:
    out = generate_text(model, tokenizer, prompt, gen)
    print('Generated:', out)
except Exception as exc:
    logger.error('Generation failed: %s', exc)
    raise
