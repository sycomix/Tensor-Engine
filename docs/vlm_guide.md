# Vision-Language Model (VLM) Guide

This guide documents how to use the `MultimodalLLM` and associated IO utilities in `tensor-engine`.

## Prefill / Decode Usage

To optimize multimodal decoding, the `MultimodalLLM` supports a prefill phase followed by decode steps:

- `prefill(images, input_ids=None)` computes and caches the image features and optional text prefix embeddings in a `ModalMemoryContext`.
- `decode_step(memory, new_input_ids)` appends `new_input_ids` token embeddings into the memory context and returns logits and an updated `ModalMemoryContext` for further decoding.

This allows efficient autoregressive generation without recomputing image features at each token step.

## ImageTextDataLoader

`ImageTextDataLoader` loads image/caption pairs from a manifest file where each line is:

```
<image_path>\t<caption>
```

Usage example (Rust):

```rust
use tensor_engine::io::image_text_dataloader::ImageTextDataLoader;
let loader = ImageTextDataLoader::new_from_manifest("manifest.txt", (224,224), 8, true, true)?;
let (images, captions) = loader.load_batch(0)?;
```

For Python, use the `ImageTextDataLoader` wrapper when compiled with `python_bindings` and `vision` features:

```python
from tensor_engine import ImageTextDataLoader
loader = ImageTextDataLoader('manifest.txt', 224, 224, 8, True, True)
images, captions = loader.load_batch(0)
```

## Tokenizer

If `with_tokenizers` and `python_bindings` are enabled, `PyTokenizer` exposes `encode(text)` and `decode(ids)`.

## Examples

See `examples/generate_llava.py` and `examples/train_llava.py` for end-to-end training and inference.
