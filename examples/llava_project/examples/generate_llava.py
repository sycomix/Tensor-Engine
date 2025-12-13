#!/usr/bin/env python3
# pylint: disable=E1101,E1102,E0401,W0611,C0415,C0301,C0103
# type: ignore
"""
Simple generation example using the toy LLAVA-like model.
"""
from __future__ import annotations
import argparse
from typing import Any
import logging
from pathlib import Path
import numpy as np

try:
    import tensor_engine as te  # type: ignore
except ImportError:  # pragma: no cover
    te = None  # type: ignore


def simple_tokenize(text: str, vocab: dict, bos: int = 1, eos: int = 2) -> list[int]:
    """A simple whitespace tokenizer into ids using provided vocab.

    This is intentionally simplistic for the demo dataset.
    """
    return [bos] + [vocab.get(w, 0) for w in text.split()] + [eos]


def main() -> None:
    """CLI entry point to run simple generation with the toy model."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Describe the image 0")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--model_path", default="examples/models/llava_model.safetensors")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    if te is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")
    # Validate presence of core Python bindings
    required = ['Tensor', 'VisionTransformer', 'MultimodalLLM']
    missing = [r for r in required if not hasattr(te, r)]
    if missing:
        raise RuntimeError(f"tensor_engine Python bindings missing required classes: {missing}. Build with python_bindings and vision features enabled.")

    d_model = 32
    vocab_size = 256

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    words = "Describe the image 0 Synthetic image description".split()
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)

    if args.config:
        import json
        cfg = json.load(open(args.config, 'r', encoding='utf-8'))
        vt_class: Any = getattr(te, 'VisionTransformer', None)
        mm_class: Any = getattr(te, 'MultimodalLLM', None)
        if vt_class is None or mm_class is None:
            raise RuntimeError('tensor_engine module does not expose VisionTransformer and/or MultimodalLLM. Rebuild the package with python_bindings and vision enabled.')
        if not callable(vt_class):
            raise RuntimeError('VisionTransformer class is not callable')
        # vt_class is a runtime class from pyo3 bindings; disable static not-callable warning
        # pylint: disable=not-callable
        vision = vt_class(
            cfg.get('c', 3),
            cfg.get('patch_size', 8),
            cfg.get('d_model', d_model),
            cfg.get('d_ff', d_model * 4),
            cfg.get('num_heads', 4),
            cfg.get('depth', 2),
            cfg.get('max_len', 512),
        )
        if not callable(mm_class):
            raise RuntimeError('MultimodalLLM class is not callable')
        # pylint: disable=not-callable
        model = mm_class(
            vision,
            cfg.get('vocab_size', vocab_size),
            cfg.get('d_model', d_model),
            cfg.get('d_ff', d_model * 4),
            cfg.get('num_heads', 4),
            cfg.get('depth', 2),
        )
    else:
        vt_class: Any = getattr(te, 'VisionTransformer', None)
        mm_class: Any = getattr(te, 'MultimodalLLM', None)
        if vt_class is None or mm_class is None:
            raise RuntimeError('tensor_engine module does not expose VisionTransformer and/or MultimodalLLM. Rebuild the package with python_bindings and vision enabled.')
        if not callable(vt_class):
            raise RuntimeError('VisionTransformer class is not callable')
        # pylint: disable=not-callable
        vision = vt_class(
            3,
            8,
            d_model,
            d_model * 4,
            num_heads=4,
            depth=2,
            max_len=512,
        )
        if not callable(mm_class):
            raise RuntimeError('MultimodalLLM class is not callable')
        # pylint: disable=not-callable
        model = mm_class(
            vision,
            vocab_size,
            d_model,
            d_model * 4,
            num_heads=4,
            depth=2,
        )

    hf_tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            hf_tok = AutoTokenizer.from_pretrained(args.tokenizer)
        except (ImportError, OSError, ValueError) as err:
            logger.warning("Failed to load HF tokenizer; continuing without tokenizer: %s", err)

    ids = simple_tokenize(args.prompt, vocab)
    ids_arr = np.array(ids, dtype=np.float32).reshape((1, len(ids)))
    # Use the package-provided Tensor class via getattr to avoid static lint warnings
    TensorClass: Any = getattr(te, 'Tensor', None)
    # TensorClass is a runtime-provided class from pyo3 bindings; disable static not-callable warning
    # pylint: disable=not-callable
    ids_t = TensorClass(ids_arr.flatten().tolist(), [1, ids_arr.shape[1]])

    model_path = Path(args.model_path)
    if model_path.exists():
        # attempt to load model
        try:
            if hasattr(model, 'load_state_dict_from_path'):
                model.load_state_dict_from_path(str(model_path), False, None)
                logger.info("Loaded weights from %s via model.load_state_dict_from_path", model_path)
            else:
                with open(model_path, "rb") as fh:
                    b = fh.read()
                try:
                    model.load_state_dict(b, False, None)
                    logger.info("Loaded weights from %s", model_path)
                except (ValueError, TypeError, RuntimeError) as err:
                    py_load_fn = getattr(te, 'py_load_safetensors_into_module', None)
                    if callable(py_load_fn):
                        try:
                            py_load_fn(b, False, model, None)
                            logger.info("Loaded weights from %s using py_load_safetensors_into_module", model_path)
                        except (RuntimeError, TypeError, ValueError) as inner_err:
                            logger.debug("Couldn't load model weights; continuing with random init: %s", inner_err)
                    else:
                        logger.debug("Couldn't load model weights; continuing with random init: %s", err)
        except (OSError, ValueError, RuntimeError, AttributeError) as err:
            logger.error("Failed to load model via path loader: %s", err)

    # pylint: disable=not-callable
    zeros_img = TensorClass(np.zeros((1, 3, 32, 32), dtype=np.float32).flatten().tolist(), [1, 3, 32, 32])
    try:
        logits = model.forward(zeros_img, ids_t)
        last_logits = logits[:, -1, :]
        out_id = int(np.argmax(np.array(last_logits.get_data())))
    except (RuntimeError, ValueError, TypeError) as err:
        logger.exception("Model forward failed during generation: %s", err)
        raise

    if hf_tok:
        try:
            decoded = hf_tok.decode([out_id])
            logger.info("Generated token text: %s", decoded)
        except (ValueError, TypeError, IndexError) as err:
            logger.exception("Failed to decode generated token with HF tokenizer: %s", err)
            logger.info("Generated token id: %s", out_id)
    else:
        inv_vocab = {v: k for k, v in vocab.items()}
        logger.info("Generated token text: %s", inv_vocab.get(out_id, str(out_id)))

    seq_ids = ids.copy()
    for step in range(int(args.steps)):
        ids_arr = np.array(seq_ids, dtype=np.float32).reshape((1, len(seq_ids)))
        # pylint: disable=not-callable
        ids_t = TensorClass(ids_arr.flatten().tolist(), [1, ids_arr.shape[1]])
        try:
            logits = model.forward(zeros_img, ids_t)
            last_logits = logits[:, -1, :]
            next_id = int(np.argmax(np.array(last_logits.get_data())))
        except (RuntimeError, ValueError, TypeError) as err:
            logger.exception("Generation failed at step %s: %s", step+1, err)
            raise
        seq_ids.append(next_id)
        logger.info("Step %d, next token id: %d", step + 1, next_id)

    if hf_tok:
        decoded = hf_tok.decode(seq_ids)
        logger.info("Decoded text: %s", decoded)
    else:
        inv_vocab = {v: k for k, v in vocab.items()}
        logger.info("Decoded text (via inv_vocab): %s", ' '.join(inv_vocab.get(x, str(x)) for x in seq_ids))


if __name__ == "__main__":
    main()
