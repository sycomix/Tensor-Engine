#!/usr/bin/env python3
# pylint: disable=E1101,E1102,E0401,W0611,C0415,C0301,C0103
# type: ignore
"""
Simple generation example using a LLAVA-like model.

This script loads a trained model and runs greedy decoding conditioned on a real
image path.
"""
from __future__ import annotations

import argparse
import json
import logging
import numpy as np
import tempfile
from pathlib import Path
from typing import Any

try:
    import tensor_engine as te  # type: ignore
except ImportError:  # pragma: no cover
    te = None  # type: ignore


def simple_tokenize(text: str, vocab: dict, bos: int = 1, eos: int = 2) -> list[int]:
    """A simple whitespace tokenizer into ids using provided vocab.

    This is intentionally simplistic for the demo dataset.
    """
    return [bos] + [vocab.get(w, 0) for w in text.split()] + [eos]


def _resolve_tokenizer_json_path(tokenizer_arg: str) -> Path:
    p = Path(tokenizer_arg)
    if p.is_dir():
        candidate = p / "tokenizer.json"
        if candidate.exists():
            return candidate
    return p


def _infer_special_ids(tok: Any) -> tuple[int, int, int]:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    try:
        for t in ("<pad>", "[PAD]", "<PAD>"):
            v = tok.token_to_id(t)
            if v is not None:
                pad_id = int(v)
                break
        for t in ("<bos>", "<s>", "[BOS]", "<BOS>"):
            v = tok.token_to_id(t)
            if v is not None:
                bos_id = int(v)
                break
        for t in ("<eos>", "</s>", "[EOS]", "<EOS>"):
            v = tok.token_to_id(t)
            if v is not None:
                eos_id = int(v)
                break
    except (AttributeError, TypeError, ValueError):
        return pad_id, bos_id, eos_id
    return pad_id, bos_id, eos_id


def _load_image_tensor_from_path(te_mod: Any, image_path: Path, image_w: int, image_h: int,
                                 parallel: bool = False) -> Any:
    """Load one image using the Rust image loader via ImageTextDataLoader."""
    LoaderClass: Any = getattr(te_mod, 'ImageTextDataLoader', None)
    TensorClass: Any = getattr(te_mod, 'Tensor', None)
    if LoaderClass is None or TensorClass is None:
        raise RuntimeError("tensor_engine missing ImageTextDataLoader/Tensor; rebuild with python_bindings,vision")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Build a temporary manifest for a single item so we reuse the exact same preprocessing.
    with tempfile.TemporaryDirectory() as td:
        manifest = Path(td) / "one.txt"
        manifest.write_text(f"{str(image_path.resolve())}\tplaceholder\n", encoding="utf-8")
        # pylint: disable=not-callable
        loader = LoaderClass(str(manifest), int(image_w), int(image_h), 1, False, False, bool(parallel))
        images, _ = loader.load_batch(0)
        if not images:
            raise RuntimeError("Failed to load image via manifest")
        # image is [1,C,H,W]
        return TensorClass.cat(images, 0)


def main() -> None:
    """CLI entry point to run simple generation with the toy model."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Describe the image 0")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--model_path", default="examples/models/llava_model.safetensors")
    parser.add_argument("--config", default=None, help="Optional explicit config JSON; defaults to <model>.config.json")
    parser.add_argument("--tokenizer-json", default=None, help="Path to tokenizer.json (or directory containing it)")
    parser.add_argument("--image", default=None, help="Path to an image file (required)")
    parser.add_argument("--parallel-io", action="store_true")
    args = parser.parse_args()

    if te is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")
    # Validate presence of core Python bindings
    required = ['Tensor', 'VisionTransformer', 'MultimodalLLM']
    missing = [r for r in required if not hasattr(te, r)]
    if missing:
        raise RuntimeError(
            f"tensor_engine Python bindings missing required classes: {missing}. Build with python_bindings and vision features enabled.")

    model_path = Path(args.model_path)
    cfg_path = Path(args.config) if args.config else model_path.with_suffix(".config.json")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg_path}. Run training with examples.train_llava to generate <model>.config.json, "
            "or pass --config explicitly."
        )

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    d_model = int(cfg.get('d_model', 32))
    vocab_size = int(cfg.get('vocab_size', 256))
    d_ff = int(cfg.get('d_ff', d_model * 4))
    num_heads = int(cfg.get('num_heads', 4))
    depth = int(cfg.get('depth', 2))
    patch_size = int(cfg.get('patch_size', 8))
    max_len = int(cfg.get('max_len', 512))
    image_w = int(cfg.get('image_w', 224))
    image_h = int(cfg.get('image_h', 224))
    bos_id = int(cfg.get('bos_id', 1))
    eos_id = int(cfg.get('eos_id', 2))
    default_prompt = str(cfg.get('prompt', 'Describe the image.'))
    if args.prompt == "Describe the image 0":
        # Preserve backwards-compatible default, but prefer a real prompt if present.
        args.prompt = default_prompt

    vt_class: Any = getattr(te, 'VisionTransformer', None)
    mm_class: Any = getattr(te, 'MultimodalLLM', None)
    if vt_class is None or mm_class is None:
        raise RuntimeError(
            'tensor_engine module does not expose VisionTransformer and/or MultimodalLLM. Rebuild the package with python_bindings and vision enabled.')
    if not callable(vt_class):
        raise RuntimeError('VisionTransformer class is not callable')
    # pylint: disable=not-callable
    vision = vt_class(3, patch_size, d_model, d_ff, num_heads=num_heads, depth=depth, max_len=max_len)
    if not callable(mm_class):
        raise RuntimeError('MultimodalLLM class is not callable')
    # pylint: disable=not-callable
    model = mm_class(vision, vocab_size, d_model, d_ff, num_heads=num_heads, depth=depth)

    tok_obj = None
    if args.tokenizer_json:
        if not hasattr(te, 'Tokenizer'):
            raise RuntimeError("tensor_engine.Tokenizer not available; rebuild with with_tokenizers")
        tok_path = _resolve_tokenizer_json_path(args.tokenizer_json)
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer JSON not found: {tok_path}")
        TokenizerClass: Any = getattr(te, 'Tokenizer', None)
        if TokenizerClass is None:
            raise RuntimeError('tensor_engine module does not expose Tokenizer')
        # pylint: disable=not-callable
        tok_obj = TokenizerClass.from_file(str(tok_path))
        # If config didn't encode special IDs, infer best-effort.
        if 'pad_id' not in cfg or 'bos_id' not in cfg or 'eos_id' not in cfg:
            _inf_pad, inf_bos, inf_eos = _infer_special_ids(tok_obj)
            bos_id, eos_id = inf_bos, inf_eos

    if args.image is None:
        raise ValueError("--image is required")
    img_tensor = _load_image_tensor_from_path(te, Path(args.image), image_w=image_w, image_h=image_h,
                                              parallel=bool(args.parallel_io))

    if tok_obj is not None:
        prompt_ids_u32 = tok_obj.encode(args.prompt)
        seq_ids: list[int] = [bos_id] + [int(x) for x in prompt_ids_u32]
    else:
        # Fallback legacy whitespace vocab for minimal debugging.
        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        words = "Describe the image".split()
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
        seq_ids = simple_tokenize(args.prompt, vocab, bos=bos_id, eos=eos_id)

    ids_arr = np.array(seq_ids, dtype=np.float32).reshape((1, len(seq_ids)))
    # Use the package-provided Tensor class via getattr to avoid static lint warnings
    TensorClass: Any = getattr(te, 'Tensor', None)
    # TensorClass is a runtime-provided class from pyo3 bindings; disable static not-callable warning
    # pylint: disable=not-callable
    ids_t = TensorClass(ids_arr.flatten().tolist(), [1, ids_arr.shape[1]])

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

    try:
        logits = model.forward(img_tensor, ids_t)
        last_logits = logits[:, -1, :]
        out_id = int(np.argmax(np.array(last_logits.get_data())))
    except (RuntimeError, ValueError, TypeError) as err:
        logger.exception("Model forward failed during generation: %s", err)
        raise

    if tok_obj is not None:
        try:
            logger.info("First token: %s", tok_obj.decode([int(out_id)]))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            logger.info("First token id: %s", out_id)
    else:
        logger.info("First token id: %s", out_id)

    for step in range(int(args.steps)):
        ids_arr = np.array(seq_ids, dtype=np.float32).reshape((1, len(seq_ids)))
        # pylint: disable=not-callable
        ids_t = TensorClass(ids_arr.flatten().tolist(), [1, ids_arr.shape[1]])
        try:
            logits = model.forward(img_tensor, ids_t)
            last_logits = logits[:, -1, :]
            next_id = int(np.argmax(np.array(last_logits.get_data())))
        except (RuntimeError, ValueError, TypeError) as err:
            logger.exception("Generation failed at step %s: %s", step + 1, err)
            raise
        seq_ids.append(next_id)
        logger.info("Step %d, next token id: %d", step + 1, next_id)

        if next_id == eos_id:
            logger.info("EOS reached; stopping")
            break

    if tok_obj is not None:
        try:
            decoded = tok_obj.decode([int(x) for x in seq_ids])
            logger.info("Decoded text: %s", decoded)
        except (AttributeError, RuntimeError, TypeError, ValueError) as err:
            logger.warning("Tokenizer decode failed: %s", err)
            logger.info("Decoded token ids: %s", seq_ids)
    else:
        logger.info("Decoded token ids: %s", seq_ids)


if __name__ == "__main__":
    main()
