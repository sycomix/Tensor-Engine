#!/usr/bin/env python3
"""
Simple generation example using the toy LLAVA-like model.

This script constructs the same architecture as the training example, creates a random model (or optionally loads one),
then performs autoregressive greedy decoding on a prompt.
"""
from __future__ import annotations
import numpy as np
import argparse
from pathlib import Path
import logging

try:
    import tensor_engine as te  # type: ignore
except ImportError:  # pragma: no cover
    te = None  # type: ignore


def simple_tokenize(text: str, vocab: dict, bos=1, eos=2):
    idxs = [bos] + [vocab.get(w, 0) for w in text.split()] + [eos]
    return idxs


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Describe the image 0")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    if te is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")

    d_model = 32
    vocab_size = 256
    parser.add_argument("--model_path", default="examples/models/llava_model.safetensors")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tokenizer", default=None, help='Optional HF tokenizer name to decode generated ids, e.g. bert-base-uncased')

    # toy vocabulary
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    # add a few words
    words = "Describe the image 0 Synthetic image description".split()
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)

    # Use MultimodalLLM constructed via config or default
    if args.config:
        import json
        with open(args.config, 'r', encoding='utf-8') as fh:
            cfg = json.load(fh)
        vision = te.VisionTransformer(cfg.get('c', 3), cfg.get('patch_size', 8), cfg.get('d_model', d_model), cfg.get('d_ff', d_model*4), cfg.get('num_heads', 4), cfg.get('depth', 2), cfg.get('max_len', 512))
        model = te.MultimodalLLM(vision, cfg.get('vocab_size', vocab_size), cfg.get('d_model', d_model), cfg.get('d_ff', d_model*4), cfg.get('num_heads', 4), cfg.get('depth', 2))
    else:
        vision = te.VisionTransformer(3, 8, d_model, d_model * 4, num_heads=4, depth=2, max_len=512)
        model = te.MultimodalLLM(vision, vocab_size, d_model, d_model * 4, num_heads=4, depth=2)

    # instantiate hf tokenizer if required
    hf_tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            hf_tok = AutoTokenizer.from_pretrained(args.tokenizer)
        except (ImportError, OSError) as e:
            logger.warning("Failed to load HF tokenizer; continuing without tokenizer: %s", e)

    # create prompt ids
    ids = simple_tokenize(args.prompt, vocab)
    ids_arr = np.array(ids, dtype=np.float32).reshape((1, len(ids)))
    ids_t = te.Tensor(ids_arr.flatten().tolist(), [1, ids_arr.shape[1]])

    # Embedding: random init matrix
    # optionally load model weights
    model_path = Path(args.model_path)
    if model_path.exists():
        try:
            # prefer a dedicated path-based loader if available
            if hasattr(model, 'load_state_dict_from_path'):
                model.load_state_dict_from_path(str(model_path), False, None)
                logger.info("Loaded weights from %s via model.load_state_dict_from_path", model_path)
            else:
                with open(model_path, "rb") as fh:
                    b = fh.read()

                # Try to load using the module's load_state_dict from bytes
                try:
                    model.load_state_dict(b, False, None)
                    logger.info("Loaded weights from %s", model_path)
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.debug("model.load_state_dict failed: %s", e)
                    # Try Python fallback loader that may exist in bindings
                    try:
                        te.py_load_safetensors_into_module(b, False, model, None)
                        logger.info("Loaded weights from %s with py_load_safetensors_into_module", model_path)
                    except (RuntimeError, ValueError, TypeError) as e2:
                        logger.debug("py_load_safetensors_into_module failed: %s", e2)
                        # fallback: try loading a numpy npz if available
                        try:
                            data = np.load(str(model_path))
                            # set params directly
                            for (name, param) in model.named_parameters(""):
                                if name in data:
                                    arr = data[name]
                                    param.set_data(arr.flatten().tolist())
                            logger.info("Loaded model params from npz %s", model_path)
                        except (IOError, ValueError, KeyError) as e3:
                            logger.warning("No safetensors/npz fallback available; model remains randomly initialized: %s", e3)
        except (RuntimeError, IOError, OSError, ValueError) as e:
            logger.error("Failed to load model via path loader or bytes loader: %s", e)

    # initial input embeddings (embedding lookup is optional and not used in this example)
    # Use a default zero image tensor: [1, 3, 32, 32]
    zeros_img = te.Tensor(np.zeros((1, 3, 32, 32), dtype=np.float32).flatten().tolist(), [1, 3, 32, 32])
    # Use prefill/logit-from-memory path if available
    try:
        mem = model.prefill(zeros_img, ids_t)
        logits = model.logits_from_memory(mem)
        last_logits = logits[:, -1, :]
        # Greedy token selection and update memory using decode_step
        out_id = int(np.argmax(np.array(last_logits.get_data())))
        new_id_t = te.Tensor([float(out_id)], [1, 1])
        _logits, mem = model.decode_step(mem, new_id_t)
        logger.info("Generated token id: %d", out_id)
    except (AttributeError, TypeError, RuntimeError):
        # fallback to direct forward
        logits = model.forward(zeros_img, ids_t)
        last_logits = logits[:, -1, :]
    # Greedy sampling
    out_id = int(np.argmax(np.array(last_logits.get_data())))
    logger.info("Generated token id: %d", out_id)
    # decode generated id to string
    if hf_tok:
        decoded = hf_tok.decode([out_id])
        logger.info("Generated token text: %s", decoded)
    else:
        inv_vocab = {v: k for k, v in vocab.items()}
        logger.info("Generated token text (via inv vocab): %s", inv_vocab.get(out_id, str(out_id)))

    # simple loop for autoregressive continuation
    seq_ids = ids.copy()
    for step in range(args.steps):
        # prepare Tensor for last token id
        ids_arr = np.array(seq_ids, dtype=np.float32).reshape((1, len(seq_ids)))
        ids_t = te.Tensor(ids_arr.flatten().tolist(), [1, ids_arr.shape[1]])
        # Note: For a simple generation example we omit explicit embedding lookup
        # Inference via model.forward using zero image placeholder for the sample
        zeros_img = te.Tensor(np.zeros((1, 3, 32, 32), dtype=np.float32).flatten().tolist(), [1, 3, 32, 32])
        logits = model.forward(zeros_img, ids_t)
        last_logits = logits[:, -1, :]
        last_logits_np = np.array(last_logits.get_data())
        next_id = int(np.argmax(last_logits_np))
        seq_ids.append(next_id)
        logger.info("Step %d, next token id: %d", step+1, next_id)

    if hf_tok:
        decoded = hf_tok.decode(seq_ids)
        logger.info("Decoded text: %s", decoded)
    else:
        inv_vocab = {v: k for k, v in vocab.items()}
        logger.info("Decoded ids: %s", seq_ids)
        logger.info("Decoded text (via inv_vocab): %s", ' '.join(inv_vocab.get(x, str(x)) for x in seq_ids))


if __name__ == "__main__":
    main()
