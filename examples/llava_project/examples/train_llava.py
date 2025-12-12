#!/usr/bin/env python3
"""
Simplified training script adapted from the Tensor-Engine example.
"""

from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    import tensor_engine as te  # type: ignore
except ImportError:  # pragma: no cover
    te = None  # type: ignore


def build_vocab_from_data(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Construct a token vocabulary from dataset records.

    Returns a mapping from token string to integer id.
    """
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    for rec in records:
        for tok in rec["input_text"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
        for tok in rec["target_text"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def tokenize_texts(records: List[Dict[str, Any]], vocab: Dict[str, int]) -> tuple[List[list[int]], list[list[int]]]:
    """Tokenize the input and target texts in records using the provided vocabulary.

    Returns (inputs_tokens, target_tokens).
    """
    inputs = []
    targets = []
    for rec in records:
        input_tokens = [vocab["<bos>"]] + [vocab[t] for t in rec["input_text"].split()] + [vocab["<eos>"]]
        target_tokens = [vocab["<bos>"]] + [vocab[t] for t in rec["target_text"].split()] + [vocab["<eos>"]]
        inputs.append(input_tokens)
        targets.append(target_tokens)
    return inputs, targets


def pad_and_stack_token_ids(token_list: list[list[int]], pad: int = 0) -> np.ndarray:
    max_len = max(len(lst) for lst in token_list) if token_list else 0
    arr = np.full((len(token_list), max_len), pad, dtype=np.float32)
    for i, lst in enumerate(token_list):
        arr[i, : len(lst)] = lst
    return arr


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/data/synthetic_llava.jsonl")
    parser.add_argument("--tokenizer", default=None, help="Optional HF tokenizer name or local path to tokenizer")
    parser.add_argument("--config", default=None, help="Optional JSON config path to build model (e.g., examples/llava_model_config.json)")
    parser.add_argument("--full-model", action="store_true", help="Build a full model using examples/llava_model_config.json (overrides --d_model/options unless --config specified)")
    parser.add_argument("--force", action="store_true", help="Force dataset regeneration even if it exists")
    parser.add_argument("--tokenized-data", default=None, help="Path to write tokenized dataset if using a tokenizer (separate from --data)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint file if present")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint file (defaults to save path with .ckpt.safetensors)")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Checkpoint save interval in epochs (0 to disable)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--save", default="examples/models/llava_model.safetensors")
    args = parser.parse_args()

    data_path = Path(args.data)
    tokenizer = args.tokenizer
    force_regen = bool(args.force)
    tokenized_data = args.tokenized_data
    resume = bool(args.resume)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    checkpoint_interval = int(args.checkpoint_interval)
    # If a tokenized dataset path is provided and exists, prefer that dataset for training.
    if tokenized_data:
        try:
            if Path(str(tokenized_data)).exists():
                data_path = Path(tokenized_data)
                logger.info("Using tokenized dataset from %s", tokenized_data)
        except OSError as e:
            logger.exception("Failed to stat tokenized-data path %s: %s", tokenized_data, e)
    if not data_path.exists() or force_regen:
        if data_path.exists() and force_regen:
            logger.info("Forcing dataset regeneration: removing %s", data_path)
        logger.info("Dataset regeneration: generating synthetic dataset")
        # import placed in function body to avoid top-level dependency on optional packages
        # try both package-style import (when running as package) and module-style
        # import (when running script directly).
        # pylint: disable=import-outside-toplevel
        try:
            from examples.prepare_dataset import generate_dataset as prepare_generate
        except ImportError:
            from prepare_dataset import generate_dataset as prepare_generate
        # If tokenizer and a separate tokenized-data path provided, generate both original (un-tokenized)
        # and tokenized dataset. If only tokenizer provided and no tokenized-data, replace original with tokenized version.
        if tokenized_data and tokenizer:
            # generate un-tokenized original
            prepare_generate(str(data_path), 16, 32, 32, 3, None)
            # generate tokenized dataset to the separate path
            prepare_generate(str(tokenized_data), 16, 32, 32, 3, tokenizer)
        else:
            # generate to data_path (tokenized if tokenizer provided)
            prepare_generate(str(data_path), 16, 32, 32, 3, tokenizer)
        # If tokenized_data is provided and exists, use that as the dataset for training
        if tokenized_data and Path(str(tokenized_data)).exists():
            data_path = Path(tokenized_data)

    try:
        records = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line.strip()))
    except FileNotFoundError:
        logger.error("Dataset file not found: %s", data_path)
        raise
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse JSONL dataset: %s", e)
        raise
    except (OSError, UnicodeDecodeError, ValueError) as e:
        logger.exception("Unexpected error while reading dataset: %s", e)
        raise

    if not records:
        logger.error("No records were found in %s. Aborting training.", data_path)
        raise SystemExit(1)

    # If dataset exists but has no token ids and a tokenizer was provided, re-generate dataset with tokenizer
    if tokenizer is not None and records and "input_ids" not in records[0]:
        logger.info("Dataset found without token ids; re-generating with tokenizer=%s", tokenizer)
        from examples.prepare_dataset import generate_dataset as prepare_generate
        try:
            prepare_generate(str(data_path), len(records), records[0]["height"], records[0]["width"], records[0]["channels"], tokenizer)
        except (RuntimeError, OSError, ValueError) as e:
            logger.exception("Failed to re-generate dataset with tokenizer: %s: %s", tokenizer, e)
            raise
        try:
            records = []
            with open(data_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        records.append(json.loads(line.strip()))
        except (OSError, UnicodeDecodeError, ValueError) as e:
            logger.exception("Failed to re-load regenerated dataset: %s", e)
            raise

    if "input_ids" in records[0]:
        inputs_tokens = [rec.get("input_ids", []) for rec in records]
        targets_tokens = [rec.get("target_ids", []) for rec in records]
        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    else:
        vocab = build_vocab_from_data(records)
        inputs_tokens, targets_tokens = tokenize_texts(records, vocab)

    input_ids = pad_and_stack_token_ids(inputs_tokens, pad=vocab["<pad>"])
    target_ids = pad_and_stack_token_ids(targets_tokens, pad=vocab["<pad>"])

    # Convert images to NumPy
    images = [rec["image"] for rec in records]
    h = records[0]["height"]
    w = records[0]["width"]
    c = records[0]["channels"]
    images_np = np.stack([np.array(i).reshape((h, w, c)) for i in images])
    images_np = images_np.transpose((0, 3, 1, 2))  # [B, C, H, W]

    if args.epochs <= 0:
        logger.info("Epochs set to 0 or less; exiting without building model (dataset generated).")
        return

    if te is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")

    cfg = None
    if args.config:
        try:
            cfg = json.load(open(args.config, 'r', encoding='utf-8'))
            logger.info("Using model config from %s", args.config)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.exception("Failed to load config from %s; falling back to CLI args: %s", args.config, e)
            cfg = None
    elif args.full_model:
        try:
            cfg_path = Path(__file__).parent / 'llava_model_config.json'
            if cfg_path.exists():
                cfg = json.load(open(cfg_path, 'r', encoding='utf-8'))
                logger.info("Using full model config from %s", cfg_path)
            else:
                logger.warning("Full model config not found at %s; falling back to CLI args", cfg_path)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.exception("Failed to load default full model config; falling back to CLI args: %s", e)
            cfg = None

    if cfg is not None:
        d_model = int(cfg.get('d_model', args.d_model))
        d_ff = int(cfg.get('d_ff', d_model * 4))
        num_heads = int(cfg.get('num_heads', 4))
        depth = int(cfg.get('depth', args.num_blocks))
        patch_size = int(cfg.get('patch_size', args.patch_size))
        vocab_size = int(cfg.get('vocab_size', len(vocab)))
        max_len = int(cfg.get('max_len', 512))
    else:
        d_model = args.d_model
        vocab_size = len(vocab)
        d_ff = d_model * 4
        num_heads = 4
        depth = args.num_blocks
        patch_size = args.patch_size
        max_len = 512

    # Build model using tensor_engine
    vision = te.VisionTransformer(3, patch_size, d_model, d_ff, num_heads=num_heads, depth=depth, max_len=max_len)
    model = te.MultimodalLLM(vision, vocab_size, d_model, d_ff, num_heads=num_heads, depth=depth)

    opt = te.Adam(3e-4, 0.9, 0.999, 1e-8)
    loss_fn = te.SoftmaxCrossEntropyLoss()

    num_samples = input_ids.shape[0]
    batch_size = args.batch
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Helper functions for checkpoint save/load
    def save_model_state(model, path: Path) -> bool:
        try:
            if hasattr(model, 'save_state_dict_to_path'):
                path.parent.mkdir(parents=True, exist_ok=True)
                model.save_state_dict_to_path(str(path))
                logger.info("Saved model parameters to %s via model.save_state_dict_to_path", path)
                return True
        except (AttributeError, RuntimeError, OSError) as e:
            logger.exception("model.save_state_dict_to_path failed; will try fallback: %s", e)
        # Fallback: export via safetensors or npz
        try:
            # import placed here to avoid optional dependency at module import time
            # pylint: disable=import-outside-toplevel
            # import placed here to avoid optional dependency at module import time
            # pylint: disable=import-outside-toplevel
            from safetensors.numpy import save_file
        except ImportError:
            save_file = None
        params_dict = {}
        try:
            for (name, param) in model.named_parameters(""):
                data = np.array(param.get_data(), dtype=np.float32)
                shape = tuple(param.shape)
                params_dict[name] = data.reshape(shape)
            if save_file is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file(params_dict, str(path))
                logger.info("Saved model parameters to %s via safetensors", path)
                return True
            else:
                np_path = _Path(str(path)).with_suffix('.npz')
                np_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(np_path, **params_dict)
                logger.info("Saved model parameters to %s (npz fallback)", np_path)
                return True
        except (OSError, ValueError, TypeError) as e:
            logger.exception("Failed to save model via fallback: %s", e)
            return False

    def load_model_state(model, path: Path) -> bool:
        try:
            if hasattr(model, 'load_state_dict_from_path') and path.exists():
                model.load_state_dict_from_path(str(path), False, None)
                logger.info("Loaded weights from %s via model.load_state_dict_from_path", path)
                return True
        except (AttributeError, RuntimeError, OSError) as e:
            logger.exception("model.load_state_dict_from_path failed; trying byte loaders: %s", e)
        if path.exists():
            try:
                with open(path, 'rb') as fh:
                    b = fh.read()
                try:
                    model.load_state_dict(b, False, None)
                    logger.info("Loaded weights from %s via model.load_state_dict(bytes)", path)
                    return True
                except (ValueError, TypeError, RuntimeError) as e:
                    try:
                        te.py_load_safetensors_into_module(b, False, model, None)
                        logger.info("Loaded weights from %s via py_load_safetensors_into_module", path)
                        return True
                    except (RuntimeError, TypeError, ValueError) as e:
                        logger.debug("Bytes loaders failed; attempting npz fallback: %s", e)
            except OSError as e:
                logger.exception("Failed to open model path bytes for %s: %s", path, e)
        # Try numpy .npz fallback
        try:
            npz_path = _Path(str(path)).with_suffix('.npz')
            if npz_path.exists():
                data = np.load(str(npz_path))
                for (name, param) in model.named_parameters(""):
                    if name in data:
                        arr = data[name]
                        param.set_data(arr.flatten().tolist())
                logger.info("Loaded model params from npz %s", npz_path)
                return True
        except (OSError, ValueError, TypeError, KeyError) as e:
            logger.exception("Failed to load model from npz fallback: %s", e)
        return False

    # Resolve default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = Path(args.save).with_suffix('.ckpt.safetensors')

    # Optionally resume
    if resume and checkpoint_path.exists():
        try:
            load_model_state(model, checkpoint_path)
        except (RuntimeError, OSError, ValueError) as e:
            logger.exception("Failed to resume from checkpoint: %s: %s", checkpoint_path, e)
            # continue training with random init

    try:
        for epoch in range(int(args.epochs)):
            epoch_loss = 0.0
            for b in range(num_batches):
                start = b * batch_size
                end = min((b + 1) * batch_size, num_samples)
                bs = end - start
                img_batch = images_np[start:end]
                try:
                    img_tensor = te.Tensor(img_batch.flatten().tolist(), [bs, c, h, w])
                except (TypeError, ValueError, OSError) as e:
                    logger.exception("Failed to construct image tensor for batch %s: %s", b, e)
                    raise

                try:
                    img_tokens = model.vision_forward(img_tensor)
                except (RuntimeError, TypeError, ValueError) as e:
                    logger.exception("Vision forward failed for batch %s: %s", b, e)
                    raise

                ids_batch = input_ids[start:end].astype(np.float32)
                ids_tensor = te.Tensor(ids_batch.flatten().tolist(), [bs, ids_batch.shape[1]])

                try:
                    logits = model.forward(img_tokens, ids_tensor)
                except (RuntimeError, TypeError, ValueError) as e:
                    logger.exception("Model forward failed for batch %s: %s", b, e)
                    raise

                n_image_tokens = img_tokens.shape[1]
                logits_text = logits[:, n_image_tokens:, :]

                targ = target_ids[start:end]
                flat_labels = [int(x) for row in targ.tolist() for x in row]
                try:
                    loss = loss_fn.forward_from_labels(logits_text, te.Labels(flat_labels))
                    loss.backward()
                    params = model.parameters()
                    opt.step(params)
                    opt.zero_grad(params)
                    epoch_loss += float(loss.get_data())
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.exception("Loss/backprop/optimizer step failed for batch %s: %s", b, e)
                    raise
            logger.info("Epoch %d/%d, loss=%0.4f", epoch+1, args.epochs, epoch_loss/num_batches)
            # epoch-level checkpointing
            if checkpoint_interval > 0 and ((epoch + 1) % checkpoint_interval) == 0:
                try:
                    save_model_state(model, checkpoint_path)
                except (RuntimeError, OSError, ValueError) as e:
                    logger.exception("Failed to save checkpoint at epoch %s: %s", epoch + 1, e)

    except (RuntimeError, OSError, ValueError, KeyboardInterrupt, TypeError) as e:
        logger.exception("Training failed; attempting to save partial checkpoint: %s", e)
        try:
            partial_path = _Path(str(checkpoint_path)).with_name(str(checkpoint_path.stem) + f'.partial.{int(time.time())}' + str(checkpoint_path.suffix))
            save_model_state(model, partial_path)
            logger.info("Saved partial checkpoint to %s", partial_path)
        except (RuntimeError, OSError, ValueError) as e:
            logger.exception("Failed to save partial checkpoint: %s", e)
        raise

    logger.info("Training done!")
    # Final checkpoint save
    try:
        save_model_state(model, checkpoint_path)
    except (RuntimeError, OSError, ValueError) as e:
        logger.exception("Final checkpoint save failed: %s", e)
    save_path = Path(args.save)

    # Try saving via model API
    save_used = False
    try:
        if hasattr(model, 'save_state_dict_to_path'):
            model.save_state_dict_to_path(str(save_path))
            logger.info("Saved model parameters to %s via model.save_state_dict_to_path", save_path)
            save_used = True
    except (AttributeError, RuntimeError, OSError) as e:
        logger.exception("model.save_state_dict_to_path failed: %s", e)

    if not save_used:
        try:
            # import placed here to avoid optional dependency at module import time
            # pylint: disable=import-outside-toplevel
            from safetensors.numpy import save_file
        except ImportError as e:
            save_file = None
            logger.debug("safetensors.numpy import failed, falling back to numpy save: %s", e)

        params_dict = {}
        for (name, param) in model.named_parameters(""):
            data = np.array(param.get_data(), dtype=np.float32)
            shape = tuple(param.shape)
            params_dict[name] = data.reshape(shape)

        try:
            if save_file is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(params_dict, str(save_path))
                logger.info("Saved model parameters to %s", save_path)
            else:
                save_path_npz = save_path.with_suffix('.npz')
                np.savez(save_path_npz, **params_dict)
                logger.info("Saved model parameters to %s (safetensors not available)", save_path_npz)
        except (OSError, ValueError, TypeError) as e:
            logger.exception("Failed to persist final model parameters: %s", e)

    # Optionally, if a tokenizer is present, log a small decoded output sample for sanity.
    if tokenizer is not None:
        try:
            # import placed here to avoid optional dependency at module import time
            # pylint: disable=import-outside-toplevel
            from transformers import AutoTokenizer
            hf_tok = AutoTokenizer.from_pretrained(tokenizer)
            # Decode the first sample target
            sample_ids = target_ids[0].astype(np.int64).tolist()
            # Trim padding and eos
            sample_ids = [int(x) for x in sample_ids if x != vocab.get("<pad>")]
            if len(sample_ids) > 0:
                # Remove BOS/EOS if present
                if sample_ids[0] == vocab.get("<bos>"):
                    sample_ids = sample_ids[1:]
                if len(sample_ids) and sample_ids[-1] == vocab.get("<eos>"):
                    sample_ids = sample_ids[:-1]
                if len(sample_ids):
                    decoded = hf_tok.decode(sample_ids)
                    logger.info("Sample target decoded via tokenizer: %s", decoded)
        except (ValueError, TypeError, KeyError) as e:
            logger.debug("HF tokenizer decode attempt failed; skipping: %s", e)


if __name__ == "__main__":
    main()
