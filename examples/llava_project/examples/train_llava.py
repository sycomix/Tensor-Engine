#!/usr/bin/env python3
# pylint: disable=E1101,E1102,E0401,W0611,C0415,C0301,C0103
# type: ignore
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
    # Ensure a minimum sequence length of 1 to avoid creating arrays with a zero-width
    # dimension which can cause downstream ops (e.g., matmul) to panic.
    max_len = max(1, max(len(lst) for lst in token_list)) if token_list else 1
    arr = np.full((len(token_list), max_len), pad, dtype=np.float32)
    for i, lst in enumerate(token_list):
        arr[i, : len(lst)] = lst
    return arr


def _resolve_tokenizer_json_path(tokenizer_arg: str) -> Path:
    p = Path(tokenizer_arg)
    if p.is_dir():
        candidate = p / "tokenizer.json"
        if candidate.exists():
            return candidate
    return p


def _infer_special_ids(tok: Any) -> tuple[int, int, int]:
    """Best-effort inference for (pad_id, bos_id, eos_id) from token names."""
    # Defaults match the legacy toy vocab.
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    try:
        # PAD
        for t in ("<pad>", "[PAD]", "<PAD>"):
            v = tok.token_to_id(t)
            if v is not None:
                pad_id = int(v)
                break
        # BOS
        for t in ("<bos>", "<s>", "[BOS]", "<BOS>"):
            v = tok.token_to_id(t)
            if v is not None:
                bos_id = int(v)
                break
        # EOS
        for t in ("<eos>", "</s>", "[EOS]", "<EOS>"):
            v = tok.token_to_id(t)
            if v is not None:
                eos_id = int(v)
                break
    except (AttributeError, TypeError, ValueError):
        # Tokenizer helpers may not be present if built without with_tokenizers.
        return pad_id, bos_id, eos_id

    return pad_id, bos_id, eos_id


def _pad_2d_int(seqs: list[list[int]], pad: int, max_len: int) -> np.ndarray:
    out = np.full((len(seqs), max_len), pad, dtype=np.float32)
    for i, s in enumerate(seqs):
        if not s:
            continue
        s2 = s[:max_len]
        out[i, : len(s2)] = np.array(s2, dtype=np.float32)
    return out


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="examples/data/manifest.txt",
        help="TSV manifest for real training. Each line: /abs/path/to/image\tcaption",
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic JSONL dataset instead of --manifest")
    parser.add_argument("--data", default="examples/data/synthetic_llava.jsonl", help="Synthetic JSONL path (only used with --synthetic)")
    parser.add_argument(
        "--tokenizer-json",
        default=None,
        help="Path to tokenizer.json (or directory containing tokenizer.json). Required for --manifest training.",
    )
    parser.add_argument("--config", default=None, help="Optional JSON config path to build model (e.g., examples/llava_model_config.json)")
    parser.add_argument("--full-model", action="store_true", help="Build a full model using examples/llava_model_config.json (overrides --d_model/options unless --config specified)")
    parser.add_argument("--force", action="store_true", help="Force dataset regeneration even if it exists")
    parser.add_argument("--tokenized-data", default=None, help="Path to write tokenized dataset if using a tokenizer (separate from --data)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint file if present")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint file (defaults to save path with .ckpt.safetensors)")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Checkpoint save interval in epochs (0 to disable)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--image-w", type=int, default=224)
    parser.add_argument("--image-h", type=int, default=224)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset each epoch (manifest loader only)")
    parser.add_argument("--augment", action="store_true", help="Enable light data augmentation (manifest loader only)")
    parser.add_argument("--parallel-io", action="store_true", help="Enable parallel image loading (requires Tensor-Engine built with parallel_io)")
    parser.add_argument("--prompt", default="Describe the image.", help="Prompt template used for captioning training")
    parser.add_argument("--pad-id", type=int, default=-1)
    parser.add_argument("--bos-id", type=int, default=-1)
    parser.add_argument("--eos-id", type=int, default=-1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=0, help="Override vocab size (0 = infer from tokenizer when available)")
    parser.add_argument("--save", default="examples/models/llava_model.safetensors")
    args = parser.parse_args()

    use_synthetic = bool(args.synthetic)
    data_path = Path(args.data)
    manifest_path = Path(args.manifest)
    tokenizer_json = args.tokenizer_json
    force_regen = bool(args.force)
    tokenized_data = args.tokenized_data
    resume = bool(args.resume)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    checkpoint_interval = int(args.checkpoint_interval)

    if te is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")

    # Validate presence of important Python bindings/classes
    required = ['Tensor', 'VisionTransformer', 'MultimodalLLM', 'Adam', 'SoftmaxCrossEntropyLoss', 'Labels']
    missing = [r for r in required if not hasattr(te, r)]
    if missing:
        raise RuntimeError(
            f"tensor_engine Python bindings missing required classes: {missing}. "
            "Build with python_bindings and vision features."
        )

    # Resolve tokenizer (required for manifest training)
    tok_obj = None
    pad_id = int(args.pad_id)
    bos_id = int(args.bos_id)
    eos_id = int(args.eos_id)
    if not use_synthetic:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Build one with scripts/prepare_manifest.py and rerun, "
                "or use --synthetic for a quick smoke run."
            )
        if tokenizer_json is None:
            raise ValueError("--tokenizer-json is required when training from --manifest")
        if not hasattr(te, 'Tokenizer'):
            raise RuntimeError(
                "tensor_engine.Tokenizer is not available. Rebuild Tensor-Engine with feature with_tokenizers."
            )
        tok_path = _resolve_tokenizer_json_path(tokenizer_json)
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer JSON not found: {tok_path}")
        TokenizerClass: Any = getattr(te, 'Tokenizer', None)
        if TokenizerClass is None:
            raise RuntimeError('tensor_engine module does not expose Tokenizer')
        # pylint: disable=not-callable
        tok_obj = TokenizerClass.from_file(str(tok_path))
        # Infer special IDs if not provided
        if pad_id < 0 or bos_id < 0 or eos_id < 0:
            inf_pad, inf_bos, inf_eos = _infer_special_ids(tok_obj)
            if pad_id < 0:
                pad_id = inf_pad
            if bos_id < 0:
                bos_id = inf_bos
            if eos_id < 0:
                eos_id = inf_eos

    # ---- Synthetic path (legacy smoke test) ----
    tokenizer = None  # legacy HF tokenizer arg is deprecated for real training
    # If a tokenized dataset path is provided and exists, prefer that dataset for training.
    if tokenized_data:
        try:
            if Path(str(tokenized_data)).exists():
                data_path = Path(tokenized_data)
                logger.info("Using tokenized dataset from %s", tokenized_data)
        except OSError as err:
            logger.exception("Failed to stat tokenized-data path %s: %s", tokenized_data, err)
    if use_synthetic and (not data_path.exists() or force_regen):
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

    records = []
    if use_synthetic:
        try:
            with open(data_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        records.append(json.loads(line.strip()))
        except FileNotFoundError:
            logger.error("Dataset file not found: %s", data_path)
            raise
        except json.JSONDecodeError as err:
            logger.exception("Failed to parse JSONL dataset: %s", err)
            raise
        except (OSError, UnicodeDecodeError, ValueError) as err:
            logger.exception("Unexpected error while reading dataset: %s", err)
            raise

        if not records:
            logger.error("No records were found in %s. Aborting training.", data_path)
            raise SystemExit(1)

    input_ids = None
    target_ids = None
    images_np = None
    h = int(args.image_h)
    w = int(args.image_w)
    c = 3

    if use_synthetic:
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

    cfg = None
    if args.config:
        try:
            cfg = json.load(open(args.config, 'r', encoding='utf-8'))
            logger.info("Using model config from %s", args.config)
        except (OSError, json.JSONDecodeError, ValueError) as err:
            logger.exception("Failed to load config from %s; falling back to CLI args: %s", args.config, err)
            cfg = None
    elif args.full_model:
        try:
            cfg_path = Path(__file__).parent / 'llava_model_config.json'
            if cfg_path.exists():
                cfg = json.load(open(cfg_path, 'r', encoding='utf-8'))
                logger.info("Using full model config from %s", cfg_path)
            else:
                logger.warning("Full model config not found at %s; falling back to CLI args", cfg_path)
        except (OSError, json.JSONDecodeError, ValueError) as err:
            logger.exception("Failed to load default full model config; falling back to CLI args: %s", err)
            cfg = None

    if cfg is not None:
        d_model = int(cfg.get('d_model', args.d_model))
        d_ff = int(cfg.get('d_ff', d_model * 4))
        num_heads = int(cfg.get('num_heads', 4))
        depth = int(cfg.get('depth', args.num_blocks))
        patch_size = int(cfg.get('patch_size', args.patch_size))
        vocab_size = int(cfg.get('vocab_size', args.vocab_size or 0))
        max_len = int(cfg.get('max_len', int(args.max_seq_len)))
    else:
        d_model = args.d_model
        vocab_size = int(args.vocab_size or 0)
        d_ff = d_model * 4
        num_heads = 4
        depth = args.num_blocks
        patch_size = args.patch_size
        max_len = int(args.max_seq_len)

    if not use_synthetic:
        if tok_obj is None:
            raise RuntimeError("Tokenizer unexpectedly missing for manifest training")
        if vocab_size <= 0:
            try:
                vocab_size = int(tok_obj.vocab_size())
            except Exception as err:
                raise RuntimeError(
                    "Failed to infer vocab size from tokenizer; pass --vocab-size explicitly"
                ) from err
    else:
        # Legacy synthetic vocab
        if vocab_size <= 0:
            vocab_size = 256

    # Build model using tensor_engine
    vt_class: Any = getattr(te, 'VisionTransformer', None)
    mm_class: Any = getattr(te, 'MultimodalLLM', None)
    if vt_class is None or mm_class is None:
        raise RuntimeError('tensor_engine module does not expose VisionTransformer and/or MultimodalLLM. Rebuild the package with python_bindings and vision enabled (e.g., `cargo build --features "python_bindings,vision"` or `maturin develop --release --features python_bindings,vision`).')
    if not callable(vt_class):
        raise RuntimeError('VisionTransformer class is not callable')
    # pylint: disable=not-callable
    vision: Any = vt_class(3, patch_size, d_model, d_ff, num_heads=num_heads, depth=depth, max_len=max_len)
    if not callable(mm_class):
        raise RuntimeError('MultimodalLLM class is not callable')
    # pylint: disable=not-callable
    model: Any = mm_class(vision, vocab_size, d_model, d_ff, num_heads=num_heads, depth=depth)

    # Save a sidecar config next to the model so generation can reproduce the architecture.
    save_path = Path(args.save)
    sidecar_cfg_path = save_path.with_suffix(".config.json")
    try:
        sidecar_cfg = {
            "d_model": int(d_model),
            "d_ff": int(d_ff),
            "num_heads": int(num_heads),
            "depth": int(depth),
            "patch_size": int(patch_size),
            "vocab_size": int(vocab_size),
            "max_len": int(max_len),
            "image_w": int(w),
            "image_h": int(h),
            "pad_id": int(pad_id if pad_id >= 0 else 0),
            "bos_id": int(bos_id if bos_id >= 0 else 1),
            "eos_id": int(eos_id if eos_id >= 0 else 2),
            "prompt": str(args.prompt),
        }
        sidecar_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_cfg_path.write_text(json.dumps(sidecar_cfg, indent=2), encoding="utf-8")
        logger.info("Wrote model sidecar config: %s", sidecar_cfg_path)
    except (OSError, ValueError, TypeError) as err:
        logger.warning("Failed to write sidecar config %s: %s", sidecar_cfg_path, err)

    AdamClass: Any = getattr(te, 'Adam', None)
    SoftmaxCrossEntropyLossClass: Any = getattr(te, 'SoftmaxCrossEntropyLoss', None)
    LabelsClass: Any = getattr(te, 'Labels', None)
    TensorClass: Any = getattr(te, 'Tensor', None)
    if AdamClass is None or SoftmaxCrossEntropyLossClass is None or LabelsClass is None or TensorClass is None:
        raise RuntimeError('Missing required runtime classes from tensor_engine.')
    if not callable(AdamClass):
        raise RuntimeError('Adam class not callable')
    # pylint: disable=not-callable
    opt: Any = AdamClass(3e-4, 0.9, 0.999, 1e-8)
    if not callable(SoftmaxCrossEntropyLossClass):
        raise RuntimeError('SoftmaxCrossEntropyLoss class not callable')
    # pylint: disable=not-callable
    loss_fn: Any = SoftmaxCrossEntropyLossClass()

    batch_size = int(args.batch)
    if use_synthetic:
        if input_ids is None or target_ids is None or images_np is None:
            raise RuntimeError("Synthetic dataset tensors not initialized")
        num_samples = input_ids.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
    else:
        # Use manifest loader
        if not hasattr(te, 'ImageTextDataLoader'):
            raise RuntimeError(
                "tensor_engine.ImageTextDataLoader is not available. Rebuild Tensor-Engine with feature vision."
            )
        LoaderClass: Any = getattr(te, 'ImageTextDataLoader', None)
        if LoaderClass is None:
            raise RuntimeError('tensor_engine module does not expose ImageTextDataLoader')
        # pylint: disable=not-callable
        loader = LoaderClass(
            str(manifest_path),
            int(w),
            int(h),
            batch_size,
            bool(args.shuffle),
            bool(args.augment),
            bool(args.parallel_io),
        )
        num_batches = int(loader.num_batches())

    # Helper functions for checkpoint save/load
    def save_model_state(model, path: Path) -> bool:
        try:
            if hasattr(model, 'save_state_dict_to_path'):
                path.parent.mkdir(parents=True, exist_ok=True)
                model.save_state_dict_to_path(str(path))
                logger.info("Saved model parameters to %s via model.save_state_dict_to_path", path)
                return True
        except (AttributeError, RuntimeError, OSError) as err:
            logger.exception("model.save_state_dict_to_path failed; will try fallback: %s", err)
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
                np_path = Path(str(path)).with_suffix('.npz')
                np_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(np_path, **params_dict)
                logger.info("Saved model parameters to %s (npz fallback)", np_path)
                return True
        except (OSError, ValueError, TypeError) as err:
            logger.exception("Failed to save model via fallback: %s", err)
            return False

    def load_model_state(model, path: Path) -> bool:
        try:
            if hasattr(model, 'load_state_dict_from_path') and path.exists():
                model.load_state_dict_from_path(str(path), False, None)
                logger.info("Loaded weights from %s via model.load_state_dict_from_path", path)
                return True
        except (AttributeError, RuntimeError, OSError) as err:
            logger.exception("model.load_state_dict_from_path failed; trying byte loaders: %s", err)
        if path.exists():
            try:
                with open(path, 'rb') as fh:
                    b = fh.read()
                try:
                    model.load_state_dict(b, False, None)
                    logger.info("Loaded weights from %s via model.load_state_dict(bytes)", path)
                    return True
                except (ValueError, TypeError, RuntimeError) as _:
                    py_load_fn = getattr(te, 'py_load_safetensors_into_module', None)
                    if callable(py_load_fn):
                        try:
                            py_load_fn(b, False, model, None)
                            logger.info("Loaded weights from %s via py_load_safetensors_into_module", path)
                            return True
                        except (RuntimeError, TypeError, ValueError) as inner_err:
                            logger.debug("Bytes loaders failed; attempting npz fallback: %s", inner_err)
                    else:
                        logger.debug("Bytes loaders failed; attempting npz fallback: %s", err)
            except OSError as err:
                logger.exception("Failed to open model path bytes for %s: %s", path, err)
        # Try numpy .npz fallback
        try:
            npz_path = Path(str(path)).with_suffix('.npz')
            if npz_path.exists():
                data = np.load(str(npz_path))
                for (name, param) in model.named_parameters(""):
                    if name in data:
                        arr = data[name]
                        param.set_data(arr.flatten().tolist())
                logger.info("Loaded model params from npz %s", npz_path)
                return True
        except (OSError, ValueError, TypeError, KeyError) as err:
            logger.exception("Failed to load model from npz fallback: %s", err)
        return False

    # Resolve default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = Path(args.save).with_suffix('.ckpt.safetensors')

    # Optionally resume
    if resume and checkpoint_path.exists():
        try:
            load_model_state(model, checkpoint_path)
        except (RuntimeError, OSError, ValueError) as err:
            logger.exception("Failed to resume from checkpoint: %s: %s", checkpoint_path, err)
            # continue training with random init

    try:
        for epoch in range(int(args.epochs)):
            epoch_loss = 0.0
            if not use_synthetic and bool(args.shuffle):
                try:
                    loader.shuffle_in_place()
                except (AttributeError, RuntimeError, TypeError, ValueError) as err:
                    logger.warning("Shuffle failed (continuing without shuffle): %s", err)

            for b in range(num_batches):
                if use_synthetic:
                    start = b * batch_size
                    end = min((b + 1) * batch_size, num_samples)
                    bs = end - start
                    img_batch = images_np[start:end]
                    try:
                        # pylint: disable=not-callable
                        img_tensor = TensorClass(img_batch.flatten().tolist(), [bs, c, h, w])
                    except (TypeError, ValueError, OSError) as err:
                        logger.exception("Failed to construct image tensor for batch %s: %s", b, err)
                        raise

                    ids_batch = input_ids[start:end].astype(np.float32)
                    if ids_batch.shape[1] == 0:
                        ids_batch = np.full((ids_batch.shape[0], 1), 0, dtype=np.float32)
                    # pylint: disable=not-callable
                    ids_tensor = TensorClass(ids_batch.flatten().tolist(), [bs, ids_batch.shape[1]])

                    targ = target_ids[start:end]
                    flat_labels = [int(x) for row in targ.tolist() for x in row]
                else:
                    if tok_obj is None:
                        raise RuntimeError("Tokenizer missing in manifest training loop")
                    try:
                        images_list, cap_ids = loader.load_batch_tokenized(b, tok_obj)
                    except Exception as err:
                        logger.exception("Failed to load batch %s from manifest: %s", b, err)
                        raise

                    if not images_list:
                        raise RuntimeError(f"Empty image batch {b}")

                    # Each image tensor is [1,C,H,W]; concat along batch -> [B,C,H,W]
                    try:
                        img_tensor = TensorClass.cat(images_list, 0)
                    except Exception as err:
                        logger.exception("Failed to batch images via Tensor.cat: %s", err)
                        raise

                    # Build teacher-forcing sequences: seq = [bos] + prompt + caption + [eos]
                    try:
                        prompt_ids_u32 = tok_obj.encode(str(args.prompt))
                    except Exception as err:
                        logger.exception("Tokenizer failed to encode prompt: %s", err)
                        raise
                    prompt_ids = [int(x) for x in prompt_ids_u32]

                    inputs_seqs: list[list[int]] = []
                    targets_seqs: list[list[int]] = []
                    for ids_u32 in cap_ids:
                        caption_ids = [int(x) for x in ids_u32]
                        seq = [bos_id] + prompt_ids + caption_ids + [eos_id]
                        if len(seq) < 2:
                            seq = [bos_id, eos_id]
                        inp = seq[:-1]
                        tgt = seq[1:]
                        inputs_seqs.append(inp)
                        targets_seqs.append(tgt)

                    max_len_batch = min(
                        int(args.max_seq_len),
                        max((len(s) for s in inputs_seqs), default=1),
                    )
                    ids_arr = _pad_2d_int(inputs_seqs, pad=pad_id, max_len=max_len_batch)
                    targ_arr = _pad_2d_int(targets_seqs, pad=pad_id, max_len=max_len_batch)

                    # pylint: disable=not-callable
                    ids_tensor = TensorClass(ids_arr.flatten().tolist(), [ids_arr.shape[0], ids_arr.shape[1]])
                    targ = targ_arr
                    flat_labels = [int(x) for row in targ.tolist() for x in row]
                    bs = int(ids_arr.shape[0])

                try:
                    img_tokens = model.vision_forward(img_tensor)
                except (RuntimeError, TypeError, ValueError) as err:
                    logger.exception("Vision forward failed for batch %s: %s", b, err)
                    raise

                # Debug shapes
                try:
                    img_tokens_shape = tuple(img_tokens.shape)
                except (AttributeError, TypeError):
                    img_tokens_shape = None
                try:
                    ids_shape = tuple(ids_tensor.shape)
                except (AttributeError, TypeError):
                    ids_shape = None
                logger.debug("img_tokens.shape=%s ids.shape=%s", img_tokens_shape, ids_shape)

                # Defensive check: if ids has zero-length sequence dimension, replace with padding column
                if ids_shape is not None and ids_shape[1] == 0:
                    logger.warning("Zero-length token sequence encountered in batch %s; replacing with pad token.", b)
                    ids_batch = np.full((ids_batch.shape[0], 1), vocab["<pad>"], dtype=np.float32)
                    ids_tensor = TensorClass(ids_batch.flatten().tolist(), [bs, ids_batch.shape[1]])

                # Defensive check: if image tokens sequence dimension is zero, create a zero token
                if img_tokens_shape is not None and img_tokens_shape[1] == 0:
                    logger.warning("Zero-length image tokens in batch %s; inserting zero token to avoid matmul panic", b)
                    zeros_img_tokens = [0.0] * (bs * 1 * d_model)
                    img_tokens = TensorClass(zeros_img_tokens, [bs, 1, d_model])

                try:
                    logits = model.forward(img_tensor, ids_tensor)
                except (RuntimeError, TypeError, ValueError) as err:
                    logger.exception("Model forward failed for batch %s: %s", b, err)
                    raise

                n_image_tokens = img_tokens.shape[1]
                logits_text = logits[:, n_image_tokens:, :]

                try:
                    # pylint: disable=not-callable
                    labels_obj: Any = LabelsClass(flat_labels)
                    loss = loss_fn.forward_from_labels(logits_text, labels_obj)
                    loss.backward()
                    params = model.parameters()
                    opt.step(params)
                    opt.zero_grad(params)
                    epoch_loss += float(loss.get_data())
                except (RuntimeError, ValueError, TypeError) as err:
                    logger.exception("Loss/backprop/optimizer step failed for batch %s: %s", b, err)
                    raise
            logger.info("Epoch %d/%d, loss=%0.4f", epoch+1, args.epochs, epoch_loss/num_batches)
            # epoch-level checkpointing
            if checkpoint_interval > 0 and ((epoch + 1) % checkpoint_interval) == 0:
                try:
                    save_model_state(model, checkpoint_path)
                except (RuntimeError, OSError, ValueError) as err:
                    logger.exception("Failed to save checkpoint at epoch %s: %s", epoch + 1, err)

    except (RuntimeError, OSError, ValueError, KeyboardInterrupt, TypeError) as err:
        logger.exception("Training failed; attempting to save partial checkpoint: %s", err)
        try:
            partial_path = Path(str(checkpoint_path)).with_name(str(checkpoint_path.stem) + f'.partial.{int(time.time())}' + str(checkpoint_path.suffix))
            save_model_state(model, partial_path)
            logger.info("Saved partial checkpoint to %s", partial_path)
        except (RuntimeError, OSError, ValueError) as save_err:
            logger.exception("Failed to save partial checkpoint: %s", save_err)
        raise

    logger.info("Training done!")
    # Final checkpoint save
    try:
        save_model_state(model, checkpoint_path)
    except (RuntimeError, OSError, ValueError) as err:
        logger.exception("Final checkpoint save failed: %s", err)
    save_path = Path(args.save)

    # Try saving via model API
    save_used = False
    try:
        if hasattr(model, 'save_state_dict_to_path'):
            model.save_state_dict_to_path(str(save_path))
            logger.info("Saved model parameters to %s via model.save_state_dict_to_path", save_path)
            save_used = True
    except (AttributeError, RuntimeError, OSError) as err:
        logger.exception("model.save_state_dict_to_path failed: %s", err)

    if not save_used:
        try:
            # import placed here to avoid optional dependency at module import time
            # pylint: disable=import-outside-toplevel
            from safetensors.numpy import save_file
        except ImportError as err:
            save_file = None
            logger.debug("safetensors.numpy import failed, falling back to numpy save: %s", err)

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
        except (OSError, ValueError, TypeError) as err:
            logger.exception("Failed to persist final model parameters: %s", err)

    if tok_obj is not None and not use_synthetic:
        try:
            logger.info("Tokenizer vocab size: %s", tok_obj.vocab_size())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


if __name__ == "__main__":
    main()
