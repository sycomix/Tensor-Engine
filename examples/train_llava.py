#!/usr/bin/env python3
"""
Minimal training loop for a toy LLaVA-like model using `tensor_engine` Python bindings.

The script uses a synthetic dataset and builds a tiny multimodal model by projecting
image patches and text token ids into a shared `d_model` embedding space, concatenating
image tokens as prefix tokens followed by text embeddings, applying a few `TransformerBlock`s,
then projecting to a vocabulary size for language modeling.

This minimal example is intended for quick dev/test/CI use and does not attempt to
improve model quality. It demonstrates how to use the Python bindings for multimodal flows.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, ParamSpec, cast

P = ParamSpec("P")


def _import_optional(module_name: str) -> Any | None:
    """Import a module by name, returning None if it is not installed."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


def _require_module(module_name: str, purpose: str) -> Any:
    """Import a required module, raising a helpful error message if missing."""
    mod = _import_optional(module_name)
    if mod is None:
        raise RuntimeError(
            f"Missing Python dependency '{module_name}' ({purpose}). "
            "Install project requirements and retry."
        )
    return mod


te = _import_optional("tensor_engine")

# runtime bindings via getattr to avoid static analyzer warnings
if te is not None:
    vision_transformer_class: Any = getattr(te, "VisionTransformer", None)
    multimodal_llm_class: Any = getattr(te, "MultimodalLLM", None)
    adam_class: Any = getattr(te, "Adam", None)
    softmax_cross_entropy_loss_class: Any = getattr(te, "SoftmaxCrossEntropyLoss", None)
    tensor_class: Any = getattr(te, "Tensor", None)
    labels_class: Any = getattr(te, "Labels", None)
else:
    vision_transformer_class = None
    multimodal_llm_class = None
    adam_class = None
    softmax_cross_entropy_loss_class = None
    tensor_class = None
    labels_class = None


def prepare_synthetic_dataset(path: Path, num_examples: int, h: int, w: int, c: int) -> None:
    """Create a small random synthetic dataset on disk as JSONL."""
    data = []
    np_mod = _require_module("numpy", "synthetic dataset generation")
    rng = np_mod.random.default_rng(42)
    for i in range(num_examples):
        img = (rng.random((h, w, c), dtype=np_mod.float32) * 2 - 1).astype(np_mod.float32)
        inp = f"Describe the image {i}"
        tgt = f"Synthetic image {i} description"
        data.append({
            "image": img.flatten().tolist(),
            "height": h,
            "width": w,
            "channels": c,
            "input_text": inp,
            "target_text": tgt,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in data:
            fh.write(json.dumps(rec) + "\n")


def build_vocab_from_data(records: list[dict[str, Any]]) -> dict[str, int]:
    """Build a trivial whitespace vocab from dataset records."""
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    for rec in records:
        for tok in rec["input_text"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
        for tok in rec["target_text"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def tokenize_texts(
        records: list[dict[str, Any]],
        vocab: dict[str, int],
) -> tuple[list[list[int]], list[list[int]]]:
    """Tokenize input/target texts into integer ids using the given vocab."""
    inputs = []
    targets = []
    for rec in records:
        input_tokens = [vocab["<bos>"]] + [vocab[t] for t in rec["input_text"].split()] + [vocab["<eos>"]]
        target_tokens = [vocab["<bos>"]] + [vocab[t] for t in rec["target_text"].split()] + [vocab["<eos>"]]
        inputs.append(input_tokens)
        targets.append(target_tokens)
    return inputs, targets


def pad_and_stack_token_ids(token_list: list[list[int]], pad: int = 0) -> Any:
    """Pad a ragged list of token id lists and return a float32 NumPy array."""
    np_mod = _require_module("numpy", "token id padding")
    # Ensure a minimum sequence length of 1 to avoid creating arrays with a zero-width
    # dimension which can cause downstream ops (e.g., matmul) to panic.
    max_len = max(1, max(len(lst) for lst in token_list)) if token_list else 1
    arr = np_mod.full((len(token_list), max_len), pad, dtype=np_mod.float32)
    for i, lst in enumerate(token_list):
        arr[i, : len(lst)] = lst
    return arr


def image_to_patches(
        images: list[list[float]],
        h: int,
        w: int,
        c: int,
        patch_size: int = 8,
) -> Any:
    """Convert flattened images into a (B, n_patches, patch_flat) NumPy array."""
    np_mod = _require_module("numpy", "image patch conversion")
    # images: list of flattened arrays per sample
    imgs = [np_mod.array(img, dtype=np_mod.float32).reshape((h, w, c)) for img in images]
    batch = []
    for img in imgs:
        patches = []
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = img[y: y + patch_size, x: x + patch_size, :]
                patches.append(patch.flatten())
        batch.append(np_mod.stack(patches))
    # shape (B, n_patches, patch_flat)
    return np_mod.stack(batch)


def main() -> None:
    """Run a tiny training loop for a toy multimodal model."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/data/synthetic_llava.jsonl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--save", default="examples/models/llava_model.safetensors")
    args = parser.parse_args()

    np_mod = _require_module("numpy", "training loop")

    data_path = Path(args.data)
    if not data_path.exists():
        logger.info("Dataset not found; generating synthetic dataset")
        prepare_synthetic_dataset(data_path, num_examples=16, h=32, w=32, c=3)

    # Load dataset
    records = []
    with open(data_path, "r", encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line.strip()))

    # If records already contain tokenized ids (e.g. prepared with a tokenizer), use those.
    if "input_ids" in records[0]:
        inputs_tokens = [rec.get("input_ids", []) for rec in records]
        targets_tokens = [rec.get("target_ids", []) for rec in records]
        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        # We don't reconstruct a full vocab here since token ids are present already
    else:
        vocab = build_vocab_from_data(records)
        inputs_tokens, targets_tokens = tokenize_texts(records, vocab)

    # pad and convert to numpy arrays
    input_ids = pad_and_stack_token_ids(inputs_tokens, pad=vocab["<pad>"])
    target_ids = pad_and_stack_token_ids(targets_tokens, pad=vocab["<pad>"])

    # images to patches
    images = [rec["image"] for rec in records]
    h = records[0]["height"]
    w = records[0]["width"]
    c = records[0]["channels"]
    # Convert images to NumPy array in channel-first layout for model.forward
    images_np = np_mod.stack([np_mod.array(i).reshape((h, w, c)) for i in images])
    images_np = images_np.transpose((0, 3, 1, 2))  # [B, C, H, W]

    # Build model using tensor_engine
    if te is None:
        raise RuntimeError(
            "tensor_engine Python package not found. Build with 'maturin develop --release'."
        )

    d_model = args.d_model
    vocab_size = len(vocab)

    # Instantiate Vision + Multimodal model using Python bindings via runtime lookup
    if vision_transformer_class is None or multimodal_llm_class is None:
        raise RuntimeError(
            "tensor_engine module does not expose VisionTransformer and/or MultimodalLLM; "
            "rebuild with python bindings and vision features"
        )
    if not callable(vision_transformer_class):
        raise RuntimeError("VisionTransformer class is not callable")
    vt_ctor: Callable[P, Any] = cast(Callable[P, Any], vision_transformer_class)
    vision: Any = vt_ctor(
        3,
        args.patch_size,
        d_model,
        d_model * 4,
        num_heads=4,
        depth=args.num_blocks,
        max_len=512,
    )

    if not callable(multimodal_llm_class):
        raise RuntimeError("MultimodalLLM class is not callable")
    mm_ctor: Callable[P, Any] = cast(Callable[P, Any], multimodal_llm_class)
    model: Any = mm_ctor(vision, vocab_size, d_model, d_model * 4, num_heads=4, depth=args.num_blocks)

    if adam_class is None or softmax_cross_entropy_loss_class is None:
        raise RuntimeError("Missing optimizer/loss runtime classes from tensor_engine")
    adam_ctor: Callable[P, Any] = cast(Callable[P, Any], adam_class)
    opt: Any = adam_ctor(3e-4, 0.9, 0.999, 1e-8)
    softmax_ctor: Callable[P, Any] = cast(Callable[P, Any], softmax_cross_entropy_loss_class)
    loss_fn: Any = softmax_ctor()

    num_samples = input_ids.shape[0]
    batch_size = args.batch
    num_batches = (num_samples + batch_size - 1) // batch_size

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, num_samples)
            bs = end - start

            # Image tensor: shape (B, C, H, W)
            img_batch = images_np[start:end]
            # pylint: disable=not-callable
            if tensor_class is None:
                raise RuntimeError("tensor_engine did not expose Tensor")
            img_tensor: Any = tensor_class(img_batch.flatten().tolist(), [bs, c, h, w])
            # The model's vision encoder expects images shape [B, C, H, W] or preprocessed patches (we'll reuse a patch-based flow similar to earlier)
            # For this toy example, use the VisionTransformer directly to produce image token embeddings.
            # Convert patches back to images if needed (simple reshape)
            # However to keep the example simple, we can reuse patch projection via model.vision_encoder.patch_embed if exposed; otherwise call model.vision_encoder.forward
            # Let the model's vision encoder compute patch tokens internally via model.forward
            # For the multimodal LLM, we pass images to model.forward directly and let the encoder run.
            img_tokens = model.vision_forward(img_tensor)

            # text embeddings via embedding lookup (provided by the model's text embedding trainable weight)
            ids_batch = input_ids[start:end].astype(np_mod.float32)
            # Safety: ensure seq length at least 1
            if ids_batch.shape[1] == 0:
                ids_batch = np_mod.full(
                    (ids_batch.shape[0], 1),
                    vocab["<pad>"],
                    dtype=np_mod.float32,
                )
            # pylint: disable=not-callable
            ids_tensor: Any = tensor_class(ids_batch.flatten().tolist(), [bs, ids_batch.shape[1]])

            # Defensive check: ensure ids_tensor second dimension > 0 (no zero-width sequences). If it happens, replace with pad and log.
            if ids_tensor.shape[1] == 0:
                logger.warning("Zero-length token sequence encountered; replacing with pad token to avoid matmul panic")
                ids_batch = np_mod.full(
                    (ids_batch.shape[0], 1),
                    vocab["<pad>"],
                    dtype=np_mod.float32,
                )
                ids_tensor = tensor_class(ids_batch.flatten().tolist(), [bs, ids_batch.shape[1]])

            # create combined sequence: image tokens, then text tokens
            # Forward through the Multimodal model using images and ids
            logits = model.forward(img_tensor, ids_tensor)

            # take text part of logits only for loss
            n_image_tokens = img_tokens.shape[1]
            logits_text = logits[:, n_image_tokens:, :]
            # targets for text: (B, seq)
            targ = target_ids[start:end]
            # flatten labels for the SoftmaxCrossEntropyLoss forward_from_labels convenience
            flat_labels = [int(x) for row in targ.tolist() for x in row]
            # We don't explicitly need a targ_tensor for this workflow — the Labels object is created below

            # pylint: disable=not-callable
            if labels_class is None:
                raise RuntimeError("tensor_engine did not expose Labels")
            labels_ctor: Callable[P, Any] = cast(Callable[P, Any], labels_class)
            labels_obj: Any = labels_ctor(flat_labels)
            loss = loss_fn.forward_from_labels(logits_text, labels_obj)
            loss.backward()
            # Use model.parameters() to collect parameters
            params = model.parameters()
            opt.step(params)
            # clear grads
            opt.zero_grad(params)
            epoch_loss += loss.get_data()

        logger.info("Epoch %d/%d, loss=%0.4f", epoch + 1, args.epochs, epoch_loss / num_batches)

    logger.info("Training done!")
    save_path = Path(args.save)
    # Try to use model-level Save API if available (preferred), else fallback to safetensors.numpy save_file or .npz
    save_used = False
    try:
        if hasattr(model, 'save_state_dict_to_path'):
            model.save_state_dict_to_path(str(save_path))
            logger.info("Saved model parameters to %s via model.save_state_dict_to_path", save_path)
            save_used = True
    except (AttributeError, RuntimeError, OSError) as e:
        logger.error("model.save_state_dict_to_path failed: %s", e)

    if not save_used:
        # Export model parameters as SafeTensors (NumPy format) if safetensors is available
        save_file = None
        safetensors_mod = _import_optional("safetensors")
        if safetensors_mod is not None:
            try:
                safetensors_numpy = importlib.import_module("safetensors.numpy")
                save_file = getattr(safetensors_numpy, "save_file", None)
            except (ImportError, ModuleNotFoundError, AttributeError):
                save_file = None

        params_dict = {}
        for (name, param) in model.named_parameters(""):
            # param: PyTensor
            data = np_mod.array(param.get_data(), dtype=np_mod.float32)
            shape = tuple(param.shape)
            arr = data.reshape(shape)
            params_dict[name] = arr

        if save_file is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(params_dict, str(save_path))
            logger.info("Saved model parameters to %s", save_path)
        else:
            # fallback: save as numpy .npz
            save_path_npz = save_path.with_suffix('.npz')
            np_mod.savez(save_path_npz, **params_dict)
            logger.info("Saved model parameters to %s (safetensors not available)", save_path_npz)


if __name__ == "__main__":
    main()
