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
import json
from pathlib import Path
import argparse
import logging
import numpy as np
from typing import Any, Callable, cast
# Linting: these tests require the `tensor_engine` package to be installed in the selected Python environment; editors without our venv may report import errors. Disable some lint warnings for this example.
# pylint: disable=import-error,reimported,wrong-import-position,missing-module-docstring,unused-import,no-member,no-name-in-module,not-callable

try:
    import tensor_engine as te  # type: ignore
except ImportError:  # pragma: no cover
    te = None  # type: ignore

# runtime bindings via getattr to avoid static analyzer warnings
if te is not None:
    VTClass: Any = getattr(te, 'VisionTransformer', None)
    MMClass: Any = getattr(te, 'MultimodalLLM', None)
    AdamClass: Any = getattr(te, 'Adam', None)
    SoftmaxCrossEntropyLossClass: Any = getattr(te, 'SoftmaxCrossEntropyLoss', None)
    TensorClass: Any = getattr(te, 'Tensor', None)
    LabelsClass: Any = getattr(te, 'Labels', None)
else:
    VTClass = None
    MMClass = None
    AdamClass = None
    SoftmaxCrossEntropyLossClass = None
    TensorClass = None
    LabelsClass = None


def prepare_synthetic_dataset(path: Path, num_examples: int, h: int, w: int, c: int):
    data = []
    rng = np.random.default_rng(42)
    for i in range(num_examples):
        img = (rng.random((h, w, c), dtype=np.float32) * 2 - 1).astype(np.float32)
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


def build_vocab_from_data(records):
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    for rec in records:
        for tok in rec["input_text"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
        for tok in rec["target_text"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def tokenize_texts(records, vocab):
    inputs = []
    targets = []
    for rec in records:
        input_tokens = [vocab["<bos>"]] + [vocab[t] for t in rec["input_text"].split()] + [vocab["<eos>"]]
        target_tokens = [vocab["<bos>"]] + [vocab[t] for t in rec["target_text"].split()] + [vocab["<eos>"]]
        inputs.append(input_tokens)
        targets.append(target_tokens)
    return inputs, targets


def pad_and_stack_token_ids(token_list, pad=0):
    # Ensure a minimum sequence length of 1 to avoid creating arrays with a zero-width
    # dimension which can cause downstream ops (e.g., matmul) to panic.
    max_len = max(1, max(len(lst) for lst in token_list)) if token_list else 1
    arr = np.full((len(token_list), max_len), pad, dtype=np.float32)
    for i, lst in enumerate(token_list):
        arr[i, : len(lst)] = lst
    return arr


def image_to_patches(images, h, w, c, patch_size=8):
    # images: list of flattened arrays per sample
    imgs = [np.array(img, dtype=np.float32).reshape((h, w, c)) for img in images]
    batch = []
    for img in imgs:
        patches = []
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = img[y : y + patch_size, x : x + patch_size, :]
                patches.append(patch.flatten())
        batch.append(np.stack(patches))
    # shape (B, n_patches, patch_flat)
    return np.stack(batch)


def main():
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
    images_np = np.stack([np.array(i).reshape((h, w, c)) for i in images])
    images_np = images_np.transpose((0, 3, 1, 2))  # [B, C, H, W]

    # Build model using tensor_engine
    if te is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")

    d_model = args.d_model
    vocab_size = len(vocab)

    # Instantiate Vision + Multimodal model using Python bindings via runtime lookup
    if VTClass is None or MMClass is None:
        raise RuntimeError('tensor_engine module does not expose VisionTransformer and/or MultimodalLLM; rebuild with python_bindings and vision features')
    if not callable(VTClass):
        raise RuntimeError('VisionTransformer class is not callable')
    VT_ctor: Callable[..., Any] = cast(Callable[..., Any], VTClass)
    vision: Any = VT_ctor(3, args.patch_size, d_model, d_model * 4, num_heads=4, depth=args.num_blocks, max_len=512)

    if not callable(MMClass):
        raise RuntimeError('MultimodalLLM class is not callable')
    MM_ctor: Callable[..., Any] = cast(Callable[..., Any], MMClass)
    model: Any = MM_ctor(vision, vocab_size, d_model, d_model * 4, num_heads=4, depth=args.num_blocks)

    if AdamClass is None or SoftmaxCrossEntropyLossClass is None:
        raise RuntimeError('Missing optimizer/loss runtime classes from tensor_engine')
    Adam_ctor: Callable[..., Any] = cast(Callable[..., Any], AdamClass)
    opt: Any = Adam_ctor(3e-4, 0.9, 0.999, 1e-8)
    SoftmaxCtor: Callable[..., Any] = cast(Callable[..., Any], SoftmaxCrossEntropyLossClass)
    loss_fn: Any = SoftmaxCtor()

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
            img_tensor: Any = TensorClass(img_batch.flatten().tolist(), [bs, c, h, w])
            # The model's vision encoder expects images shape [B, C, H, W] or preprocessed patches (we'll reuse a patch-based flow similar to earlier)
            # For this toy example, use the VisionTransformer directly to produce image token embeddings.
            # Convert patches back to images if needed (simple reshape)
            # However to keep the example simple, we can reuse patch projection via model.vision_encoder.patch_embed if exposed; otherwise call model.vision_encoder.forward
            # Let the model's vision encoder compute patch tokens internally via model.forward
            # For the multimodal LLM, we pass images to model.forward directly and let the encoder run.
            img_tokens = model.vision_forward(img_tensor)

            # text embeddings via embedding lookup (provided by the model's text embedding trainable weight)
            ids_batch = input_ids[start:end].astype(np.float32)
            # Safety: ensure seq length at least 1
            if ids_batch.shape[1] == 0:
                ids_batch = np.full((ids_batch.shape[0], 1), vocab["<pad>"], dtype=np.float32)
            # pylint: disable=not-callable
            ids_tensor: Any = TensorClass(ids_batch.flatten().tolist(), [bs, ids_batch.shape[1]])

            # create combined sequence: image tokens, then text tokens
            # Forward through the Multimodal model using images and ids
            logits = model.forward(img_tokens, ids_tensor)

            # take text part of logits only for loss
            n_image_tokens = img_tokens.shape[1]
            logits_text = logits[:, n_image_tokens :, :]
            # targets for text: (B, seq)
            targ = target_ids[start:end]
            # flatten labels for the SoftmaxCrossEntropyLoss forward_from_labels convenience
            flat_labels = [int(x) for row in targ.tolist() for x in row]
            # We don't explicitly need a targ_tensor for this workflow — the Labels object is created below

            # pylint: disable=not-callable
            Labels_ctor: Callable[..., Any] = cast(Callable[..., Any], LabelsClass)
            labels_obj: Any = Labels_ctor(flat_labels)
            loss = loss_fn.forward_from_labels(logits_text, labels_obj)
            loss.backward()
            # Use model.parameters() to collect parameters
            params = model.parameters()
            opt.step(params)
            # clear grads
            opt.zero_grad(params)
            epoch_loss += loss.get_data()

        logger.info("Epoch %d/%d, loss=%0.4f", epoch+1, args.epochs, epoch_loss / num_batches)

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
        try:
            from safetensors.numpy import save_file  # type: ignore
        except ImportError:
            save_file = None

        params_dict = {}
        for (name, param) in model.named_parameters(""):
            # param: PyTensor
            data = np.array(param.get_data(), dtype=np.float32)
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
            np.savez(save_path_npz, **params_dict)
            logger.info("Saved model parameters to %s (safetensors not available)", save_path_npz)


if __name__ == "__main__":
    main()
