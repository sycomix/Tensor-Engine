#!/usr/bin/env python3
"""Train a MultimodalLLM using streaming image+text batches.

This script is the scalable successor to `examples/train_llava.py`.

It follows `next.md` requirements:
- Streams data via `tensor_engine.ImageTextDataLoader` (no full dataset in RAM)
- Uses a real tokenizer via `tensor_engine.Tokenizer.from_file(...)`
- Initializes the projector explicitly based on config
- Supports optional CUDA backend if available
- Saves checkpoints periodically
- Logs average loss

Notes:
- This is a minimal, practical trainer for development workflows.
- It assumes the `tensor_engine` Python extension is built with:
  - `--features python_bindings,vision`
  - plus `with_tokenizers` for tokenization support
  - plus `safe_tensors` for checkpoint saving
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    config_path: Path
    manifest_path: Path
    tokenizer_path: Path
    output_dir: Path
    epochs: int
    batch_size: int
    image_w: int
    image_h: int
    shuffle: bool
    augment: bool
    parallel: bool
    lr: float
    beta1: float
    beta2: float
    eps: float
    checkpoint_every: int
    device: str
    pad_id: int


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return obj


def _pad_2d(seqs: Sequence[Sequence[int]], pad_id: int) -> Tuple[List[float], int, int]:
    """Pad variable-length int sequences into a flat float list and shape."""
    b = len(seqs)
    max_len = max((len(s) for s in seqs), default=1)
    max_len = max(1, max_len)
    flat: List[float] = []
    flat_extend = flat.extend
    for s in seqs:
        row = list(s)
        if not row:
            row = [pad_id]
        if len(row) < max_len:
            row = row + [pad_id] * (max_len - len(row))
        # Tensor constructor expects f32 values
        flat_extend([float(x) for x in row])
    return flat, b, max_len


def _require_attr(obj: Any, name: str) -> Any:
    v = getattr(obj, name, None)
    if v is None:
        raise RuntimeError(f"Required attribute '{name}' is not available in tensor_engine bindings")
    return v


def _maybe_set_backend(te: Any, device: str) -> None:
    device_l = device.lower().strip()
    if device_l not in {"cpu", "cuda", "auto"}:
        raise ValueError("--device must be one of: cpu, cuda, auto")

    set_cpu = getattr(te, "set_cpu_backend", None)
    set_cuda = getattr(te, "set_cuda_backend", None)

    if device_l == "cpu":
        if callable(set_cpu):
            set_cpu()
        _LOG.info("Backend set to CPU")
        return

    if device_l == "cuda":
        if not callable(set_cuda):
            raise RuntimeError("CUDA backend is not available in this build")
        set_cuda()
        _LOG.info("Backend set to CUDA")
        return

    # auto
    if callable(set_cuda):
        try:
            set_cuda()
            _LOG.info("Backend set to CUDA (auto)")
            return
        except (RuntimeError, OSError, ValueError, TypeError) as e:
            _LOG.warning("CUDA backend not usable; falling back to CPU: %s", e)
    if callable(set_cpu):
        set_cpu()
    _LOG.info("Backend set to CPU (auto)")


def _save_checkpoint(model: Any, out_dir: Path, epoch: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"checkpoint_epoch_{epoch:04d}.safetensors"
    save_fn = getattr(model, "save_state_dict_to_path", None)
    if not callable(save_fn):
        raise RuntimeError(
            "Model does not expose save_state_dict_to_path; build with feature 'safe_tensors'"
        )
    save_fn(str(ckpt_path))
    return ckpt_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train tensor_engine MultimodalLLM from an image+caption manifest"
    )
    parser.add_argument("--config", required=True, help="Path to JSON model config")
    parser.add_argument("--manifest", required=True, help="Path to manifest.txt (TSV)")
    parser.add_argument("--tokenizer", required=True, help="Path to HuggingFace tokenizer.json")
    parser.add_argument("--output", default="runs/multimodal_v1", help="Output directory")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-w", type=int, default=224)
    parser.add_argument("--image-h", type=int, default=224)

    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--parallel", action="store_true")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save every N epochs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pad-id", type=int, default=0, help="Token id used for padding")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = TrainConfig(
        config_path=Path(args.config).expanduser().resolve(),
        manifest_path=Path(args.manifest).expanduser().resolve(),
        tokenizer_path=Path(args.tokenizer).expanduser().resolve(),
        output_dir=Path(args.output).expanduser().resolve(),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        image_w=int(args.image_w),
        image_h=int(args.image_h),
        shuffle=bool(args.shuffle),
        augment=bool(args.augment),
        parallel=bool(args.parallel),
        lr=float(args.lr),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        eps=float(args.eps),
        checkpoint_every=max(1, int(args.checkpoint_every)),
        device=str(args.device),
        pad_id=int(args.pad_id),
    )

    if cfg.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if cfg.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    # Import tensor_engine late so logging is configured first.
    try:
        import tensor_engine as te  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "tensor_engine Python package not found. Build it with 'maturin develop --release'."
        ) from e

    _maybe_set_backend(te, cfg.device)

    # Validate required bindings.
    ImageTextDataLoader = _require_attr(te, "ImageTextDataLoader")
    MultimodalLLM = _require_attr(te, "MultimodalLLM")
    Adam = _require_attr(te, "Adam")
    SoftmaxCrossEntropyLoss = _require_attr(te, "SoftmaxCrossEntropyLoss")
    Labels = _require_attr(te, "Labels")
    Tokenizer = getattr(te, "Tokenizer", None)
    if Tokenizer is None:
        raise RuntimeError(
            "Tokenizer bindings not available. Build with '--features with_tokenizers,python_bindings'."
        )

    model_cfg = _load_json(cfg.config_path)

    # Create loader/tokenizer.
    loader = ImageTextDataLoader(
        manifest_path=str(cfg.manifest_path),
        image_w=cfg.image_w,
        image_h=cfg.image_h,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        augment=cfg.augment,
        parallel=cfg.parallel,
    )

    tokenizer = Tokenizer.from_file(str(cfg.tokenizer_path))

    # Create model and projector.
    model = MultimodalLLM.from_config(str(cfg.config_path))
    projector_type = str(model_cfg.get("projector_type", "mlp")).lower().strip()
    if projector_type == "mlp":
        d_model = int(model_cfg.get("d_model", 768))
        model.set_projector_mlp(hidden_dim=d_model)
        _LOG.info("Initialized projector: MLP (hidden_dim=%d)", d_model)
    elif projector_type == "linear":
        # Allow user to set linear projector later; MLP is the recommended default.
        raise RuntimeError(
            "projector_type 'linear' is not configured automatically by this script. "
            "Use projector_type 'mlp' or extend the script to construct te.Linear and call set_projector_linear."
        )
    else:
        raise ValueError(f"Unknown projector_type: {projector_type}")

    opt = Adam(cfg.lr, cfg.beta1, cfg.beta2, cfg.eps)
    loss_fn = SoftmaxCrossEntropyLoss()

    num_batches = int(loader.num_batches())
    if num_batches <= 0:
        raise RuntimeError("DataLoader reported zero batches")

    run_dir = cfg.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_meta.json"
    meta = {
        "config": str(cfg.config_path),
        "manifest": str(cfg.manifest_path),
        "tokenizer": str(cfg.tokenizer_path),
        "started_at": time.time(),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "image_w": cfg.image_w,
        "image_h": cfg.image_h,
        "device": cfg.device,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _LOG.info("Starting training: epochs=%d batches/epoch=%d", cfg.epochs, num_batches)

    for epoch in range(1, cfg.epochs + 1):
        if cfg.shuffle and hasattr(loader, "shuffle_in_place"):
            loader.shuffle_in_place()

        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch_idx in range(num_batches):
            # Streaming load from Rust.
            images, tokenized = loader.load_batch_tokenized(batch_idx, tokenizer)

            if not images or not tokenized:
                continue
            if len(images) != len(tokenized):
                raise RuntimeError("Loader returned mismatched images and tokenized lengths")

            # Build [B, C, H, W] image tensor by concatenating per-sample tensors along batch axis.
            # The loader returns tensors with shape [1, 3, H, W].
            images_batch = te.Tensor.cat(list(images), axis=0)

            # Pad token ids and build [B, seq] token tensor.
            token_seqs: List[List[int]] = [list(map(int, seq)) for seq in tokenized]
            flat_ids, bsz, seq_len = _pad_2d(token_seqs, pad_id=cfg.pad_id)
            input_ids = te.Tensor(flat_ids, [bsz, seq_len])

            # Forward.
            logits = model.forward(images_batch, input_ids)

            # Determine number of image tokens so we can isolate text logits.
            img_tokens = model.vision_forward(images_batch)
            n_img = int(img_tokens.shape[1])
            logits_text = logits[:, n_img:, :]

            # Labels: flatten token ids for convenience API.
            flat_labels: List[int] = []
            flat_labels_extend = flat_labels.extend
            for seq in token_seqs:
                if not seq:
                    seq = [cfg.pad_id]
                if len(seq) < seq_len:
                    seq = seq + [cfg.pad_id] * (seq_len - len(seq))
                flat_labels_extend(seq)

            labels = Labels([int(x) for x in flat_labels])
            loss = loss_fn.forward_from_labels(logits_text, labels)
            loss.backward()

            params = model.parameters()
            opt.step(params)
            opt.zero_grad(params)

            loss_val = float(loss.get_data())
            epoch_loss_sum += loss_val
            epoch_steps += 1

            if batch_idx % 50 == 0:
                _LOG.info("epoch=%d batch=%d/%d loss=%0.6f", epoch, batch_idx, num_batches, loss_val)

        avg_loss = epoch_loss_sum / max(1, epoch_steps)
        _LOG.info("epoch=%d avg_loss=%0.6f steps=%d", epoch, avg_loss, epoch_steps)

        if epoch % cfg.checkpoint_every == 0:
            ckpt = _save_checkpoint(model, run_dir, epoch)
            _LOG.info("Saved checkpoint: %s", ckpt)

    final_ckpt = _save_checkpoint(model, run_dir, cfg.epochs)
    _LOG.info("Training complete. Final checkpoint: %s", final_ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
