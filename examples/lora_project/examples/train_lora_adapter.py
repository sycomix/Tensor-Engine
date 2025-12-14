#!/usr/bin/env python3
"""Train a LoRA-style low-rank adapter on top of a pretrained base checkpoint.

This example applies the low-rank update to the vocab head:

  logits = base_logits + (alpha/r) * B(A(hidden))

Only LoRA parameters are optimized.

Expected base checkpoint format: same as `examples/pretrain_project`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Protocol, Sequence, Tuple, cast

try:
    import numpy as np  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError("numpy is required for this example. Install it via requirements.txt") from e

try:
    import tensor_engine as te  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError("tensor_engine Python package not found. Build with maturin develop --release.") from e

te_mod: types.ModuleType = cast(types.ModuleType, te)


class TensorLike(Protocol):
    shape: Sequence[int]

    def get_data(self) -> Sequence[float]:
        raise NotImplementedError

    def set_data(self, data: Sequence[float]) -> None:
        raise NotImplementedError

    def reshape(self, shape: Sequence[int]) -> "TensorLike":
        raise NotImplementedError

    def backward(self) -> None:
        raise NotImplementedError

    def softmax_cross_entropy_with_logits(self, targets: "TensorLike") -> "TensorLike":
        raise NotImplementedError

    def __add__(self, other: "TensorLike") -> "TensorLike":
        raise NotImplementedError

    def __mul__(self, other: "TensorLike") -> "TensorLike":
        raise NotImplementedError


class OptimizerLike(Protocol):
    def zero_grad(self, params: Sequence[TensorLike]) -> None:
        raise NotImplementedError

    def step(self, params: Sequence[TensorLike]) -> None:
        raise NotImplementedError


class LinearLike(Protocol):
    def forward(self, x: TensorLike) -> TensorLike:
        raise NotImplementedError

    def parameters(self) -> Sequence[TensorLike]:
        raise NotImplementedError

    def named_parameters(self, prefix: str) -> Sequence[Tuple[str, TensorLike]]:
        raise NotImplementedError


class LinearWithWeightLike(LinearLike, Protocol):
    weight: TensorLike


class TransformerBlockLike(Protocol):
    def forward(self, x: TensorLike) -> TensorLike:
        raise NotImplementedError

    def parameters(self) -> Sequence[TensorLike]:
        raise NotImplementedError

    def named_parameters(self, prefix: str) -> Sequence[Tuple[str, TensorLike]]:
        raise NotImplementedError


class TokenizerLike(Protocol):
    def encode(self, text: str) -> Sequence[int]:
        raise NotImplementedError

    def token_to_id(self, token: str) -> int | None:
        raise NotImplementedError


def _get_tensor_ctor() -> Callable[[Sequence[float], Sequence[int]], TensorLike]:
    Tensor = getattr(te_mod, "Tensor", None)
    if not callable(Tensor):
        raise RuntimeError("tensor_engine.Tensor not available; build with python_bindings.")
    return cast(Callable[[Sequence[float], Sequence[int]], TensorLike], Tensor)


def te_tensor(data: Sequence[float], shape: Sequence[int]) -> TensorLike:
    ctor = _get_tensor_ctor()
    return ctor([float(x) for x in data], [int(x) for x in shape])


def te_embedding_lookup(emb: TensorLike, ids: TensorLike) -> TensorLike:
    Tensor = getattr(te_mod, "Tensor", None)
    fn = getattr(Tensor, "embedding_lookup", None)
    if not callable(fn):
        raise RuntimeError("tensor_engine.Tensor.embedding_lookup not available")
    return cast(TensorLike, fn(emb, ids))


def te_stack(tensors: Sequence[TensorLike], axis: int) -> TensorLike:
    Tensor = getattr(te_mod, "Tensor", None)
    fn = getattr(Tensor, "stack", None)
    if not callable(fn):
        raise RuntimeError("tensor_engine.Tensor.stack not available")
    return cast(TensorLike, fn(list(tensors), axis=axis))


log = logging.getLogger(__name__)


def _set_backend() -> None:
    fn = getattr(te_mod, "set_cpu_backend", None)
    if callable(fn):
        fn()


def _np_from_tensor(t: TensorLike) -> np.ndarray:
    data = np.asarray(t.get_data(), dtype=np.float32)
    return data.reshape(tuple(t.shape))


def _init_linear_small(linear: LinearWithWeightLike, std: float) -> None:
    w = linear.weight
    arr = np.random.randn(*w.shape).astype(np.float32) * std
    w.set_data(list(arr.ravel()))


def _init_linear_zeros(linear: LinearWithWeightLike) -> None:
    w = linear.weight
    arr = np.zeros(tuple(w.shape), dtype=np.float32)
    w.set_data(list(arr.ravel()))


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    d_ff: int
    num_heads: int
    depth: int
    max_len: int
    rope: bool = True
    llama_bias: bool = False


class TextCausalLM:
    """Same model as pretrain_project, with an exposed hidden-state forward."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        scale = 0.02
        self.tok_emb = te_tensor(
            (np.random.randn(cfg.vocab_size, cfg.d_model).astype(np.float32) * scale).ravel().tolist(),
            [cfg.vocab_size, cfg.d_model],
        )
        self.pos_emb = te_tensor(
            (np.random.randn(cfg.max_len, cfg.d_model).astype(np.float32) * scale).ravel().tolist(),
            [cfg.max_len, cfg.d_model],
        )

        TB = getattr(te_mod, "TransformerBlock", None)
        if not callable(TB):
            raise RuntimeError("tensor_engine.TransformerBlock not available; build with python_bindings.")
        self.blocks = [
            cast(
                TransformerBlockLike,
                TB(
                cfg.d_model,
                cfg.d_ff,
                num_heads=cfg.num_heads,
                kv_heads=None,
                use_rope=cfg.rope,
                nl_oob_config=None,
                nl_oob_max_scale=None,
                llama_style=True,
                llama_bias=cfg.llama_bias,
                ),
            )
            for _ in range(cfg.depth)
        ]

        Linear = getattr(te_mod, "Linear", None)
        if not callable(Linear):
            raise RuntimeError("tensor_engine.Linear not available; build with python_bindings.")
        self.lm_head = cast(LinearLike, Linear(cfg.d_model, cfg.vocab_size, False))

    def named_parameters(self) -> List[Tuple[str, TensorLike]]:
        named: List[Tuple[str, TensorLike]] = [
            ("tok_emb", self.tok_emb),
            ("pos_emb", self.pos_emb),
        ]
        for i, b in enumerate(self.blocks):
            for (n, t) in b.named_parameters(f"blocks.{i}."):
                named.append((n, t))
        for (n, t) in self.lm_head.named_parameters("lm_head."):
            named.append((n, t))
        return named

    def forward_hidden(self, input_ids: TensorLike) -> TensorLike:
        bsz, seq = input_ids.shape
        if seq > self.cfg.max_len:
            raise ValueError(f"seq_len {seq} > max_len {self.cfg.max_len}")
        x = te_embedding_lookup(self.tok_emb, input_ids)
        pos_ids = te_tensor([float(i) for i in range(int(seq))], [int(seq)])
        pos = te_embedding_lookup(self.pos_emb, pos_ids)
        pos_b = te_stack([pos] * int(bsz), axis=0)
        x = x + pos_b
        for blk in self.blocks:
            x = blk.forward(x)
        return x

    @staticmethod
    def load_npz(path: Path) -> "TextCausalLM":
        cfg_path = path.with_suffix(path.suffix + ".config.json")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config sidecar: {cfg_path}")
        cfg = ModelConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
        model = TextCausalLM(cfg)
        ckpt = np.load(path, allow_pickle=False)
        expected = [n for (n, _) in model.named_parameters()]
        missing = [n for n in expected if n not in ckpt]
        extra = [n for n in ckpt.files if n not in expected]
        if missing:
            raise ValueError(f"Checkpoint missing keys ({len(missing)}): {missing}")
        if extra:
            raise ValueError(f"Checkpoint has unexpected keys ({len(extra)}): {extra}")
        for name, t in model.named_parameters():
            arr = np.asarray(ckpt[name], dtype=np.float32)
            if list(arr.shape) != list(t.shape):
                raise ValueError(f"Shape mismatch for {name}: ckpt {arr.shape} vs tensor {t.shape}")
            t.set_data(list(arr.ravel()))
        return model


@dataclass
class LoRAConfig:
    r: int
    alpha: float


class LoRAHead:
    def __init__(self, d_model: int, vocab_size: int, cfg: LoRAConfig):
        Linear = getattr(te_mod, "Linear", None)
        if not callable(Linear):
            raise RuntimeError("tensor_engine.Linear not available; build with python_bindings.")
        self.cfg = cfg
        self.a = cast(LinearWithWeightLike, Linear(d_model, cfg.r, False))
        self.b = cast(LinearWithWeightLike, Linear(cfg.r, vocab_size, False))

        # LoRA practice: A small init, B zeros so the adapter starts as a no-op.
        _init_linear_small(self.a, std=0.01)
        _init_linear_zeros(self.b)

    def parameters(self) -> List[TensorLike]:
        return list(self.a.parameters()) + list(self.b.parameters())

    def named_parameters(self) -> List[Tuple[str, TensorLike]]:
        named: List[Tuple[str, TensorLike]] = []
        for (n, t) in self.a.named_parameters("lora.a."):
            named.append((n, t))
        for (n, t) in self.b.named_parameters("lora.b."):
            named.append((n, t))
        return named

    def delta_logits(self, hidden_flat: TensorLike) -> TensorLike:
        # hidden_flat: [N, D]
        d1 = self.a.forward(hidden_flat)  # [N, r]
        d2 = self.b.forward(d1)           # [N, vocab]
        scale = float(self.cfg.alpha) / float(self.cfg.r)
        scale_t = te_tensor([scale], [1])
        return d2 * scale_t

    def save_npz(self, path: Path, base_ckpt: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for name, t in self.named_parameters():
            arrays[name] = _np_from_tensor(t)
        np.savez(path, **arrays)
        meta = {
            "base_checkpoint": base_ckpt,
            **asdict(self.cfg),
        }
        path.with_suffix(path.suffix + ".config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_tokenizer(tokenizer_json: str) -> TokenizerLike:
    TokClass = getattr(te_mod, "Tokenizer", None)
    if not callable(TokClass):
        raise RuntimeError("tensor_engine.Tokenizer not available; rebuild with python_bindings,with_tokenizers")
    p = Path(tokenizer_json)
    if not p.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_json}")
    from_file = getattr(TokClass, "from_file", None)
    if not callable(from_file):
        raise RuntimeError("Tokenizer wrapper is present but missing from_file(); rebuild tensor_engine.")
    return cast(TokenizerLike, from_file(str(p)))


def read_lines(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"No non-empty lines in dataset: {path}")
    return lines


def build_stream(lines: List[str], tok: TokenizerLike) -> List[int]:
    ids: List[int] = []
    sep = int(tok.token_to_id("[SEP]") or 0)
    for ln in lines:
        ids.extend(int(x) for x in tok.encode(ln))
        ids.append(sep)
    if len(ids) < 64:
        raise ValueError("Dataset too small after tokenization.")
    return ids


def sample_batch(stream: List[int], batch: int, seq_len: int) -> Tuple[TensorLike, TensorLike]:
    x_all: List[float] = []
    y_all: List[float] = []
    max_off = len(stream) - (seq_len + 1)
    if max_off <= 0:
        raise ValueError("Token stream shorter than seq_len+1.")
    for _ in range(batch):
        off = random.randint(0, max_off)
        chunk = stream[off : off + seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        x_all.extend(float(t) for t in x)
        y_all.extend(float(t) for t in y)
    return te_tensor(x_all, [batch, seq_len]), te_tensor(y_all, [batch, seq_len])


def freeze_params(params: Sequence[TensorLike]) -> None:
    for p in params:
        # Tensor objects expose requires_grad setter
        try:
            setattr(p, "requires_grad", False)
        except (AttributeError, TypeError, ValueError):
            # If p is not a Tensor (e.g., module wrapper), ignore.
            pass


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _set_backend()

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base checkpoint path (*.npz) from pretrain_project")
    ap.add_argument("--text", required=True, help="Fine-tuning text file (one example per line)")
    ap.add_argument("--tokenizer-json", required=True, help="Path to tokenizer.json")
    ap.add_argument("--save", required=True, help="Output adapter path (*.npz)")

    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)

    args = ap.parse_args()

    base_path = Path(args.base)
    if base_path.suffix.lower() != ".npz":
        raise ValueError("--base must be a .npz checkpoint")

    save_path = Path(args.save)
    if save_path.suffix.lower() != ".npz":
        raise ValueError("--save must end with .npz")

    tok = load_tokenizer(args.tokenizer_json)
    lines = read_lines(args.text)
    stream = build_stream(lines, tok)

    base = TextCausalLM.load_npz(base_path)
    # Freeze base tensors (including head)
    freeze_params([t for (_, t) in base.named_parameters()])

    lora = LoRAHead(base.cfg.d_model, base.cfg.vocab_size, LoRAConfig(r=int(args.r), alpha=float(args.alpha)))

    Adam = getattr(te_mod, "Adam", None)
    if not callable(Adam):
        raise RuntimeError("tensor_engine.Adam not available")
    opt = cast(OptimizerLike, Adam(float(args.lr), 0.9, 0.999, 1e-8))

    train_params: List[TensorLike] = lora.parameters()

    for step in range(int(args.steps)):
        opt.zero_grad(train_params)
        x, y = sample_batch(stream, int(args.batch), int(args.seq_len))
        hidden = base.forward_hidden(x)

        bsz, seq, d = hidden.shape
        hidden_flat = hidden.reshape([int(bsz) * int(seq), int(d)])

        base_logits = base.lm_head.forward(hidden_flat)
        delta = lora.delta_logits(hidden_flat)
        logits = (base_logits + delta).reshape([int(bsz), int(seq), base.cfg.vocab_size])

        logits2 = logits.reshape([int(bsz) * int(seq), base.cfg.vocab_size])
        y2 = y.reshape([int(bsz) * int(seq)])
        loss = logits2.softmax_cross_entropy_with_logits(y2)
        loss.backward()
        opt.step(train_params)

        if step % 10 == 0 or step == int(args.steps) - 1:
            loss_val = float(loss.get_data()[0]) if loss.get_data() else math.nan
            log.info("step=%d loss=%.6f", step, loss_val)

    lora.save_npz(save_path, base_ckpt=str(base_path))
    log.info("Saved adapter to %s", str(save_path))


if __name__ == "__main__":
    main()
