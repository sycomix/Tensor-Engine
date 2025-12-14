#!/usr/bin/env python3
"""Pretrain a tiny causal language model from scratch using Tensor Engine.

This script intentionally keeps the model small and CPU-friendly.
It demonstrates:

- HF tokenizer loading via `tensor_engine.Tokenizer` (requires `with_tokenizers` feature)
- token/position embedding via `Tensor.embedding_lookup`
- causal Transformer blocks via `TransformerBlock` (with `llama_style=True` and `use_rope=True`)
- checkpoint save/load using `.npz` (no external model framework)

Checkpoint format:
- `<save>.npz`: tensors by name
- `<save>.config.json`: model hyperparameters
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

    def vocab_size(self) -> int:
        raise NotImplementedError


log = logging.getLogger(__name__)


def _set_backend() -> None:
    # Be defensive: not all builds expose set_cpu_backend.
    fn = getattr(te_mod, "set_cpu_backend", None)
    if callable(fn):
        fn()


def _get_tensor_ctor() -> Callable[[Sequence[float], Sequence[int]], TensorLike]:
    Tensor = getattr(te_mod, "Tensor", None)
    if not callable(Tensor):
        raise RuntimeError("tensor_engine.Tensor not available; build with python_bindings.")
    return cast(Callable[[Sequence[float], Sequence[int]], TensorLike], Tensor)


def te_tensor(data: Sequence[float], shape: Sequence[int]) -> TensorLike:
    ctor = _get_tensor_ctor()
    # Ensure plain Python types cross the FFI boundary.
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


def _tensor_from_np(arr: np.ndarray) -> TensorLike:
    return te_tensor(arr.astype(np.float32).ravel().tolist(), list(arr.shape))


def _np_from_tensor(t: TensorLike) -> np.ndarray:
    data = np.asarray(t.get_data(), dtype=np.float32)
    return data.reshape(tuple(t.shape))


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
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        # Trainable parameters
        scale = 0.02
        self.tok_emb = _tensor_from_np(np.random.randn(cfg.vocab_size, cfg.d_model).astype(np.float32) * scale)
        self.pos_emb = _tensor_from_np(np.random.randn(cfg.max_len, cfg.d_model).astype(np.float32) * scale)

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

    def parameters(self) -> List[TensorLike]:
        params: List[TensorLike] = [self.tok_emb, self.pos_emb]
        for i, b in enumerate(self.blocks):
            ps = list(b.parameters())
            if not ps:
                raise RuntimeError(f"TransformerBlock[{i}] returned no parameters; cannot train.")
            params.extend(ps)
        params.extend(self.lm_head.parameters())
        return params

    def named_parameters(self) -> List[Tuple[str, TensorLike]]:
        named: List[Tuple[str, TensorLike]] = [
            ("tok_emb", self.tok_emb),
            ("pos_emb", self.pos_emb),
        ]
        for i, b in enumerate(self.blocks):
            # Requires the Python binding method we expose in this repo.
            for (n, t) in b.named_parameters(f"blocks.{i}."):
                named.append((n, t))
        for (n, t) in self.lm_head.named_parameters("lm_head."):
            named.append((n, t))
        return named

    def forward(self, input_ids: TensorLike) -> TensorLike:
        """input_ids: Tensor [B, S] (float ids); returns logits [B, S, vocab]."""
        bsz, seq = input_ids.shape
        if seq > self.cfg.max_len:
            raise ValueError(f"seq_len {seq} > max_len {self.cfg.max_len}. Increase --max-len.")

        x = te_embedding_lookup(self.tok_emb, input_ids)

        pos_ids = te_tensor([float(i) for i in range(int(seq))], [int(seq)])
        pos = te_embedding_lookup(self.pos_emb, pos_ids)  # [S, D]
        pos_b = te_stack([pos] * int(bsz), axis=0)  # [B, S, D]
        x = x + pos_b

        for blk in self.blocks:
            x = blk.forward(x)

        x2 = x.reshape([int(bsz) * int(seq), self.cfg.d_model])
        logits = self.lm_head.forward(x2).reshape([int(bsz), int(seq), self.cfg.vocab_size])
        return logits

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for name, t in self.named_parameters():
            arrays[name] = _np_from_tensor(t)
        np.savez(path, **arrays)
        cfg_path = path.with_suffix(path.suffix + ".config.json")
        cfg_path.write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")

    @staticmethod
    def load_npz(path: Path) -> "TextCausalLM":
        cfg_path = path.with_suffix(path.suffix + ".config.json")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config sidecar: {cfg_path}")
        cfg = ModelConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
        model = TextCausalLM(cfg)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = np.load(path, allow_pickle=False)
        # Fill tensors by name; enforce exact key match to avoid silent corruption.
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


def _tokenizer_vocab_size(tok: object) -> int:
    # Some wrappers expose vocab_size() as a method; be defensive.
    vs = getattr(tok, "vocab_size", None)
    if callable(vs):
        res = vs()
        if isinstance(res, int):
            return res
        if isinstance(res, float):
            return int(res)
        return 0
    if isinstance(vs, int):
        return int(vs)
    return 0


def load_tokenizer(tokenizer_json: str) -> TokenizerLike:
    Tok = getattr(te_mod, "Tokenizer", None)
    if not callable(Tok):
        raise RuntimeError(
            "tensor_engine.Tokenizer not available. Rebuild with features: python_bindings,with_tokenizers"
        )
    p = Path(tokenizer_json)
    if not p.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_json}")
    from_file = getattr(Tok, "from_file", None)
    if not callable(from_file):
        raise RuntimeError("Tokenizer wrapper is present but missing from_file(); rebuild tensor_engine.")
    return cast(TokenizerLike, from_file(str(p)))


def read_corpus_lines(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text dataset not found: {path}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"No non-empty lines in dataset: {path}")
    return lines


def build_training_stream(lines: List[str], tok: TokenizerLike) -> List[int]:
    ids: List[int] = []
    for i, ln in enumerate(lines):
        try:
            toks = tok.encode(ln)
        except Exception as e:  # pragma: no cover
            log.exception("Tokenization failed at line %d", i)
            raise RuntimeError(f"Tokenization failed at line {i}") from e
        ids.extend(int(x) for x in toks)
        # simple separator
        ids.append(int(tok.token_to_id("[SEP]") or 0))
    if len(ids) < 64:
        raise ValueError("Dataset too small after tokenization; provide more text.")
    return ids


def sample_batch(stream: List[int], batch: int, seq_len: int) -> Tuple[TensorLike, TensorLike]:
    # We sample seq_len+1 tokens and create (input, target) shifted by 1.
    x_all: List[float] = []
    y_all: List[float] = []
    max_off = len(stream) - (seq_len + 1)
    if max_off <= 0:
        raise ValueError("Token stream shorter than seq_len+1.")

    for _ in range(batch):
        off = random.randint(0, max_off)
        chunk = stream[off: off + seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        x_all.extend(float(t) for t in x)
        y_all.extend(float(t) for t in y)

    x_t = te_tensor(x_all, [batch, seq_len])
    y_t = te_tensor(y_all, [batch, seq_len])
    return x_t, y_t


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _set_backend()

    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Path to UTF-8 text file (one example per line)")
    ap.add_argument("--tokenizer-json", required=True, help="Path to tokenizer.json")
    ap.add_argument("--save", required=True, help="Output checkpoint path (*.npz)")
    ap.add_argument("--resume", default=None, help="Resume from an existing checkpoint (*.npz)")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256)

    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_json)
    vocab_size = _tokenizer_vocab_size(tok)
    if vocab_size <= 0:
        raise RuntimeError("Tokenizer vocab size is invalid.")

    lines = read_corpus_lines(args.text)
    stream = build_training_stream(lines, tok)

    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        depth=args.depth,
        max_len=args.max_len,
        rope=True,
        llama_bias=False,
    )

    save_path = Path(args.save)
    if save_path.suffix.lower() != ".npz":
        raise ValueError("--save must end with .npz")

    if args.resume:
        model = TextCausalLM.load_npz(Path(args.resume))
        if asdict(model.cfg) != asdict(cfg):
            raise ValueError("Resume checkpoint config does not match current CLI config.")
        log.info("Resumed from %s", args.resume)
    else:
        model = TextCausalLM(cfg)

    Adam = getattr(te_mod, "Adam", None)
    if not callable(Adam):
        raise RuntimeError("tensor_engine.Adam not available; build with python_bindings.")
    opt = cast(OptimizerLike, Adam(float(args.lr), 0.9, 0.999, 1e-8))

    params = model.parameters()

    # training loop
    for step in range(int(args.steps)):
        opt.zero_grad(params)
        x, y = sample_batch(stream, int(args.batch), int(args.seq_len))
        logits = model.forward(x)
        # Flatten for CE: [B*S, V], targets [B*S]
        bsz, seq, vocab = logits.shape
        logits2 = logits.reshape([int(bsz) * int(seq), int(vocab)])
        y2 = y.reshape([int(bsz) * int(seq)])

        loss = logits2.softmax_cross_entropy_with_logits(y2)
        loss.backward()
        opt.step(params)

        if step % 10 == 0 or step == int(args.steps) - 1:
            loss_val = float(loss.get_data()[0]) if loss.get_data() else math.nan
            log.info("step=%d loss=%.6f", step, loss_val)

    model.save_npz(save_path)
    log.info("Saved checkpoint to %s", str(save_path))


if __name__ == "__main__":
    main()
