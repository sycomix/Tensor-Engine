#!/usr/bin/env python3
"""Fine-tune an existing Tensor Engine text checkpoint.

This script loads a base `.npz` checkpoint created by `examples/pretrain_project` and continues
training on a new dataset.

It uses the same model architecture as pretrain_project.
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
log = logging.getLogger(__name__)


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


def _set_backend() -> None:
    fn = getattr(te_mod, "set_cpu_backend", None)
    if callable(fn):
        fn()


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
        # initialized; will be overwritten on load
        self.tok_emb = te_tensor([0.0] * (cfg.vocab_size * cfg.d_model), [cfg.vocab_size, cfg.d_model])
        self.pos_emb = te_tensor([0.0] * (cfg.max_len * cfg.d_model), [cfg.max_len, cfg.d_model])

        TB = getattr(te_mod, "TransformerBlock", None)
        if not callable(TB):
            raise RuntimeError("tensor_engine.TransformerBlock not available")
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
            raise RuntimeError("tensor_engine.Linear not available")
        self.lm_head = cast(LinearLike, Linear(cfg.d_model, cfg.vocab_size, False))

    def named_parameters(self) -> List[Tuple[str, TensorLike]]:
        named: List[Tuple[str, TensorLike]] = [("tok_emb", self.tok_emb), ("pos_emb", self.pos_emb)]
        for i, b in enumerate(self.blocks):
            for (n, t) in b.named_parameters(f"blocks.{i}."):
                named.append((n, t))
        for (n, t) in self.lm_head.named_parameters("lm_head."):
            named.append((n, t))
        return named

    def parameters(self) -> List[TensorLike]:
        return [t for (_, t) in self.named_parameters()]

    def forward(self, input_ids: TensorLike) -> TensorLike:
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
        x2 = x.reshape([int(bsz) * int(seq), self.cfg.d_model])
        logits = self.lm_head.forward(x2).reshape([int(bsz), int(seq), self.cfg.vocab_size])
        return logits

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for name, t in self.named_parameters():
            arrays[name] = _np_from_tensor(t)
        np.savez(path, **arrays)
        path.with_suffix(path.suffix + ".config.json").write_text(json.dumps(asdict(self.cfg), indent=2),
                                                                  encoding="utf-8")

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
        chunk = stream[off: off + seq_len + 1]
        x_all.extend(float(t) for t in chunk[:-1])
        y_all.extend(float(t) for t in chunk[1:])
    return te_tensor(x_all, [batch, seq_len]), te_tensor(y_all, [batch, seq_len])


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _set_backend()

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base checkpoint (*.npz) from pretrain_project")
    ap.add_argument("--text", required=True, help="Fine-tuning text file")
    ap.add_argument("--tokenizer-json", required=True, help="Path to tokenizer.json")
    ap.add_argument("--save", required=True, help="Output checkpoint (*.npz)")

    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)

    args = ap.parse_args()

    base_path = Path(args.base)
    if base_path.suffix.lower() != ".npz":
        raise ValueError("--base must end with .npz")

    save_path = Path(args.save)
    if save_path.suffix.lower() != ".npz":
        raise ValueError("--save must end with .npz")

    tok = load_tokenizer(args.tokenizer_json)
    lines = read_lines(args.text)
    stream = build_stream(lines, tok)

    model = TextCausalLM.load_npz(base_path)

    Adam = getattr(te_mod, "Adam", None)
    if not callable(Adam):
        raise RuntimeError("tensor_engine.Adam not available")
    opt = cast(OptimizerLike, Adam(float(args.lr), 0.9, 0.999, 1e-8))

    params: List[TensorLike] = model.parameters()

    for step in range(int(args.steps)):
        opt.zero_grad(params)
        x, y = sample_batch(stream, int(args.batch), int(args.seq_len))
        logits = model.forward(x)
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
    log.info("Saved fine-tuned checkpoint to %s", str(save_path))


if __name__ == "__main__":
    main()
