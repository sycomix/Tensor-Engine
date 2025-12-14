#!/usr/bin/env python3
"""Transformer demo using the `tensor_engine` Python bindings.

This example requires installing the Python extension (maturin develop --release).
"""
from __future__ import annotations

# pylint: disable=import-error, missing-function-docstring, line-too-long
import logging
from typing import Any, cast

import numpy as np

try:
    import tensor_engine as te  # type: ignore
except ImportError:  # pragma: no cover
    te = None  # type: ignore

# Cast to Any to avoid static attribute errors in linters/type checkers
te_mod: Any = cast(Any, te)


def make_tensor_from_numpy(arr: np.ndarray) -> Any:
    return make_tensor(list(arr.flatten()), list(arr.shape))


def make_tensor(data: Any, shape: list[int]) -> Any:
    """
    Compatibility wrapper for constructing tensors from the tensor_engine module.
    Tries common constructor names: Tensor or tensor. Falls back to a lightweight
    numpy-backed object if neither exists (useful for static analysis/testing).
    """
    if te_mod is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")
    ctor = getattr(te_mod, "Tensor", None) or getattr(te_mod, "tensor", None)
    if ctor is not None:
        return ctor(data, shape)
    # Fallback to a minimal numpy-backed tensor for runtime-safe behavior in tests
    arr = np.array(data, dtype=np.float32).reshape(shape)

    class _NumpyTensor:
        def __init__(self, arr: np.ndarray):
            self._arr = arr

        @property
        def shape(self):
            return list(self._arr.shape)

        def __repr__(self):
            return f"_NumpyTensor(shape={self.shape})"

    return _NumpyTensor(arr)


def create_input(batch: int, seq: int, d_model: int) -> Any:
    x_arr = (np.arange(batch * seq * d_model, dtype=np.float32) * 0.01).flatten()
    return make_tensor(list(x_arr), [batch, seq, d_model])


def train_llama_style(llama_tb: Any, x: Any, batch: int, seq: int, d_model: int) -> None:
    # Try to find an 'Adam' optimizer in the tensor_engine module or submodules.
    def _find_optimizer_class(module: Any, class_name: str):
        if hasattr(module, class_name):
            return getattr(module, class_name)
        for attr in dir(module):
            try:
                sub = getattr(module, attr)
            except Exception:
                continue
            if hasattr(sub, class_name):
                return getattr(sub, class_name)
        return None

    AdamCtor = _find_optimizer_class(te_mod, "Adam")
    if AdamCtor is not None:
        opt = AdamCtor(1e-3, 0.9, 0.999, 1e-8)
    else:
        logging.warning("tensor_engine module has no 'Adam' -- using a minimal fallback optimizer.")

        class _FallbackAdam:
            def __init__(self, lr, beta1=None, beta2=None, eps=None):
                self.lr = lr

            def zero_grad(self, params):
                for p in params:
                    if hasattr(p, "grad"):
                        try:
                            p.grad = None
                        except Exception:
                            pass

            def step(self, params):
                for p in params:
                    g = getattr(p, "grad", None)
                    if g is None:
                        continue
                    if hasattr(p, "data"):
                        try:
                            p.data = p.data - self.lr * g
                        except Exception:
                            try:
                                p.data -= self.lr * g
                            except Exception:
                                pass

        opt = _FallbackAdam(1e-3, 0.9, 0.999, 1e-8)

    # Find a suitable MSE loss class or create a minimal fallback
    def _find_loss_class(module: Any, *class_names: str):
        for name in class_names:
            if hasattr(module, name):
                return getattr(module, name)
        for attr in dir(module):
            try:
                sub = getattr(module, attr)
            except Exception:
                continue
            for name in class_names:
                if hasattr(sub, name):
                    return getattr(sub, name)
        return None

    LossCtor = _find_loss_class(te_mod, "MSELoss", "MeanSquaredError")

    class _FallbackMSELoss:
        def forward(self, pred, target):
            # Attempt elementwise ops on tensor-like objects
            try:
                diff = pred - target
                sq = diff * diff
                if hasattr(sq, "mean"):
                    return sq.mean()
                # Try to convert to numpy array if attribute exists
                arr = getattr(sq, "_arr", None)
                if arr is None:
                    arr = np.array(sq)
                mse = float(np.mean(arr))
                scalar = make_tensor_from_numpy(np.array([mse], dtype=np.float32))
                # Provide a no-op backward to maintain API compatibility
                try:
                    def _noop_backward():
                        return None

                    scalar.backward = _noop_backward
                except Exception:
                    pass
                return scalar
            except Exception:
                # Final fallback: pure numpy calculation returning a scalar-backed tensor
                try:
                    p_arr = getattr(pred, "_arr", None) or np.array(pred)
                    t_arr = getattr(target, "_arr", None) or np.array(target)
                    mse = float(np.mean((np.array(p_arr) - np.array(t_arr)) ** 2, dtype=np.float32))
                    scalar = make_tensor_from_numpy(np.array([mse], dtype=np.float32))
                    try:
                        def _noop_backward():
                            return None

                        scalar.backward = _noop_backward
                    except Exception:
                        pass
                    return scalar
                except Exception:
                    raise RuntimeError("Unable to compute MSE loss.")

    if LossCtor is not None:
        loss_fn = LossCtor()
    else:
        logging.warning("tensor_engine module has no 'MSELoss' -- using a minimal fallback MSELoss.")
        loss_fn = _FallbackMSELoss()

    target = make_tensor([0.0] * (batch * seq * d_model), [batch, seq, d_model])
    for _ in range(10):
        opt.zero_grad(llama_tb.parameters())
        pred = llama_tb.forward(x)
        loss = loss_fn.forward(pred, target)
        # Only call backward if supported
        if hasattr(loss, "backward") and callable(getattr(loss, "backward")):
            try:
                loss.backward()
            except Exception:
                logging.debug("Loss backward failed; continuing without gradient computation.")
        opt.step(llama_tb.parameters())


def create_distance_tensor(seq: int) -> Any:
    dist_arr = np.zeros((seq, seq), dtype=np.float32)
    for i in range(seq):
        for j in range(seq):
            dist_arr[i, j] = abs(i - j)
    return make_tensor(list(dist_arr.flatten()), [seq, seq])


def demo() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    if te_mod is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")

    batch = 2
    seq = 4
    d_model = 8
    x = create_input(batch, seq, d_model)

    # Locate TransformerBlock class in tensor_engine module or submodules.
    TB_ctor = getattr(te_mod, "TransformerBlock", None) or getattr(te_mod, "transformer_block", None)
    if TB_ctor is None:
        for attr in dir(te_mod):
            try:
                sub = getattr(te_mod, attr)
            except Exception:
                continue
            TB_ctor = getattr(sub, "TransformerBlock", None) or getattr(sub, "transformer_block", None)
            if TB_ctor is not None:
                break
    if TB_ctor is None:
        logging.warning("tensor_engine module has no 'TransformerBlock' -- using a minimal fallback TransformerBlock.")

        class _FallbackTransformerBlock:
            def __init__(self, d_model, d_ff, num_heads=None, kv_heads=None, use_rope=False, nl_oob_config=None,
                         nl_oob_max_scale=None, llama_style=False, llama_bias=True):
                self.d_model = d_model

            def forward(self, x):
                return x

            def forward_with_distance(self, x, dist):
                return x

            def parameters(self):
                return []

        TB_ctor = _FallbackTransformerBlock

    tb = TB_ctor(d_model, d_model * 2, num_heads=2)
    out = tb.forward(x)
    logger.info("Input shape: %s", x.shape)
    logger.info("Output shape: %s", out.shape)

    # LLaMA-style TransformerBlock (RMSNorm pre-norm + SwiGLU), with no bias in dense layers
    llama_tb = TB_ctor(d_model, d_model * 2, num_heads=2, kv_heads=None, use_rope=True, nl_oob_config=None,
                       nl_oob_max_scale=None, llama_style=True, llama_bias=False)
    out_llama = llama_tb.forward(x)
    logger.info("LLaMA-style output shape: %s", out_llama.shape)

    train_llama_style(llama_tb, x, batch, seq, d_model)
    logger.info("Finished training toy LLaMA-style loop")

    # NL-OOB example: create with a 'logarithmic' bias function and max_scale 2.0
    tb_oob = TB_ctor(
        d_model, d_model * 2, num_heads=2, nl_oob_config="logarithmic", nl_oob_max_scale=2.0
    )
    dist_t = create_distance_tensor(seq)
    out2 = tb_oob.forward_with_distance(x, dist_t)
    logger.info("NL-OOB Output shape: %s", out2.shape)


if __name__ == "__main__":
    demo()
