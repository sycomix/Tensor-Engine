#!/usr/bin/env python3
"""Tiny LoopLM POC using Tensor-Engine LoopedTransformer + NL‑OOB (example)

This script trains a small LoopedTransformer on a synthetic regression task to
exercise the loop + gate mechanism. It's a quick smoke/demo and not paper-scale.
"""
from __future__ import annotations

import logging
import numpy as np

try:
    import tensor_engine as te  # type: ignore
except ImportError:  # pragma: no cover
    te = None  # type: ignore


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    if te is None:
        raise RuntimeError("tensor_engine Python package not found. Build with 'maturin develop --release'.")

    batch = 4
    seq = 6
    d_model = 16
    d_ff = 32
    num_heads = 2
    t_max = 4
    beta = 0.05

    rng = np.random.default_rng(0)
    x = te.Tensor(rng.normal(size=(batch * seq * d_model)).astype(np.float32).tolist(), [batch, seq, d_model])
    # target is a small random tensor (regression/MSE target)
    y = te.Tensor(rng.normal(size=(batch * seq * d_model)).astype(np.float32).tolist(), [batch, seq, d_model])

    # distance matrix (abs token distance)
    dist = np.abs(np.subtract.outer(np.arange(seq), np.arange(seq))).astype(np.float32)
    dist_t = te.Tensor(dist.flatten().tolist(), [seq, seq])

    lt = te.LoopedTransformer(
        d_model,
        d_ff,
        num_heads,
        "logarithmic",
        2.0,
        t_max,
        beta,
    )

    opt = te.Adam(1e-3, 0.9, 0.999, 1e-8)
    loss_fn = te.MSELoss()

    for step in range(20):
        opt.zero_grad(lt.parameters())
        per_step_outs, p_phi = lt.forward_looped(x, dist_t)
        # per-step scalar losses (MSE averaged across batch)
        per_step_losses = [loss_fn.forward(o, y) for o in per_step_outs]
        # stack losses into shape [1, t_max]
        losses_vec = te.Tensor.stack(per_step_losses, 0).reshape([1, t_max])

        # compute mean p_phi across batch for each t: p_phi.shape = [B, t_max]
        # -> permute to [t_max, B] and matmul with ones[B,1] to sum over batch, then divide
        ones = te.Tensor([1.0] * batch, [batch, 1])
        p_phi_mean = p_phi.permute([1, 0]).matmul(ones).reshape([1, t_max]).mul_scalar(1.0 / float(batch))

        # expected loss = sum_t mean(p_phi[:,t]) * L_t
        expected = p_phi_mean.mul(losses_vec).sum()
        # entropy regularizer: -beta * mean_i H(p_phi_i)
        entropy = p_phi.mul(p_phi.log()).sum().mul_scalar(-1.0 / float(batch))
        loss = expected.add(entropy.mul_scalar(beta * -1.0 + 0.0))

        loss.backward()
        opt.step(lt.parameters())

        if (step + 1) % 5 == 0:
            logger.info("step=%d loss=%.6f expected=%.6f entropy=%.6f", step + 1, loss.get_data()[0], expected.get_data()[0], entropy.get_data()[0])
            logger.info("p_phi mean per-step: %s", p_phi_mean.get_data())

    logger.info("Done - quick LoopLM POC")


if __name__ == "__main__":
    main()
