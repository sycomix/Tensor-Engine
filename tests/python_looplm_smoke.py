"""Smoke test for LoopedTransformer Python wrapper."""
import numpy as np

import tensor_engine as te  # type: ignore


def test_looped_smoke():
    batch = 2
    seq = 4
    d_model = 8
    d_ff = 16
    num_heads = 2
    t_max = 3
    beta = 0.01

    rng = np.random.default_rng(1)
    x = te.Tensor(rng.normal(size=(batch * seq * d_model)).astype(np.float32).tolist(), [batch, seq, d_model])
    y = te.Tensor(rng.normal(size=(batch * seq * d_model)).astype(np.float32).tolist(), [batch, seq, d_model])

    lt = te.LoopedTransformer(d_model, d_ff, num_heads, None, None, t_max, beta)
    opt = te.Adam(1e-3, 0.9, 0.999, 1e-8)
    loss_fn = te.MSELoss()

    per_step_outs, p_phi = lt.forward_looped(x, None)
    assert len(per_step_outs) == t_max
    assert p_phi.shape()[0] == batch and p_phi.shape()[1] == t_max

    # quick training step: scalar per-step losses
    per_step_losses = [loss_fn.forward(o, y) for o in per_step_outs]
    losses_vec = te.Tensor.stack(per_step_losses, 0).reshape([1, t_max])

    # use the new Stage-II helper (compute expected cumulative + entropy reg)
    stage2 = lt.stage2_loss(p_phi, losses_vec)

    opt.zero_grad(lt.parameters())
    stage2.backward()
    opt.step(lt.parameters())

    # ensure gate received gradient via p_phi (approx check)
    g = p_phi.get_grad()
    assert g is not None
    assert any(abs(v) > 0.0 for v in g)
