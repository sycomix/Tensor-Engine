# RVQ EMA Updates & Scheduling

This document outlines recommended settings and behaviors for the RVQ `update_ema()` method.

Key behaviors:

- `RVQ::update_ema(inputs, indices, decay)` computes per-level residual statistics and updates codebooks using an
  unbiased EMA using per-code `ema_counts`.
- `set_ema_update_every(n)` will only perform the update every `n` calls to reduce CPU/GPU overhead. Default is `1` (
  every call).
- `set_reinit_empty_codes(true)` will reinitialize empty codes (those with zero recent counts) from random residual
  vectors to avoid dead codes.

Recommended values for training:

- `decay = 0.999` (slow moving average — common schedule used in vector quantization like VQ-VAE and RVQ).
- `ema_update_every = 1` for small batch sizes, `ema_update_every = 32` for very large batch training to amortize cost.
- `reinit_empty_codes = true` when training from scratch to avoid dead codes; once training stabilizes, you can turn it
  off.

Scheduling examples:

- Update per batch (default): `rvq.set_ema_update_every(1); rvq.update_ema(&inputs, &indices, 0.999);`
- Update every 16 batches to reduce overhead: `rvq.set_ema_update_every(16);` — call `update_ema()` every training step
  as usual; internal scheduling will only run on the correct step.

Notes:

- For distributed training, ensure `ema_counts` and codebooks are synchronized across replicas or perform server-side
  updates after accumulation.
- For empty cluster reinitialization, the code picks a random residual to reseed the codebook entry and sets an initial
  `ema_count` small positive value so it will participate later.

See `src/nn/quantization.rs` for implementation details.
