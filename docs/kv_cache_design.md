# KV Cache Design (short)

Status: Draft (Dec 2025)

## Goals
- Provide per-layer KV cache support for efficient incremental decoding.
- Avoid unnecessary copies during append by leveraging op-level concat (`KVCacheAppend`) when possible.
- Provide simple, well-tested API and safe fallbacks to full computation when caches are absent.

## Data layout
- Packed storage shape (recommended canonical): (batch, seq, dim)
  - For keys/values store two tensors: `packed_keys` and `packed_values`.
  - Per-layer `KVCache` will hold both packed handles.
- `dim` equals `kv_heads * head_dim` (or `num_heads * head_dim` after expansion).

## API sketch

Rust types & methods (minimal):

- struct KVCache {
    packed_keys: Option<Tensor>, // shape (b, seq, dim)
    packed_values: Option<Tensor>,
}

- impl KVCache {
    pub fn new() -> Self;
    pub fn append_packed(&mut self, new_keys: &Tensor, new_values: &Tensor) -> Result<(), String>; // uses op-level append when possible
    pub fn set_packed(&mut self, keys: Tensor, values: Tensor);
    pub fn packed_keys(&self) -> Option<Tensor>;
    pub fn packed_values(&self) -> Option<Tensor>;
    pub fn clear(&mut self);
}

- TransformerBlock changes:
    - Add `kv_cache: Option<KVCache>` field
    - Provide accessors: `get_kv_cache_mut(&mut self) -> &mut Option<KVCache>`
    - Update forward signature for incremental decode: accept `kv_cache: Option<&mut KVCache>` and prefer cached packed keys/values when present.

- Tensor helper
    - `Tensor::kvcache_append(cache: &Tensor, new_kv: &Tensor, axis: usize) -> Tensor` (thin wrapper over `KVCacheAppend` op)

## Integration points
- Prefill flow: when performing encoder/prefill, build per-layer packed k/v and set `kv_cache`.
- Step decode flow: for each step:
  - For each layer, compute new_k/new_v for this step.
  - Call `kv_cache.append_packed(&new_k, &new_v)` or use `Tensor::kvcache_append` to avoid data copies.
  - Use packed k/v to compute attention (supports q_len <= kv_len; ALiBi & causal mask generalized to (q_len, kv_len)).

## Tests & Acceptance
- Unit tests:
  - KVCache `append_packed` correctness and failure on mismatched shapes.
  - `Tensor::kvcache_append` op correctness & gradient checks (existing `new_ops_test.rs`).
- Integration tests:
  - Incremental decode parity: step-by-step decode using per-layer caches should match full decode over same input (numeric tolerance 1e-5).
  - One-shot generation smoke test: load tiny SafeTensors fixture and call generator for one token (CI job, <2 min).
- Bench:
  - Microbench per-token latency comparison: concatenation via ndarray vs op-level concat.

## Risks & mitigations
- Memory copies: initial implementation may copy on append; op-level append reduces copies.
- Lock ordering: prefer operating on `Tensor` op-level functions to reduce mutex churn.
- Shape mismatches: include strict checks and error messages to make debugging easier.

## Timeline
- Draft design doc & TODOs (this doc) — now (1–2 hours)
- Implement per-layer KV caches + tests (1–2 days)
- Replace concat with op-level append (small follow-up, 1 day)
- Generator integration + tests (1–2 days)

---

If this looks good I’ll begin implementing items (3) and (4) in the TODOs next (per your approval).