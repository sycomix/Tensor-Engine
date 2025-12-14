# Bench Descriptions — tensor_engine (Criterion benches)

This document explains the purpose and intent behind each Criterion bench in `benches/matmul_bench.rs`.

General Notes

- Group names: benches are grouped into `matmul`, `ops`, `nn`, `training`, `batched_block_quant`, and a
  `quantized_dequant_compare` sub-group. Group names show up in the Criterion output and help categorize microbenchmarks
  by domain.
- CI gating: The repository uses an environment variable `CI_BENCH` to shorten test duration in CI. Set `CI_BENCH` to
  any value to use shorter measurement times, smaller sample sizes, and shorter warmup times.
- Feature flags: Some benches are conditional on features such as `dtype_f16`, `dtype_bf16` and `openblas`. When running
  local benches add those feature flags with `--features "openblas,dtype_f16"` as needed.
- Running a specific bench group: `cargo bench matmul`, `cargo bench ops`, `cargo bench nn`, `cargo bench training`,
  `cargo bench batched_block_quant`; to run the whole bench suite, run `cargo bench`.

Bench group: "matmul"
Purpose: Measure core matrix multiply performance across several sizes, data types, and paths (float, quantized), plus
autograd (forward+backward) impact.

Key benches:

- `matmul_{}x{}` (sizes: 10, 50, 100, 200): Measure the raw float matmul throughput for small to medium square
  matrices — useful for micro-optimizing CPU matrix multiply or BLAS-backed matmul.
- `matmul_forward_backward_{}x{}`: Measures the cost of forward + backward (autograd) for matmul at the same sizes; this
  highlights autograd overhead and memory traffic of creating and backpropagating through computational graphs.
- `quantized_matmul_{}x{}`: Measures the runtime of `QuantizedMatMul` when the right-hand side is stored as int8 with
  scale. Use this to compare the quantized fast path vs float matmul and dequantization costs.
- `matmul_f16_{}x{}` / `matmul_bf16_{}x{}` (feature-gated): Test matmul when using the f16/bf16 simulated dtypes; useful
  to validate multi-precision path and any compile-time flags or vendor acceleration.
- `matmul_{}x{}x{}` (rectangular case 64x128*128x64): Measures non-square matmul behavior (memory access, cache
  effects).
- Large matmul sizes (512, 1024): Gate under `CI_BENCH` false; measure more realistic large-matrix performance and
  compare int8 quantization to dequantized float path (the bench includes `dequantized_matmul_{}x{}` to measure
  dequantization + float matmul). These are heavier tests; use them for local profiling or nightly runs.

Bench group: "quantized_dequant_compare"
Purpose: Direct microbenchmark comparing the quantized matmul runtime vs an explicit dequantize-then-multiply (float)
path.

Key benches:

- `quantized_matmul_int8_128x128`: quantized matmul with 128x128 size.
- `dequantized_matmul_128x128`: dequantize int8 storage to float and then perform row-major float matmul. Use this pair
  to compare the speedup (or slowdown) from keeping the right-hand side int8 vs dequantizing it to float.

Bench group: "ops"
Purpose: Microbench elemental operators and small utilities (e.g., stack/concat) for throughput-focused operations and
autograd overhead on small tensors.

Key benches:

- `add`, `mul`, `relu`, `sigmoid`, `tanh`, `log`, `pow`, `sum`, `mean`: Small-element ops that show per-element
  computation and memory throughput.
- `stack`, `concat`: Memory and copy performance; important for batching or preprocessing logic.
- `softmax_axis1`, `log_softmax_axis1`: Softmax and log-softmax performance for 2D inputs; important for classification
  and training workloads.
- `ternary_forward`, `ternary_forward_backward`: Ternary quantization forward pass & autograd loop (forward+backward) —
  helps validate quantization-friendly ops.
- `add_forward_backward`: Autograd overhead for a small element-wise op.

Bench group: "nn"
Purpose: High-level neural network building blocks, including linear layers, normalization, dropout, convolutional
layers (1D/2D/3D), pooling, attention layers and multi-head attention variants.

Key benches & rationale:

- `linear_10_5`, `linear_5_1`: Forward pass throughput for small dense layers — baseline for small networks.
- `linear_10_5_backward`: Backward pass overhead for linear layers (gradable gradients), important for training
  performance assessment.
- `layernorm_1x10`, and `layernorm_1x10_backward`: LayerNorm forward and backward cost which is often a small but
  important per-token cost in transformers.
- `dropout_training` vs `dropout_eval`: Compare runtime of Dropout in training and eval modes (eval should be identity
  path).
- `conv2d_3x64x64`, `conv2d_3x64x64_backward`: 2D convolution forward/backward — measure compute and memory effects for
  convolutional layers.
- `maxpool2d_3x64x64`: MaxPool forward throughput.
- `conv1d_3x128`, `conv3d_3x8x32x32`, `depthwise_separable_conv2d_3x64x64`, `convtranspose2d_3x64x64`,
  `avgpool2d_3x64x64`, `adaptive_avgpool2d_3x64x64`: Op-level benches for different convolution and pooling ops used in
  vision and other workloads.
- `absolute_positional_embedding_1x128x64`: Embedding forward cost, useful for transformer-like inputs.
- `mha_forward_64_128`, `mha_forward_alibi_64_128`: MultiHeadAttention forward, with and without ALiBi, to measure
  attentional compute; important for sequence model baselines.
- NL-OOB (mha forward with distance matrix) args through several sequence lengths and batch sizes: measure overhead of
  non-local out-of-bound biases (distance-based attention) and scaling of memory/computation as seq_len and batch size
  increase. Also includes backward variants to measure training costs.
- `mha_forward_backward_64_128`, `mha_forward_backward_alibi_64_128`: MHA backward cost that includes gradient
  propagation and KV state cost.

Bench group: "batched_block_quant"
Purpose: Batched matmul performance and block-wise quantization throughput.

Key benches:

- `batched_matmul_16_64_128_64`: Measures batched matmul throughput using an op-level BatchedMatMul. Useful for
  production-style throughput where multiple sequences/batches compute matmul in bulk.
- `block_quantized_matmul_512x512_block64`: (heavy, gated under CI_BENCH false) Splits the RHS into column blocks,
  quantizes each block and performs repeated quantized matmul segments — this emulates blockwise quantization used by
  many quantization schemes (e.g., AWQ/GPTQ) and measures block-chunked matmul throughput.

Bench group: "training"
Purpose: Measure training loop micro-benchmarks including forward, backward, optimizer steps, and data loading overhead.

Key benches:

- `training_step`: A small model forward pass, compute loss (MSE), — bench that measures forward-only training
  throughput.
- `training_step_backward`: Forward + backward pass — measures gradient computation overhead and shows end-to-end step
  costs without optimizer updates.
- Optimizer benches: `sgd_step`, `adam_step`: Bench the optimizer `step` + `zero_grad` for a small model. Particularly
  useful to compare optimizer overhead on parameter updates and memory usage patterns.
- DataLoader bench: `dataloader_shuffle`, `dataloader_next_batch` — measure shuffle cost and iteration cost across small
  datasets to ensure batching and shuffle logic don't become bottlenecks.

How to use these benches:

- ShortCI mode: `CI_BENCH=1 cargo bench matmul` runs shorter measurement times to make CI pass faster; useful when
  verifying code changes quickly.
- Local profile: `cargo bench` or `cargo bench matmul` without `CI_BENCH` runs longer measurements for a stable result —
  recommended for profiling and performance tuning.
- Test with features: include `--features "openblas multi_precision dtype_f16"` to run with vendor libs or simulated
  dtypes.

Interpretation tips:

- Compare `quantized_matmul_*` vs `dequantized_matmul_*` to see if keeping compressed int8 data yields runtime
  improvement on your target hardware. Also check the memory usage — dequantizing increases memory and can hurt cache
  locality.
- Matmul forward vs forward+backward shows autograd/graph overhead; for training systems, forward+backward cost is often
  far more relevant than pure forward.
- Pay attention to `mha_` group results and NL-OOB tests to understand how attention scales with sequence length and
  batch size — most transformer tuning happens here.
- Use `benches/` as a regression test: add new bench or compare commit results with prior runs to identify regressions
  or improvements.

Example run commands (Windows / Linux / macOS):

Quick (CI-like short measurements):

```powershell
# Run only the matmul benches quickly (CI_BENCH short run)
$env:CI_BENCH = 1
cargo bench matmul --features "openblas"  # or omit features as required
```

Full suite (local profiling):

```bash
# Run full criteria benches locally (longer execution time, better statistics)
cargo bench --features "openblas multi_precision dtype_f16"
```

Reproduce a specific bench (e.g., quantized vs dequantize comparison):

```bash
# Quantized vs dequantized microbench for 128x128
cargo bench quantized_dequant_compare
```

Notes for bench authors and contributors

- Consistent RNG seeds ensure repeatable inputs and stable comparisons across runs.
- Use `CI_BENCH` for CI-friendly runs to keep jobs fast; separate nightly/profile runs for full statistics.
- If testing vendor acceleration or dtype differences, gate them under feature flags and document results as part of the
  PR.
- Add microbench notes in `docs/bench_descriptions.md` whenever adding new benches so others can easily understand the
  rationale.

If you'd like, I can:

- Add a script that runs selected benches and logs results to `bench_reports/` (with timestamped CSVs), or
- Create a simple CI job that runs a subset of performance-critical benches nightly and stores the artifacts.

---
End of bench explanation document.
