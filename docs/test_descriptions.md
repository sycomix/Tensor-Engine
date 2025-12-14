# Test Descriptions — tensor_engine

This document explains the purpose of unit and integration tests under `tests/` and the `src/nn/tests/` folder. Use this
as a quick reference to understand what each test validates and why it exists.

General notes

- Feature-gated tests: Many tests are conditionally compiled behind feature flags. See `#[cfg(feature = "openblas")]` (
  and similar) attributes in the files. Examples include `with_tch`, `safe_tensors`, and dtype feature flags like
  `dtype_f16`.
- TorchScript fixtures: Some TorchScript loader tests use checked-in `.pt` files in `tests/assets/`. Where fixtures are
  absent, tests try to decode a `.b64` file or generate fixtures using `python` and `torch` at runtime (skipping if
  Python+torch isn't available).
- Numeric/grad checks: Several tests implement finite difference numeric gradients to validate analytic autograd
  computations.
- CI behavior: CI can be configured via environment variables such as `CI_BENCH`/`CI_TEST` and feature flags recommended
  for faster CI runs. See `.github/workflows/ci.yml`.

Tests in `tests/`

- `tokenizer_test.rs`
    - test_tokenizer_load_and_encode: Verify that tokenizer loader fails gracefully when a tokenizer file does not
      exist. Ensures the wrapper handles file errors gracefully.
    - Feature: `with_tokenizers`.

- `state_dict_default.rs`
    - test_default_load_state_dict_applies_named_parameters: Verify that the module `Sequential` default state_dict
      loader applies parameters from a `HashMap<String, Tensor>` using `named_parameters` keys. Ensures loading and key
      binding works for common module naming patterns.

- `safetensors_loader_full.rs` (feature: `safe_tensors`, `multi_precision`)
    - test_safetensors_full_state_dict_load: Create a safetensors fixture and load it into `MultiHeadAttention` module;
      confirm weights of `linear_q` are updated correctly. Tests the safetensors loader end-to-end and multi-precision
      parsing.

- `safetensors_test.rs` (feature: `safe_tensors` and `multi_precision`)
    - test_safetensors_loader_invalid_bytes: Ensures invalid bytes cause an error during safetensor load.
    - test_safetensors_parse_f32_and_f16: Confirms safetensors parsing works for F32, F16, and BF16 encodings.

- `quantized_matmul_test.rs`
    - test_quantized_matmul_basic: Ensures the quantized matmul path runs and returns the expected shape for a simple
      input. This is a shape & basic execution check for the int8 path.

- `pytorch_loader_test.rs` (feature: `with_tch`)
    - missing_file_returns_error: Check loader returns error with helpful message when file is missing.
    - cmodule_extraction_via_python_jit: Tests dispatcher that extracts named_parameters & state_dict from a TorchScript
      CModule created via traced Python code, using baked fixture or generated file. Skips if Python+torch isn't
      available.
    - cmodule_nested_state_dict_extraction: Ensures loader can parse nested state_dict results from TorchScript (when
      state_dict returns a dict with nested keys).
    - cmodule_state_dict_list_pairs_extraction: Tests the loader handles `state_dict` returning a list of (name,tensor)
      pairs.
    - cmodule_state_dict_hashmap_extraction: Test the loader handles a `state_dict` that aliases keys (hashmap) and maps
      them to parameter names.

- `optimizer_test.rs`
    - test_linear_forward / test_sequential: Basic shape checks for `Linear` and `Sequential` modules.
    - test_sgd_step, test_sgd_momentum_behaviour: Validate step semantics of SGD with and without momentum and parameter
      updates.
    - test_zero_grad / test_optimizer_zero_grad: Ensure `zero_grad()` works on single `Tensor` and Optimizer-level
      zeroing.
    - test_sgd_cast_params_dtype_changes: Tests casting to F8 changes dtype and affects stored values.
    - test_adamw_weight_decay_reduces_param_more_than_adam: Confirms `AdamW` weight decay effect vs `Adam` when weight
      decay is non-zero.
    - test_adamw_matches_adam_with_zero_weight_decay: `AdamW` with zero weight decay matches Adam behavior.
    - test_linear_warmup_scheduler, test_cosine_annealing_scheduler, test_rmsprop_step: Basic checks for scheduler
      outputs and RMSProp step behavior.

- `nn_extra_test.rs`
    - test_rnncell_forward_backward: RNNCell forward/backward shapes & gradients.
    - test_lstmcell_forward_backward: LSTMCell forward/backward shapes & gradient checks.
    - test_attention_forward_shape_and_grad: Self-Attention forward/backward shapes & gradient propagation.
    - test_transformer_forward_shape: TransformerBlock shape check.
    - test_convblock_and_gan_forward: ConvBlock forward shape and simple generator/discriminator forward shape checks.

- `nl_oob_test.rs` (NL-OOB = Non-Local / Out-of-Bounds attention biasing)
    - test_nl_oob_forward_affects_logits: Verify using NL-OOB distance-based bias alters attention outputs when provided
      an appropriately configured `MultiHeadAttention` with slopes and bias function.

- `new_ops_test.rs`
    - test_rmsnorm_forward_backward_shapes: Ensure RMSNorm forward shape & backward gradient are computed.
    - test_swiglu_forward_backward_shapes: Ensure SwiGLU forward/backward correctness and shapes.
    - test_embedding_lookup_forward_backward: Embedding lookup forward shape and gradient.
    - test_kvcache_append: KV cache append op shape, result, and gradient flow.
    - test_numeric_gradient_*: Finite-difference numerical gradient checks for new ops (RMSNorm, SwiGLU) to detect
      autograd correctness.

- `dtype_tests.rs`
    - test_astype_f8_roundtrip_and_dtype: Verify dtype cast to F8 and shape preservation works and dtype is set.
    - test_astype_f16_roundtrip_and_loss: (feature gated `dtype_f16`) Tests F16 conversion shape and numerical loss vs
      F32.
    - test_astype_bf16_roundtrip_and_loss: (feature gated `dtype_bf16`) BF16 conversion and round-trip check.

- `blas_matmul_test.rs`
    - test_blas_matmul_matches_ndarray_for_various_shapes: Compare numeric matmul result of our BLAS-backed matmul
      against `ndarray` dot across shapes; verifies numerical parity across shapes.

- `autograd_test.rs` (large file)
    - A thorough set of tests for core autograd correctness, including:
        - Scalar arithmetic ops (add, mul, sub, div, pow, exp, tanh, sigmoid) — forward values and backward gradients.
        - Element reductions (sum, mean, max, min) — forward and backward semantics and tie behavior (splitting
          gradients across ties).
        - Matmul forward/backward gradients and shapes.
        - Activation functions: relu, gelu; shape tests and numerical grad checks
        - Ternary operator tests (ternary quant) — forward/grad behavior.
        - Broadcast behavior and gradient reduction across broadcasted dims for add & mul.
        - Numeric gradient tests for add, mul, broadcast add/mul, pow, sigmoid, gelu, exp — finite-difference gradient
          checks.
        - MSE/cross-entropy/NLL Loss gradient checks: validate loss gradients are computed as per analytical formulae.
        - DataLoader shuffle/next_batch semantics — random shuffle and batch iteration.
        - Dropout forward/backward in training and evaluation modes.
        - Convolution and pooling ops (conv2d, conv3d, conv1d, depthwise separable conv, conv transpose)
          forward/backward correctness and gradient shapes.
        - MaxPool/AvgPool forward/backward checks with expected gradient values in pooled windows.
        - LayerNorm forward and backward numeric checks.

- `attention_variants_test.rs`
    - test_flash_ref_and_chunked_match_baseline: Validate that FlashRef and Chunked attention produce numerically
      similar outputs to baseline (within tolerance), ensuring alternate kernels are functionally equivalent.

- `attention_grad_parity_test.rs`
    - test_attention_grad_parity_flashref and test_attention_grad_parity_chunked: Compare gradient parity between the
      baseline attention implementation and either FlashRef or Chunked variants, ensuring their gradients (backprop) are
      numerically close to the baseline.

Tests in `src/nn/tests/` (module-level tests)

- `transformer_tests.rs` (transformer & NL-OOB tests)
    - transformer_block_forward_shape: Basic shape check for `TransformerBlock`.
    - mha_forward_with_distance_applies_penalty: Ensure the NL-OOB `forward_with_distance` produces different outputs
      than base attention.
    - mha_slopes_are_learnable_and_receive_grad: Ensure slopes parameter exists and receives gradients.

- `transformer_rope_gqa_tests.rs`
    - transformer_block_rope_and_gqa_shapes: Validate TransformerBlock with RoPE and Grouped Query Attention (GQA)
      combinations preserve input-output shapes.

- `transformer_nl_oob_tests.rs`
    - A collection of NL-OOB focused tests:
        - mha_forward_with_distance_shapes_and_slopes_present (detect slopes and parameter names)
        - mha_forward_with_distance_batch_and_gaussian (batchwise distances)
        - mha_forward_with_distance_mismatched_batch_returns_input (test mismatch behavior where distance batch shape
          mismatched -> identity return)
        - transformer_block_forward_with_distance_integrates_nl_oob (ensure TransformerBlock integrates NL-OOB MHA
          correctly)
        - transformer_block_builder_with_nl_oob_works (module builder creates named slopes param)
        - load_state_dict_sets_nl_oob_config_from_state (loader sets slope config and slopes and requires grad
          correctly)
        - slopes_receive_grad_on_backward (slopes receive grad backprop)
        - mha_forward_with_distance_2d_phi_broadcasts_to_batch (2D distance broadcast semantics)

- `flatten_tests.rs` (small module tests)
    - flatten_works_on_4d: ensure `Flatten` reduces N-D to 2-D for conv outputs.
    - flatten_integrates_with_sequential: Validate `Flatten` integrates with `Sequential` and subsequent layers.

How to run the test suite

- Basic unit tests (default features):

```bash
cargo test
```

- Run a specific test file or test name (useful for debugging):

```bash
# Run the test file with Rust's `--test` harness name
cargo test --test quantized_matmul_test
# or a specific test name
cargo test test_quantized_matmul_basic
```

- Run tests that require optional features (e.g., `with_tch`, `safe_tensors`):

```bash
cargo test --features "with_tch"  # for TorchScript loader tests
cargo test --features "safe_tensors multi_precision"  # safe tensor parsing tests
```

- Notes for developers
    - RNG seeds are fixed in tests to ensure deterministic, reproducible results across runs.
    - Some tests are intentionally skipped if a feature or runtime dependency (Python/Torch) isn't present. This keeps
      CI stable.
    - Many tests run heavy numeric checks — use short or subset runs while iterating on code.

If you want, I can add:

- A single `scripts/run_tests.sh` or `scripts/run_tests.ps1` that runs common test subsets (fast, full, feature-gated).
- A CI matrix example that runs extended tests nightly and marks heavy tests as `allow_failures: false` for catching
  regressions.

End of `docs/test_descriptions.md`.
