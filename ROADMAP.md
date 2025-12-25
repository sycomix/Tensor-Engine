# Tensor Engine Roadmap: Complete LLM, Diffusion, and Audio Model Support

This document outlines all features and components needed to train and run leading Large Language Models (LLMs),
diffusion models, and audio generation models using the tensor_engine library.

## Updates (Dec 2025) ✅

- **Completed / Verified**

  - [x] **Windows test blocker fixed**: made the Python FFI (`cffi`) optional and gated under the `python_bindings` feature to avoid linker issues on Windows and enable running the full test suite.
  - [x] **Transformer load-state hardening**: added unit & integration tests covering kv-head expansion, transposed k_proj handling, and LLaMA-style key mappings (`src/nn/tests/transformer_load_state_tests.rs`, `src/nn/tests/transformer_integration_tests.rs`).
  - [x] **Rules-compliant example**: rewrote `examples/chat_safetensors.py` to load embeddings & per-layer weights from SafeTensors, apply per-layer state, enforce `rules.md` (no placeholder tensors), tie LM head to embeddings when necessary, and add a one-shot `--message` mode plus a naive greedy generator; validated end-to-end with Llama-3.2-1B safetensors.
  - [x] **Tests & CI readiness**: ran full `cargo test --all` locally after fixes and confirmed tests pass.

- **Short-term (High priority)**

  - [ ] Implement an **optimized decoding path** (robust KV cache + attention caching + generator integration) — owner: core, ETA: 2-4 weeks. 🔧
  - [ ] Add a **lightweight CI smoke test** that loads a small SafeTensors checkpoint and runs a one-step generation (guard regressions without heavy runtime cost) — owner: infra, ETA: 1 week. ⚠️
  - [ ] Create **microbenchmarks** for SafeTensors load/apply operations and generator steps; add to `benches/` and gate heavy runs behind `CI_BENCH` — owner: perf, ETA: 1-2 weeks. 📊
  - [ ] Add **attention caching & batched decode** support and integrate with the example generator — owner: core, ETA: 3-6 weeks. 🚀
  - [ ] Add **documentation + smoke test** for `examples/chat_safetensors.py` and a short usage example in the README — owner: docs, ETA: 3 days. 📚

- **Mid-term (Strategic / Roadmapped)**
  - [ ] GPU acceleration and attention kernel integration (priority for production throughput, ETA: Q1 2026)
  - [ ] Production quantization (AWQ/GPTQ) and 8-bit optimizer support (ETA: Q1–Q2 2026)
  - [ ] Speculative decoding and batched speculative decoding research for latency improvements

**Why this matters:** the recent fixes remove test blockers and improve compatibility with real-world LLM checkpoints. Short-term objectives focus on making the loading + generation path robust and guarded by CI while we plan performance work and quantization for production use.

---

## 1. Core Tensor Operations & Infrastructure

### 1.1 Basic Operations (Status: Mostly Complete)

- [x] Element-wise operations (Add, Mul, Sub, Div) (see: `src/ops.rs`)
- [x] Matrix operations (MatMul, Transpose) (see: `src/ops.rs` / `MatMul` / `PermuteAxes`)
- Reduction operations:
  - [x] Sum (`src/ops.rs`)
  - [x] Mean (`src/ops.rs`)
  - [x] Max (`src/ops.rs`)
  - [x] Min (`src/ops.rs`)
- Activation functions:
  - [x] ReLU (`src/ops.rs`)
  - [x] Sigmoid (`src/ops.rs`)
  - [x] Tanh (`src/ops.rs`)
  - [x] GELU (`src/ops.rs`)
  - [x] Swish/SiLU (`src/ops.rs`) - standalone SiLU op implemented
- Power and logarithmic operations:
  - [x] Pow (`src/ops.rs`)
  - [x] Log (`src/ops.rs`)
  - [x] Exp (`src/ops.rs`)
- [x] Comparison operations (Equal, Greater, Less) (`src/ops.rs`)
- [x] Broadcasting support for element-wise operations (see broadcasting logic in `src/tensor.rs`)
- [x] Advanced broadcasting verification for complex patterns (`tensor::Tensor::broadcast_shapes`)
- [x] Mixed precision operations (FP16/BF16 round-trip conversions; INT8 quantization helpers implemented) (
      `src/dtype.rs`)

### 1.2 Advanced Operations

- Convolution operations:
  - [x] Conv1D (`src/ops.rs`, `src/nn.rs`)
  - [x] Conv2D (`src/ops.rs`, `src/nn.rs`)
  - [x] Conv3D (`src/ops.rs`, `src/nn.rs`)
- [x] Depthwise separable convolutions (`src/ops.rs`, `src/nn.rs`)
- [x] Transposed convolutions (`src/ops.rs`, `src/nn.rs`) - ConvTranspose1D op & module added
- Pooling operations:
  - [x] MaxPool (`src/ops.rs`, `src/nn.rs`)
  - [x] AvgPool (`src/ops.rs`, `src/nn.rs`)
- [x] Adaptive pooling (`src/ops.rs`, `src/nn.rs`)
- Normalization:
  - [x] LayerNorm (`src/ops.rs`/`src/nn.rs`)
  - [x] RMSNorm (`src/ops.rs`/`src/nn.rs`)
- [x] Dropout (`src/ops.rs`)
- [x] Attention mechanisms (MultiHeadAttention) (`src/nn/transformer.rs`)
- Positional embeddings:

  - [x] RoPE / Rotary Positional Embeddings (`src/ops.rs`, `src/nn/transformer.rs`)
  - [x] Absolute positional embeddings (`src/nn.rs` / `AbsolutePositionalEmbedding`) — basic implementation + unit
        test
  - [x] ALiBi positional embeddings (`src/nn/transformer.rs` / `with_alibi`) — ALiBi slopes + unit test present
  - [x] ALiBi: add robust validation tests (edge cases where bias doesn't affect outputs) and additional integration
        checks

  - [ ] Complex number operations for RoPE (not implemented)

- [x] FlashAttentionRef & ChunkedAttention (reference implementations and op-level variants; see `src/ops.rs` and
      `src/nn/transformer.rs`)
- [x] Memory-efficient attention variants (Chunked attention implemented; optimized vendor kernels not integrated)

### 1.3 Optimization & Performance

- [ ] CUDA/GPU acceleration
- [x] OpenBLAS integration
- [ ] MKL support
- [ ] Tensor cores utilization
- [ ] Memory pooling and reuse
- [ ] Asynchronous operations
- [ ] Multi-threading optimizations
- [ ] Gradient checkpointing
- [ ] Automatic mixed precision (AMP)

## 2. Neural Network Layers & Components

### 2.1 Basic Layers

- [x] Linear/Dense layers (`src/nn.rs` / `Linear`)
- [x] Convolutional layers (`src/nn.rs` / `Conv2D`)
- Recurrent layers:
  - [x] LSTM (`src/nn.rs` / `LSTMCell`)
  - [ ] GRU (not implemented)
- [x] Transformer layers (`src/nn/transformer.rs` / `TransformerBlock`)
- [x] Embedding layers (`src/ops.rs` / `EmbeddingLookup`)
- [ ] Sparse embedding layers (not implemented)
- [ ] Adaptive embedding layers (not implemented)

### 2.2 Advanced Layers

- [x] Multi-head attention (`src/nn/transformer.rs`)
- [x] Grouped Query Attention (GQA) (supported by transformer tests; see `src/nn/tests/transformer_rope_gqa_tests.rs`)
- [x] Cross-attention ops: `FlashAttentionRef`/`ChunkedAttention` and op-level interfaces accept separate Q/K/V (
      op-level cross-attn supported). Note: `TransformerBlock` default forward is self-attention; encoder-decoder
      cross-attention wrapper is not pre-built.

- [ ] Sliding window attention (not implemented)
- [ ] Sparse attention patterns (not implemented)
- [x] Feed-forward networks (MLP) (`src/nn/transformer.rs` / feed-forward layers)
- [x] SwiGLU activation (`src/ops.rs` / `SwiGLU`)
- [ ] GeGLU, ReGLU variants (not implemented)
- [ ] MoE (Mixture of Experts) layers (not implemented)
- [ ] Parallel experts implementation (not implemented)
- [ ] Routing mechanisms (not implemented)

### 2.3 Normalization & Regularization

- [x] Layer Normalization (`src/ops.rs`, `src/nn.rs`)
- [x] RMS Normalization (`src/ops.rs`, `src/nn.rs`)
- [ ] Batch Normalization (not implemented)
- [x] Group Normalization (`src/ops.rs`, `src/nn.rs`)
- [ ] Instance Normalization (not implemented)
- [x] Dropout (`src/ops.rs` / `src/nn.rs`)
- [ ] DropPath/Stochastic Depth (not implemented)
- [ ] Weight decay (optimizer feature; limited/no support)
- [ ] Gradient clipping (not implemented)

## 3. Model Architectures

### 3.1 Language Models

- [x] Transformer blocks
- [ ] GPT-style decoder-only models (not implemented; `TransformerBlock` exists)
- [ ] BERT-style encoder-only models (not implemented; `TransformerBlock` exists)
- [x] Encoder-decoder wrapper implemented (`src/nn/transformer.rs::EncoderDecoderTransformer`), full T5 is not
      implemented
- [ ] Llama architecture variants (1, 2, 3, 3.1, 3.2)
  - [x] Llama-style TransformerBlock (RMSNorm pre-norm + SwiGLU, RoPE applied to Q/K, optional biasless dense)
        implemented in `src/nn/transformer_cleaned.rs` via `new_llama_style` constructor.
- [ ] Mistral architecture
- [ ] Phi models
- [ ] Qwen models
- [ ] Gemma models
- [ ] Grok architecture
- [ ] MoE architectures (Mixtral, DeepSeek)
- [ ] Sparse models (ALBERT, DistilBERT)

### 3.2 Vision Models

- [x] Vision Transformer (ViT) (`src/nn/vision.rs`) - PatchEmbed and ViT basics implemented
- [ ] Swin Transformer
- [ ] CLIP architecture
- [ ] DINO models
- [ ] SAM (Segment Anything Model)

### 3.3 Multimodal Models

- [x] Multimodal LLM (fusion/decoder basics) (`src/nn/multimodal.rs`) - basic fusion/decoder scaffolding implemented
- [ ] CLIP (Contrastive Language-Image Pretraining)
- [ ] LLaVA (Large Language and Vision Assistant)
- [ ] BLIP models
- [ ] ImageBind
- [ ] Audio-Visual models

## 4. Training Infrastructure

### 4.1 Optimizers

- [x] Adam optimizer (`src/nn.rs` / `Adam`)
- [x] AdamW optimizer (`src/nn.rs` / `AdamW`)
- [x] SGD (basic) (`src/nn.rs` / `SGD`)
- [x] SGD with momentum (implemented via `SGD::new(lr, momentum)`; momentum parameter is supported)

- [x] RMSProp (`src/nn/mod.rs`) implemented
- [ ] Adagrad
- [ ] Lion optimizer
- [ ] 8-bit optimizers (bitsandbytes)
- [ ] Zero Redundancy Optimizer (ZeRO)
- [ ] Gradient accumulation

### 4.2 Loss Functions

- [x] Cross-entropy loss (`src/ops.rs` / `CrossEntropyLogits` & `SoftmaxCrossEntropyLogits`)
- [x] Mean squared error (MSE) (`src/nn.rs` / `MSELoss`)
- [ ] Binary cross-entropy (not implemented)
- [ ] Focal loss
- [ ] Label smoothing
- [ ] KL divergence
- [ ] Contrastive loss
- [ ] Triplet loss

### 4.3 Learning Rate Schedulers

- [x] Cosine annealing (`src/nn/mod.rs::CosineAnnealing`) implemented
- [x] Linear warmup (`src/nn/mod.rs::LinearWarmup`) implemented
- [ ] Exponential decay
- [ ] Step decay
- [ ] Polynomial decay
- [ ] Cyclic learning rates

### 4.4 Distributed Training

- [ ] Data parallelism
- [ ] Model parallelism
- [ ] Pipeline parallelism
- [ ] Tensor parallelism
- [ ] DeepSpeed integration
- [ ] Megatron-LM style parallelism

## 5. Data Loading & Preprocessing

### 5.1 Data Loaders

- [x] Batch data loading (`src/nn.rs` Dataset & `src/io/dataloader.rs` WavDataLoader) with `batch_size` support and
      `load_batch()` helpers
- [x] Shuffle and sampling (`Dataset::shuffle`, `tests::autograd_test::test_dataloader_shuffle_next_batch`)
- [ ] Distributed data loading
- [ ] Memory mapping for large datasets
- [ ] Streaming data loading

### 5.2 Tokenization

- [x] Hugging Face tokenizers integration (feature-gated wrapper + simple test: `src/io/tokenizers.rs`,
      `tests/tokenizer_test.rs`, enable with `--features with_tokenizers`)
- [ ] BPE (Byte Pair Encoding)
- [ ] WordPiece
- [ ] SentencePiece
- [ ] Tiktoken (OpenAI)
- [ ] Custom tokenizer training

### 5.3 Data Processing

- [ ] Text preprocessing pipelines
- [x] Image preprocessing (resize & normalize) (`src/io/image.rs::load_image_to_tensor`) implemented
- [ ] Image preprocessing (augment)
- [ ] Audio preprocessing (MFCC, spectrograms)
- [ ] Data augmentation
- [ ] Sequence padding and masking

## 6. Model Loading & Saving

### 6.1 Model Formats

- [x] SafeTensors format support (implemented via `src/io/safetensors_loader.rs` behind `safe_tensors` feature;
      transpose flag and `apply_safetensors_bytes_to_module_bytes` helper exist)
  - [x] Kronos SafeTensors mapping: `apply_kronos_bytes_to_module_bytes` maps `vision_encoder`, `text_embedding`,
        `projector`, `decoder_blocks`, and `head` to `MultimodalLLM` fields (see `kronos-modal-format.md` /
        `docs/kronos_integration.md`)
- [x] PyTorch state_dict loading (VarStore loader implemented under feature `with_tch`; TorchScript fallback now
      attempts to extract parameters via CModule::named_parameters() and calls `state_dict()` via IValue to extract buffers
      when possible. Still recommend `examples/convert_torch_to_safetensors.py` for complex pickled modules.) (partial)
  - Improvements: Added CModule fallback, state_dict(IValue) parsing for Vec<(IValue,IValue)>, key normalization and
    fixture-based CI tests. Added recursive parsing for nested GenericDict and tuple entries; added tests for nested
    state_dict and list-of-pairs. `TryFrom<IValue>` conversions for `Vec<(String, Tensor)>` and
    `HashMap<String, Tensor>` are not supported by `tch` so we rely on `Vec<(IValue,IValue)>` and GenericDict parsing
    instead. Added base64-encoded TorchScript fixtures in `tests/assets` so CI does not require Python to build
    fixtures. (See `src/io/pytorch_loader.rs`, `tests/pytorch_loader_test.rs`)
  - Next: Additional edge-case parsing (deeply nested constructs, mixed variant types), streaming large tensors
    without decode to memory, and more robust checks for `IValue` variant conversions. Add CI improvements for Windows
    runtime alignment: ensure libtorch is built with matching MSVC runtime or pin a known-good shared libtorch build;
    consider test matrix that builds libtorch from source under the pinned MSVC toolchain for Windows runners.
- [ ] Hugging Face model hub integration
- [ ] ONNX format support
- [ ] GGUF format (llama.cpp)
- [ ] Custom binary formats

### 6.2 Weight Management

- [x] Automatic weight transposition (PyTorch to custom format) — `safetensors` & `pytorch` loaders accept `transpose`
      flag and perform 2D weight transpose when required (see `src/io/safetensors_loader.rs`, `src/io/pytorch_loader.rs`)
- [x] Quantized MatMul helper (dequantizes INT8 to float and performs matmul; see `src/ops.rs::QuantizedMatMul`).
  - Improvements: Added quantized MatMul op with basic tests (`tests/quantized_matmul_test.rs`).
  - Next: Add microbenchmarks in `benches/` and extend tests for more cases and protocol types (per-layer scales,
    blockwise formats). Done: Added quantized_matmul benches to `benches/matmul_bench.rs` for sizes 10/50/100/200 and
    larger sizes/batched/blockwise quantized variants (gated under CI_BENCH to avoid heavy CI runtime).
- [ ] Production-grade quantization (AWQ/GPTQ & runtime support)

- [ ] LoRA (Low-Rank Adaptation)
- [ ] QLoRA
- [ ] Weight pruning
- [ ] Knowledge distillation

## 7. Inference Optimization

### 7.1 Runtime Optimizations

- [x] KV cache implementation (basic) (`src/ops.rs` / `KVCacheAppend` + `Tensor::kvcache_append`)
- [ ] Attention caching (not implemented)
- [ ] Memory management
- [ ] Batch processing
- [ ] Continuous batching
- [ ] Speculative decoding
- [ ] Medusa heads

### 7.2 Quantization

- [x] Storage/round-trip quantization helpers (F8/I8 emulation and f16/bf16 round-trip conversion in `src/dtype.rs`)
- [ ] Dynamic quantization (runtime/inference support)
- [ ] Static quantization (compiled quantized models)
- [ ] Quantization-aware training (training-aware quantization)
- [ ] Mixed precision inference (runtime mixed-precision optimization)
- [ ] AWQ (Activation-aware Weight Quantization)
- [ ] GPTQ (GPT Quantization)

### 7.3 Acceleration

- [ ] CPU optimizations
- [ ] GPU acceleration
- [ ] TPU support
- [ ] WebGPU/WebAssembly
- [ ] Mobile optimizations

## 8. Diffusion Models

### 8.1 Core Components

- [ ] Denoising diffusion probabilistic models (DDPM)
- [ ] Denoising diffusion implicit models (DDIM)
- [ ] Stable Diffusion architecture
- [ ] Latent Diffusion Models (LDM)
- [ ] ControlNet
- [ ] Inpainting models
- [ ] Image-to-image translation

### 8.2 Components Needed

- [ ] U-Net architecture
- [ ] Variational Autoencoder (VAE)
- [ ] CLIP text encoder
- [ ] Noise schedulers (linear, cosine, etc.)
- [ ] CFG (Classifier-Free Guidance)
- [ ] Self-attention in U-Net
- [ ] Cross-attention for text conditioning

### 8.3 Training Features

- [ ] Diffusion model training loops
- [ ] VAE training
- [ ] Text encoder fine-tuning
- [ ] LoRA training for diffusion models

## 9. Audio Generation Models

### 9.1 Speech Synthesis

- [ ] Tacotron architecture
- [ ] FastSpeech models
- [ ] VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
- [ ] Bark (multilingual TTS)
- [ ] Tortoise TTS

### 9.2 Music Generation

- [ ] Jukebox
- [ ] MusicGen
- [ ] AudioLM
- [ ] MuseNet

### 9.3 Audio Processing Components

- [ ] Mel-spectrogram computation
- [ ] STFT (Short-time Fourier Transform)
- [ ] WaveNet layers
- [ ] HiFi-GAN vocoder
- [ ] Universal audio tokenizer
- [x] Audio encoder/decoder (`src/nn/audio.rs`) implemented using Conv1D/ConvTranspose1D stacks
- [x] Residual Vector Quantizer (RVQ) (`src/nn/quantization.rs`) implemented (hierarchical RVQ, quantize & dequantize)
  - [x] RVQ: add EMA updates (unbiased counts), reinit empty codes, and scheduling (implemented in
        `src/nn/quantization.rs`)
- [x] WAV I/O utilities (`src/io/audio.rs`) implemented (load and write WAV via `hound`)
- [x] Audio resampling (linear fallback + `rubato::FftFixedIn`) implemented; `src/io/dataloader.rs` includes resample
      support and `tests/dataloader_resample_test.rs` validates both methods

### 9.4 Training Infrastructure

- [x] Audio data loading (`src/io/dataloader.rs` WavDataLoader) with optional resampling (linear + `rubato`) and
      integration with `examples/train_codec.rs` and `examples/text_to_audio.rs`.
- [ ] Spectrogram preprocessing
- [ ] Multi-speaker support
- [ ] Voice conversion
- [ ] Audio augmentation

## 10. Utilities & Tools

### 10.1 Development Tools

- [ ] Model visualization
- [ ] Gradient flow debugging
- [ ] Memory profiling
- [x] Performance benchmarking (Criterion benches added/expanded in `benches/matmul_bench.rs` including quantized
      variants; heavy benches gated by `CI_BENCH`)
- [x] Unit testing framework (new tests + fixtures for TorchScript, quantized ops, tokenizer wrapper present)
- [x] `as_any_mut` verification script (`scripts/verify_as_any_mut.py`) to enforce Module impl changes and guard
      Operation impls from regressions (add to CI: `ci/verify_as_any_mut.sh`).
- [x] Integration testing (PyO3 wrappers, tokenizers & quantized ops integration tests added)
- [x] Documentation site generation (MkDocs) + build scripts and CI (`mkdocs.yml`, `scripts/build_docs.*`, `.github/workflows/docs.yml`)

### 10.2 Deployment & Serving

- [ ] Model serving infrastructure
- [ ] REST API endpoints
- [ ] gRPC services
- [ ] Streaming inference
- [ ] Model versioning
- [ ] A/B testing framework

### 10.3 Monitoring & Observability

- [ ] Training metrics logging
- [ ] Inference latency monitoring
- [ ] Memory usage tracking
- [ ] Error rate monitoring
- [ ] Custom metrics

### 10.4 Documentation & Examples

- [ ] Comprehensive API documentation
- [ ] Tutorial notebooks
- [ ] Model zoo with pre-trained weights
- [x] Performance benchmarks (`benches/` + `docs/bench_descriptions.md` present)
- [x] Migration guides (`docs/backend_migration_plan.md` present)
- [x] Quickstart (`docs/quickstart.md` present)
- [x] HTML docs site (MkDocs) + build scripts & CI (`mkdocs.yml`, `scripts/build_docs.*`, `.github/workflows/docs.yml`)
- [x] Audio codec examples: `examples/train_codec.rs` and `examples/text_to_audio.rs` (training loop and inference example added)

## 11. Research & Advanced Features

### 11.1 Cutting-Edge Techniques

- [ ] Retentive Networks (RetNet)
- [ ] Mamba architecture
- [ ] RWKV models
- [ ] Hyena hierarchy
- [ ] Liquid Neural Networks
- [ ] Kolmogorov-Arnold Networks (KAN)

### 11.2 Efficiency Improvements

- [ ] Linear attention mechanisms
- [ ] Performer (FAVOR+) attention
- [ ] LongRoPE for extended context
- [ ] Ring Attention for infinite context
- [ ] Dynamic sparse attention

### 11.3 Multimodal Advancements

- [ ] Unified multimodal architectures
- [ ] 3D understanding
- [ ] Video generation models
- [ ] Embodied AI components

## Implementation Priority

### High Priority (Essential for Basic LLM Training/Inference)

1. Hugging Face tokenizers integration (feature-gated wrapper implemented; unit test included `tests/tokenizer_test.rs`)
2. Complete optimizer implementations (Adam, AdamW present; review and expand scheduling/weight-decay)
3. Learning rate schedulers (basic warmup & cosine implemented; other schedules like exponential, polynomial and cyclic
   remain pending)
4. Distributed training primitives (missing)
5. Production-quality quantization support (ongoing: `QuantizedMatMul` implemented and benches added; AWQ/GPTQ,
   block/rowwise quantization formats and runtime support still pending)
6. KV cache optimization (basic KV cache implemented; optimize memory & caching for inference)
7. Windows builder/runtime alignment for `libtorch` (pin MSVC runtime or build libtorch from source to avoid runtime
   mismatches in CI)

### Medium Priority (Advanced LLM Features)

1. MoE layers
2. Flash Attention
3. Gradient checkpointing
4. Mixed precision training
5. Model parallelism
6. Sparse attention patterns

### Low Priority (Research/Diffusion/Audio)

1. Diffusion model components
2. Audio processing pipelines
3. Advanced architectures (Mamba, RetNet)
4. Multimodal fusion layers
5. Research model implementations

## Dependencies to Add

### Core Dependencies

- `safetensors` - Model weight loading
- `tokenizers` - Text tokenization
- `serde` - Serialization
- `rayon` - Parallel processing
- `crossbeam` - Concurrent utilities

### Optional/Feature-gated Dependencies

- `candle-core` - Alternative tensor operations
- `tch` - PyTorch integration
- `ort` - ONNX runtime
- `tract` - ONNX inference
- `rten` - ONNX models in Rust

### Audio-specific Dependencies

- [x] `hound` - WAV file I/O (added to Cargo.toml; feature `audio` available)
- [x] `rubato` - Audio resampling (`FftFixedIn` integration and tests)
- `realfft` - FFT computations

### GPU/Acceleration Dependencies

- `cudarc` - CUDA acceleration
- `wgpu` - WebGPU support
- `metal` - Apple Metal support

This comprehensive roadmap covers all major components needed for a production-ready deep learning framework capable of
handling modern LLMs, diffusion models, and audio generation tasks.

## Advisory / Recommendations

- Hugging Face tokenizers: Feature-gated `tokenizers` wrapper has been added (`src/io/tokenizers.rs`) with unit tests (
  `tests/tokenizer_test.rs`). If not enabled by default, expose an optional CLI wrapper for tokenization and example
  usage in `examples/`.
- PyTorch/state_dict: Prefer `safetensors` as the canonical format. Implemented `tch`-based loader with VarStore load
  and a robust TorchScript fallback (`src/io/pytorch_loader.rs`) that parses `IValue` and handles GenericDict/Vec<(
  IValue,IValue)> tuples. Added base64 TorchScript fixtures and generator scripts in `tests/assets`/`scripts/` for CI
  that avoids Python dependency. Keep `examples/convert_torch_to_safetensors.py` for more complex pickled modules.
- Quantization: `QuantizedMatMul` implemented and tested (`src/ops.rs`, `tests/quantized_matmul_test.rs`). Criterion
  benches updated to include quantized variants (`benches/matmul_bench.rs`). Next: add per-layer quantization helpers,
  block/rowwise quantization formats (AWQ/GPTQ), runtime support for quantized Conv, and a `quantize_weights` utility.
- GPU acceleration: Create a GPU backend ABI (cudarc or wgsl): implement a `backend` trait and start with a `cpu` and
  `wgpu` reference backend. Target `cudarc` in a later phase.
- Cross-attention & seq2seq: Add a TransformerBlock builder that supports `cross_attn` with separate K/V inputs, and
  expose an encoder-decoder example in `examples/`.
- ALiBi / NL-OOB tests: Add focused unit tests covering zero-initialized proj edge cases and end-to-end model tests with
  NL-OOB enabled.
- CI & builds: Added Linux `test_with_tch` job to CI using CPU torch wheel; added Windows `test_with_tch_windows` job
  that downloads shared libtorch and sets env variables to mitigate MSVC runtime mismatch (installs `vc_redist` and
  optionally Visual Studio Build Tools via Chocolatey). Add recommendations for a future step: pin MSVC runtime and
  matching libtorch builds, or build libtorch from source in CI for Windows to remove runtime mismatch artifacts.

- Docs & examples: `docs/quickstart.md` added; HTML docs site generation added via MkDocs (`mkdocs.yml`), build scripts (`scripts/build_docs.ps1`, `scripts/build_docs.sh`), and a GitHub Action (`.github/workflows/docs.yml`). Stay mindful that **comprehensive API reference** and **tutorial notebooks** are still outstanding and should be added as docs evolve.

These recommendations prioritize integration and small, deliverable steps that enable adoption by broader ML tooling and
developer workflows.
