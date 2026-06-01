# Tensor Engine Roadmap: Complete LLM, Diffusion, and Audio Model Support

This document outlines all features and components needed to train and run leading Large Language Models (LLMs),
diffusion models, and audio generation models using the tensor_engine library.

## Updates (Jun 2026) ✅

- **Inspection completed**: Full recursive codebase inspection (130 Rust files, 26 Python examples, 90+ test files)
- **Status**: ROADMAP.md updated with verified implementation status from actual source code
- **Key additions**: CLIP full implementation, MoE layer, LoopedTransformer, decoders, paged attention, distributed training scaffold, AWQ integration, continuous thought module

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
- [x] Additional ops verified in ops.rs: Abs, Sign, Sqrt, Rsqrt, Clamp, Floor, Ceil, Round, Trunc, Frac, IsInf, IsNaN, Tril, Triu, Reciprocal
- [x] Reduction ops verified: Prod, All, Any
- [x] Indexing ops verified: IndexSelect, Gather, Scatter, ScatterAdd, MaskedScatter
- [x] Array ops verified: Concat, Stack, PermuteAxes, Slice, Unfold2D, Fold2D
- [x] Search ops verified: TopK, Sort, ArgSort
- [x] Loss ops verified: CrossEntropyLogits, SoftmaxCrossEntropyLogits, NLLLoss, BinaryCrossEntropy, BinaryCrossEntropyWithLogits
- [x] Image ops verified: Interpolate, GridSample, UpSampleNearest2D
- [x] FFT ops verified: FFT, IFFT, RFFT, IRFFT
- [x] Other ops verified: Where, CumSum, CumProd, CumMax, CumMin, ComplexMul, ComplexConj, BatchNorm
- [x] Checkpoint op for gradient checkpointing (`src/autograd.rs::checkpoint`)
- [x] FlashAttentionRef: CPU reference FlashAttention with forward+backward
- [x] Operation trait: forward(), backward(), as_any()
- [x] reduce_grad_to_shape() helper for broadcasting gradient reduction
- [x] permute_to_last() / permute_back() helpers

### 1.2 Advanced Operations

- Convolution operations:
    - [x] Conv1D (`src/ops.rs`, `src/nn.rs`)
    - [x] Conv2D (`src/ops.rs`, `src/nn.rs`)
    - [x] Conv3D (`src/nn/conv.rs` / `Conv3D` with `Conv3DConfig`; op-level `Conv3D` in ops.rs)
    - [x] DepthwiseSeparableConv2D (`src/nn/conv.rs` / `DepthwiseSeparableConv2D` with depthwise + pointwise weights)
- [x] Transposed convolutions (`src/ops.rs`, `src/nn.rs`) - ConvTranspose1D op & module added
- [x] ConvTranspose2D (src/nn/conv.rs)
- Pooling operations:
    - [x] MaxPool (`src/ops.rs`, `src/nn.rs`)
    - [x] AvgPool2D (`src/nn/conv.rs` / `AvgPool2D` with kernel_size + stride)
    - [x] AdaptiveAvgPool2D (`src/nn/conv.rs` / `AdaptiveAvgPool2D` with out_h/out_w)
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
- [x] Memory pooling and reuse (`src/memory_pool.rs` - TensorPool exists)
- [x] Asynchronous operations (`src/async_ops.rs` - feature-gated)
- [ ] Multi-threading optimizations
- [x] Gradient checkpointing (`src/autograd.rs::checkpoint`)
- [x] Automatic mixed precision (AMP) (`src/amp.rs`)

## 2. Neural Network Layers & Components

### 2.1 Basic Layers

- [x] Linear/Dense layers (`src/nn.rs` / `Linear`)
- [x] Convolutional layers (`src/nn.rs` / `Conv2D`)
- Recurrent layers:
    - [x] LSTM (`src/nn.rs` / `LSTMCell`)
    - [x] GRU (`src/nn.rs` / `GRUCell` with `forward_step()`)
- [x] Transformer layers (`src/nn/transformer.rs` / `TransformerBlock`)
- [x] Embedding layers (`src/ops.rs` / `EmbeddingLookup`)
- [x] Sparse embedding layers (`src/nn/embedding.rs` / `SparseEmbedding`)
- [x] Adaptive embedding layers (`src/nn/embedding.rs` / `AdaptiveEmbedding` with head/tail clusters, cutoffs, div_value)

### 2.2 Advanced Layers

- [x] Multi-head attention (`src/nn/transformer.rs`)
- [x] Grouped Query Attention (GQA) (supported by transformer tests; see `src/nn/tests/transformer_rope_gqa_tests.rs`)
- [x] Cross-attention ops: `FlashAttentionRef`/`ChunkedAttention` and op-level interfaces accept separate Q/K/V (
  op-level cross-attn supported). Note: `TransformerBlock` default forward is self-attention; encoder-decoder
  cross-attention wrapper is not pre-built.
- [x] Sliding window attention (`src/nn/transformer.rs` / `SlidingWindowAttention` wrapper)
- [ ] Sparse attention patterns (not implemented)
- [x] Feed-forward networks (MLP) (`src/nn/transformer.rs` / feed-forward layers)
- [x] SwiGLU activation (`src/ops.rs` / `SwiGLU`)
- [ ] GeGLU, ReGLU variants (not implemented)
- [x] MoE (Mixture of Experts) layers (`src/nn/moe.rs` / `MoELayer` with top-k routing, softmax weights, scatter-add)
- [x] Parallel experts implementation (`src/nn/moe.rs` / `Expert` struct with w1/w2/w3)
- [x] Routing mechanisms (`src/nn/moe.rs` / gate + topk + softmax routing)

### 2.3 Normalization & Regularization

- [x] Layer Normalization (`src/ops.rs`, `src/nn.rs`)
- [x] RMS Normalization (`src/ops.rs`, `src/nn.rs`)
- [x] Batch Normalization (`src/nn/mod.rs` / `BatchNorm1d`, `BatchNorm2d`)
- [x] Group Normalization (`src/ops.rs`, `src/nn.rs`)
- [ ] Instance Normalization (not implemented)
- [x] Dropout (`src/ops.rs` / `src/nn.rs`)
- [x] DropPath/Stochastic Depth (`src/nn/mod.rs` / `DropPath`)
- [ ] Weight decay (optimizer feature; limited/no support)
- [ ] Gradient clipping (not implemented)

## 3. Model Architectures

### 3.1 Language Models

- [x] Transformer blocks
- [x] GPT-style decoder-only models (`src/nn/transformer.rs` / `GPTDecoder`)
- [x] BERT-style encoder-only models (`src/nn/transformer.rs` / `BERTEncoder`)
- [x] Encoder-decoder wrapper implemented (`src/nn/transformer.rs::EncoderDecoderTransformer`)
- [x] T5EncoderDecoder (`src/nn/transformer.rs`)
- [x] Llama architecture variants (1, 2, 3, 3.1, 3.2)
    - [x] Llama-style TransformerBlock (RMSNorm pre-norm + SwiGLU, RoPE applied to Q/K, optional biasless dense)
      implemented in `src/nn/transformer.rs` via `new_llama_style` constructor.
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
- [x] CLIP architecture (`src/nn/clip.rs` - full CLIP: CLIPConfig, QuickGELU, CLIPAttention, CLIPMLP, CLIPEncoderLayer, CLIPVisionTransformer, CLIPTextTransformer, CLIP model)
- [ ] DINO models
- [ ] SAM (Segment Anything Model)

### 3.3 Multimodal Models

- [x] Multimodal LLM (fusion/decoder basics) (`src/nn/multimodal.rs`) - basic fusion/decoder scaffolding implemented
- [x] CLIP (Contrastive Language-Image Pretraining) (`src/nn/clip.rs` - full implementation)
- [ ] LLaVA (Large Language and Vision Assistant)
- [ ] BLIP models
- [ ] ImageBind
- [ ] Audio-Visual models

## 4. Training Infrastructure

### 4.1 Optimizers

- [x] Adam optimizer (`src/optim.rs` / `Adam`)
- [x] AdamW optimizer (`src/optim.rs` / `AdamW`)
- [x] SGD (basic) (`src/optim.rs` / `SGD`)
- [x] SGD with momentum (implemented via `SGD::with_momentum()`)
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
- [x] Exponential decay (`src/lr_scheduler.rs::ExponentialLR`) implemented
- [x] Step decay (`src/lr_scheduler.rs::StepLR`) implemented
- [x] Polynomial decay (`src/lr_scheduler.rs::PolynomialLR`) implemented
- [ ] Cyclic learning rates

### 4.4 Distributed Training

- [x] Data parallelism (`src/distributed/data_parallel.rs` / `DataParallel` with `ShardedBatch`)
- [ ] Model parallelism
- [ ] Pipeline parallelism
- [ ] Tensor parallelism
- [ ] DeepSpeed integration
- [ ] Megatron-LM style parallelism
- [x] Gradient sync (`src/distributed/all_reduce.rs` / `AllReduce` with `ReduceOp`)
- [x] Distributed checkpointing (`src/distributed/checkpoint.rs` / `DistributedCheckpoint` with `CheckpointConfig`)
- [x] Distributed context (`src/distributed/context.rs` / `DistributedContext`, `DistributedConfig`, `DeviceId`)

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
  when possible. Still recommend `examples/convert_torch_to_safetensors.py` for complex pickled modules.)  (partial)
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
- [x] HuggingFace model loading (`src/hf_compat/huggingface_loader.rs` + `src/compat/rllama/huggingface_loader.rs`)
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
- [x] KV cache (`src/nn/kv_cache.rs` / `KVCache` for incremental decoding)
- [x] Paged KV cache (`src/nn/paged_kv_cache.rs` / `PagedKVCache`)
- [x] Paged attention (`src/nn/paged_attention.rs`)
- [ ] Attention caching (not implemented)
- [ ] Memory management
- [ ] Batch processing
- [ ] Continuous batching
- [x] Speculative decoding (`src/generation/speculative.rs`)
- [ ] Medusa heads

### 7.2 Quantization

- [x] Storage/round-trip quantization helpers (F8/I8 emulation and f16/bf16 round-trip conversion in `src/dtype.rs`)
- [ ] Dynamic quantization (runtime/inference support)
- [ ] Static quantization (compiled quantized models)
- [ ] Quantization-aware training (training-aware quantization)
- [ ] Mixed precision inference (runtime mixed-precision optimization)
- [x] AWQ (Activation-aware Weight Quantization) (`src/quantization/awq.rs` - AWQ module exists)
- [ ] GPTQ (GPT Quantization)

### 7.3 Acceleration

- [ ] CPU optimizations
- [ ] GPU acceleration
- [ ] TPU support
- [ ] WebGPU/WebAssembly
- [ ] Mobile optimizations

## 8. Diffusion Models

### 8.1 Core Components

- [x] Denoising diffusion probabilistic models (DDPM) (`src/nn/diffusion.rs` / `DDPMScheduler` with linear beta schedule, q_sample, predict_eps, step)
- [ ] Denoising diffusion implicit models (DDIM)
- [ ] Stable Diffusion architecture
- [ ] Latent Diffusion Models (LDM)
- [ ] ControlNet
- [ ] Inpainting models
- [ ] Image-to-image translation

### 8.2 Components Needed

- [x] U-Net architecture (`src/nn/diffusion.rs` / `UNetModel` with `ResNetBlock` stack and time embedding injection)
- [ ] Variational Autoencoder (VAE)
- [x] CLIP text encoder (`src/nn/clip.rs` / `CLIPTextTransformer`)
- [x] Noise schedulers (linear, cosine, etc.) (`src/nn/diffusion.rs` / `DDPMScheduler::new_linear`)
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
- [x] Monitoring (`src/monitoring.rs` - feature-gated)
- [x] Server (`src/server/mod.rs` - feature-gated)
- [x] HF Bridge (`src/hf_bridge.rs` - feature-gated)
- [x] Python Bindings (`src/python_bindings.rs` - feature-gated)
- [x] Config (`src/config.rs`)
- [x] Error handling (`src/error.rs`)
- [x] Labels (`src/labels.rs`)
- [x] Tensor utils (`src/tensor_utils.rs`)
- [x] Compat BLAS (`src/compat_blas.rs`)

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

- [x] Continuous thought module (`src/nn/continuous_thought.rs` / `ContinuousThoughtModule`)
- [x] LoopedTransformer (`src/nn/looped_transformer.rs` / `LoopedTransformer` with weight-tied block application, Stage-II gate objective)
- [ ] Linear attention mechanisms
- [ ] Performer (FAVOR+) attention
- [ ] LongRoPE for extended context
- [ ] Ring Attention for infinite context
- [ ] Dynamic sparse attention

### 11.3 Multimodal Advancements

- [x] Unified multimodal architectures (`src/nn/multimodal.rs` / `MultimodalLLM`, `GenerationConfig`, `ModalMemoryContext`)
- [x] Decoders (`src/nn/decoders.rs` / `TextDecoder`, `ImageDecoder`, `VideoDecoder`)
- [x] CLIP (`src/nn/clip.rs` / full CLIP implementation)
- [ ] 3D understanding
- [ ] Video generation models
- [ ] Embodied AI components

## Implementation Priority

### High Priority (Essential for Basic LLM Training/Inference)

1. Hugging Face tokenizers integration (feature-gated wrapper implemented; unit test included `tests/tokenizer_test.rs`)
2. Complete optimizer implementations (Adam, AdamW present; review and expand scheduling/weight-decay)
3. Learning rate schedulers (basic warmup & cosine implemented; other schedules like exponential, polynomial and cyclic
   remain pending)
4. Distributed training primitives (scaffold exists: `src/distributed/` with DataParallel, AllReduce, DistributedCheckpoint)
5. Production-quality quantization support (ongoing: `QuantizedMatMul` implemented and benches added; AWQ module exists at `src/quantization/awq.rs`; block/rowwise quantization formats and runtime support still pending)
6. KV cache optimization (basic KV cache implemented; `PagedKVCache` and paged attention exist at `src/nn/`)
7. Windows builder/runtime alignment for `libtorch` (pin MSVC runtime or build libtorch from source to avoid runtime
   mismatches in CI)

### Medium Priority (Advanced LLM Features)

1. MoE layers (implemented at `src/nn/moe.rs` - SMoE with top-k routing)
2. Flash Attention (FlashAttentionRef CPU reference exists; vendor kernels pending)
3. Gradient checkpointing (implemented at `src/autograd.rs::checkpoint`)
4. Mixed precision training (AMP module exists at `src/amp.rs`)
5. Model parallelism
6. Sparse attention patterns

### Low Priority (Research/Diffusion/Audio)

1. Diffusion model components (UNet skeleton + DDPMScheduler exist at `src/nn/diffusion.rs`; full pipeline pending)
2. Audio processing pipelines (AudioEncoder/AudioDecoder exist at `src/nn/audio.rs`; full pipeline pending)
3. Advanced architectures (Mamba, RetNet)
4. Multimodal fusion layers (MultimodalLLM scaffold exists at `src/nn/multimodal.rs`)
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
  benches updated to include quantized variants (`benches/matmul_bench.rs`). AWQ module exists at `src/quantization/awq.rs`.
  Next: add per-layer quantization helpers, block/rowwise quantization formats (AWQ/GPTQ), runtime support for quantized Conv, and a `quantize_weights` utility.
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

- **New (Jun 2026 inspection findings):**
  - CLIP is fully implemented (`src/nn/clip.rs`) — consider adding CLIP model loading from HuggingFace checkpoints
  - MoE layer is fully implemented (`src/nn/moe.rs`) — add MoE-specific examples and benchmarks
  - LoopedTransformer (`src/nn/looped_transformer.rs`) implements weight-tied transformer with Stage-II gate objective — add example
  - ContinuousThoughtModule (`src/nn/continuous_thought.rs`) exists — needs integration testing
  - PagedKVCache and paged attention (`src/nn/paged_kv_cache.rs`, `src/nn/paged_attention.rs`) — add integration tests
  - Distributed training scaffold (`src/distributed/`) — needs NCCL/RDMA integration for production use
  - Diffusion UNet (`src/nn/diffusion.rs`) is skeleton-level — add encoder/decoder/attention blocks for full pipeline
  - Audio encoder/decoder (`src/nn/audio.rs`) exists — add full text-to-audio pipeline example
  - Server module (`src/server/mod.rs`) is feature-gated — needs production hardening
  - Python bindings (`src/python_bindings.rs`) are feature-gated — needs more coverage
  - Monitoring (`src/monitoring.rs`) is feature-gated — needs metrics export
  - Metal backend (`src/backend/metal/`) is feature-gated — needs full testing
  - CUDA kernels (`src/backend/cuda_kernels.rs`) exist — needs integration
  - OpenBLAS on Windows blocked by `#[cfg(not(target_os = "windows"))]` in `compat_blas.rs`
  - F16/BF16 storage currently emulated via round-trip conversion — real half storage needs `multi_precision` feature
  - rllama compat (`src/compat/rllama/`) and HF compat (`src/hf_compat/`) are full compatibility layers — need ongoing maintenance

These recommendations prioritize integration and small, deliverable steps that enable adoption by broader ML tooling and
developer workflows.
