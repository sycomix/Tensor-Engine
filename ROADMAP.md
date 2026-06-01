# Tensor Engine - Project Roadmap

> Last inspected: 2026-06-01
> Source tree: 130 Rust files across 14 directories, 26 Python examples, 90+ test files

---

## 1. Core Tensor Operations

### 1.1 Tensor Core (src/tensor.rs)
- [x] Tensor struct wrapping Arc<Mutex<TensorData>>
- [x] new(), ones(), zeros(), from_scalar()
- [x] new_with_dtype() - F32/F16/BF16/F8/I8/I8Rowwise/I8Blockwise/U8
- [x] apply() - operation dispatch with autograd graph building
- [x] quantized_matmul() - int8 weight matmul
- [x] quantize_weights() - per-tensor rowwise/blockwise quantization
- [x] shape(), to_vec(), to_f32_array(), dtype(), is_same()
- [x] lock() / Deref to Arc<Mutex<TensorData>>
- [x] build_topo() - topological sort for autograd
- [x] detach(), requires_grad(), set_requires_grad(), zero_grad()
- [x] backward() - delegates to AutogradEngine
- [x] Memory pooling: new_pooled(), zeros_pooled(), ones_pooled()
- [x] reshape(), transpose(), permute()
- [x] rope() - rotary positional embeddings
- [x] concat(), stack()
- [x] batch_norm() with BatchNormConfig
- [x] PartialEq, Eq, Hash implementations

### 1.2 Data Types (src/dtype.rs)
- [x] DType enum: F32, F16, BF16, F8, I8, I8Rowwise, I8Blockwise, U8
- [x] TensorStorage enum: F32, F16, BF16, F8, I8, I8Rowwise, I8Blockwise, U8
- [x] QuantizedMatrix enum for validated 2D quantized weight views
- [x] f16_helpers module: to_f16(), from_f16(), to_bf16(), from_bf16()
- [x] f8 module: quantize_to_f8(), dequantize_from_f8()
- [x] int8 module: quantize_to_i8(), dequantize_from_i8(), rowwise/blockwise variants
- [x] Storage trait impl for TensorStorage
- [x] as_f32_view(), try_as_quantized_matrix_2d()

### 1.3 Operations (src/ops.rs) - 40+ ops
- [x] Element-wise: Add, Sub, Mul, Div, Pow, Neg, Abs, Sign, Sqrt, Rsqrt, Exp, Log, Sin, Cos, Tanh, Sigmoid, Reciprocal, Clamp, Floor, Ceil, Round, Trunc, Frac, IsInf, IsNaN, Tril, Triu
- [x] Reductions: Sum, Mean, Max, Min, Prod, All, Any
- [x] Linear algebra: MatMul, QuantizedMatMul, Determinant, Inverse
- [x] Embedding: EmbeddingLookup, EmbeddingBag
- [x] Activation: ReLU, Sigmoid, Tanh, Softmax, LogSoftmax, GELU, SiLU, SwiGLU, Ternary
- [x] Normalization: LayerNorm, RMSNorm
- [x] Indexing: IndexSelect, Gather, Scatter, ScatterAdd, MaskedScatter
- [x] Array ops: Concat, Stack, PermuteAxes, Slice, Unfold2D, Fold2D
- [x] Search: TopK, Sort, ArgSort
- [x] Loss: CrossEntropyLogits, SoftmaxCrossEntropyLogits, NLLLoss, BinaryCrossEntropy, BinaryCrossEntropyWithLogits
- [x] Image: Interpolate, GridSample, UpSampleNearest2D
- [x] FFT: FFT, IFFT, RFFT, IRFFT
- [x] Other: Where, CumSum, CumProd, CumMax, CumMin, ComplexMul, ComplexConj, BatchNorm
- [x] Checkpoint op for gradient checkpointing
- [x] FlashAttentionRef: CPU reference FlashAttention with forward+backward
- [x] Operation trait: forward(), backward(), as_any()
- [x] reduce_grad_to_shape() helper for broadcasting gradient reduction
- [x] permute_to_last() / permute_back() helpers

### 1.4 Automatic Differentiation (src/autograd.rs)
- [x] AutogradEngine::backward() - topological sort + reverse pass
- [x] Gradient accumulation for shared nodes
- [x] checkpoint() function for gradient checkpointing

### 1.5 Optimizers (src/optim.rs)
- [x] SGD with momentum support
- [x] Adam with bias correction

### 1.6 Learning Rate Schedulers (src/lr_scheduler.rs)
- [x] Learning rate scheduling (file exists)

### 1.7 Memory Pool (src/memory_pool.rs)
- [x] TensorPool for memory reuse

### 1.8 AMP (src/amp.rs)
- [x] Automatic Mixed Precision support

### 1.9 Async Operations (src/async_ops.rs)
- [x] Async operation support (feature-gated)

---

## 2. Neural Network Layers

### 2.1 Module System (src/nn/mod.rs)
- [x] Module trait: forward(), parameters(), named_parameters(), load_state_dict(), set_training(), as_any(), as_any_mut()
- [x] AbsolutePositionalEmbedding
- [x] ConvBlock (Conv2D -> ReLU -> optional MaxPool)
- [x] Generator (GAN)
- [x] Discriminator (GAN)
- [x] RNNCell (Elman RNN)
- [x] LSTMCell with forward_step()
- [x] GRUCell with forward_step()
- [x] BatchNorm1d, BatchNorm2d
- [x] MSELoss, CrossEntropyLoss, CrossEntropyLogitsLoss, NLLLossLayer
- [x] DataLoader (simple in-memory)
- [x] DropPath (stochastic depth)

### 2.2 Transformer (src/nn/transformer.rs)
- [x] MultiHeadAttention - Q/K/V projections, RoPE, ALiBi, NL-OOB biases
- [x] GroupedQueryAttention wrapper (GQA)
- [x] CrossAttention wrapper
- [x] SlidingWindowAttention wrapper (Mistral-style)
- [x] TransformerBlock - decoder and encoder variants
- [x] GPTDecoder - token embedding, position embedding, blocks, ln, lm_head
- [x] BERTEncoder - token/position/token_type embeddings, pooler
- [x] EncoderDecoderTransformer
- [x] T5EncoderDecoder
- [x] Llama model
- [x] TransformerConfig
- [x] compute_alibi_slopes()
- [x] AttentionVariant enum: Baseline, FlashRef, Chunked, SlidingWindow
- [x] BiasFunction enum: Logarithmic, Gaussian

### 2.3 Convolution (src/nn/conv.rs)
- [x] Conv1D
- [x] Conv2D
- [x] ConvTranspose1D
- [x] ConvTranspose2D

### 2.4 Embedding (src/nn/embedding.rs)
- [x] Embedding layer
- [x] EmbeddingBag

### 2.5 Linear Dispatch (src/nn/linear_dispatch.rs)
- [x] LinearLayer with dispatch for quantized/f32 paths

### 2.6 MoE (src/nn/moe.rs)
- [x] Mixture of Experts layer

### 2.7 KV Cache (src/nn/kv_cache.rs)
- [x] KVCache for incremental decoding

### 2.8 Paged KV Cache (src/nn/paged_kv_cache.rs)
- [x] PagedKVCache

### 2.9 Paged Attention (src/nn/paged_attention.rs)
- [x] Paged attention implementation

### 2.10 Quantization (src/nn/quantization.rs)
- [x] RVQ (Residual Vector Quantization)

### 2.11 Quantized Modules (src/nn/quantized.rs)
- [x] Quantized module implementations

### 2.12 CLIP (src/nn/clip.rs)
- [x] CLIP model components

### 2.13 Looped Transformer (src/nn/looped_transformer.rs)
- [x] LoopedTransformer

### 2.14 Continuous Thought (src/nn/continuous_thought.rs)
- [x] ContinuousThoughtModule

### 2.15 Decoders (src/nn/decoders.rs)
- [x] TextDecoder, ImageDecoder, VideoDecoder

### 2.16 Multimodal (src/nn/multimodal.rs)
- [x] MultimodalLLM, GenerationConfig, ModalMemoryContext, get_decode_count(), reset_decode_count()

### 2.17 Vision (src/nn/vision.rs)
- [x] VisionTransformer

### 2.18 Latent (src/nn/latent.rs)
- [x] Latent utilities

### 2.19 Multi-Head Attention Module (src/nn/multi_head_attention_module.rs)
- [x] MultiHeadAttention variant

### 2.20 Flatten (src/nn/flatten.rs)
- [x] Flatten utilities

### 2.21 NN Migration (src/nn/nn_migrated_from_old_rs.rs)
- [x] Migration layer from old Rust code

### 2.22 NN Tests (src/nn/tests/)
- [x] 20+ test files: bert, clip, cross-attention, droppath, flatten, gpt, gqa, kv-cache, looped, mha, multimodal, optimizer, paged-attention, reshape, scheduler, sliding-window, t5, transformer, vision

---

## 3. Model Architectures

### 3.1 Language Models
- [x] Llama
- [x] GPTDecoder
- [x] BERTEncoder
- [x] EncoderDecoderTransformer
- [x] T5EncoderDecoder

### 3.2 Multimodal
- [x] MultimodalLLM (Kronos format)
- [x] VisionTransformer
- [x] CLIP components

### 3.3 Diffusion
- [x] UNetModel
- [x] ResNetBlock
- [x] GroupNorm
- [x] TimestepEmbedding
- [x] DDPMScheduler

### 3.4 Audio
- [x] AudioEncoder (Conv1D stack)
- [x] AudioDecoder (ConvTranspose1D stack)

### 3.5 Other
- [x] Generator (GAN)
- [x] Discriminator (GAN)
- [x] RNNCell, LSTMCell, GRUCell

---

## 4. Training Infrastructure

### 4.1 Autograd
- [x] AutogradEngine with topological sort
- [x] Gradient checkpointing

### 4.2 Optimizers
- [x] SGD with momentum
- [x] Adam with bias correction

### 4.3 LR Schedulers
- [x] Learning rate scheduling

### 4.4 AMP
- [x] Automatic Mixed Precision

### 4.5 Distributed Training (src/distributed/)
- [x] DistributedContext - rank/world management
- [x] DataParallel - batch sharding
- [x] AllReduce / ReduceOp - gradient sync
- [x] DistributedCheckpoint - distributed save/load
- [x] DistributedConfig, DeviceId

---

## 5. Data Loading and Tokenization

### 5.1 SafeTensors (src/io/safetensors_loader.rs)
- [x] load_safetensors_from_bytes() - F32/F16/BF16/U16
- [x] apply_state_dict_to_module() - with 3D->2D transpose fallback
- [x] apply_safetensors_bytes_to_module_bytes()
- [x] apply_kronos_bytes_to_module_bytes() - Kronos format detection
- [x] save_module_to_safetensors_bytes()
- [x] augment_state_dict_for_compat() - HF key mapping
- [x] apply_kronos_bytes_to_module_bytes() - re-exported from lib

### 5.2 Tokenizers (src/io/tokenizers.rs)
- [x] HuggingFace tokenizers integration (feature-gated)

### 5.3 PyTorch Loader (src/io/pytorch_loader.rs)
- [x] PyTorch model loading (feature-gated)

### 5.4 Image (src/io/image.rs)
- [x] Image loading (feature-gated)

### 5.5 Image-Text Dataloader (src/io/image_text_dataloader.rs)
- [x] Image-text dataloader

### 5.6 Audio Dataloader (src/io/audio.rs)
- [x] Audio dataloader

### 5.7 Tokenizer (src/tokenizer.rs)
- [x] Tokenizer utilities

---

## 6. Model Loading and Saving

### 6.1 SafeTensors
- [x] Load/save F32/F16/BF16
- [x] State dict application with fallback
- [x] Kronos format support

### 6.2 HuggingFace Compat (src/hf_compat/)
- [x] data_source.rs - data source handling
- [x] embedding.rs - embedding loading
- [x] huggingface_loader.rs - HF model loading
- [x] tokenizer.rs - tokenizer compat
- [x] token_sampler.rs - token sampling compat
- [x] transformer.rs - transformer compat
- [x] unpickler.rs - pickle unpickling

### 6.3 rllama Compat (src/compat/rllama/)
- [x] data_source.rs
- [x] embedding.rs
- [x] entrypoint.rs
- [x] huggingface_loader.rs
- [x] model_params.rs
- [x] semaphore.rs
- [x] simd_support.rs
- [x] tensor.rs
- [x] tensor_opencl_support.rs
- [x] tokenizer.rs
- [x] token_sampler.rs
- [x] transformer.rs
- [x] unpickler.rs
- [x] weight_compression.rs
- [x] protomodels/ - sentencepiece model parsing
- [x] benches/benchmark.rs

---

## 7. Inference Optimization

### 7.1 Sampling (src/generation/sampling.rs)
- [x] Sampling strategies

### 7.2 Speculative Decoding (src/generation/speculative.rs)
- [x] Speculative decoding

### 7.3 KV Cache
- [x] KVCache
- [x] PagedKVCache
- [x] Paged attention

### 7.4 Quantization
- [x] RVQ
- [x] AWQ (src/quantization/awq.rs)
- [x] Quantized modules
- [x] Per-tensor/rowwise/blockwise int8

### 7.5 Backend Dispatch
- [x] CPU backend
- [x] WGPU backend (feature-gated)
- [x] GPU memory management (feature-gated)

### 7.6 Async Operations
- [x] Async operation support

---

## 8. Diffusion Models

### 8.1 UNet (src/nn/diffusion.rs)
- [x] UNetModel - ResNet block stack with time embedding injection
- [x] ResNetBlock - GroupNorm -> SiLU -> Conv2D + time injection
- [x] GroupNorm - NCHW group normalization
- [x] TimestepEmbedding - sinusoidal + linear
- [x] DDPMScheduler - linear beta schedule, q_sample, predict_eps, step

### 8.2 Examples
- [x] examples/sample_diffusion.rs
- [x] examples/train_diffusion.py

---

## 9. Audio Generation

### 9.1 Audio Modules (src/nn/audio.rs)
- [x] AudioEncoder - Conv1D downsampling stack
- [x] AudioDecoder - ConvTranspose1D upsampling stack

### 9.2 Audio Dataloader (src/io/audio.rs)
- [x] Audio dataloader

### 9.3 Examples
- [x] examples/text_to_audio.py
- [x] examples/train_codec.py

---

## 10. Utilities and Tools

### 10.1 Config (src/config.rs)
- [x] Configuration management

### 10.2 Error Handling (src/error.rs)
- [x] Error types and handling

### 10.3 Labels (src/labels.rs)
- [x] Labels utilities

### 10.4 Tensor Utils (src/tensor_utils.rs)
- [x] Tensor utility functions

### 10.5 Monitoring (src/monitoring.rs)
- [x] Monitoring/metrics (feature-gated)

### 10.6 Server (src/server/mod.rs)
- [x] Server module (feature-gated)

### 10.7 HF Bridge (src/hf_bridge.rs)
- [x] HuggingFace bridge (feature-gated)

### 10.8 Python Bindings (src/python_bindings.rs)
- [x] Python bindings (feature-gated)

### 10.9 Compat BLAS (src/compat_blas.rs)
- [x] BLAS compatibility layer

---

## 11. Research and Advanced Features

### 11.1 Continuous Thought
- [x] ContinuousThoughtModule

### 11.2 Mixture of Experts
- [x] MoE layer

### 11.3 CLIP
- [x] CLIP model components

### 11.4 Looped Transformer
- [x] LoopedTransformer

### 11.5 Decoders
- [x] TextDecoder, ImageDecoder, VideoDecoder

### 11.6 Latent Utilities
- [x] Latent utilities

---

## 12. Examples (26 Python + 10 Rust)

### Python Examples
- [x] chat_batched.py, chat_llama.py, chat_safetensors.py
- [x] convert_torch_to_safetensors.py
- [x] data_curation.py, prepare_dataset.py
- [x] deepdream_clip.py
- [x] diagnose_llama.py
- [x] generate_llava.py, train_llava.py
- [x] linear_regression.py, matrix_multiply.py
- [x] load_model.py
- [x] phase1_encoders.py, phase2_fusion.py, phase3_integration.py
- [x] run_demo_chat.py, server_example.py, simple_server_example.py
- [x] sweep_sampling.py
- [x] test_tokenizer.py
- [x] train_looplm.py, train_multimodal.py, train_nl_oob.py
- [x] transformer_demo.py
- [x] sample_diffusion.py, train_diffusion.py
- [x] text_to_audio.py, train_codec.py

### Rust Examples
- [x] backend_demo.rs, blas_check.rs
- [x] distributed_verification.rs
- [x] gru_demo.rs, loss_functions_demo.rs
- [x] mnist_parity.rs
- [x] sample_diffusion.rs, speculative_decoding.rs
- [x] scheduler_demo.rs
- [x] examples/server/main.rs

### Project Subdirectories
- [x] finetune_project/ - finetune examples
- [x] llava_project/ - LLaVA project files
- [x] lora_project/ - LoRA examples
- [x] pretrain_project/ - pretraining examples
- [x] NL-OOB/ - NL-OOB implementation

---

## 13. Tests (90+ files)

### Rust Tests
- [x] alibi_test.rs, as_any_mut_verification.rs
- [x] attention_cross_test.rs, attention_grad_parity_test.rs, attention_variants_test.rs
- [x] audio_multimodal_test.rs
- [x] autograd_test.rs
- [x] awq_integration_test.rs
- [x] backend_test.rs
- [x] batched_decode_test.rs
- [x] blas_matmul_test.rs
- [x] checkpoint_test.rs
- [x] complex_ops_test.rs
- [x] conv1d_transpose_test.rs
- [x] cumulative_ops_test.rs
- [x] dataloader_resample_test.rs
- [x] distributed_test.rs
- [x] dtype_tests.rs
- [x] embedding_bag_test.rs, embedding_oob_test.rs, embedding_test.rs
- [x] embed_transpose_test.rs
- [x] error_handling_tests.rs
- [x] fft_ops_test.rs
- [x] gather_test.rs
- [x] generation_test.rs
- [x] hf_bridge_sampler_tests.rs, hf_bridge_tests.rs
- [x] index_select_test.rs
- [x] kronos_loader_test.rs
- [x] linear_algebra_ops_test.rs
- [x] llama_forward_test.rs, llama_style_transformer_test.rs
- [x] llava_smoke_test.rs
- [x] lock_ordering_test.rs
- [x] masked_scatter_test.rs
- [x] matmul_shape_test.rs
- [x] metal_backend_test.rs
- [x] mha_shape_test.rs
- [x] mlp_concat_test.rs
- [x] moe_test.rs
- [x] multimodal_prefill_decode.rs, multimodal_prefill_parity.rs
- [x] new_ops_test.rs
- [x] nl_oob_test.rs
- [x] nn_extra_test.rs
- [x] optimizer_test.rs
- [x] phase3_tests.rs
- [x] pytorch_loader_test.rs
- [x] quantization_test.rs, quantized_loading_test.rs, quantized_matmul_test.rs
- [x] restored_tensor_test.rs
- [x] rvq_dequantize_test.rs, rvq_ema_test.rs, rvq_test.rs
- [x] safetensors_heuristics_test.rs, safetensors_loader_full.rs, safetensors_test.rs
- [x] scatter_ops_test.rs
- [x] softmax_backward.rs
- [x] sort_ops_test.rs
- [x] state_dict_default.rs
- [x] tiny_train_convergence.rs
- [x] tokenizer_test.rs
- [x] transformer_llama_numeric_grad.rs
- [x] unfold_fold_ops_test.rs
- [x] wgpu_backend_test.rs, wgpu_quantized_test.rs
- [x] where_ops_test.rs

### Python Tests
- [x] python_llava_smoke.py, python_looplm_smoke.py, python_smoke_test.py
- [x] test_api_parity.py, test_bce.py, test_chat_safetensors_smoke.py
- [x] test_ci_gen_smoke.py, test_gradients.py

### Benchmarks
- [x] benches/matmul_bench.rs
- [x] benches/safetensors_bench.rs

---

## 14. Configuration

- [x] configs/multimodal_v1.json - Multimodal config

---

## 15. Documentation

- [x] README.md - Project readme
- [x] README_SERVER.md - Server documentation
- [x] mkdocs.yml - MkDocs configuration
- [x] docs/ - Documentation directory
- [x] site/ - Built site
- [x] kronos-modal-format.md - Kronos format spec
- [x] next.md - Next steps planning
- [x] rules.md - Project rules
- [x] .github/copilot-instructions.md - Copilot instructions

---

## 16. Build and Deployment

### Build System
- [x] Cargo.toml - Rust package manifest
- [x] Cargo.lock - Dependency lockfile
- [x] build.rs - Build script
- [x] pyproject.toml - Python package config
- [x] pyrightconfig.json - Python type checking config
- [x] pytest.ini - pytest config

### Docker
- [x] Dockerfile.debian - Debian Docker image
- [x] .dockerignore

### CI/CD
- [x] .github/workflows/rules-enforcement.yml - Rules enforcement workflow

### Installation
- [x] install.sh - Linux/macOS installer
- [x] Unity-TE-setup.sh - Unity setup (Linux/macOS)
- [x] Unity-TE-setup.ps1 - Unity setup (Windows)

### Configuration
- [x] .cargo/config.toml - Cargo configuration
- [x] .gitignore

---

## 17. External Dependencies and Integrations

### External Libraries
- [x] OpenBLAS-0.3.30-x64-64/ - BLAS library (Windows)
- [x] vendor/ - External dependencies
- [x] xla/ - XLA integration

### Unity Integration
- [x] unity-tensor-engine/ - Unity plugin

### Python Environment
- [x] .venv/ - Python virtual environment
- [x] .pytest_cache/ - pytest cache

---

## 18. Gaps and TODO

### Missing / Partial
- [ ] Metal backend - src/backend/metal/ exists but feature-gated; needs full testing
- [ ] CUDA kernels - src/backend/cuda_kernels.rs exists; needs integration
- [ ] GPU memory management - src/backend/gpu_memory.rs exists; feature-gated
- [ ] OpenBLAS on Windows - compat_blas.rs has not(target_os = "windows") guard; Windows BLAS needs work
- [ ] F16/BF16 actual storage - Currently emulated via round-trip conversion; real half storage needs multi_precision feature
- [ ] Server module - Feature-gated; needs production hardening
- [ ] Python bindings - Feature-gated; needs more coverage
- [ ] Monitoring - Feature-gated; needs metrics export
- [ ] Async ops - Feature-gated; needs more operations
- [ ] Distributed training - Scaffold exists; needs NCCL/RDMA integration
- [ ] Diffusion UNet - Skeleton only; missing encoder/decoder/attention blocks
- [ ] Audio generation - Encoder/decoder exist; full pipeline incomplete
- [ ] Vision transformer - Exists; needs multimodal integration testing
- [ ] LoRA support - Examples exist; core implementation needs verification
- [ ] AWQ integration - Module exists; needs full pipeline testing
- [ ] rllama compat - Full compatibility layer; needs ongoing maintenance
- [ ] HF compat - Full compatibility layer; needs ongoing maintenance

### Performance
- [ ] WGPU backend matmul optimization
- [ ] Quantized matmul kernel optimization
- [ ] Paged attention memory efficiency
- [ ] Batched inference throughput

### Testing
- [ ] Metal backend test coverage
- [ ] CUDA backend test coverage
- [ ] Distributed training integration tests
- [ ] End-to-end diffusion pipeline tests
- [ ] End-to-end audio generation tests
