    Tensor Engine Unity Extension v1.0.0-beta.1

    Package Structure

    Assets/TensorEngine/
    ├── Runtime/
    │   ├── TensorEngine.asmdef                 # C# assembly definition
    │   └── Scripts/
    │       ├── Core/
    │       │   ├── Tensor.cs                   # Multi-dim tensor + 25+ ops
    │       │   └── Layers.cs                   # Linear, Embedding, LayerNorm, RMSNorm
    │       ├── Models/
    │       │   └── Transformers.cs             # MHA, SwiGLUFFN, TransformerBlock, LlamaDecoder, ViT, Conv2D
    │       ├── Bridge/
    │       │   └── PythonBridgeService.cs      # HTTP bridge to Rust engine.exe (OpenAI API, SSE streaming)
    │       └── Agents/
    │           ├── NeuralAgent.cs              # AI agent with dialogue/action generation, SSE streaming
    │           ├── MonoBrain.cs                # Central brain: model discovery, agent management
    │           └── NeuralEnvironment.cs        # World simulation state (time, weather, events)
    ├── Editor/
    │   └── TensorEngine.Editor.asmdef
    ├── PythonBridge/                           # Legacy — kept for reference
    │   ├── services/inference_server.py        # Previously Flask server, now replaced by direct engine.exe bridge
    │   ├── requirements.txt
    │   └── setup.py
    └── Examples/
        └── Scripts/
            ├── ExampleNPC.cs                  # NPC with neural dialogue + proximity detection
            │   (includes ExampleProceduralDialogue and ExampleNeuralBehavior)
            └── ExampleSceneSetup.cs            # Complete demo scene auto-setup

    Architecture

    Unity ───HTTP (OpenAI API)───> Rust engine.exe
           ◄──SSE streaming──────

    PythonBridgeService.cs:
    - Spawns engine.exe directly as a subprocess
    - Derives tokenizer path from model directory
    - Communicates via the Rust engine's native Rocket HTTP server (:8080 default)
    - Uses OpenAI-compatible endpoints:
        GET  /v1/models              — discover available models
        POST /v1/chat/completions    — streaming chat (SSE)
        POST /v1/completions         — streaming text completion (SSE)
    - Supports temperature, top_p, top_k, repetition_penalty, max_tokens

    What's Included

    C# Runtime (mirrors Tensor-Engine Rust API):
    - Tensor class with 25+ ops (Add, Sub, MatMul, Softmax, ReLU, Gelu, etc.)
    - Layers: Linear, Embedding, LayerNorm, RMSNorm
    - Models: MultiHeadAttention, SwiGLUFFN, StandardFFN, TransformerBlock,
              LlamaDecoder, MultimodalLLM, VisionTransformer, Conv2D

    Unity Integration:
    - PythonBridgeService — spawns engine.exe, OpenAI HTTP + SSE streaming, top_k/repetition_penalty
    - NeuralAgent — dialogue/action generation with SSE token streaming
    - MonoBrain — model discovery, agent management
    - NeuralEnvironment — world simulation state with dynamic events
    - NeuralBehavior — behavior tree with neural selection

    Examples:
    - ExampleNPC — NPC with dialogue UI, proximity detection
    - ExampleProceduralDialogue — multi-agent conversation (in ExampleNPC.cs)
    - ExampleNeuralBehavior — neural behavior selection (in ExampleNPC.cs)
    - ExampleSceneSetup — one-click demo scene creation with NeuralEnvironment

    Dependencies

    - com.unity.nuget.newtonsoft-json (3.2.1) — JSON serialization for OpenAI API

    To Use It

    1. Build engine.exe: cd Tensor-Engine && cargo build --release --features opencl,cffi
    2. Copy target/release/engine.exe next to your Unity project or set EnginePath
    3. Copy Packages/unity-tensor-engine and Assets/TensorEngine into your Unity project
    4. Ensure the model directory contains tokenizer.model or tokenizer.json
    5. Attach MonoBrain to a GameManager, set ModelPath to your model directory
    6. Attach NeuralAgent to NPCs, add ExampleNPC for dialogue UI
    7. Run the scene — MonoBrain auto-starts engine.exe and discovers models
