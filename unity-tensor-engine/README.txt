    Tensor Engine Unity Extension v0.5.1

    Package Structure

    Package Manager install path:

    Packages/unity-tensor-engine/
    ├── package.json                              # UPM manifest
    ├── Runtime/                                  # Runtime assembly and scripts
    ├── Editor/                                   # Tensor Engine menu, control panel, inspectors
    ├── PythonBridge/                             # Legacy reference bridge files
    └── Samples~/TensorEngineExamples/            # Importable Unity samples

    Legacy direct-copy path:

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
    │   ├── TensorEngine.Editor.asmdef
    │   └── TensorEngineEditorTools.cs            # Menus, control panel, setup helpers, inspectors
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
    - Tensor Engine menu — Control Panel, setup commands, agent creation, asset refresh
    - Custom inspectors — MonoBrain and PythonBridgeService runtime actions
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
    2. In Unity Package Manager, add the local package at Packages/unity-tensor-engine
    3. Copy target/release/engine.exe next to your Unity project or set Engine Path in Tensor Engine > Control Panel
    4. Ensure the model directory contains tokenizer.model or tokenizer.json
    5. Use Tensor Engine > Control Panel > Create Complete Scene Setup
    6. Select an NPC GameObject and use Tensor Engine > Agents > Add NeuralAgent To Selection
    7. Run the scene — MonoBrain auto-starts engine.exe and discovers models

    Important install note:

    Use either the Package Manager path or the legacy Assets/TensorEngine direct-copy path, not both in the same Unity project.
    Installing both copies at once creates duplicate C# assemblies and Unity will suppress menus until compile errors are resolved.
