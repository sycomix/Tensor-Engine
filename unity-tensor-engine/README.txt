    Tensor Engine Unity Extension v0.5.0

    Package Structure

    Assets/TensorEngine/
    ├── Runtime/
    │   ├── TensorEngine.asmdef                 # C# assembly definition
    │   └── Scripts/
    │       ├── Core/
    │       │   ├── Tensor.cs                   # Multi-dim tensor + 25+ ops
    │       │   └── Layers.cs                   # Linear, Embedding, LayerNorm, RMSNorm
    │       ├── Models/
    │       │   └── Transformers.cs             # MHA, SwiGLUFFN, TransformerBlock, LlamaDecoder, etc.
    │       ├── Bridge/
    │       │   └── PythonBridgeService.cs      # HTTP bridge to Rust engine.exe (OpenAI API, SSE streaming)
    │       └── Agents/
    │           ├── NeuralAgent.cs              # AI agent with dialogue/action generation, SSE streaming
    │           └── MonoBrain.cs                # Central brain: model discovery, agent management
    ├── Editor/
    │   └── TensorEngine.Editor.asmdef
    ├── PythonBridge/                           # Legacy — kept for reference
    │   ├── services/inference_server.py        # Previously Flask server, now replaced by direct engine.exe bridge
    │   ├── requirements.txt
    │   └── setup.py
    └── Examples/
        └── Scripts/
            ├── ExampleNPC.cs                  # NPC with neural dialogue + proximity detection
            ├── ExampleProceduralDialogue.cs    # Multi-agent turn-based conversation
            ├── ExampleNeuralBehavior.cs        # Neural decision-making for behavior selection
            └── ExampleSceneSetup.cs            # Complete demo scene auto-setup

    Architecture (Updated)

    Unity ───HTTP (OpenAI API)───> Rust engine.exe
           ◄──SSE streaming──────

    The Python Flask bridge has been replaced. PythonBridgeService.cs now:
    - Spawns engine.exe directly as a subprocess
    - Communicates via the Rust engine's native Rocket HTTP server
    - Uses OpenAI-compatible endpoints:
        GET  /v1/models              — discover available models
        POST /v1/chat/completions    — streaming chat (SSE)
        POST /v1/completions         — streaming text completion (SSE)

    What's Included

    C# Runtime (mirrors Tensor-Engine Rust API):
    - Tensor class with 25+ ops
    - Layers: Linear, Embedding, LayerNorm, RMSNorm
    - Transformers: MultiHeadAttention, SwiGLUFFN, TransformerBlock, LlamaDecoder, MultimodalLLM, ViT

    Unity Integration:
    - PythonBridgeService — spawns engine.exe, OpenAI HTTP + SSE streaming
    - NeuralAgent — dialogue/action generation with SSE token streaming
    - MonoBrain — model discovery, agent management
    - NeuralBehavior — behavior tree with neural selection
    - NeuralMemory — persistent memory with keyword search

    Examples:
    - ExampleNPC — NPC with dialogue UI, proximity detection
    - ExampleProceduralDialogue — multi-agent conversation
    - ExampleNeuralBehavior — neural behavior selection
    - ExampleSceneSetup — one-click demo scene creation

    To Use It

    1. Build engine.exe: cd Tensor-Engine && cargo build --release --features opencl,cffi
    2. Copy target/release/engine.exe next to your Unity project or set EnginePath
    3. Copy Packages/unity-tensor-engine and Assets/TensorEngine into your Unity project
    4. Attach MonoBrain to a GameManager, set ModelPath to your model directory
    5. Attach NeuralAgent to NPCs, add ExampleNPC for dialogue UI
    6. Run the scene — MonoBrain auto-starts engine.exe and discovers models
