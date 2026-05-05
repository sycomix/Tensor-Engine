    Package Structure


    /tmp/unity-tensor-engine/
    ├── Packages/unity-tensor-engine/
    │   └── package.json                          # Unity package manifest
    ├── Assets/TensorEngine/
    │   ├── Runtime/
    │   │   ├── TensorEngine.asmdef               # C# assembly definition
    │   │   └── Scripts/
    │   │       ├── Core/
    │   │       │   ├── Tensor.cs                 # Multi-dim tensor + 25+ ops (matmul, relu, softmax, etc.)
    │   │       │   └── Layers.cs                 # Linear, Embedding, LayerNorm, RMSNorm
    │   │       ├── Models/
    │   │       │   └── Transformers.cs           # MultiHeadAttention, SwiGLUFFN, TransformerBlock, LlamaDecoder, MultimodalLLM, Rope, VisionTransformer
    │   │       ├── Bridge/
    │   │       │   └── PythonBridgeService.cs    # HTTP bridge to Python inference server
    │   │       └── Agents/
    │   │           ├── NeuralAgent.cs            # AI agent with dialogue/action generation, tokenizer, sampling
    │   │           └── MonoBrain.cs              # Central brain: model management, inference queue, memory, environment
    │   ├── Editor/
    │   │   └── TensorEngine.Editor.asmdef
    │   ├── PythonBridge/
    │   │   ├── services/
    │   │   │   └── inference_server.py           # Flask server for model loading, inference, tensor ops
    │   │   ├── requirements.txt                  # Python deps (flask, numpy, torch, safetensors, transformers)
    │   │   └── setup.py                          # Setup/install script
    │   ├── Examples/
    │   │   ├── Scripts/
    │   │   │   ├── ExampleNPC.cs                 # Basic NPC with neural dialogue + proximity detection
    │   │   │   ├── ExampleProceduralDialogue.cs  # Multi-agent turn-based conversation system
    │   │   │   ├── ExampleNeuralBehavior.cs      # Neural decision-making for behavior selection
    │   │   │   └── ExampleSceneSetup.cs          # Complete demo scene auto-setup
    │   │   └── README.md
    │   └── Docs/
    │       └── README.md                         # Full API reference, architecture diagram, setup guide


    What's Included

    C# Runtime (mirrors Tensor-Engine Rust API):
    - Tensor class with 25+ operations (matmul, relu, gelu, softmax, log_softmax, concat, stack, slice, transpose, embedding lookup)
    - Linear, Embedding, LayerNorm, RMSNorm layers
    - MultiHeadAttention with causal masking and RoPE
    - SwiGLUFFN (Llama-style) and StandardFFN (GELU)
    - TransformerBlock (both standard and Llama variants)
    - LlamaDecoder — full decoder model with weight-tied LM head
    - MultimodalLLM — vision encoder + text decoder

    Unity Integration:
    - PythonBridgeService — manages Flask subprocess, HTTP API for model load/inference/compute
    - NeuralAgent — AI agent with dialogue generation, action generation, sampling (greedy/topK/topP/temp), tokenizer, history
    - MonoBrain — central brain: model registry, inference queue, neural memory, environment context
    - NeuralBehavior — behavior tree with neural selection
    - NeuralMemory — persistent memory system with search

    Examples:
    - ExampleNPC — NPC with dialogue UI, proximity detection, auto-interact
    - ExampleProceduralDialogue — multi-agent conversation system
    - ExampleNeuralBehavior — neural behavior selection
    - ExampleSceneSetup — one-click demo scene creation

    To Use It

    1. Clone Tensor-Engine: git clone https://github.com/sycomix/Tensor-Engine.git
    2. Install Python deps: pip install -r Assets/TensorEngine/PythonBridge/requirements.txt
    3. Copy Packages/unity-tensor-engine and Assets/TensorEngine into your Unity project
    4. Run python3 Assets/TensorEngine/PythonBridge/setup.py
    5. Attach MonoBrain to a GameManager, add NeuralAgent to NPCs, add ExampleNPC for dialogue

    The whole thing is at /tmp/unity-tensor-engine/ — ready to copy into a Unity project.