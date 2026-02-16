import os
import sys
import tempfile
import numpy as np
try:
    import tensor_engine as torch
    import tensor_engine.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from safetensors.numpy import save_file

# Add project root to path to find tensor_engine
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    import tensor_engine as te
except ImportError:
    print("Failed to import tensor_engine. Make sure it is installed or built.")
    sys.exit(1)

def test_ci_gen_smoke():
    print("Starting CI Smoke Test...")
    
    # 1. Define Tiny Config
    vocab_size = 128
    d_model = 32
    d_ff = 64
    num_layers = 2
    num_heads = 4
    kv_heads = 4
    
    print(f"Config: vocab={vocab_size}, d_model={d_model}, layers={num_layers}")
    
    # 2. Instantiate Model to get shapes
    model = te.Llama(vocab_size, d_model, num_layers, d_ff, num_heads, kv_heads)
    
    # 3. Generate Random Weights
    # We use numpy for generation to be framework agnostic where possible, 
    # but safetensors.numpy.save_file handles dict of numpy arrays.
    
    tensors = {}
    params = dict(model.named_parameters(""))
    
    print(f"Model has {len(params)} parameters.")
    
    head_dim = d_model // num_heads
    
    for name, py_tensor in params.items():
        # Get shape from PyTensor
        # PyTensor -> Tensor -> lock -> storage -> shape
        # But PyTensor doesn't expose .shape directly in Python?
        # Let's check lib.rs. PyTensor has no .shape binding visible in previous view?
        # But we know the shapes from config!
        
        # Let's deduce shape from name:
        shape = None
        if "embed_tokens" in name:
            shape = (vocab_size, d_model)
        elif "lm_head" in name:
            shape = (d_model, vocab_size)
        elif "norm" in name or "rms_" in name:
            shape = (d_model,)
        elif "q_proj" in name:
            shape = (d_model, num_heads * head_dim)
        elif "k_proj" in name:
            shape = (d_model, kv_heads * head_dim)
        elif "v_proj" in name:
            shape = (d_model, kv_heads * head_dim)
        elif "o_proj" in name:
            # o_proj maps from concatenated heads back to d_model
            shape = (num_heads * head_dim, d_model)
        elif "linear_q" in name or "linear_k" in name or "linear_v" in name or "linear_o" in name:
             # Fallback if internal names leak
             shape = (d_model, d_model)
        elif "linear1" in name:
            # MLP Up/Gate. Llama style: (d_model, 2*d_ff)
            shape = (d_model, 2 * d_ff)
        elif "linear2" in name:
            # MLP Down. (d_ff, d_model)
            shape = (d_ff, d_model)
            
        if shape is None:
            print(f"WARNING: Could not deduce shape for {name}. Using default.")
            continue
            
        # Generate random data
        # Use float32
        data = np.random.randn(*shape).astype(np.float32) * 0.1
        tensors[name] = data
        
    # 4. Save to SafeTensors
    # Create temp file
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    
    print(f"Saving random weights to {path}...")
    try:
        save_file(tensors, path)
        
        # 5. Load back
        print("Loading weights into model...")
        # transpose=False because we generated [in, out] shapes matching TE internals
        if hasattr(te, 'py_load_safetensors_into_module'):
            # Load bytes
            with open(path, "rb") as f:
                mod_bytes = f.read()
                
            te.py_load_safetensors_into_module(mod_bytes, False, model, "")
            print("Load successful.")
        else:
            print("ERROR: te.py_load_safetensors_into_module not found.")
            sys.exit(1)
            
    finally:
        if os.path.exists(path):
            os.remove(path)
            
    # 6. Run Inference
    print("Running Inference Step...")
    model.set_kv_cache(True)
    
    # Input: [1, 5]
    batch_size = 1
    seq_len = 5
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.float32)
    # te.Tensor(value: List[float], shape: List[int], dtype: Optional[str])
    t_input = te.Tensor(input_ids.flatten().tolist(), list(input_ids.shape))
    
    # Forward
    logits = model.forward(t_input)
    print(f"Logits shape: {logits.shape}")
    
    # Verify shape [1, 5, vocab]
    # logits might be PyTensor. Does it have .shape?
    # `chat_batched.py` used `logits.shape`.
    if list(logits.shape) != [batch_size, seq_len, vocab_size]:
        print(f"FAILURE: Expected shape {[batch_size, seq_len, vocab_size]}, got {logits.shape}")
        sys.exit(1)
        
    print("Smoke Test Passed!")

if __name__ == "__main__":
    test_ci_gen_smoke()
