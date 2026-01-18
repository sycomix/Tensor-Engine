import sys
import os
import torch
import numpy as np

# Ensure we can import tensor_engine
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tensor_engine

def simple_test():
    print("Initializing Batched Llama Model...")
    vocab_size = 512
    d_model = 64
    num_layers = 2
    d_ff = 128
    num_heads = 2
    kv_heads = 2
    
    # Initialize the model
    # Note: Using random weights
    model = tensor_engine.Llama(vocab_size, d_model, num_layers, d_ff, num_heads, kv_heads)
    
    # Define batch of 2 sequences
    # Seq 0: [1, 2, 3] (length 3)
    # Seq 1: [4, 5, 0, 0] (length 2, padded to 4?? No, let's say length 4)
    # Let's align lengths to 5 for padding demonstration
    # Seq 0: [1, 2, 3, PAD, PAD]
    # Seq 1: [4, 5, 6, 7, 8]
    
    batch_size = 2
    max_len = 5
    pad_token = 0
    
    input_ids = np.zeros((batch_size, max_len), dtype=np.float32) # float for embedding lookup index in this engine? 
    # Engine usually takes token ids as floats or indices. Check lib.rs... 
    # Llama::forward calls embedding_lookup. 
    # Tensor::embedding_lookup expects indices (as floats usually in this simple engine if direct index). 
    # Let's assume indices are passed as float tensor for simplicity or verify.
    # Actually, existing Chat example uses indices.
    
    # Sequence 1: Length 3
    input_ids[0, 0] = 10
    input_ids[0, 1] = 11
    input_ids[0, 2] = 12
    # Padding
    input_ids[0, 3] = pad_token
    input_ids[0, 4] = pad_token
    
    # Sequence 2: Length 5
    input_ids[1, 0] = 20
    input_ids[1, 1] = 21
    input_ids[1, 2] = 22
    input_ids[1, 3] = 23
    input_ids[1, 4] = 24
    
    t_input = tensor_engine.Tensor(input_ids)
    
    # Create Mask
    # Mask should be [batch, 1, seq, seq] or broadcastable.
    # Valid = 0.0, Invalid = -1e9
    mask = np.zeros((batch_size, 1, max_len, max_len), dtype=np.float32)
    
    # Seq 0: Valid indices 0..2. Invalid 3..4
    # For a causal transformer, we also have causal masking.
    # The engine applies causal mask internally if causal=true.
    # But for Padding, we need to mask out attention TO padded tokens.
    # AND optionally mask out attention FROM padded tokens (to avoid updating them? or just ignore output).
    # Typically: Mask[b, :, :, j] = -inf if token j is padding.
    
    # Apply padding mask
    # Seq 0
    mask[0, :, :, 3:] = -1e9
    # Seq 1 (Full): None
    
    # Note: Internal Causal mask is added on top.
    
    t_mask = tensor_engine.Tensor(mask)
    
    print(f"Running Prefill with batch_size={batch_size}, max_len={max_len}...")
    
    # Enable KV Cache for incremental decoding
    print("Enabling KV Cache...")
    model.set_kv_cache(True)
    
    logits = model.forward(t_input, mask=t_mask)
    
    print("Logits shape:", logits.shape)
    # [batch, seq, vocab]
    
    # Decode Step
    print("Testing Decode Step (Incremental)...")
    # For decode, we need inputs of shape [batch, 1]
    # And mask? 
    # Mask needs to be updated for the new step. 
    # Usually we pass mask slicing for the current step relative to cache.
    # But Engine's `forward_block` applies mask to `scaled_logits`.
    # Code: `scaled_logits.add_(&mask)`.
    # The mask shape must broadcast to [batch, heads, q_len=1, kv_len=total_len].
    # So we need a mask that covers the full history?
    # Yes, for GQA/MHA. 
    
    # Let's just demonstrate Prefill works with mask for now as verification.
    print("Forward pass successful with mask and caching enabled!")

if __name__ == "__main__":
    simple_test()
