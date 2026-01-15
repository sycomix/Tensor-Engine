#!/usr/bin/env python3
"""Smoke test: verify LLaMA model respects RoPE config and does not add absolute pos embeddings."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_llama import ModelConfig, LlamaModel


def test_llama_rope_no_pos():
    cfg = ModelConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=256,
        num_attention_heads=8,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        use_rope=True,
    )
    model = LlamaModel(cfg)
    assert model.pos_emb is None, "pos_emb must be None for RoPE-enabled models"

    # create dummy ids with batch=1, seq=4 as a tensor_engine Tensor
    import tensor_engine as te
    ids_arr = np.array([[1, 2, 3, 4]], dtype=np.int32).ravel().tolist()
    ids = te.Tensor(ids_arr, [1, 4])
    # forward should return logits [1, seq, vocab]
    logits = model.forward(ids)
    if hasattr(logits, 'shape'):
        shape = logits.shape
    else:
        # fallback: try get_data length
        data_len = len(logits.get_data())
        shape = (1, 4, data_len // (1 * 4))
    assert shape[0] == 1 and shape[1] == 4 and shape[2] == cfg.vocab_size

    # Also verify that passing NumPy arrays (instead of te.Tensor) works
    ids_np = np.array([[1, 2, 3, 4]], dtype=np.int32)
    logits_np = model.forward(ids_np)
    if hasattr(logits_np, 'shape'):
        shape2 = logits_np.shape
    else:
        data_len2 = len(logits_np.get_data())
        shape2 = (1, 4, data_len2 // (1 * 4))
    assert shape2[0] == 1 and shape2[1] == 4 and shape2[2] == cfg.vocab_size

    print("test_llama_rope_no_pos: OK")


if __name__ == '__main__':
    test_llama_rope_no_pos()