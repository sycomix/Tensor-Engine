"""Parity test: compare Tensor-Engine outputs to HuggingFace (PyTorch) reference.

This test is optional (skips if torch/transformers aren't installed) and runs a short prompt
through both models and compares the final hidden states and logits.

Requirements to run locally:
- Python env with torch, transformers, safetensors, numpy, tensor_engine installed from local wheel
- The included model at examples/Llama-3.2-1B/model.safetensors
"""

import numpy as np
import os
import sys
from pathlib import Path

# Ensure `examples` directory is importable so we can import `chat_llama` helper
examples_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(examples_dir))


try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    import tensor_engine as te
    TE_AVAILABLE = True
except Exception:
    TE_AVAILABLE = False


def load_hf(model_dir, device="cpu"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        model.to(device)
        model.eval()
        return tokenizer, model
    except ValueError as exc:
        # Unrecognized HF model type/config; skip tests that require HF reference files
        print(f"HF model load failed (unsupported): {exc}; skipping HF parity tests")
        raise SystemExit(0) from exc


def load_te(model_path):
    # Construct LlamaModel helper from the examples and load weights via the Rust loader
    from chat_llama import load_config_json, LlamaModel
    cfg = load_config_json(Path(model_path))
    model = LlamaModel(cfg)
    model.load_weights(Path(model_path))
    return model


def forward_hf(tokenizer, model, text, device="cpu"):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple(len=num_layers+1) including embeddings as [0]
    hidden_states = [h.cpu().numpy() for h in out.hidden_states]
    logits = out.logits.cpu().numpy()
    return hidden_states, logits, inputs


def forward_te(model, tokenizer, text):
    # Tokenize using HF tokenizer for exact tokenization parity
    toks = tokenizer.encode(text)
    ids_arr = np.array(toks, dtype=np.int32).reshape(1, len(toks))
    # Create te.Tensor for ids using the helper from chat_llama
    from chat_llama import create_tensor, embedding_lookup, stack_tensors
    ids_tensor = create_tensor(ids_arr.ravel().tolist(), [1, len(toks)])

    # Embedding lookup
    x = embedding_lookup(model.tok_emb, ids_tensor)
    # capture embedding as layer 0
    hidden_states = [np.array(x.get_data(), dtype=np.float32).reshape(list(x.shape))]

    # Skip absolute pos embedding when RoPE is used
    if not model.config.use_rope and model.pos_emb is not None:
        pos_ids_data = [float(i) for i in range(len(toks))]
        pos_ids = create_tensor(pos_ids_data, [1, len(toks)])
        pos_emb = embedding_lookup(model.pos_emb, pos_ids)
        pos_emb_batched = stack_tensors([pos_emb], axis=0)  # batch_size=1
        combined = [a + b for a, b in zip(x.get_data(), pos_emb_batched.get_data())]
        x = create_tensor(combined, list(x.shape))
        hidden_states[0] = np.array(x.get_data(), dtype=np.float32).reshape(list(x.shape))

    # Pass through layers and capture each layer output
    for layer in model.layers:
        x = layer.forward(x)
        hidden_states.append(np.array(x.get_data(), dtype=np.float32).reshape(list(x.shape)))

    # capture last hidden
    last_hidden = hidden_states[-1]

    # compute logits via lm_head
    b, seq, hidden = list(x.shape)
    x_flat = x.reshape([b * seq, hidden])
    logits_flat = model.lm_head.forward(x_flat)
    logits = np.array(logits_flat.get_data(), dtype=np.float32).reshape((b, seq, model.config.vocab_size))

    return hidden_states, logits


def test_parity_short_prompt():
    model_dir = "examples/Llama-3.2-1B"
    model_path = os.path.join(model_dir, "model.safetensors")
    text = "Hello world"

    if not HF_AVAILABLE or not TE_AVAILABLE:
        print("Skipping parity test: HF or Tensor-Engine python bindings not available")
        return

    tokenizer, hf_model = load_hf(model_dir)
    te_model = load_te(model_path)

    hf_hidden_states, hf_logits, hf_inputs = forward_hf(tokenizer, hf_model, text)
    te_hidden_states, te_logits = forward_te(te_model, tokenizer, text)

    # Compare shapes for final outputs
    hf_last = hf_hidden_states[-1]
    te_last = te_hidden_states[-1]
    assert hf_last.shape == te_last.shape, f"last_hidden shape mismatch: {hf_last.shape} vs {te_last.shape}"
    assert hf_logits.shape == te_logits.shape, f"logits shape mismatch: {hf_logits.shape} vs {te_logits.shape}"

    # Compare values (relative tolerance); start with loose tolerance then tighten
    max_abs_last = np.max(np.abs(hf_last - te_last))
    max_abs_logits = np.max(np.abs(hf_logits - te_logits))
    print(f"max_abs_last={max_abs_last}, max_abs_logits={max_abs_logits}")

    # Per-layer diagnostics
    n_layers = len(te_hidden_states) - 1
    print("Per-layer max abs diffs:")
    for i in range(len(te_hidden_states)):
        if i >= len(hf_hidden_states):
            print(f"  layer {i}: no HF state (hf len={len(hf_hidden_states)})")
            continue
        a = hf_hidden_states[i]
        b = te_hidden_states[i]
        try:
            d = np.max(np.abs(a - b))
        except Exception as exc:
            d = float('nan')
        print(f"  layer {i:2d}: max_abs_diff = {d}")

    # Inspect first block internals (q/k after RoPE) to localize divergence
    try:
        from chat_llama import create_tensor, embedding_lookup
        toks = tokenizer.encode(text)
        ids_arr = np.array(toks, dtype=np.int32).reshape(1, len(toks))
        ids_tensor = create_tensor(ids_arr.ravel().tolist(), [1, len(toks)])
        x_emb = embedding_lookup(te_model.tok_emb, ids_tensor)
        layer0 = te_model.layers[0]
        dbg = layer0.debug_forward(x_emb)
        te_q_rope = dbg.get('q_rope')
        if te_q_rope is not None:
            te_q_np = np.array(te_q_rope.get_data(), dtype=np.float32).reshape(list(te_q_rope.shape))
            # compute HF q_rope
            hf_emb = hf_hidden_states[0]
            # get q_proj weight from HF model
            q_w = None
            # attempt common LLaMA layout
            try:
                q_w = hf_model.model.layers[0].self_attn.q_proj.weight.detach().cpu().numpy()
            except Exception:
                # fallback to different attribute names
                q_w = hf_model.model.layers[0].attention.q_proj.weight.detach().cpu().numpy()
            # compute q_pre = emb @ Wq.T
            q_pre = np.matmul(hf_emb, q_w.T)
            # compute RoPE in numpy (match ops.rs implementation)
            b, seq, d = q_pre.shape
            num_heads = te_model.config.num_attention_heads
            head_dim = d // num_heads
            theta = te_model.config.rope_theta
            pair = head_dim // 2
            inv_freq = np.array([1.0 / (theta ** ((2.0 * i) / (head_dim))) for i in range(pair)], dtype=np.float32)
            pos = np.arange(seq, dtype=np.float32)[:, None]
            freqs = pos * inv_freq[None, :]
            sin = np.sin(freqs)
            cos = np.cos(freqs)
            # debug shapes
            print(f"ROPE debug: b={b}, seq={seq}, d={d}, num_heads={num_heads}, head_dim={head_dim}, pair={pair}, inv_freq.shape={inv_freq.shape}, sin.shape={sin.shape}")
            # build full sin/cos along head dim
            sin_full = np.repeat(sin, 2, axis=1)
            cos_full = np.repeat(cos, 2, axis=1)
            print(f"sin_full.shape={sin_full.shape}, cos_full.shape={cos_full.shape}")
            # apply rotation
            q_pre_reshaped = q_pre.reshape(b, seq, num_heads, head_dim)
            q_rope_ref = np.empty_like(q_pre_reshaped)
            for bi in range(b):
                for si in range(seq):
                    for h in range(num_heads):
                        vec = q_pre_reshaped[bi, si, h]
                        interleaved = np.empty_like(vec)
                        for pair_i in range(pair):
                            e_idx = 2 * pair_i
                            o_idx = e_idx + 1
                            even = vec[e_idx]
                            odd = vec[o_idx]
                            cosv = cos[si, pair_i]
                            sinv = sin[si, pair_i]
                            out_even = even * cosv - odd * sinv
                            out_odd = even * sinv + odd * cosv
                            interleaved[e_idx] = out_even
                            interleaved[o_idx] = out_odd
                        q_rope_ref[bi, si, h] = interleaved
            q_rope_ref = q_rope_ref.reshape(list(te_q_np.shape))
            q_diff = np.max(np.abs(q_rope_ref - te_q_np))
            print(f"First-block q_rope max_abs_diff = {q_diff}")
            # compute k_rope and scaled logits to compare
            try:
                # show debug keys
                print('DEBUG KEYS:', list(dbg.keys()))
                te_scaled = dbg.get('scaled_logits_final')
                if te_scaled is not None:
                    print('TE scaled tensor shape attr:', tuple(te_scaled.shape), 'len(flat):', len(te_scaled.get_data()))
                # k
                k_w = None
                try:
                    k_w = hf_model.model.layers[0].self_attn.k_proj.weight.detach().cpu().numpy()
                except Exception:
                    k_w = hf_model.model.layers[0].attention.k_proj.weight.detach().cpu().numpy()
                k_pre = np.matmul(hf_emb, k_w.T)
                k_pre_reshaped = k_pre.reshape(b, seq, num_heads, head_dim)
                k_rope_ref = np.empty_like(k_pre_reshaped)
                for bi in range(b):
                    for si in range(seq):
                        for h in range(num_heads):
                            vec = k_pre_reshaped[bi, si, h]
                            interleaved = np.empty_like(vec)
                            for pair_i in range(pair):
                                e_idx = 2 * pair_i
                                o_idx = e_idx + 1
                                even = vec[e_idx]
                                odd = vec[o_idx]
                                cosv = cos[si, pair_i]
                                sinv = sin[si, pair_i]
                                out_even = even * cosv - odd * sinv
                                out_odd = even * sinv + odd * cosv
                                interleaved[e_idx] = out_even
                                interleaved[o_idx] = out_odd
                            k_rope_ref[bi, si, h] = interleaved
                # compute batched matmul q2 @ k2t
                q2 = q_rope_ref.reshape(b * num_heads, seq, head_dim)
                k2 = k_rope_ref.reshape(b * num_heads, seq, head_dim)
                k2t = np.transpose(k2, (0, 2, 1))
                qk = np.matmul(q2, k2t)
                scale = 1.0 / (head_dim ** 0.5)
                scaled = qk * scale
                # reshape to (b,num_heads,seq,seq)
                scaled_logits4 = scaled.reshape(b, num_heads, seq, seq)
                hf_scaled = scaled_logits4
                # compare to te's scaled_logits_final
                te_scaled = dbg.get('scaled_logits_final')
                if te_scaled is not None:
                    te_scaled_np = np.array(te_scaled.get_data(), dtype=np.float32)
                    te_shape = tuple(te_scaled.shape)
                    print(f"TE scaled raw shape: {te_shape}, flattened len={len(te_scaled_np)}")
                    # try possible shapes: (b*num_heads, seq, seq) or (b, num_heads, seq, seq)
                    try:
                        if te_shape == (b, num_heads, seq, seq):
                            te_scaled_reshaped = te_scaled_np.reshape((b, num_heads, seq, seq))
                        elif te_shape == (b * num_heads, seq, seq):
                            te_scaled_reshaped = te_scaled_np.reshape((b, num_heads, seq, seq))
                        elif te_shape == (b * num_heads, seq * seq):
                            te_scaled_reshaped = te_scaled_np.reshape((b, num_heads, seq, seq))
                        else:
                            # as a last resort, try to reshape by inferring one dimension
                            te_scaled_reshaped = te_scaled_np.reshape((b, num_heads, seq, seq))
                        s_diff = np.max(np.abs(hf_scaled - te_scaled_reshaped))
                        print(f"First-block scaled_logits max_abs_diff = {s_diff}")
                    except Exception as exc:
                        print(f"First-block scaled logits shape mismatch: hf={hf_scaled.shape} te_raw_shape={te_shape} err={exc}")
                # compare attention probs
                hf_attn = np.exp(hf_scaled - np.max(hf_scaled, axis=-1, keepdims=True))
                hf_attn = hf_attn / np.sum(hf_attn, axis=-1, keepdims=True)
                te_attn = dbg.get('attn_probs')
                if te_attn is not None:
                    te_attn_np = np.array(te_attn.get_data(), dtype=np.float32).reshape(list(te_attn.shape))
                    if te_attn_np.shape == hf_attn.shape:
                        a_diff = np.max(np.abs(hf_attn - te_attn_np))
                        print(f"First-block attn_probs max_abs_diff = {a_diff}")
                    else:
                        print(f"First-block attn_probs shape mismatch: hf={hf_attn.shape} te={te_attn_np.shape}")
            except Exception as exc:
                print(f"Detailed scaled logits inspection failed: {exc}")
    except Exception as exc:
        print(f"Internal block inspection failed: {exc}")

    # Fail if overall too large
    assert np.allclose(hf_last, te_last, atol=1e-4, rtol=1e-5), "last_hidden diverged beyond tolerance"
    assert np.allclose(hf_logits, te_logits, atol=1e-4, rtol=1e-5), "logits diverged beyond tolerance"


if __name__ == "__main__":
    test_parity_short_prompt()
    print("Parity test finished (passed)")
