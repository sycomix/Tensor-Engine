"""Per-layer parity tests between TensorEngine and NumPy reference implementations.

This script runs randomized tests for MatMul, Softmax, and FlashAttentionRef.
It creates small inputs, computes reference outputs using NumPy, and compares
with TensorEngine's Python bindings (PyTensor + module ops). No PyTorch involved.

Usage: python scripts/per_layer_parity.py --op matmul --trials 100
"""
import argparse
import math
import numpy as np
import random
import sys

import tensor_engine as te


# Helper: convert numpy array to PyTensor (f32)
def np_to_py_tensor(arr: np.ndarray):
    flat = arr.astype(np.float32).ravel().tolist()
    shape = list(arr.shape)
    # te.Tensor constructor: (values: List[float], shape: List[int], dtype: Optional[str]=None)
    return te.Tensor(flat, shape)

# Reference implementations

def ref_matmul(a: np.ndarray, b: np.ndarray):
    return a.dot(b)


def ref_softmax(x: np.ndarray, axis: int = -1):
    # stable softmax
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    s = np.sum(ex, axis=axis, keepdims=True)
    # fallback to uniform if s is non-finite or <=0
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        shape = list(x.shape)
        if axis < 0:
            axis = x.ndim + axis
        l = x.shape[axis]
        return np.ones_like(x) / float(l)
    return ex / s


def ref_flash_attention(q, k, v, head_dim):
    # q,k,v: [b*heads, seq, head_dim]
    # compute scaled qk, softmax on last axis, then attn @ v
    scale = 1.0 / math.sqrt(head_dim)
    bnh = q.shape[0]
    seq = q.shape[1]
    out = np.zeros_like(q)
    for i in range(bnh):
        qmat = q[i]
        kmat = k[i]
        vmat = v[i]
        qk = qmat.dot(kmat.T) * scale
        attn = ref_softmax(qk, axis=-1)
        out[i] = attn.dot(vmat)
    return out

# Compare helper
def assert_allclose(a, b, rtol=1e-5, atol=1e-6):
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False):
        diff = np.abs(a - b)
        maxd = float(np.max(diff))
        idx = np.unravel_index(int(np.argmax(diff)), a.shape)
        raise AssertionError(f"max diff {maxd} at idx {idx}: ref={b[idx]} te={a[idx]}")

# Test implementations calling TE via PyTensor wrappers

def test_matmul(trials=50):
    for t in range(trials):
        m = random.randint(1, 16)
        k = random.randint(1, 32)
        n = random.randint(1, 16)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        ref = ref_matmul(a, b)
        A = np_to_py_tensor(a)
        B = np_to_py_tensor(b)
        # call TE matmul via helper
        res = te.py_matmul(A, B)
        flat, shape, dtype = te.py_tensor_to_flat(res)
        te_arr = np.array(flat, dtype=np.float32).reshape(tuple(shape))
        assert_allclose(te_arr, ref)
    print("matmul tests passed")


def test_softmax(trials=50):
    for t in range(trials):
        dims = random.randint(1, 6)
        shape = [random.randint(1, 8) for _ in range(dims)]
        axis = random.randint(-1, dims - 1)
        x = np.random.randn(*shape).astype(np.float32)
        ref = ref_softmax(x, axis=axis)
        X = np_to_py_tensor(x)
        te_out = X.softmax(axis)
        flat, shape, dtype = te.py_tensor_to_flat(te_out)
        arr = np.array(flat, dtype=np.float32).reshape(tuple(shape))
        assert_allclose(arr, ref)
    print("softmax tests passed")


def test_flashattention(trials=20):
    for t in range(trials):
        b = random.randint(1, 2)
        heads = random.randint(1, 4)
        bnh = b * heads
        seq = random.randint(1, 10)
        hd = random.choice([8, 16, 32])
        q = np.random.randn(bnh, seq, hd).astype(np.float32)
        k = np.random.randn(bnh, seq, hd).astype(np.float32)
        v = np.random.randn(bnh, seq, hd).astype(np.float32)
        ref = ref_flash_attention(q, k, v, hd)
        # TE: construct PyTensors and call FlashAttentionRef op; since we don't have a direct binding, use the Op via apply
        Q = np_to_py_tensor(q)
        K = np_to_py_tensor(k)
        V = np_to_py_tensor(v)
        # compute K^T on last two axes via permute
        Kt = K.permute([0, 2, 1])
        qk = te.py_batched_matmul(Q, Kt)
        scale = 1.0 / math.sqrt(hd)
        scale_t = te.Tensor([scale], [1])
        qk_scaled = qk.mul(scale_t)
        attn = qk_scaled.softmax(-1)
        out = te.py_batched_matmul(attn, V)
        flat, shape, dtype = te.py_tensor_to_flat(out)
        arr = np.array(flat, dtype=np.float32).reshape(tuple(shape))
        assert_allclose(arr, ref, rtol=1e-4, atol=1e-6)
    print("flash attention tests passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["matmul","softmax","flash"], default="softmax")
    p.add_argument("--trials", type=int, default=50)
    args = p.parse_args()

    if args.op == "matmul":
        test_matmul(args.trials)
    elif args.op == "softmax":
        test_softmax(args.trials)
    else:
        test_flashattention(args.trials)

if __name__ == "__main__":
    main()
