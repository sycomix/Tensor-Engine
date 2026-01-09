"""Compare BF16->F32 parity between safetensors (python) and TensorEngine loader.

Usage:
  python scripts/check_bf16_parity.py /path/to/model.safetensors [key]

If no key is provided, the script lists keys and chooses the first BF16 or the first tensor.

Output: max absolute difference and a small sample of mismatched elements.
"""
import sys
import numpy as np
from safetensors import safe_open
import tensor_engine as te


def bf16_to_f32_from_uint16(arr_uint16: np.ndarray) -> np.ndarray:
    # arr_uint16 assumed dtype=np.uint16 (little-endian bf16 bit patterns)
    # convert by shifting left 16 bits and reinterpret as float32
    u32 = arr_uint16.astype(np.uint32) << 16
    return u32.view(np.float32)


def load_te_tensor_dict(model_path: str):
    with open(model_path, "rb") as f:
        data = f.read()
    state = te.py_load_safetensors(data, transpose=False)
    return state


def load_safetensors_tensor(model_path: str, key: str):
    """Load tensor using safetensors numpy backend only.

    If the numpy backend cannot decode bfloat16 for this key, raise RuntimeError
    so the caller can skip comparison (no PyTorch fallback per user request).
    """
    try:
        with safe_open(model_path, framework="np") as f:
            arr = f.get_tensor(key)
        return arr
    except TypeError:
        # numpy backend couldn't decode this dtype (likely bfloat16); caller should skip
        raise RuntimeError("safetensors numpy backend cannot decode this tensor (likely bfloat16); no PyTorch fallback")


def sample_values_from_py_tensor(py_tensor, sample_idxs):
    # Given a PyTensor, fetch scalar values at sample indices using __getitem__.
    # sample_idxs: list of index tuples
    vals = []
    for idx in sample_idxs:
        try:
            t = py_tensor.__getitem__(idx)
        except Exception:
            # single index may be provided as an int
            t = py_tensor.__getitem__(idx)
        # t is a PyTensor of scalar; use __str__ to get string of the scalar ndarray and parse
        s = t.__str__()
        # s may look like '1.234' or 'array(1.234)' or 'Tensor(data=..., shape=..., ...)'; try float conversion
        try:
            v = float(s.strip())
        except Exception:
            # fallback: try to extract number from the string
            import re
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if m:
                v = float(m.group(0))
            else:
                raise
        vals.append(v)
    return np.array(vals, dtype=np.float32)



def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_bf16_parity.py /path/to/model.safetensors [key]")
        return
    model = sys.argv[1]
    key_arg = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Opening {model} for parity check")
    with safe_open(model, framework="np") as f:
        keys = f.keys()
        print(f"Found {len(keys)} keys. First 10: {list(keys)[:10]}")

        selected_key = key_arg
        if selected_key is None:
            # prefer bf16 tensors
            found = None
            for k in keys:
                try:
                    arr = f.get_tensor(k)
                except TypeError:
                    # numpy backend may not understand bfloat16; treat as candidate
                    found = found or k
                    continue
                if arr.dtype == np.uint16:
                    found = k
                    break
                if found is None:
                    found = k
            selected_key = found
        print(f"Selected key: {selected_key}")

    te_state = load_te_tensor_dict(model)
    if selected_key is None:
        print("No suitable key found in safetensors file.")
        return
    selected_key = str(selected_key)
    if selected_key not in te_state:
        print(f"Key {selected_key} not present in TensorEngine parsed dict; available keys: {list(te_state.keys())[:10]}")
        # proceed to compare only if present in both
        return
    else:
        te_py_tensor = te_state[selected_key]
        # TensorEngine summary
        print(f"TensorEngine: dtype={te_py_tensor.dtype}, shape={te_py_tensor.shape}")

        # load with safetensors directly (numpy backend only)
        try:
            raw = load_safetensors_tensor(model, selected_key)
        except RuntimeError as e:
            print(f"Skipping comparison for key {selected_key}: {e}")
            return

        # raw is expected to be a numpy array; check dtype
        raw_dtype = getattr(raw, 'dtype', None)
        print(f"safetensors raw dtype: {raw_dtype}, shape: {raw.shape}")
        if raw_dtype is not None and str(raw_dtype).startswith('uint16'):
            # bf16 encoded as uint16 in numpy backend
            sa = bf16_to_f32_from_uint16(raw)
        else:
            # numpy float types
            sa = np.array(raw, dtype=np.float32)
        print(f"safetensors converted: dtype={sa.dtype}, min={sa.min()}, max={sa.max()}")

        # shape check
        if sa.shape != tuple(te_py_tensor.shape):
            print(f"Shape mismatch: safetensors {sa.shape} vs TE {tuple(te_py_tensor.shape)}")
            return

        # sample indices to compare (up to 20 random locations)
        shape = sa.shape
        import random
        total = int(np.prod(shape))
        samples = min(20, total)
        idxs = []
        for _ in range(samples):
            # pick random flat index and convert to multi-index
            fi = random.randrange(total)
            multi = np.unravel_index(fi, shape)
            idxs.append(tuple(int(x) for x in multi))
        te_vals = sample_values_from_py_tensor(te_py_tensor, idxs)
        sa_vals = np.array([sa[idx] for idx in idxs], dtype=np.float32)
        diffs = np.abs(te_vals - sa_vals)
        print(f"sampled {samples} indices; max_abs_diff = {float(np.max(diffs))}, mean_abs_diff = {float(np.mean(diffs))}")
        for i, idx in enumerate(idxs):
            if diffs[i] > 1e-6:
                print(idx, sa_vals[i], te_vals[i], diffs[i])
        if float(np.max(diffs)) == 0.0:
            print("Sampled parity: sampled elements match exactly.")
        else:
            print("Non-zero differences found in sampled elements.")



if __name__ == "__main__":
    main()
