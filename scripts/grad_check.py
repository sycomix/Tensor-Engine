"""Utilities for numerical gradient checking with TensorEngine.

Functions:
 - numerical_gradient(param, loss_fn, eps=1e-3) -> numpy array of same shape

Usage: import from tests or scripts to perform finite-diff checks.
"""
import numpy as np


def numerical_gradient(param, loss_fn, eps=1e-3):
    """Compute numerical gradient of scalar loss_fn wrt param PyTensor.

    param: PyTensor (supports get_data(), set_data())
    loss_fn: callable() -> scalar float (should call forward using current param state)
    eps: finite difference step
    """
    import tensor_engine as te
    flat, shape, _ = te.py_tensor_to_flat(param)
    flat = np.array(flat, dtype=np.float32)
    grad = np.zeros_like(flat)
    orig = flat.copy()
    for i in range(len(flat)):
        flat[i] = orig[i] + eps
        param.set_data(flat.tolist())
        lp = float(loss_fn())
        flat[i] = orig[i] - eps
        param.set_data(flat.tolist())
        lm = float(loss_fn())
        grad[i] = (lp - lm) / (2 * eps)
        flat[i] = orig[i]
    # restore
    param.set_data(orig.tolist())
    return grad.reshape(tuple(shape))
