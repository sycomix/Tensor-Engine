import numpy as np
import tensor_engine as te
from scripts.grad_check import numerical_gradient


def test_matmul_gradients():
    m, k, n = 3, 4, 5
    A = te.Tensor(np.random.randn(m, k).astype(np.float32).ravel().tolist(), [m, k])
    B = te.Tensor(np.random.randn(k, n).astype(np.float32).ravel().tolist(), [k, n])
    # ensure requires_grad true on these tensors (PyTensor constructor sets requires_grad True)

    def loss_fn():
        C = te.py_matmul(A, B)
        # scalar loss: sum of all elements
        vals, shape, _ = te.py_tensor_to_flat(C)
        return float(sum(vals))

    # analytic gradients via backward
    C = te.py_matmul(A, B)
    # scalar loss: sum using TE ops so graph links back to A,B
    loss = C.sum()
    loss.backward()
    # get grads
    ga = A.get_grad()
    gb = B.get_grad()
    # numerical grads
    num_ga = numerical_gradient(A, loss_fn, eps=1e-3)
    num_gb = numerical_gradient(B, loss_fn, eps=1e-3)
    # flatten and compare
    ga_arr = np.array(ga).reshape((m, k))
    gb_arr = np.array(gb).reshape((k, n))
    assert np.allclose(ga_arr, num_ga, rtol=1e-2, atol=1e-3)
    assert np.allclose(gb_arr, num_gb, rtol=1e-2, atol=1e-3)


def test_softmax_gradient():
    x = np.random.randn(4, 6).astype(np.float32)
    X = te.Tensor(x.ravel().tolist(), list(x.shape))

    def loss_fn():
        sm = X.softmax(-1)
        vals, shape, _ = te.py_tensor_to_flat(sm)
        return float(sum(vals))

    sm = X.softmax(-1)
    loss = sm.sum()
    loss.backward()
    gx = X.get_grad()
    num_gx = numerical_gradient(X, loss_fn, eps=1e-3)
    gx_arr = np.array(gx).reshape(x.shape)
    assert np.allclose(gx_arr, num_gx, rtol=1e-2, atol=1e-3)
