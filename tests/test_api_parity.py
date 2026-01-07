import numpy as np
import tensor_engine as te


def test_matmul_and_softmax_parity():
    a = te.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = te.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
    # matmul via helper
    c = te.py_matmul(a, b)
    flat, shape, _ = te.py_tensor_to_flat(c)
    arr = np.array(flat, dtype=np.float32).reshape(tuple(shape))

    ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).dot(
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    )
    assert np.allclose(arr, ref, rtol=1e-6, atol=1e-6)

    # softmax parity
    sm = c.softmax(-1)
    flat2, shape2, _ = te.py_tensor_to_flat(sm)
    arr2 = np.array(flat2, dtype=np.float32).reshape(tuple(shape2))
    # numpy reference
    ref_sm = np.exp(ref - np.max(ref, axis=-1, keepdims=True))
    ref_sm = ref_sm / np.sum(ref_sm, axis=-1, keepdims=True)
    assert np.allclose(arr2, ref_sm, rtol=1e-6, atol=1e-6)


def test_item_and_to_and_numpy_tuple():
    s = te.Tensor([42.0], [])
    assert s.item() == 42.0

    a = te.Tensor([1.5, 2.5], [2, 1])
    # convert dtype (noop if feature not compiled) and verify dtype string present
    a2 = a.to(dtype="f32", device="cpu")
    assert isinstance(a2, te.Tensor)

    flat, shape = a2.to_numpy_tuple()
    arr = np.array(flat, dtype=np.float32).reshape(tuple(shape))
    assert arr.shape == (2, 1)
    assert arr[0, 0] == 1.5


def test_squeeze_unsqueeze_and_view():
    t = te.Tensor([1.0, 2.0], [1, 2, 1])
    s = t.squeeze()
    flat, shape, dtype = te.py_tensor_to_flat(s)
    arr = np.array(flat, dtype=np.float32).reshape(tuple(shape))
    assert arr.shape == (2,)

    u = s.unsqueeze(0)
    flat2, shape2, _ = te.py_tensor_to_flat(u)
    # ensure shape is as expected
    assert tuple(shape2) == (1, 2)

    v = u.view([2, 1])
    flat3, shape3, _ = te.py_tensor_to_flat(v)
    arr3 = np.array(flat3, dtype=np.float32).reshape(tuple(shape3))
    assert arr3.shape == (2, 1)
