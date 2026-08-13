import numpy as np

import tensor_engine as te


def test_matmul_and_softmax_parity():
    a = te.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = te.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = a.matmul(b)
    arr = np.array(c.get_data(), dtype=np.float32).reshape(c.shape)

    ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).dot(
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    )
    assert np.allclose(arr, ref, rtol=1e-6, atol=1e-6)

    # softmax parity
    sm = c.softmax(1)
    arr2 = np.array(sm.get_data(), dtype=np.float32).reshape(sm.shape)
    # numpy reference
    ref_sm = np.exp(ref - np.max(ref, axis=-1, keepdims=True))
    ref_sm = ref_sm / np.sum(ref_sm, axis=-1, keepdims=True)
    assert np.allclose(arr2, ref_sm, rtol=1e-6, atol=1e-6)


def test_tensor_data_and_reshape_parity():
    a = te.Tensor([1.5, 2.5], [2, 1])
    reshaped = a.reshape([1, 2])
    assert reshaped.shape == [1, 2]
    assert reshaped.get_data() == [1.5, 2.5]
