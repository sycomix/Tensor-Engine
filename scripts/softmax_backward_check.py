import numpy as np
import tensor_engine as te
x = np.random.randn(4,6).astype(np.float32)
X = te.Tensor(x.ravel().tolist(), list(x.shape))
sm = X.softmax(-1)
sm_vals = np.array(te.py_tensor_to_flat(sm)[0]).reshape(x.shape)
# manual expected gradient for loss=sum(sm) -> dL/dx = y * (1 - sum_j 1*y_j) = y*(1 - 1) = 0
expected = np.zeros_like(sm_vals)
# compute y*(1 - s) using same formula
s = np.sum(sm_vals * 1.0, axis=1, keepdims=True)
manual = sm_vals * (1.0 - s)
print('max abs expected:', np.max(np.abs(expected - manual)))
# now check X.get_grad after backward
loss = sm.sum()
loss.backward()
Xg = np.array(X.get_grad()).reshape(x.shape)
print('max abs analytic:', np.max(np.abs(Xg)))
print('manual max abs:', np.max(np.abs(manual)))
print('sm sample row:', sm_vals[0])
print('manual row:', manual[0])
print('Xg row:', Xg[0])
