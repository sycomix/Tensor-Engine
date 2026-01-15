import numpy as np

import tensor_engine as te

x = np.random.randn(4,6).astype(np.float32)
X = te.Tensor(x.ravel().tolist(), list(x.shape))
sm = X.softmax(-1)
loss = sm.sum()
loss.backward()
print('sm.get_grad() len:', len(sm.get_grad()))
print('sm.get_grad() sample:', sm.get_grad())
print('X.get_grad() len:', len(X.get_grad()))
print('X.get_grad() sample:', X.get_grad())
