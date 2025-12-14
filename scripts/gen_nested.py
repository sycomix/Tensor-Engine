import base64

import torch
import torch.nn as nn


# Generate nested state dict
class SimpleNested(nn.Module):
    def __init__(self):
        super(SimpleNested, self).__init__()
        self.l = nn.Linear(4, 2)

    def forward(self, x):
        return self.l(x)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        return {'nested': sd}


model = SimpleNested()
traced = torch.jit.trace(model, torch.randn(1, 4))
out = 'tests/assets/simple_linear_nested.pt'
traced.save(out)
with open(out, 'rb') as f:
    data = f.read()
with open(out + '.b64', 'wb') as f:
    f.write(base64.b64encode(data))
print('wrote nested', out + '.b64', 'size', len(data))
