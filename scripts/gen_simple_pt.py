import base64
import os

import torch
import torch.nn as nn

os.makedirs('tests/assets', exist_ok=True)


class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.l = nn.Linear(4, 2)

    def forward(self, x):
        return self.l(x)


model = Simple()
traced = torch.jit.trace(model, torch.randn(1, 4))
out = 'tests/assets/simple_linear.pt'
traced.save(out)
with open(out, 'rb') as f:
    print(base64.b64encode(f.read()).decode('ascii'))
