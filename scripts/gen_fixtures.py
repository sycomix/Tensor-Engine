import base64
import logging

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.l = nn.Linear(4, 2)

    def forward(self, x):
        return self.l(x)


# Nested
model = Simple()
traced = torch.jit.trace(model, torch.randn(1, 4))
out = 'tests/assets/simple_linear_nested.pt'
traced.save(out)
with open(out, 'rb') as f:
    data = f.read()
with open(out + '.b64', 'wb') as f:
    f.write(base64.b64encode(data))
logger.info('wrote %s size %d', out + '.b64', len(data))


# list pairs
class PairState(Simple):
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        return list(sd.items())


model = PairState()
traced = torch.jit.trace(model, torch.randn(1, 4))
out = 'tests/assets/simple_linear_pairs.pt'
traced.save(out)
with open(out, 'rb') as f:
    data = f.read()
with open(out + '.b64', 'wb') as f:
    f.write(base64.b64encode(data))
logger.info('wrote %s size %d', out + '.b64', len(data))


# hashmap alias
class HashMapState(Simple):
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        return {'a.weight': sd['l.weight'], 'a.bias': sd['l.bias']}


model = HashMapState()
traced = torch.jit.trace(model, torch.randn(1, 4))
out = 'tests/assets/simple_linear_hashmap.pt'
traced.save(out)
with open(out, 'rb') as f:
    data = f.read()
with open(out + '.b64', 'wb') as f:
    f.write(base64.b64encode(data))
logger.info('wrote %s size %d', out + '.b64', len(data))
