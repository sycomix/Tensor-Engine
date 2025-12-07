import torch
import torch.nn as nn
import base64

class Simple(nn.Module):
    def __init__(self):
        super(Simple,self).__init__()
        self.l = nn.Linear(4,2)
    def forward(self,x):
        return self.l(x)

# nested via explicit submodule
class SimpleNested2(nn.Module):
    def __init__(self):
        super(SimpleNested2,self).__init__()
        self.nested = Simple()
    def forward(self,x):
        return self.nested(x)

model = SimpleNested2()
traced = torch.jit.trace(model, torch.randn(1,4))
out = 'tests/assets/simple_linear_nested.pt'
traced.save(out)
with open(out,'rb') as f:
    data = f.read()
with open(out + '.b64', 'wb') as f:
    f.write(base64.b64encode(data))
print('wrote nested2', out + '.b64', 'size', len(data))

# alias module a.l = same as l
class HashAlias(nn.Module):
    def __init__(self):
        super(HashAlias,self).__init__()
        self.l = nn.Linear(4,2)
        self.a = self.l
    def forward(self, x):
        return self.l(x)

model = HashAlias()
traced = torch.jit.trace(model, torch.randn(1,4))
out = 'tests/assets/simple_linear_hashmap.pt'
traced.save(out)
with open(out,'rb') as f:
    data = f.read()
with open(out + '.b64', 'wb') as f:
    f.write(base64.b64encode(data))
print('wrote alias', out + '.b64', 'size', len(data))
