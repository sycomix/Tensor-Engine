import torch

m = torch.jit.load('tests/assets/simple_linear_nested.pt')
sd = m.state_dict()
print('repr sd:', repr(sd))
try:
    keys = list(sd.keys())
    print('keys', keys)
    for k in keys:
        v = sd[k]
        print('key', k, 'type(v):', type(v))
        if isinstance(v, dict):
            print('innerkeys', list(v.keys()))
except Exception as e:
    print('Exception accessing sd keys:', e)
