import torch

m = torch.jit.load('tests/assets/simple_linear_nested.pt')
sd = m.state_dict()
print(type(sd))
print(sorted(sd.keys()))
for k, v in sd.items():
    print('key', k, 'type', type(v))
    if isinstance(v, dict):
        print('nested keys', sorted(list(v.keys())))
