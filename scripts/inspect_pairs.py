import torch
m = torch.jit.load('tests/assets/simple_linear_pairs.pt')
sd = m.state_dict()
print(type(sd))
try:
    print(list(sd.keys()))
except Exception as e:
    print('sd is not mapping; repr:', repr(sd))
    # if sd is list of pairs
    try:
        print('list len', len(sd))
        for el in sd:
            print(type(el), repr(el))
    except Exception as ee:
        print('error inspecting sd as list', ee)
