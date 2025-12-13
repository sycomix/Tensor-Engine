from importlib import import_module
import sys
sys.path.insert(0, 'D:/Tensor-Engine')
print('Starting isolated test')
import tensor_engine as te
Tensor = getattr(te, 'Tensor', None)
if Tensor is None:
    raise ImportError("No name 'Tensor' in module 'tensor_engine'")
print('1: Basic grad test')
logits = Tensor([1.0, 2.0, -1.0, 0.1, 0.2, 0.3], [2, 3])
targets = Tensor([1.0, 2.0], [2])
loss = logits.softmax_cross_entropy_with_logits(targets)
loss.backward()
print('Loss data', loss.get_data())
assert logits.get_grad() is not None
print('1 done OK')
print('2 NLL test')
logits = Tensor([1.0, 2.0, -1.0], [1, 3])
logits_grad = logits.log_softmax(1)
targets = Tensor([1.0], [1])
NLLLoss = getattr(te, 'NLLLoss', None)
if NLLLoss is not None:
    loss2 = NLLLoss().forward(logits_grad, targets)
    loss2.backward()
    assert logits_grad.get_grad() is not None
    print('2 NLL done')
else:
    print('NLL not available - skipping')
print('3 SoftmaxCrossEntropy')
SoftmaxCrossEntropyLoss = getattr(te, 'SoftmaxCrossEntropyLoss', None)
if SoftmaxCrossEntropyLoss is not None:
    loss3 = SoftmaxCrossEntropyLoss().forward(logits, targets)
    loss3.backward()
    print('Softmax OK')
else:
    print('Skip Softmax')
print('4 CrossEntropyLogitsLoss')
CrossEntropyLogitsLoss = getattr(te, 'CrossEntropyLogitsLoss', None)
if CrossEntropyLogitsLoss is not None:
    cel = CrossEntropyLogitsLoss()
    loss4 = cel.forward(logits, targets)
    loss4.backward()
    print('CrossEnt OK')
else:
    print('Skip CrossEnt')
print('5 Quantized matmul test')
wa = Tensor([1.0, 2.0], [1, 2])
wb = Tensor([1.0, 0.0, 0.0, 1.0], [2, 2])
qw = wb.quantize_weights('i8_rowwise', None)
o = wa.quantized_matmul(qw)
assert o.get_data() is not None
print('Quantize OK')
print('6 getitem tests')
# Now the new tests
print('Starting getitem block')
t = Tensor([1.0, 2.0, 3.0], [3])
print('t', t.get_data())
assert t[0].get_data()[0] == 1.0
assert t[-1].get_data()[0] == 3.0

t2 = Tensor([0.0] * 6, [2, 3])
# set scalar into single index
print('Before assign', t2.get_data())
t2[1, 1] = 42.0
print('After scalar set', t2.get_data())
assert t2[1, 1].get_data()[0] == 42.0
# assign a row
print('Before row assign', t2.get_data())
t2[0, :] = Tensor([1.0, 2.0, 3.0], [3])
print('After row assign', t2.get_data())
assert t2[0, 0].get_data()[0] == 1.0
assert t2[0, 1].get_data()[0] == 2.0
assert t2[0, 2].get_data()[0] == 3.0
print('All getitem/setitem assertions passed')
print('Done all tests OK')
