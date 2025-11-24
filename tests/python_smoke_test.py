import tensor_engine as te
import numpy as np

logits = te.Tensor([1.0, 2.0, -1.0, 0.1, 0.2, 0.3], [2, 3])
targets = te.Tensor([1.0, 2.0], [2])
loss = logits.softmax_cross_entropy_with_logits(targets)
loss.backward()
print('loss:', loss.get_data())
print('logits_grad:', logits.get_grad())
# Assert not raising and gradient exists
assert logits.get_grad() is not None
print('OK')
from tensor_engine import NLLLoss, SoftmaxCrossEntropyLoss

logits = te.Tensor([1.0, 2.0, -1.0], [1, 3])
logits_grad = logits.log_softmax(1)
targets = te.Tensor([1.0], [1])
loss2 = NLLLoss().forward(logits_grad, targets)
loss2.backward()
print('nll_loss:', loss2.get_data())
print('log_probs_grad:', logits_grad.get_grad())
assert logits_grad.get_grad() is not None
print('NLL OK')
from tensor_engine import SoftmaxCrossEntropyLoss
loss3 = SoftmaxCrossEntropyLoss().forward(logits, targets)
loss3.backward()
print('SoftmaxCrossEntropyLoss wrapper OK')
from tensor_engine import CrossEntropyLogitsLoss
cel = CrossEntropyLogitsLoss()
loss4 = cel.forward(logits, targets)
loss4.backward()
print('CrossEntropyLogitsLoss layer OK')

# Test forward_from_labels convenience methods
from tensor_engine import Labels
labels = Labels([0, 2])
loss5 = SoftmaxCrossEntropyLoss().forward_from_labels(logits, labels)
loss5.backward()
print('SoftmaxCrossEntropyLoss forward_from_labels OK')
labels2 = Labels([0])
loss6 = NLLLoss().forward_from_labels(logits_grad, labels2)
loss6.backward()
print('NLLLoss forward_from_labels OK')
