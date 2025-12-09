"""Python smoke tests for the tensor_engine package.

This file checks a selection of operations exposed to Python through the
PyO3 bindings. To run this with linting in an editor, select the venv in
`.venv` as the Python interpreter so the `tensor_engine` package is resolvable.
"""

# Linting: these tests require the `tensor_engine` package to be installed
# in the selected Python environment; editors without our venv may report
# import errors. Disable a few lint warnings for this test file.
# pylint: disable=import-error,reimported,wrong-import-position,missing-module-docstring,unused-import,no-member,no-name-in-module

import tensor_engine as te  # type: ignore

# Ensure backend is explicitly set to CPU via Python binding (no-op if already set)
te.set_cpu_backend()

logits = te.Tensor([1.0, 2.0, -1.0, 0.1, 0.2, 0.3], [2, 3])
targets = te.Tensor([1.0, 2.0], [2])
loss = logits.softmax_cross_entropy_with_logits(targets)
loss.backward()
print('loss:', loss.get_data())
print('logits_grad:', logits.get_grad())
# Assert not raising and gradient exists
assert logits.get_grad() is not None
print('OK')
from tensor_engine import (  # type: ignore
	NLLLoss,
	SoftmaxCrossEntropyLoss,
	CrossEntropyLogitsLoss,
	Labels,
	Tensor,
)

logits = te.Tensor([1.0, 2.0, -1.0], [1, 3])
logits_grad = logits.log_softmax(1)
targets = te.Tensor([1.0], [1])
loss2 = NLLLoss().forward(logits_grad, targets)
loss2.backward()
print('nll_loss:', loss2.get_data())
print('log_probs_grad:', logits_grad.get_grad())
assert logits_grad.get_grad() is not None
print('NLL OK')
loss3 = SoftmaxCrossEntropyLoss().forward(logits, targets)
loss3.backward()
print('SoftmaxCrossEntropyLoss wrapper OK')
cel = CrossEntropyLogitsLoss()
loss4 = cel.forward(logits, targets)
loss4.backward()
print('CrossEntropyLogitsLoss layer OK')

# Test forward_from_labels convenience methods
labels = Labels([0, 2])
loss5 = SoftmaxCrossEntropyLoss().forward_from_labels(logits, labels)
loss5.backward()
print('SoftmaxCrossEntropyLoss forward_from_labels OK')
labels2 = Labels([0])
loss6 = NLLLoss().forward_from_labels(logits_grad, labels2)
loss6.backward()
print('NLLLoss forward_from_labels OK')

# Quick quantize + quantized_matmul test
wa = Tensor([1.0, 2.0], [1, 2])
wb = Tensor([1.0, 0.0, 0.0, 1.0], [2, 2])
qw = wb.quantize_weights('i8_rowwise', None)
o = wa.quantized_matmul(qw)
print('quantized_matmul output:', o.get_data())
assert o.get_data() is not None
print('Quantize + quantized_matmul OK')
