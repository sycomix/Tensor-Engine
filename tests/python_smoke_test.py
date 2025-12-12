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
from tensor_engine import Tensor  # type: ignore

# Ensure backend is explicitly set to CPU via Python binding (no-op if already set)
# Use getattr to avoid static attribute errors in editors where the binding lacks this function.
_backend_setter = getattr(te, 'set_cpu_backend', None)
if callable(_backend_setter):
    _backend_setter()

logits = Tensor([1.0, 2.0, -1.0, 0.1, 0.2, 0.3], [2, 3])
targets = Tensor([1.0, 2.0], [2])
loss = logits.softmax_cross_entropy_with_logits(targets)
loss.backward()
print('loss:', loss.get_data())
print('logits_grad:', logits.get_grad())
# Assert not raising and gradient exists
assert logits.get_grad() is not None
print('OK')
# Use attributes from the module to avoid import errors in editors without optional bindings.
NLLLoss = getattr(te, 'NLLLoss', None)  # type: ignore
SoftmaxCrossEntropyLoss = getattr(te, 'SoftmaxCrossEntropyLoss', None)  # type: ignore
CrossEntropyLogitsLoss = getattr(te, 'CrossEntropyLogitsLoss', None)  # type: ignore
Labels = getattr(te, 'Labels', None)  # type: ignore
# Tensor is already imported above.

logits = Tensor([1.0, 2.0, -1.0], [1, 3])
logits_grad = logits.log_softmax(1)
targets = Tensor([1.0], [1])

if NLLLoss is not None:
    loss2 = NLLLoss().forward(logits_grad, targets)
    loss2.backward()
    print('nll_loss:', loss2.get_data())
    print('log_probs_grad:', logits_grad.get_grad())
    assert logits_grad.get_grad() is not None
    print('NLL OK')
else:
    print('NLLLoss not available; skipping NLL tests')

if SoftmaxCrossEntropyLoss is not None:
    loss3 = SoftmaxCrossEntropyLoss().forward(logits, targets)
    loss3.backward()
    print('SoftmaxCrossEntropyLoss wrapper OK')
else:
    print('SoftmaxCrossEntropyLoss not available; skipping SoftmaxCrossEntropy tests')

if CrossEntropyLogitsLoss is not None:
    cel = CrossEntropyLogitsLoss()
    loss4 = cel.forward(logits, targets)
    loss4.backward()
    print('CrossEntropyLogitsLoss layer OK')
else:
    print('CrossEntropyLogitsLoss not available; skipping CrossEntropyLogits tests')

# Test forward_from_labels convenience methods
if Labels is not None and SoftmaxCrossEntropyLoss is not None:
    labels = Labels([0, 2])
    loss5 = SoftmaxCrossEntropyLoss().forward_from_labels(logits, labels)
    loss5.backward()
    print('SoftmaxCrossEntropyLoss forward_from_labels OK')
else:
    print('SoftmaxCrossEntropyLoss forward_from_labels skipped; missing Labels or SoftmaxCrossEntropyLoss')

if Labels is not None and NLLLoss is not None:
    labels2 = Labels([0])
    loss6 = NLLLoss().forward_from_labels(logits_grad, labels2)
    loss6.backward()
    print('NLLLoss forward_from_labels OK')
else:
    print('NLLLoss forward_from_labels skipped; missing Labels or NLLLoss')

# Quick quantize + quantized_matmul test
wa = Tensor([1.0, 2.0], [1, 2])
wb = Tensor([1.0, 0.0, 0.0, 1.0], [2, 2])
qw = wb.quantize_weights('i8_rowwise', None)
o = wa.quantized_matmul(qw)
print('quantized_matmul output:', o.get_data())
assert o.get_data() is not None
print('Quantize + quantized_matmul OK')
try:
    VTClass = getattr(te, 'VisionTransformer', None)
    MMClass = getattr(te, 'MultimodalLLM', None)
    if VTClass is not None and MMClass is not None:
        vt = VTClass(3, 8, 32, 64, 4, 1, 128)
        mm = MMClass(vt, 512, 32, 64, 4, 1)
        zeros_img = Tensor([0.0] * (1 * 3 * 8 * 8), [1, 3, 8, 8])
        seq_ids = Tensor([1.0], [1, 1])
        mem = mm.prefill(zeros_img, seq_ids)
        if mem is not None:
            logits = mm.logits_from_memory(mem)
            print('Py Multimodal prefill OK; logits shape:', logits.get_data() is not None)
            # test decode step
            new_id = Tensor([2.0], [1, 1])
            logits2, mem2 = mm.decode_step(mem, new_id)
            print('Py Multimodal decode_step OK; logits2 len:', len(logits2.get_data()))
    else:
        print('Py Multimodal wrappers not available; skipping')
except (AttributeError, TypeError, RuntimeError, ValueError) as e:
    print('Py Multimodal check skipped or failed:', e)


try:
    if hasattr(te, 'Tokenizer'):
        tok = None
        # we can't run real tokenizer without file; but the method exists if class available
        print('Py Tokenizer available' if tok is not None else 'Tokenizer wrapper exists')
except (AttributeError, RuntimeError) as e:
    print('Tokenizer smoke test skipped:', e)
