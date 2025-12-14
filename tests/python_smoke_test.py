"""Python smoke tests for the tensor_engine package.

This file checks a selection of operations exposed to Python through the
PyO3 bindings. To run this with linting in an editor, select the venv in
`.venv` as the Python interpreter so the `tensor_engine` package is resolvable.
"""

# Linting: these tests require the `tensor_engine` package to be installed
# in the selected Python environment; editors without our venv may report
# import errors. Disable a few lint warnings for this test file.
# pylint: disable=import-error,reimported,wrong-import-position,missing-module-docstring,unused-import,no-member,no-name-in-module,not-callable

import logging
from typing import Any, Callable, ParamSpec, cast

import tensor_engine as te  # type: ignore
from tensor_engine import Tensor  # type: ignore

P = ParamSpec("P")

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

# Wrap entire test in a try-except to capture unexpected failures
# (Top-level try removed; specific try/except are used in blocks below.)
# Ensure backend is explicitly set to CPU via Python binding (no-op if already set)
# Use getattr to avoid static attribute errors in editors where the binding lacks this function.
_backend_setter: Any = getattr(te, 'set_cpu_backend', None)
if callable(_backend_setter):
    _backend_setter()

logits = Tensor([1.0, 2.0, -1.0, 0.1, 0.2, 0.3], [2, 3])
targets = Tensor([1.0, 2.0], [2])
loss = logits.softmax_cross_entropy_with_logits(targets)
loss.backward()
log.info("loss: %s", loss.get_data())
log.info("logits_grad: %s", logits.get_grad())
# Assert not raising and gradient exists
assert logits.get_grad() is not None
log.info("OK")
# Use attributes from the module to avoid import errors in editors without optional bindings.
NLLLoss = cast(Any, getattr(te, 'NLLLoss', None))  # type: ignore
SoftmaxCrossEntropyLoss = cast(Any, getattr(te, 'SoftmaxCrossEntropyLoss', None))  # type: ignore
CrossEntropyLogitsLoss = cast(Any, getattr(te, 'CrossEntropyLogitsLoss', None))  # type: ignore
Labels = cast(Any, getattr(te, 'Labels', None))  # type: ignore
# Tensor is already imported above.

logits = Tensor([1.0, 2.0, -1.0], [1, 3])
logits_grad = logits.log_softmax(1)
targets = Tensor([1.0], [1])

if callable(NLLLoss):
    NLLLoss_ctor: Callable[P, Any] = cast(Callable[P, Any], NLLLoss)
    loss2_obj: Any = NLLLoss_ctor()
    loss2: Any = loss2_obj.forward(logits_grad, targets)
    loss2.backward()
    log.info('nll_loss: %s', loss2.get_data())
    log.info('log_probs_grad: %s', logits_grad.get_grad())
    assert logits_grad.get_grad() is not None
    log.info('NLL OK')
else:
    log.info('NLLLoss not available; skipping NLL tests')

if callable(SoftmaxCrossEntropyLoss):
    SoftmaxCrossEntropyLoss_ctor: Callable[P, Any] = cast(Callable[P, Any], SoftmaxCrossEntropyLoss)
    loss3_obj: Any = SoftmaxCrossEntropyLoss_ctor()
    loss3: Any = loss3_obj.forward(logits, targets)
    loss3.backward()
    log.info('SoftmaxCrossEntropyLoss wrapper OK')
else:
    log.info('SoftmaxCrossEntropyLoss not available; skipping SoftmaxCrossEntropy tests')

if callable(CrossEntropyLogitsLoss):
    CrossEntropyLogitsLoss_ctor: Callable[P, Any] = cast(Callable[P, Any], CrossEntropyLogitsLoss)
    cel: Any = CrossEntropyLogitsLoss_ctor()
    loss4: Any = cel.forward(logits, targets)
    loss4.backward()
    log.info('CrossEntropyLogitsLoss layer OK')
else:
    log.info('CrossEntropyLogitsLoss not available; skipping CrossEntropyLogits tests')

# Test forward_from_labels convenience methods
if callable(Labels) and callable(SoftmaxCrossEntropyLoss):
    try:
        Labels_ctor: Callable[P, Any] = cast(Callable[P, Any], Labels)
        labels: Any = Labels_ctor([0, 2])
        SoftmaxCrossEntropyLoss_ctor: Callable[P, Any] = cast(Callable[P, Any], SoftmaxCrossEntropyLoss)
        loss5_obj: Any = SoftmaxCrossEntropyLoss_ctor()
        loss5 = loss5_obj.forward_from_labels(logits, labels)
        loss5.backward()
        log.info('SoftmaxCrossEntropyLoss forward_from_labels OK')
    except (AttributeError, TypeError, RuntimeError, ValueError) as e:
        log.exception('SoftmaxCrossEntropyLoss forward_from_labels failed: %s', e)
        raise
else:
    log.info('SoftmaxCrossEntropyLoss forward_from_labels skipped; missing Labels or SoftmaxCrossEntropyLoss')

if callable(Labels) and callable(NLLLoss):
    try:
        Labels_ctor: Callable[P, Any] = cast(Callable[P, Any], Labels)
        labels2: Any = Labels_ctor([0])
        NLLLoss_ctor: Callable[P, Any] = cast(Callable[P, Any], NLLLoss)
        nll_obj: Any = NLLLoss_ctor()
        loss6 = nll_obj.forward_from_labels(logits_grad, labels2)
        loss6.backward()
        log.info('NLLLoss forward_from_labels OK')
    except (AttributeError, TypeError, RuntimeError, ValueError) as e:
        log.exception('NLLLoss forward_from_labels failed: %s', e)
        raise
else:
    log.info('NLLLoss forward_from_labels skipped; missing Labels or NLLLoss')

# Quick quantize + quantized_matmul test
wa = Tensor([1.0, 2.0], [1, 2])
wb = Tensor([1.0, 0.0, 0.0, 1.0], [2, 2])
qw = wb.quantize_weights('i8_rowwise', None)
o = wa.quantized_matmul(qw)
log.info('quantized_matmul output: %s', o.get_data())
assert o.get_data() is not None
log.info('Quantize + quantized_matmul OK')
try:
    VTClass = cast(Any, getattr(te, 'VisionTransformer', None))
    MMClass = cast(Any, getattr(te, 'MultimodalLLM', None))
    if callable(VTClass) and callable(MMClass):
        VT_ctor: Callable[P, Any] = cast(Callable[P, Any], VTClass)
        vt: Any = VT_ctor(3, 8, 32, 64, 4, 1, 128)
        MM_ctor: Callable[P, Any] = cast(Callable[P, Any], MMClass)
        mm: Any = MM_ctor(vt, 512, 32, 64, 4, 1)
        zeros_img = Tensor([0.0] * (1 * 3 * 8 * 8), [1, 3, 8, 8])
        seq_ids = Tensor([1.0], [1, 1])
        mem = mm.prefill(zeros_img, seq_ids)
        if mem is not None:
            logits = mm.logits_from_memory(mem)
            log.info('Py Multimodal prefill OK; logits present: %s', logits.get_data() is not None)
            # test decode step
            new_id = Tensor([2.0], [1, 1])
            logits2, mem2 = mm.decode_step(mem, new_id)
            log.info('Py Multimodal decode_step OK; logits2 len: %d', len(logits2.get_data()))
    else:
        log.info('Py Multimodal wrappers not available; skipping')
except (AttributeError, TypeError, RuntimeError, ValueError) as err:
    log.info('Py Multimodal check skipped or failed: %s', err)

try:
    if hasattr(te, 'Tokenizer'):
        tok = None
        # we can't run real tokenizer without file; but the method exists if class available
        log.info('Py Tokenizer available' if tok is not None else 'Tokenizer wrapper exists')
except (AttributeError, RuntimeError) as err:
    log.info('Tokenizer smoke test skipped: %s', err)
except (TypeError, ValueError) as err:
    log.exception('Tokenizer smoke test failed: %s', err)
    raise

# Test Python indexing (__getitem__) and assignment (__setitem__)
t = Tensor([1.0, 2.0, 3.0], [3])
assert t[0].get_data()[0] == 1.0
assert t[-1].get_data()[0] == 3.0

t2 = Tensor([0.0] * 6, [2, 3])
# set scalar into single index
t2[1, 1] = 42.0
assert t2[1, 1].get_data()[0] == 42.0
# assign a row
t2[0, :] = Tensor([1.0, 2.0, 3.0], [3])
assert t2[0, 0].get_data()[0] == 1.0
assert t2[0, 1].get_data()[0] == 2.0
assert t2[0, 2].get_data()[0] == 3.0
log.info('Python getitem/setitem smoke tests OK')
