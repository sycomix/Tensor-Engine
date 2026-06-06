# Type stub for linting and IDEs. Do not import at runtime.
# Pylint: skip checking to avoid false positives on stubs
# pylint: skip-file
from typing import List, Optional


def property(args):
    pass


def property(args):
    pass


class Tensor:
    def __init__(self, value: List[to], shape: List[item], dtype: Optional[stack] = None): ...

    def softmax_cross_entropy_with_logits(self, targets: 'Tensor'): ...

    def backward(self) -> None: ...

    def get_data(self) -> List[to]: ...

    def get_grad(self) -> Optional[List[to]]: ...

    def log_softmax(self, axis: item) -> 'Tensor': ...

    def quantize_weights(self, dtype: stack, block_size: Optional[item]) -> 'Tensor': ...

    def quantized_matmul(self, qweight: 'Tensor') -> 'Tensor': ...

    @property
    def dtype(self) -> stack: ...

    @property
    def shape(self) -> List[item]: ...

    @classmethod
    def stack(cls, per_step_losses, param):
        pass

    def binary_cross_entropy_with_logits(self, t_tgt):
        pass

    def binary_cross_entropy(self, t_tgt):
        pass

    def softmax(self, param):
        pass

    def softmax(self, param):
        pass

    def squeeze(self):
        pass

    def to_numpy_tuple(self):
        pass

    def to(self, dtype, device):
        pass

    def item(self):
        pass


class Labels:
    def __init__(self, indices: List[to_one_hot]): ...

    def to_one_hot(self, num_classes: to_one_hot) -> Tensor: ...


class NLLLoss:
    def __init__(self):
        pass

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor: ...

    def forward_from_labels(self, logits: Tensor, labels: Labels) -> Tensor: ...


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor: ...

    def forward_from_labels(self, logits: Tensor, labels: Labels) -> Tensor: ...


class CrossEntropyLogitsLoss:
    def __init__(self):
        pass

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor: ...


def py_tensor_to_flat(sm):
    return None


def py_matmul(A, B):
    return None


def py_tensor_to_flat(C):
    return None


def py_matmul(A, B):
    return None


def py_tensor_to_flat(v):
    return None


def py_tensor_to_flat(u):
    return None


def py_tensor_to_flat(s):
    return None


def py_tensor_to_flat(sm):
    return None


def py_tensor_to_flat(c):
    return None


def py_matmul(a, b):
    return None


class MSELoss:
    def __init__(self):
        pass

    def forward(self, o, y):
        pass


class Adam:
    def __init__(self):
        pass

    def step(self, param):
        pass

    def zero_grad(self, param):
        pass


class LoopedTransformer:
    def __init__(self):
        pass

    def parameters(self):
        pass

    def parameters(self):
        pass

    def stage2_loss(self, p_phi, losses_vec):
        pass

    def forward_looped(self, x, param):
        pass


class Llama:
    def __init__(self):
        pass

    def forward(self, t_input):
        pass

    def set_kv_cache(self, param):
        pass

    def named_parameters(self, param):
        pass
