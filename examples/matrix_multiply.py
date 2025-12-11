#!/usr/bin/env python3
"""
Matrix multiplication and automatic differentiation example.
"""

import tensor_engine as te
import numpy as np
import logging

# Create matrices
a = te.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b = te.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])

logging.basicConfig(level=logging.INFO)
logging.info("Matrix A:")
logging.info("%s", a)
logging.info("Matrix B:")
logging.info("%s", b)

# Matrix multiplication
c = a.matmul(b)
logging.info("A @ B:")
logging.info("%s", c)

# Automatic differentiation
a = te.Tensor([1.0, 2.0], [2])
b = te.Tensor([3.0, 4.0], [2])
c = a * b
loss = c.sum()

logging.info("Before backward: a.grad = %s, b.grad = %s", a.grad, b.grad)

loss.backward()

logging.info("After backward: a.grad = %s, b.grad = %s", a.grad, b.grad)
