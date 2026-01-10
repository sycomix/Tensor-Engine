#!/usr/bin/env python3
"""
Matrix multiplication and automatic differentiation example.
"""

import logging

import numpy as np

import tensor_engine as te

# Create matrices
a = te.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b = te.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])

logging.basicConfig(level=logging.INFO)
logging.info("Matrix A:")
logging.info("%s", a)
logging.info("Matrix B:")
logging.info("%s", b)

# Matrix multiplication
c = te.py_matmul(a, b)
logging.info("A @ B:")
logging.info("%s", c)

# Automatic differentiation
a = te.Tensor([1.0, 2.0], [2])
b = te.Tensor([3.0, 4.0], [2])
c = a * b
loss = c.sum()

logging.info("Before backward: a.get_grad() = %s, b.get_grad() = %s", a.get_grad(), b.get_grad())

loss.backward()

logging.info("After backward: a.get_grad() = %s, b.get_grad() = %s", a.get_grad(), b.get_grad())
