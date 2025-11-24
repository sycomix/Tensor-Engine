#!/usr/bin/env python3
"""
Matrix multiplication and automatic differentiation example.
"""

import tensor_engine as te
import numpy as np

# Create matrices
a = te.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b = te.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])

print("Matrix A:")
print(a)
print("Matrix B:")
print(b)

# Matrix multiplication
c = a.matmul(b)
print("A @ B:")
print(c)

# Automatic differentiation
a = te.Tensor([1.0, 2.0], [2])
b = te.Tensor([3.0, 4.0], [2])
c = a * b
loss = c.sum()

print(f"\nBefore backward: a.grad = {a.grad}, b.grad = {b.grad}")

loss.backward()

print(f"After backward: a.grad = {a.grad}, b.grad = {b.grad}")
