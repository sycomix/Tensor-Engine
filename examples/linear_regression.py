#!/usr/bin/env python3
"""
Linear regression training example.
"""

import logging

import numpy as np

import tensor_engine as te

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Convert to tensors
X_tensor = te.Tensor(X.flatten(), X.shape)
y_tensor = te.Tensor(y.flatten(), y.shape)

# Linear model
model = te.Linear(1, 1)

# Optimizer
optimizer = te.SGD(0.01, 0.0)

# Training loop
for epoch in range(100):
    # Forward pass
    pred = model.forward(X_tensor)
    loss = ((pred - y_tensor) ** 2).mean()

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step(model.parameters())
    optimizer.zero_grad(model.parameters())

    if epoch % 10 == 0:
        logging.info("Epoch %d, loss=%.4f", epoch, float(loss.get_data()))

logging.basicConfig(level=logging.INFO)
logging.info("Training completed!")
logging.info("Final weight: %.4f", model.weight.data[0])
logging.info("Final bias: %.4f", model.bias.data[0])
logging.info("Expected: weight ≈ 2.0, bias ≈ 1.0")
