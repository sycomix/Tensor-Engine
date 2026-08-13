import math
import numpy as np
import sys

import tensor_engine as te


def manual_bce(input, target):
    # input is prob (0..1)
    # Clamp to avoid log(0)
    eps = 1e-12
    input = max(eps, min(1.0 - eps, input))
    return - (target * math.log(input) + (1 - target) * math.log(1 - input))


def manual_bce_logits(logits, target):
    # max(logits, 0) - logits * target + log(1 + exp(-abs(logits)))
    return max(logits, 0) - logits * target + math.log(1 + math.exp(-abs(logits)))
def test_bce():
    print("Testing Binary Cross Entropy...")
    # shape [2]
    # Use inputs clearly separated from 0/1 to avoid epsilon dependency issues in rough verification
    input_vals = [0.2, 0.8, 0.5]
    target_vals = [0.0, 1.0, 0.5]

    t_in = te.Tensor(input_vals, [3])
    t_tgt = te.Tensor(target_vals, [3])

    try:
        out = t_in.binary_cross_entropy(t_tgt)
        out_data = out.get_data()

        print(f"BCE Output: {out_data}")

        for i in range(3):
            ref = manual_bce(input_vals[i], target_vals[i])
            diff = abs(out_data[i] - ref)
            print(f"Index {i}: Val={out_data[i]}, Ref={ref}, Diff={diff}")
            assert diff < 1e-4, f"Mismatch at index {i}"

        print("BCE Forward Passed")
    except AttributeError:
        print("Error: binary_cross_entropy method not found. Bindings might need update.")
        sys.exit(1)
def test_bce_logits():
    print("\nTesting BCE With Logits...")
    input_vals = [-1.0, 2.0, 0.0]
    target_vals = [0.0, 1.0, 0.5]

    t_in = te.Tensor(input_vals, [3])
    t_tgt = te.Tensor(target_vals, [3])

    try:
        out = t_in.binary_cross_entropy_with_logits(t_tgt)
        out_data = out.get_data()

        print(f"BCEWithLogits Output: {out_data}")

        for i in range(3):
            ref = manual_bce_logits(input_vals[i], target_vals[i])
            diff = abs(out_data[i] - ref)
            print(f"Index {i}: Val={out_data[i]}, Ref={ref}, Diff={diff}")
            assert diff < 1e-4, f"Mismatch at index {i}"

        print("BCE With Logits Forward Passed")
    except AttributeError:
        print("Error: binary_cross_entropy_with_logits method not found. Bindings might need update.")
        sys.exit(1)
if __name__ == "__main__":
    try:
        test_bce()
        test_bce_logits()
        print("\nAll Smoke Tests Passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        sys.exit(1)
