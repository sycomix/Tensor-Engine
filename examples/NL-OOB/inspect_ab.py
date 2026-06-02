import logging

import tensor_engine as te

logging.basicConfig(level=logging.INFO)

try:
    l = te.Linear(10, 10, False)
    print("Linear attributes:", dir(l))
    print("Linear weight type:", type(l.weight))
    print("Tensor attributes:", dir(l.weight))

    # Try to set
    try:
        l.weight = te.Tensor([0.0] * 100, [10, 10])
        print("Set weight attribute SUCCESS")
    except Exception as e:
        print(f"Set weight attribute FAILED: {e}")

    # Try inplace
    if hasattr(l.weight, 'copy_'):
        print("Has copy_")
    if hasattr(l.weight, 'assign'):
        print("Has assign")

except Exception as e:
    print(e)
