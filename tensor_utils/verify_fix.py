import ctypes
import os
import sys
import unittest

class TestRustBF16Binding(unittest.TestCase):
    def setUp(self):
        lib_name = "tensor_utils"
        if sys.platform == "win32":
            lib_file = f"{lib_name}.dll"
            prefix = ""
        elif sys.platform == "darwin":
            lib_file = f"lib{lib_name}.dylib"
            prefix = "target/release/"
        else:
            lib_file = f"lib{lib_name}.so"
            prefix = "target/release/"
        self.lib_path = os.path.abspath(os.path.join(os.getcwd(), prefix, lib_file))
        if not os.path.exists(self.lib_path):
            self.skipTest(f"Library not found at {self.lib_path}. Did you run 'cargo build --release'?")
        self.lib = ctypes.CDLL(self.lib_path)
        self.lib.convert_bf16_to_f32_buffer.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self.lib.convert_bf16_to_f32_buffer.restype = None

    def test_garbage_vs_gold(self):
        input_val = 16256
        count = 1
        src_array = (ctypes.c_uint16 * count)(input_val)
        dst_array = (ctypes.c_float * count)()
        self.lib.convert_bf16_to_f32_buffer(src_array, dst_array, count)
        result = dst_array[0]
        print(f"\n[Input: 0x{input_val:04X}] -> [Output: {result}]")
        if result == 16256.0:
            self.fail("FAILURE: Rust is performing numeric cast! Model will output garbage.")
        self.assertAlmostEqual(result, 1.0, places=6, msg=f"Expected 1.0, got {result}")

    def test_buffer_integrity(self):
        inputs = [0x3F80, 0x4000, 0x3F00]
        expected = [1.0, 2.0, 0.5]
        count = len(inputs)
        src = (ctypes.c_uint16 * count)(*inputs)
        dst = (ctypes.c_float * count)()
        self.lib.convert_bf16_to_f32_buffer(src, dst, count)
        for i, (expect, actual) in enumerate(zip(expected, dst)):
            self.assertAlmostEqual(expect, actual, places=6, msg=f"Index {i} mismatch")

if __name__ == "__main__":
    unittest.main()
