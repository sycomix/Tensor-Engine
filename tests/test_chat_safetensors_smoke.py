import json
import os
import subprocess
import sys
import tempfile
import unittest


class ImportError:
    def __init__(self):
        pass


try:
    import numpy as np
except ImportError:
    np = None


class ImportError:
    def __init__(self):
        pass


try:
    from safetensors.numpy import save_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


class TestChatSafetensors(unittest.TestCase):
    def setUp(self):
        self.script_path = os.path.join(os.path.dirname(__file__), "..", "examples", "chat_safetensors.py")
        self.script_path = os.path.abspath(self.script_path)

    def test_smoke_demo(self, print=None, print=None):
        """
        Runs examples/chat_safetensors.py with no arguments.
        Expects exit code 0 and some output indicating the smoke demo ran.
        """
        # We need to run this with the current python interpreter
        cmd = [sys.executable, self.script_path]

        # Capture output
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        self.assertEqual(result.returncode, 0, "Chat smoke demo failed")
        # The script prints "Smoke demo output shape:" when running in no-arg mode
        self.assertIn("Smoke demo output shape", result.stdout)

    def test_with_dummy_model(self, print=None, print=None, open=None, print=None):
        """
        If safetensors is available, create a dummy model and run the script against it.
        """
        if not SAFETENSORS_AVAILABLE or np is None:
            print("Skipping test_with_dummy_model: safetensors or numpy not installed.")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.safetensors")
            config_path = os.path.join(tmpdir, "config.json")

            # Create dummy weights
            d_model = 32
            d_ff = 64
            num_heads = 4
            vocab_size = 50

            tensors = {
                "model.embed_tokens.weight": np.random.randn(vocab_size, d_model).astype(np.float32),
                "model.layers.0.self_attn.q_proj.weight": np.random.randn(d_model, d_model).astype(np.float32),
                "model.layers.0.self_attn.k_proj.weight": np.random.randn(d_model, d_model).astype(np.float32),
                "model.layers.0.self_attn.v_proj.weight": np.random.randn(d_model, d_model).astype(np.float32),
                "model.layers.0.self_attn.o_proj.weight": np.random.randn(d_model, d_model).astype(np.float32),
                "model.layers.0.mlp.gate_proj.weight": np.random.randn(d_ff, d_model).astype(np.float32),
                "model.layers.0.mlp.up_proj.weight": np.random.randn(d_ff, d_model).astype(np.float32),
                "model.layers.0.mlp.down_proj.weight": np.random.randn(d_model, d_ff).astype(np.float32),
                "lm_head.weight": np.random.randn(vocab_size, d_model).astype(np.float32),
            }
            save_file(tensors, model_path)

            config = {
                "d_model": d_model,
                "d_ff": d_ff,
                "num_heads": num_heads,
                "vocab_size": vocab_size,
                "num_hidden_layers": 1
            }
            with open(config_path, "w") as f:
                json.dump(config, f)

            # Run the script with --message for one-shot generation
            # Note: We don't have a tokenizer, so this might fail if the script requires one.
            # However, the script has a naive tokenizer fallback if none is provided?
            # Creating a dummy tokenizer.json just in case or relying on naive fallback.
            # The script says: "p.add_argument("--tokenizer", etc default=None)"
            # And "if tokenizer is None: etc naive_tokenize(inp, vocab_size)" for smoke demo
            # But for main execution:
            # "if args.message is not None: if tokenizer is None: logger.error... raise SystemExit(1)"
            # So we MUST have a tokenizer for --message mode unless we modify the script to allow naive tokenization there too.
            # Looking at lines 615-617 of chat_safetensors.py:
            # if tokenizer is None: logger.error("Tokenization unavailable...")

            # So test_with_dummy_model with --message will fail without a tokenizer.
            # Let's just run it in interactive mode (but pipe exit) or skip the message part?
            # Actually, we can provoke the "Tokenization unavailable" error and assert it exits with 1.
            # OR we can assert that it loads successfully up to the loop. 

            # Let's try to run with a nonexistent tokenizer and see it fail gracefully or load successfully
            # interactive mode with input "exit"

            cmd = [
                sys.executable,
                self.script_path,
                model_path,
                "--config", config_path,
                "--seq_len", "16"
            ]

            # Pipe "exit\n" to stdin
            result = subprocess.run(cmd, input="exit\n", capture_output=True, text=True)

            if result.returncode != 0:
                print("Dummy model test STDOUT:", result.stdout)
                print("Dummy model test STDERR:", result.stderr)

            self.assertEqual(result.returncode, 0, "Chat with dummy model failed")
            # Log messages go to stderr
            self.assertIn("Model loaded", result.stderr)
            self.assertIn("Interactive Model Diagnostic REPL", result.stderr)


if __name__ == "__main__":
    unittest.main()
