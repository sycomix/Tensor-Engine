import http.server
import json
import logging
import numpy as np
import socketserver
import tensor_engine as te

from model import ProteinStabilityTransformer
from tokenizer import AminoAcidTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = 8001
TOKENIZER_PATH = "tokenizer.pkl"
class InferenceHandler(http.server.BaseHTTPRequestHandler):
    model = None
    tokenizer = None

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data)
                sequence = data.get('sequence', '')

                if not sequence:
                    self.send_error(400, "Missing 'sequence' field")
                    return

                # Inference
                encoded = self.tokenizer.encode(sequence)
                # Batch of 1
                batch_x = [encoded]

                # Forward
                # Model returns [Batch, Seq, 1]
                pred_tensor = self.model.forward(batch_x)

                # Extract mean prediction from the tensor
                # Get data as list/numpy
                pred_data = pred_tensor.get_data()  # List of floats
                pred_val = float(np.mean(pred_data))

                response = {"stability": pred_val}

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404)
def run_server( load_file=None, safetensors=None):
    # Load Resources
    logger.info("Loading Tokenizer...")
    try:
        tokenizer = AminoAcidTokenizer.load(TOKENIZER_PATH)
    except:
        logger.warning("Could not load tokenizer, using fresh one.")
        tokenizer = AminoAcidTokenizer()

    logger.info("Initializing Model...")
    # Matches training config
    MAX_LEN = 512 + 2
    # Matches training config
    model = ProteinStabilityTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=64,
        d_ff=128,
        num_heads=4,
        num_layers=2,
        max_len=MAX_LEN
    )

    # Load Weights
    MODEL_PATH = "model.safetensors"
    try:
        from safetensors.numpy import load_file
        import numpy as np
        import os

        if hasattr(te, 'Tensor'):
            TensorCtor = te.Tensor
        else:
            TensorCtor = te.tensor  # fallback

        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading weights from {MODEL_PATH}...")
            state_dict = load_file(MODEL_PATH)

            with open(MODEL_PATH, "rb") as f:
                model_bytes = f.read()

            # Load Embedding (Manual In-Place)
            if "embedding.weight" in state_dict:
                w_np = state_dict["embedding.weight"]
                model.embedding.weight.set_data(w_np.flatten().tolist())

            # Load Blocks
            for i, layer in enumerate(model.layers):
                if hasattr(te, 'py_load_safetensors_into_module'):
                    te.py_load_safetensors_into_module(model_bytes, False, layer, f"layers.{i}")

            # Load Head (Manual In-Place)
            if "head.weight" in state_dict:
                w_np = state_dict["head.weight"]
                model.head.weight.set_data(w_np.flatten().tolist())
            if "head.bias" in state_dict:
                w_np = state_dict["head.bias"]
                model.head.bias.set_data(w_np.flatten().tolist())

            logger.info("Weights loaded successfully using mixed strategy (In-Place + Native).")

        else:
            logger.warning(f"{MODEL_PATH} not found. Using random weights.")

    except Exception as e:
        logger.warning(f"Could not load model: {e}")

    InferenceHandler.model = model
    InferenceHandler.tokenizer = tokenizer

    with socketserver.TCPServer(("", PORT), InferenceHandler) as httpd:
        logger.info(f"Serving at port {PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
