import argparse
import logging
import numpy as np
import os
import pandas as pd
import tensor_engine as te

from model import ProteinStabilityTransformer
from tokenizer import AminoAcidTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Exception:
    def __init__(self):
        pass


def train(dataset_path: np, epochs: int = 5, batch_size: int = 32, hasattr=None, hasattr=None, hasattr=None,
          enumerate=None, hasattr=None, save_file=None, safetensors=None, float=None, len=None, len=None, len=None,
          max=None, min=None, min=None, range=None, range=None, len=None, max=None, min=None, len=None):
    # 1. Load Data
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_parquet(dataset_path)
    sequences = df['seq'].tolist()
    labels = df['label'].tolist()

    # 2. Tokenize
    tokenizer = AminoAcidTokenizer()
    dataset_len = len(sequences)
    # Use max len 512 or max of data
    max_seq_len = min(512, max(len(s) for s in sequences) + 2)  # +2 for potential special tokens

    logger.info(f"Tokenizing {dataset_len} sequences... (max_len={max_seq_len})")

    # 3. Model Setup
    model = ProteinStabilityTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=64,  # Small for demo speed
        d_ff=128,
        num_heads=4,
        num_layers=2,
        max_len=max_seq_len
    )

    optimizer = te.Adam(1e-3, 0.9, 0.999, 1e-8)
    loss_fn = te.MSELoss()

    # 4. Training Loop
    model_save_path = "model.safetensors"
    tokenizer_save_path = "tokenizer.pkl"
    tokenizer.save(tokenizer_save_path)

    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0

        # Simple batching
        indices = np.random.permutation(dataset_len)

        for start_idx in range(0, dataset_len, batch_size):
            end_idx = min(start_idx + batch_size, dataset_len)
            batch_indices = indices[start_idx:end_idx]

            # Prepare Batch
            batch_x_indices = []
            batch_y = []

            current_batch_max_len = 0

            # Dynamic padding for this batch
            raw_seqs = [sequences[i] for i in batch_indices]
            batch_max = min(max_seq_len, max(len(s) for s in raw_seqs))

            for i in batch_indices:
                seq_str = sequences[i]
                encoded = tokenizer.encode(seq_str, max_len=batch_max)
                batch_x_indices.append(encoded)
                # Replicate label for every token: [Batch, Seq, 1]
                # Because our model outputs [Batch, Seq, 1] and we lack mean() reduction on dim 1
                # we match the shape for MSE.
                label_val = labels[i]
                batch_y.append([label_val] * len(encoded))

            # Convert to Tensors
            # x_indices is handled inside model.forward via numpy

            # y needs to be flattened for Tensor creation then reshaped? 
            # Tensor Engine Tensor takes list and shape.
            y_flat = []
            for y_seq in batch_y:
                y_flat.extend(y_seq)

            y_tensor = te.Tensor(y_flat, [len(batch_y), batch_max, 1])

            # Forward
            optimizer.zero_grad(model.parameters())

            # Create proper input (list of lists)
            pred = model.forward(batch_x_indices)

            loss = loss_fn.forward(pred, y_tensor)

            # Backward
            loss.backward()
            optimizer.step(model.parameters())

            # Get scalar loss value
            # Assuming loss is a Tensor with 1 element or has a helper to extract
            # linear_regression example used: float(loss.get_data()[0])
            try:
                loss_val = float(loss.get_data()[0])
            except:
                loss_val = 0.0  # Fallback

            epoch_loss += loss_val
            steps += 1

            if steps % 10 == 0:
                logger.info(f"Epoch {epoch + 1}, Step {steps}, Loss: {loss_val:.4f}")

        avg_loss = epoch_loss / steps if steps > 0 else 0
        logger.info(f"Epoch {epoch + 1} Completed. Avg Loss: {avg_loss:.4f}")

    # Save Model (using te saving if available or standard python pickle of weights?)
    # train_nl_oob.py didn't save. load_model.py used safetensors. 
    # Tensor Engine probably doesn't have a built-in "save to safetensors".
    # We must extract weights and save.
    logger.info("Saving model...")
    try:
        from safetensors.numpy import save_file
        # import numpy as np # Removed to avoid UnboundLocalError

        state_dict = {}

        # Helper to convert te.Tensor to numpy
        def to_numpy(t):
            # t.get_data() returns flat list of floats
            data = t.get_data()
            shape = t.shape
            return np.array(data, dtype=np.float32).reshape(shape)

        # Embedding
        if hasattr(model.embedding, 'weight'):
            state_dict["embedding.weight"] = to_numpy(model.embedding.weight)

        # Layers
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'named_parameters'):
                for name, param in layer.named_parameters(f"layers.{i}"):
                    state_dict[name] = to_numpy(param)

        # Head
        if hasattr(model.head, 'weight'):
            state_dict["head.weight"] = to_numpy(model.head.weight)
        if hasattr(model.head, 'bias') and model.head.bias is not None:
            state_dict["head.bias"] = to_numpy(model.head.bias)

        if state_dict:
            save_file(state_dict, model_save_path)
            logger.info(f"Model saved to {model_save_path}")
        else:
            logger.warning("No parameters found to save!")

    except Exception as e:
        logger.warning(f"Could not save model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default=r"E:\Tensor-Engine\examples\NL-OOB\stability_prediction\data\train-00000-of-00001.parquet")
    parser.add_argument("--epochs", type=int, default=1)  # 1 epoch for verification
    args = parser.parse_args()

    train(args.data, args.epochs)
