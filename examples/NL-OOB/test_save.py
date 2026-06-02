import logging
import os

from model import ProteinStabilityTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_save():
    model = ProteinStabilityTransformer(
        vocab_size=30,
        d_model=64,
        d_ff=128,
        num_heads=4,
        num_layers=2,
        max_len=128
    )

    save_path = "test_model.safetensors"

    logger.info("Attempting to save model...")
    try:
        from safetensors.numpy import save_file
        import numpy as np

        def to_numpy(t):
            # t.get_data() returns flat list of floats
            data = t.get_data()
            shape = t.shape
            return np.array(data, dtype=np.float32).reshape(shape)

        state_dict = {}
        if hasattr(model.embedding, 'weight'):
            state_dict["embedding.weight"] = to_numpy(model.embedding.weight)

        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'named_parameters'):
                params = layer.named_parameters(f"layers.{i}")
                for item in params:
                    name, param = item
                    state_dict[name] = to_numpy(param)

        if hasattr(model.head, 'weight'):
            state_dict["head.weight"] = to_numpy(model.head.weight)
        if hasattr(model.head, 'bias') and model.head.bias is not None:
            state_dict["head.bias"] = to_numpy(model.head.bias)

        save_file(state_dict, save_path)
        logger.info(f"Success! Saved to {save_path}")

    except Exception as e:
        logger.error(f"Save failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_save()
