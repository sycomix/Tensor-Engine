# Protein Stability Transformer (NL-OOB)

This example demonstrates the **Non-Linear Out-of-Order Bias (NL-OOB)** mechanism applied to a Protein Stability Prediction task.

## Overview

The model predicts a scalar stability score from an amino acid sequence. It leverages the **NL-OOB Logarithmic Bias** ($\phi(d) = \log(1+d)$) to model long-range interactions in the protein sequence without explicit 3D geometry, acting as a "Power Law" inductive bias.

**Key Components:**
- **Tokenizer**: Custom character-level tokenizer for 20+ amino acids.
- **Model**: `ProteinStabilityTransformer` (in `model.py`) using `te.TransformerBlock` with `nl_oob_config="logarithmic"`.
- **Training**: Trains on the `stability_prediction` dataset (Parquet format) using `MSELoss`.
- **Serving**: An HTTP Inference Server (`serve.py`) that loads the trained weights (robustly handling `te.Linear` via in-place updates) and serves predictions.

## Usage

### 1. Training

Run the training script to train the model on the local parquet data (`data/train-*.parquet`).

```bash
python train.py --epochs 1
```

This will:
- Load the dataset.
- Train the model for 1 epoch.
- Save the tokenizer to `tokenizer.pkl`.
- Save the model weights to `model.safetensors`.

### 2. Serving

Start the inference server. This will load `model.safetensors` and `tokenizer.pkl`.

```bash
python serve.py
```

The server listens on **port 8001**.

### 3. Client Test

Send a test request to the server:

```bash
python client.py
```

Or manually via curl:

```bash
curl -X POST http://localhost:8001/predict -d '{"sequence": "MKTLLILAVLLLCNNSAGSLGAPQP"}'
```

## Implementation Details

- **Weight Persistence**: Since `tensor_engine` bindings for saving are minimal, `train.py` uses `safetensors.numpy` to verify and save weights manually.
- **Weight Loading**: `serve.py` uses a mixed loading strategy:
  - **Blocks**: Loaded via `te.py_load_safetensors_into_module` (Native C++ loader).
  - **Linear Layers**: Loaded via `te.Tensor.set_data()` (Manual in-place update) to bypass Python attribute read-only restrictions.
