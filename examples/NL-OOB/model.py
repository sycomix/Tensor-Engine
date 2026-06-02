import numpy as np

import tensor_engine as te


class ProteinStabilityTransformer:
    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_layers, max_len=512):
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding: Since te.Embedding is missing, use te.Linear.
        # We will essentially do Linear(one_hot(x)) which is equivalent to lookup.
        # But efficiently: we can just index into the weight matrix if we implemented it manually,
        # but with te types, we might have to use matrix multiply.
        # For efficiency in this demo, we'll try to rely on te.Linear 
        # acting on floats. We will construct input as one-hot floats.
        self.embedding = te.Linear(vocab_size, d_model, bias=False)

        self.layers = []
        for _ in range(num_layers):
            # Using nl_oob_config="logarithmic" as per specification
            block = te.TransformerBlock(
                d_model,
                d_ff,
                num_heads,
                nl_oob_config="logarithmic",
                nl_oob_max_scale=8.0  # High scale for local interactions
            )
            self.layers.append(block)

        self.head = te.Linear(d_model, 1, bias=True)
        self.max_len = max_len

    def parameters(self):
        params = self.embedding.parameters()
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.head.parameters())
        return params

    def forward(self, x_indices, dist_tensor=None):
        """
        x_indices: List[int] or np.array of shape [batch, seq]
        dist_tensor: te.Tensor of shape [seq, seq] or [batch, seq, seq]
        """

        # 1. Convert indices to one-hot for Linear Embedding fallback
        # This is a bit heavy for Python but necessary without te.Embedding
        if hasattr(x_indices, 'shape'):
            batch, seq = x_indices.shape
        else:
            # Assume list of lists
            batch = len(x_indices)
            seq = len(x_indices[0])

        # Create one-hot input: [batch, seq, vocab_size]
        x_np = np.zeros((batch, seq, self.vocab_size), dtype=np.float32)
        if isinstance(x_indices, np.ndarray):
            for b in range(batch):
                for s in range(seq):
                    idx = int(x_indices[b, s])
                    if idx < self.vocab_size:
                        x_np[b, s, idx] = 1.0
        else:
            for b in range(batch):
                for s in range(seq):
                    idx = x_indices[b][s]
                    if idx < self.vocab_size:
                        x_np[b, s, idx] = 1.0

        x_tensor = te.Tensor(x_np.flatten().tolist(), [batch, seq, self.vocab_size])

        # 2. Embedding Projection
        h = self.embedding.forward(x_tensor)

        # 3. Transformer Blocks with Distance
        # If dist_tensor is not provided, create 1D distance
        if dist_tensor is None:
            # Create standard 1D distance matrix: |i-j|
            # Shape [seq, seq], broadcasted across batch
            d = np.abs(np.subtract.outer(np.arange(seq), np.arange(seq))).astype(np.float32)
            dist_tensor = te.Tensor(d.flatten().tolist(), [seq, seq])

        for layer in self.layers:
            h = layer.forward_with_distance(h, dist_tensor)

        # 4. Global Mean Pooling
        # Manual mean pooling since te might not have it exposed
        # We need to access the data or use a reduce operation if available.
        # Fallback to simple averaging in Python if strictly necessary? 
        # No, that breaks the graph. te.Tensor might support basic ops.
        # Assuming te.Tensor supports sum or we can implement it via matrix mult.
        # Mean = (Sum over seq) / seq
        # We can multiply by a "Summing Tensor" [batch, 1, seq] ?
        # Or simple: if te has no mean, we might just take the first token (CLS token equivalent)?
        # For simplicity and "Production" robustness without guessing API:
        # Use first token (index 0) as representative if we treated it as CLS,
        # BUT we trained on sequences.
        # Let's try to do a robust mean manually if possible, or just use H[0] if we prepend CLS.
        # Given the data, let's prepend a special token or just assume H[0] is enough?
        # Better: Let's assume h supports indexing or slicing? Unlikely in bindings.
        # We will use a dedicated "Readout" query like in Perceiver or just use the first token.
        # Let's stick to using the first token as the embedding of the sequence.
        # WAIT: The tokenizer has [PAD] and [UNK]. Let's add [CLS] at local index if we want.
        # Or just use simple global pooling by averaging (if we can).
        # Check linear_regression.py: `(pred - y).mean()`. So `mean()` exists!

        # But `mean()` usually returns a scalar (all dims). We want mean over Dim 1.
        # If `mean(dim)` is not supported, we are stuck.
        # Let's try to assume we can just use the mean of the whole tensor? No that's wrong.
        # Strategy: Use a "CLS" style token.
        # We will pretend the first token is valuable.

        # Workaround: Flatten and linear? No.
        # Let's use the property of attention: The first token attends to everything.
        # So H[0] contains info from everywhere.
        # HOWEVER, we need to extract H[0] for each batch.
        # If te does not support slicing `h[:, 0, :]`, this is hard.

        # Alternate: Linear Head on ALL tokens, then mean the logits? 
        # Output [Batch, Seq, 1]. Mean over Seq. 
        # If `mean()` reduces all, then `mean(h)` -> scalar.
        # We need `batch` outputs.

        # Let's assume for this implementation we simply use a simple trick:
        # Pass the whole sequence to the head, get [Batch, Seq, 1].
        # Then we need to average this to [Batch, 1].
        # If we can't do that, we can use `sum` if available?
        # Let's assume we can interact with numpy for the last step if not trainable? 
        # No, need grads.

        # Let's rely on standard practice: Use a Pooling Layer if available? No.
        # Let's try to use the fact that `loss` uses `mean()`.
        # Maybe we can just train on target replicated across sequence? [Batch, Seq, 1] == Target.
        # This effectively trains every token to predict the stability.
        # This is valid for global properties (every part of protein knows stability).
        h = self.head.forward(h)  # [Batch, Seq, 1]

        return h
