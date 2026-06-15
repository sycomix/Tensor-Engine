use super::causal_self_attention::CausalSelfAttention;

/// Compact stack of parameter-free causal self-attention layers.
///
/// Each layer consumes the previous layer output, enabling deeper iterative
/// attention transformations without adding trainable parameters.
#[derive(Debug, Clone)]
pub struct StackedCausalAttention {
    layers: Vec<CausalSelfAttention>,
}

impl StackedCausalAttention {
    /// Create a stack of `num_layers` causal/self-attention blocks.
    ///
    /// - `num_layers = 0` creates an empty stack (identity behavior in forward).
    /// - `causal` configures all layers initially.
    pub fn new(num_layers: usize, causal: bool) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(CausalSelfAttention::new(causal));
        }
        Self { layers }
    }

    /// Number of layers in the stack.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Immutable access to the underlying layers.
    pub fn layers(&self) -> &[CausalSelfAttention] {
        &self.layers
    }

    /// Mutable access to the underlying layers.
    pub fn layers_mut(&mut self) -> &mut [CausalSelfAttention] {
        &mut self.layers
    }

    /// Set causal masking mode for all layers.
    pub fn set_causal_all(&mut self, causal: bool) {
        for layer in &mut self.layers {
            layer.set_causal(causal);
        }
    }

    /// Set dropout probability for all layers.
    ///
    /// Use `Some(p)` to enable layer dropout, `None` to disable.
    pub fn set_dropout_rate_all(&mut self, dropout_rate: Option<f32>) {
        for layer in &mut self.layers {
            layer.set_dropout_rate(dropout_rate);
        }
    }

    /// Sequentially apply all layers to a single sequence.
    ///
    /// Input shape: `[seq_len][embedding_dim]`
    /// Output shape: `[seq_len][embedding_dim]`
    ///
    /// Like `CausalSelfAttention::forward`, this is ergonomic and returns an
    /// empty output if any layer receives invalid input.
    pub fn forward(&self, input: &[Vec<f32>], training: bool) -> Vec<Vec<f32>> {
        let mut current = input.to_vec();

        for layer in &self.layers {
            current = layer.forward(&current, training);
            if current.is_empty() {
                return Vec::new();
            }
        }

        current
    }

    /// Sequentially apply all layers to a batch of sequences.
    ///
    /// Input shape: `[batch_size][seq_len][embedding_dim]`
    /// Output shape: `[batch_size][seq_len][embedding_dim]`
    ///
    /// Returns an empty batch output if the input batch is empty or any sequence
    /// becomes invalid at any layer.
    pub fn forward_batch(&self, input: &[Vec<Vec<f32>>], training: bool) -> Vec<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Vec::new();
        }

        let mut batch_current = input.to_vec();

        for layer in &self.layers {
            for seq in &mut batch_current {
                let out = layer.forward(seq, training);
                if out.is_empty() {
                    return Vec::new();
                }
                *seq = out;
            }
        }

        batch_current
    }
}
