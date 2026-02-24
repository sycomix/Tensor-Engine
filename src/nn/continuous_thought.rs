use crate::nn::GRUCell;
use crate::tensor::Tensor;
use ndarray::IxDyn;

/// A simple recurrent module that maintains a continuous hidden "thought"
/// vector over time.  It is intended to be used inside multimodal reasoning
/// loops where each new observation (packed as a latent tensor) updates the
/// internal state.  The implementation uses a GRU cell under the hood and
/// provides a small public API suitable for both training and inference.

#[derive(Clone)]
pub struct ContinuousThoughtModule {
    pub cell: GRUCell,
    pub state: Option<Tensor>,
}

impl ContinuousThoughtModule {
    /// Create a new module whose hidden dimension equals `dim`.
    ///
    /// The GRU is initialized with zero weights; users should call standard
    /// weight-initialization routines (e.g. `nn::init::kaiming_uniform`) if they
    /// intend to train the module.
    pub fn new(dim: usize) -> Self {
        ContinuousThoughtModule {
            cell: GRUCell::new(dim, dim, true),
            state: None,
        }
    }

    /// Reset the latent state to zero.  After calling this the next `forward`
    /// call will treat the hidden state as all zeroes.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Run a forward step.  `input` is expected to have shape `[batch, dim]`.
    /// The returned tensor is the updated hidden state (which is also stored
    /// inside the module).  If the provided input has a mismatched shape, the
    /// call will log an error and return the input unchanged.
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let shape = input.lock().storage.shape();
        if shape.len() != 2 {
            log::error!(
                "ContinuousThoughtModule expected 2D input but got {:?} shape",
                shape
            );
            // use to_owned to appease analyzer
            return input.to_owned();
        }
        let batch = shape[0];
        let hidden = if let Some(h) = &self.state {
            h.clone()
        } else {
            let zeros = ndarray::Array::zeros(IxDyn(&[batch, self.cell.hidden_dim][..]));
            Tensor::new(zeros, true)
        };
        let out = self.cell.forward_step(input, &hidden);
        self.state = Some(out.to_owned());
        out
    }

    /// Accessor for the current thought vector.  This is useful for debugging
    /// or for hooking the module into larger graphs without performing another
    /// forward pass.
    pub fn get_state(&self) -> Option<Tensor> {
        self.state.clone()
    }
}
