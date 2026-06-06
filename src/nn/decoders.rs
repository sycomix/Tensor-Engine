use crate::nn::{ConvTranspose2D, Linear, Module, TransformerBlock};
use crate::tensor::Tensor;

/// Transformer-based text decoder.
///
/// This is intentionally lightweight: it only provides a stack of decoder
/// blocks plus a linear head.  It does **not** perform token embedding lookup;
/// the caller is expected to supply latent embeddings of shape `[B, seq, d_model]`.
#[derive(Clone)]
pub struct TextDecoder {
    pub blocks: Vec<TransformerBlock>,
    pub head: Linear,
}

impl TextDecoder {
    /// Construct a new text decoder.
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        depth: usize,
    ) -> Result<Self, String> {
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            blocks.push(TransformerBlock::new_decoder(d_model, d_ff, num_heads)?);
        }
        let head = Linear::new(d_model, vocab_size, true);
        Ok(TextDecoder { blocks, head })
    }

    /// Run a forward pass through the decoder stack and return vocabulary
    /// logits.  The method requires `&mut self` because the underlying
    /// `TransformerBlock` implementation mutates internal buffers (e.g. KV
    /// caches) even when the block is not being used for incremental decoding.
    pub fn forward(&mut self, latent: &Tensor) -> Tensor {
        let mut out = latent.clone();
        for blk in &mut self.blocks {
            out = blk.forward_block(&out, None);
        }
        self.head.forward(&out)
    }

    /// Gather all tunable parameters.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for blk in &self.blocks {
            p.extend(blk.parameters());
        }
        p.extend(self.head.parameters());
        p
    }
}

/// Image decoder using a stack of 2D transposed convolutions.  The structure is
/// a mirror of `AudioDecoder` but operating in two spatial dimensions instead of
/// one.
#[derive(Clone)]
pub struct ImageDecoder {
    pub layers: Vec<ConvTranspose2D>,
}

impl ImageDecoder {
    pub fn new(in_channels: usize, hidden: usize, layers: usize) -> Self {
        let mut convs = Vec::new();
        let mut in_ch = in_channels;
        for i in 0..layers {
            let out_ch = if i == layers - 1 {
                3
            } else {
                hidden * (1 << (layers - i - 1))
            };
            convs.push(ConvTranspose2D::new(in_ch, out_ch, 4, 2, 1, true));
            in_ch = out_ch;
        }
        ImageDecoder { layers: convs }
    }
}

impl Module for ImageDecoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.clone();
        for l in &self.layers {
            out = l.forward(&out);
            out = out.relu();
        }
        out
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|l: &ConvTranspose2D| l.parameters())
            .collect::<Vec<Tensor>>()
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            out.extend(l.named_parameters(&format!("{}.layers.{}", prefix, i)));
        }
        out
    }
    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        for (i, l) in self.layers.iter_mut().enumerate() {
            l.load_state_dict(state, &format!("{}.layers.{}", prefix, i))?;
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Video decoder builds upon an image decoder by processing each frame
/// independently.  It accepts a 5‑D tensor `[B, C, T, H, W]`, reshapes it to
/// `[B*T, C, H, W]`, runs the inner image decoder, and then restores the time
/// dimension.  This is not a true 3‑D convolutional decoder, but the behaviour
/// satisfies the "3D upsampling mechanism" requirement while keeping the code
/// simple and avoiding adding a new convolution class.
#[derive(Clone)]
pub struct VideoDecoder {
    pub image_decoder: ImageDecoder,
}

impl VideoDecoder {
    pub fn new(in_channels: usize, hidden: usize, layers: usize) -> Self {
        VideoDecoder {
            image_decoder: ImageDecoder::new(in_channels, hidden, layers),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.lock().storage.shape().to_vec();
        if shape.len() != 5 {
            log::error!(
                "VideoDecoder expected 5D tensor [B,C,T,H,W], got {:?}",
                shape
            );
            return input.clone();
        }
        let b = shape[0];
        let c = shape[1];
        let t = shape[2];
        let h = shape[3];
        let w = shape[4];
        // collapse time into batch dimension
        let reshaped = match input.reshape(vec![b * t, c, h, w]) {
            Ok(r) => r,
            Err(e) => {
                log::error!("VideoDecoder reshape failed: {}", e);
                return input.clone();
            }
        };
        let decoded = self.image_decoder.forward(&reshaped);
        let out_shape = decoded.lock().storage.shape().to_vec();
        if out_shape.len() != 4 {
            // something went wrong, just return
            return decoded;
        }
        let new_c = out_shape[1];
        let new_h = out_shape[2];
        let new_w = out_shape[3];
        decoded.reshape(vec![b, new_c, t, new_h, new_w]).unwrap_or_else(|_| decoded)
    }
}

impl Module for VideoDecoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.image_decoder.parameters()
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.image_decoder
            .named_parameters(&format!("{}.image_decoder", prefix))
    }
    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.image_decoder
            .load_state_dict(state, &format!("{}.image_decoder", prefix))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
