use crate::nn::{Module, VisionTransformer};
use crate::tensor::Tensor;
use ndarray::IxDyn;
use crate::nn::Linear;


pub struct MultimodalLLM {
    pub vision_encoder: VisionTransformer,
    pub text_embedding: Tensor, // [vocab, d_model]
    pub projector: Option<Linear>,
    pub decoder_blocks: Vec<crate::nn::TransformerBlock>,
    pub head: Linear,
}

impl MultimodalLLM {
    pub fn new(vision: VisionTransformer, vocab_size: usize, d_model: usize, d_ff: usize, num_heads: usize, depth: usize) -> Self {
        let text_embedding = Tensor::new(ndarray::Array::zeros(IxDyn(&[vocab_size, d_model])), true);
        let projector = None::<Linear>;
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth { blocks.push(crate::nn::TransformerBlock::new_decoder(d_model, d_ff, num_heads)); }
        let head = Linear::new(d_model, vocab_size, true);
        MultimodalLLM { vision_encoder: vision, text_embedding, projector, decoder_blocks: blocks, head }
    }

    /// Forward pass: images -> img_tokens, text indices -> txt_tokens, concat and decode.
    /// images: [B, C, H, W], input_ids: [B, seq]
    pub fn forward(&self, images: &Tensor, input_ids: &Tensor) -> Tensor {
        let img_feats = self.vision_encoder.forward(images); // [B, Np, d_model]
        let mut img_proj = img_feats.clone();
        if let Some(p) = &self.projector {
            img_proj = p.forward(&img_feats);
        }
        let txt_tokens = crate::tensor::Tensor::embedding_lookup(&self.text_embedding, input_ids);
        let mut combined = crate::tensor::Tensor::concat(&[img_proj, txt_tokens], 1);
        for b in &self.decoder_blocks {
            combined = b.forward(&combined);
        }
        let logits = self.head.forward(&combined);
        logits
    }
}

impl Module for MultimodalLLM {
    fn forward(&self, input: &Tensor) -> Tensor { input.clone() }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.vision_encoder.parameters();
        p.push(self.text_embedding.clone());
        if let Some(proj) = &self.projector { p.extend(proj.parameters()); }
        for b in &self.decoder_blocks { p.extend(b.parameters()); }
        p.extend(self.head.parameters());
        p
    }
}
