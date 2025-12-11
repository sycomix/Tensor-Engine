use crate::nn::{Conv2D, AbsolutePositionalEmbedding, Module, TransformerBlock};
use crate::tensor::Tensor;

pub struct PatchEmbed {
    pub conv: Conv2D,
    pub patch_size: usize,
}

impl PatchEmbed {
    pub fn new(in_ch: usize, embed_dim: usize, patch_size: usize) -> Self {
        PatchEmbed { conv: Conv2D::new(in_ch, embed_dim, patch_size, patch_size, 0, true), patch_size }
    }
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // input expected shape: [B, C, H, W]
        let out = self.conv.forward(input);
        // out shape: [B, embed_dim, H', W'] -> reshape to [B, H'*W', embed_dim]
        let shape = out.lock().storage.shape();
        if shape.len() != 4 { return out; }
        let b = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let seq = h * w;
        let reshaped = out
            .permute(vec![0, 2, 3, 1])
            .reshape(vec![b, seq, c])
            .expect("Reshape to (B, N_patches, D) failed - expected a 4D image tensor [B, C, H, W]");
        reshaped
    }
}

pub struct VisionTransformer {
    pub patch_embed: PatchEmbed,
    pub pos_emb: AbsolutePositionalEmbedding,
    pub blocks: Vec<TransformerBlock>,
}

impl VisionTransformer {
    pub fn new(in_ch: usize, patch_size: usize, d_model: usize, d_ff: usize, num_heads: usize, depth: usize, max_len: usize) -> Self {
        let patch = PatchEmbed::new(in_ch, d_model, patch_size);
        let pos_emb = AbsolutePositionalEmbedding::new(max_len, d_model);
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth { blocks.push(TransformerBlock::new(d_model, d_ff, num_heads)); }
        VisionTransformer { patch_embed: patch, pos_emb, blocks }
    }
    pub fn forward(&self, images: &Tensor) -> Tensor {
        // images : [B, C, H, W]
        let mut patches = self.patch_embed.forward(images); // [B, N_patches, D]
        // Add positional embeddings
        patches = self.pos_emb.forward(&patches);
        // Pass through transformer blocks
        for b in &self.blocks {
            patches = b.forward(&patches);
        }
        patches
    }
}

impl Module for VisionTransformer {
    fn forward(&self, input: &Tensor) -> Tensor { self.forward(input) }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.patch_embed.conv.weight.clone()];
        if let Some(bias) = &self.patch_embed.conv.bias { p.push(bias.clone()); }
        p.extend(self.pos_emb.parameters());
        for blk in &self.blocks { p.extend(blk.parameters()); }
        p
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        let pfx = format!("{}.patch_embed.conv", prefix);
        out.push((format!("{}.weight", pfx), self.patch_embed.conv.weight.clone()));
        if let Some(bias) = &self.patch_embed.conv.bias {
            out.push((format!("{}.bias", pfx), bias.clone()));
        }
        out.extend(self.pos_emb.named_parameters(&format!("{}.pos_emb", prefix)));
        for (i, blk) in self.blocks.iter().enumerate() {
            out.extend(blk.named_parameters(&format!("{}.blocks.{}", prefix, i)));
        }
        out
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
