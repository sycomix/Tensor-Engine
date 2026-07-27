use crate::nn::{Conv2D, LayerNorm, Linear, Module};
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// CLIP Configuration
#[derive(Clone, Debug)]
pub struct CLIPConfig {
    pub embed_dim: usize,
    // Vision
    pub image_size: usize,
    pub vision_layers: usize,
    pub vision_width: usize,
    pub vision_patch_size: usize,
    pub vision_heads: usize,
    // Text
    pub context_length: usize,
    pub vocab_size: usize,
    pub text_width: usize,
    pub text_heads: usize,
    pub text_layers: usize,
}

impl CLIPConfig {
    pub fn vit_b_32() -> Self {
        CLIPConfig {
            embed_dim: 512,
            image_size: 224,
            vision_layers: 12,
            vision_width: 768,
            vision_patch_size: 32,
            vision_heads: 12,
            context_length: 77,
            vocab_size: 49408,
            text_width: 512,
            text_heads: 8,
            text_layers: 12,
        }
    }
}

/// QuickGELU activation function: x * sigmoid(1.702 * x)
#[derive(Clone)]
pub struct QuickGELU {}

impl QuickGELU {
    pub fn new() -> Self {
        QuickGELU {}
    }
}

impl Module for QuickGELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        // x * sigmoid(1.702 * x)
        let scale = Tensor::new(Array::from_elem(IxDyn(&[1][..]), 1.702f32), false);
        let sigmoid = input.mul(&scale).sigmoid();
        input.mul(&sigmoid)
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// CLIP Multi-Head Attention
#[derive(Clone)]
pub struct CLIPAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub q_proj: Linear,
    pub out_proj: Linear,
}

impl CLIPAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        // Bias is true in standard CLIP
        CLIPAttention {
            embed_dim,
            num_heads,
            q_proj: Linear::new(embed_dim, embed_dim, true),
            k_proj: Linear::new(embed_dim, embed_dim, true),
            v_proj: Linear::new(embed_dim, embed_dim, true),
            out_proj: Linear::new(embed_dim, embed_dim, true),
        }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // x shape: [Batch, Seq, Embed]
        let x_shape = x.lock().storage.shape().to_vec();
        let b = x_shape[0];
        let seq = x_shape[1];
        let embed = x_shape[2];
        let head_dim = embed / self.num_heads;

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [Batch, Seq, NumHeads, HeadDim] and Permute to [Batch, NumHeads, Seq, HeadDim]
        let q = q
            .reshape(vec![b, seq, self.num_heads, head_dim])
            .unwrap()
            .permute(vec![0, 2, 1, 3]);
        let k = k
            .reshape(vec![b, seq, self.num_heads, head_dim])
            .unwrap()
            .permute(vec![0, 2, 1, 3]);
        let v = v
            .reshape(vec![b, seq, self.num_heads, head_dim])
            .unwrap()
            .permute(vec![0, 2, 1, 3]);

        // Attention score: Q * K^T / sqrt(d_k)
        // q and k currently have shape [Batch, NumHeads, Seq, HeadDim]
        // We'll flatten the first two dims for batched_matmul since the op
        // expects 3D tensors.  After multiplication we restore the original
        // shape so the rest of the logic remains unchanged.
        let k_t = k.permute(vec![0, 1, 3, 2]); // [Batch, NumHeads, HeadDim, Seq]

        // flatten batch and heads
        let bh = b * self.num_heads;
        let q_flat = q
            .reshape(vec![bh, seq, head_dim])
            .expect("flatten q for batched_matmul");
        let k_flat = k_t
            .reshape(vec![bh, head_dim, seq])
            .expect("flatten k_t for batched_matmul");
        let mut attn_scores = q_flat.batched_matmul(&k_flat); // [bh, seq, seq]
                                                              // restore original dims
        attn_scores = attn_scores
            .reshape(vec![b, self.num_heads, seq, seq])
            .expect("unflatten attn_scores");

        let scale = 1.0 / (head_dim as f32).sqrt();
        let scale_t = Tensor::new(Array::from_elem(IxDyn(&[1][..]), scale), false);
        let mut attn_weights = attn_scores.mul(&scale_t);

        if let Some(m) = mask {
            // mask should be broadcastable to [Batch, NumHeads, Seq, Seq]
            // Usually casual mask is [Seq, Seq]
            attn_weights = attn_weights.add(m);
        }

        let attn_probs = attn_weights.softmax(3); // Softmax over last dim (key seq dim)

        // Output: Prob * V
        // [Batch, NumHeads, Seq, Seq] * [Batch, NumHeads, Seq, HeadDim] -> [Batch, NumHeads, Seq, HeadDim]
        let output = attn_probs.matmul(&v);

        // Permute back to [Batch, Seq, NumHeads, HeadDim] and reshape to [Batch, Seq, Embed]
        let output = output
            .permute(vec![0, 2, 1, 3])
            .reshape(vec![b, seq, embed])
            .expect("clip");

        self.out_proj.forward(&output)
    }
}

impl Module for CLIPAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input, None)
    }
    fn parameters(&self) -> Vec<Tensor> {
        [
            self.q_proj.parameters(),
            self.k_proj.parameters(),
            self.v_proj.parameters(),
            self.out_proj.parameters(),
        ]
        .concat()
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        [
            self.q_proj.named_parameters(&format!("{}.q_proj", prefix)),
            self.k_proj.named_parameters(&format!("{}.k_proj", prefix)),
            self.v_proj.named_parameters(&format!("{}.v_proj", prefix)),
            self.out_proj
                .named_parameters(&format!("{}.out_proj", prefix)),
        ]
        .concat()
    }
    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.q_proj
            .load_state_dict(state, &format!("{}.q_proj", prefix))?;
        self.k_proj
            .load_state_dict(state, &format!("{}.k_proj", prefix))?;
        self.v_proj
            .load_state_dict(state, &format!("{}.v_proj", prefix))?;
        self.out_proj
            .load_state_dict(state, &format!("{}.out_proj", prefix))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// CLIP MLP: FC -> Act -> FC
#[derive(Clone)]
pub struct CLIPMLP {
    pub c_fc: Linear,
    pub c_proj: Linear,
    pub activation: QuickGELU,
}

impl CLIPMLP {
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        CLIPMLP {
            c_fc: Linear::new(embed_dim, hidden_dim, true),
            c_proj: Linear::new(hidden_dim, embed_dim, true),
            activation: QuickGELU::new(),
        }
    }
}

impl Module for CLIPMLP {
    fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.c_fc.forward(input);
        let x = self.activation.forward(&x);
        self.c_proj.forward(&x)
    }
    fn parameters(&self) -> Vec<Tensor> {
        [self.c_fc.parameters(), self.c_proj.parameters()].concat()
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        [
            self.c_fc.named_parameters(&format!("{}.c_fc", prefix)),
            self.c_proj.named_parameters(&format!("{}.c_proj", prefix)),
        ]
        .concat()
    }
    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.c_fc
            .load_state_dict(state, &format!("{}.c_fc", prefix))?;
        self.c_proj
            .load_state_dict(state, &format!("{}.c_proj", prefix))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// CLIP Encoder Layer: Pre-LN Transformer Block
#[derive(Clone)]
pub struct CLIPEncoderLayer {
    pub self_attn: CLIPAttention,
    pub layer_norm1: LayerNorm,
    pub mlp: CLIPMLP,
    pub layer_norm2: LayerNorm,
}

impl CLIPEncoderLayer {
    pub fn new(embed_dim: usize, num_heads: usize, mlp_ratio: usize) -> Self {
        CLIPEncoderLayer {
            self_attn: CLIPAttention::new(embed_dim, num_heads),
            layer_norm1: LayerNorm::new(embed_dim, 2, 1e-5),
            mlp: CLIPMLP::new(embed_dim, embed_dim * mlp_ratio),
            layer_norm2: LayerNorm::new(embed_dim, 2, 1e-5),
        }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // x = x + attn(ln1(x))
        let residual = x.clone();
        let x_norm1 = self.layer_norm1.forward(x);
        let attn_out = self.self_attn.forward(&x_norm1, mask);
        let x = residual.add(&attn_out);

        // x = x + mlp(ln2(x))
        let residual = x.clone();
        let x_norm2 = self.layer_norm2.forward(&x);
        let mlp_out = self.mlp.forward(&x_norm2);
        residual.add(&mlp_out)
    }
}

impl Module for CLIPEncoderLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input, None)
    }
    fn parameters(&self) -> Vec<Tensor> {
        [
            self.self_attn.parameters(),
            self.layer_norm1.parameters(),
            self.mlp.parameters(),
            self.layer_norm2.parameters(),
        ]
        .concat()
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        [
            self.self_attn
                .named_parameters(&format!("{}.self_attn", prefix)),
            self.layer_norm1
                .named_parameters(&format!("{}.layer_norm1", prefix)),
            self.mlp.named_parameters(&format!("{}.mlp", prefix)),
            self.layer_norm2
                .named_parameters(&format!("{}.layer_norm2", prefix)),
        ]
        .concat()
    }
    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.self_attn
            .load_state_dict(state, &format!("{}.self_attn", prefix))?;
        self.layer_norm1
            .load_state_dict(state, &format!("{}.layer_norm1", prefix))?;
        self.mlp
            .load_state_dict(state, &format!("{}.mlp", prefix))?;
        self.layer_norm2
            .load_state_dict(state, &format!("{}.layer_norm2", prefix))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// CLIP Vision Transformer
#[derive(Clone)]
pub struct CLIPVisionTransformer {
    pub conv1: Conv2D, // Patch embedding
    pub class_embedding: Tensor,
    pub positional_embedding: Tensor,
    pub ln_pre: LayerNorm,
    pub layers: Vec<CLIPEncoderLayer>,
    pub ln_post: LayerNorm,
    pub input_resolution: usize,
    pub output_dim: usize,
}

impl CLIPVisionTransformer {
    pub fn new(
        input_resolution: usize,
        patch_size: usize,
        width: usize,
        layers: usize,
        heads: usize,
        output_dim: usize,
    ) -> Self {
        let grid_size = input_resolution / patch_size;
        // conv1: in_channels=3, out=width, kernel=patch_size, stride=patch_size
        let conv1 = Conv2D::new(3, width, patch_size, patch_size, 0, false);

        let _scale = width as f32; // scale for init? standard is randn / sqrt(width) usually or just randn
        let class_embedding = Tensor::new(
            Array::from_shape_fn(IxDyn(&[width][..]), |_| 0.0f32), // Should be learned
            true,
        );
        let positional_embedding = Tensor::new(
            Array::from_shape_fn(IxDyn(&[grid_size * grid_size + 1, width][..]), |_| 0.0f32),
            true,
        );
        let ln_pre = LayerNorm::new(width, 2, 1e-5);

        let mut encoder_layers = Vec::with_capacity(layers);
        for _ in 0..layers {
            encoder_layers.push(CLIPEncoderLayer::new(width, heads, 4));
        }

        let ln_post = LayerNorm::new(width, 2, 1e-5);

        CLIPVisionTransformer {
            conv1,
            class_embedding,
            positional_embedding,
            ln_pre,
            layers: encoder_layers,
            ln_post,
            input_resolution,
            output_dim,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [N, 3, H, W]
        let x = self.conv1.forward(x); // [N, Width, Grid, Grid]

        // Flatten grid to sequence: [N, Width, Grid*Grid] -> [N, Grid*Grid, Width]
        let shape = x.lock().storage.shape().to_vec();
        let b = shape[0];
        let width = shape[1];
        let grid = shape[2] * shape[3];
        let x = x
            .reshape(vec![b, width, grid])
            .unwrap()
            .permute(vec![0, 2, 1]); // [N, Seq, Width]

        // Add class token
        // class_embedding: [Width] -> broadcast to [N, 1, Width]
        let cls = self
            .class_embedding
            .reshape(vec![1, 1, width])
            .expect("clip");
        // Broadcast CLS token to batch size using broadcasting
        let cls_batch = if b == 1 {
            cls.clone()
        } else {
            // Use broadcasting: add CLS [1, 1, width] to zeros [b, 1, width]
            let zeros = Tensor::new(Array::zeros(IxDyn(&[b, 1, width][..])), false);
            zeros.add(&cls)
        };

        let x = Tensor::concat(&vec![cls_batch, x], 1); // [N, Grid*Grid+1, Width]

        // Add positional embedding
        let x = x.add(&self.positional_embedding);

        let mut x = self.ln_pre.forward(&x);

        for layer in &self.layers {
            x = layer.forward(&x, None);
        }

        let x = self.ln_post.forward(&x);

        // Take class token (index 0)
        // using slice
        // x[:, 0, :]
        // Implement slicing helper or use Op
        // Slice op is available via tensor.slice in python binding but not directly exposed as method in tensor.rs easily?
        // Wait, slice_n is in LSTMCell.
        // I should stick to manual slicing via op if needed or implement `slice` on Tensor.
        // Actually, for CLS token we can just take the first element if we flatten? No.

        // We need a Slice operation.
        // `crate::ops::Slice` exists.
        // Slice::new(axis, start, length)
        let cls_out = Tensor::apply(
            Arc::new(crate::ops::Slice::new(1, 0, 1)),
            std::slice::from_ref(&x),
        ); // [N, 1, Width]

        let cls_out = cls_out.reshape(vec![b, width]).expect("clip");

        // Project to output_dim?
        // OpenAI CLIP VisionTransformer output IS the state after ln_post.
        // Then there is a `visual_projection` parameter in the CLIP model itself.
        // So VisionTransformer returns [N, Width].

        cls_out
    }
}

impl Module for CLIPVisionTransformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }
    // Parameters
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.conv1.parameters();
        p.push(self.class_embedding.clone());
        p.push(self.positional_embedding.clone());
        p.extend(self.ln_pre.parameters());
        for l in &self.layers {
            p.extend(l.parameters());
        }
        p.extend(self.ln_post.parameters());
        p
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut p = self.conv1.named_parameters(&format!("{}.conv1", prefix));
        p.push((
            format!("{}.class_embedding", prefix),
            self.class_embedding.clone(),
        ));
        p.push((
            format!("{}.positional_embedding", prefix),
            self.positional_embedding.clone(),
        ));
        p.extend(self.ln_pre.named_parameters(&format!("{}.ln_pre", prefix)));
        for (i, l) in self.layers.iter().enumerate() {
            p.extend(l.named_parameters(&format!("{}.resblocks.{}", prefix, i)));
        }
        p.extend(
            self.ln_post
                .named_parameters(&format!("{}.ln_post", prefix)),
        );
        p
    }
    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.conv1
            .load_state_dict(state, &format!("{}.conv1", prefix))?;
        // embeddings might be loaded manually as they are tensors not modules
        let key_cls = format!("{}.class_embedding", prefix);
        if let Some(t) = state.get(&key_cls) {
            self.class_embedding = t.clone();
        }
        let key_pos = format!("{}.positional_embedding", prefix);
        if let Some(t) = state.get(&key_pos) {
            self.positional_embedding = t.clone();
        }

        self.ln_pre
            .load_state_dict(state, &format!("{}.ln_pre", prefix))?;
        for (i, l) in self.layers.iter_mut().enumerate() {
            l.load_state_dict(state, &format!("{}.transformer.resblocks.{}", prefix, i))?;
        }
        self.ln_post
            .load_state_dict(state, &format!("{}.ln_post", prefix))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// CLIP Text Transformer
#[derive(Clone)]
pub struct CLIPTextTransformer {
    pub token_embedding: crate::nn::embedding::SparseEmbedding, // vocab -> width
    pub positional_embedding: Tensor,                           // [context_length, width]
    pub layers: Vec<CLIPEncoderLayer>,
    pub ln_final: LayerNorm,
    pub width: usize,
    pub context_length: usize,
}

impl CLIPTextTransformer {
    pub fn new(
        vocab_size: usize,
        width: usize,
        context_length: usize,
        layers: usize,
        heads: usize,
    ) -> Self {
        let token_embedding = crate::nn::embedding::SparseEmbedding::new(vocab_size, width);
        let positional_embedding = Tensor::new(
            Array::from_shape_fn(IxDyn(&[context_length, width][..]), |_| 0.0f32),
            true,
        );
        let mut encoder_layers = Vec::with_capacity(layers);
        for _ in 0..layers {
            encoder_layers.push(CLIPEncoderLayer::new(width, heads, 4));
        }
        let ln_final = LayerNorm::new(width, 2, 1e-5);

        CLIPTextTransformer {
            token_embedding,
            positional_embedding,
            layers: encoder_layers,
            ln_final,
            width,
            context_length,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [N, Seq] (indices)
        let x = self.token_embedding.forward(x);
        // Add positional embedding
        // positional_embedding is [Seq, Width]
        // Broadcast add to [N, Seq, Width]
        let x = x.add(&self.positional_embedding);

        let mut x = x;
        // Construct causal mask [Seq, Seq]
        // Lower traingular is 0, upper is -inf (for attention weights)
        // CLIP matches PyTorch convention where mask=0 is keep, mask=-inf is ignore? usually.
        // Let's create proper mask.
        // For now, assuming masking is handled inside attention if needed or we pass it?
        // CLIP uses causal attention for text.
        // We need to construct a causal mask.
        let shape = x.lock().storage.shape().to_vec();
        let seq = shape[1];
        let mut mask_arr = Array::from_elem(IxDyn(&[seq, seq][..]), -f32::INFINITY);
        for i in 0..seq {
            for j in 0..=i {
                mask_arr[[i, j]] = 0.0;
            }
        }
        let mask = Tensor::new(mask_arr, false);

        for layer in &self.layers {
            x = layer.forward(&x, Some(&mask));
        }

        let x = self.ln_final.forward(&x);

        // EOT token features.
        // In CLIP, x is indices, usually EOT is the last token or we pass argmax.
        // For standard usage, we need to gather standard EOT indices.
        // For now, let's return the whole sequence or just the last one?
        // Select the highest-scoring token independently for each batch row.
        // We don't have indices passed here.
        // Let's return the whole sequence [N, Seq, Width] and let the caller handle pooling/indexing.
        x
    }
}

impl Module for CLIPTextTransformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }
    // Parameters implementation
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.token_embedding.parameters();
        p.push(self.positional_embedding.clone());
        for l in &self.layers {
            p.extend(l.parameters());
        }
        p.extend(self.ln_final.parameters());
        p
    }
    // Named parameters
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut p = self
            .token_embedding
            .named_parameters(&format!("{}.token_embedding", prefix));
        p.push((
            format!("{}.positional_embedding", prefix),
            self.positional_embedding.clone(),
        ));
        for (i, l) in self.layers.iter().enumerate() {
            p.extend(l.named_parameters(&format!("{}.transformer.resblocks.{}", prefix, i)));
        }
        p.extend(
            self.ln_final
                .named_parameters(&format!("{}.ln_final", prefix)),
        );
        p
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.token_embedding
            .load_state_dict(state, &format!("{}.token_embedding", prefix))?;
        let key_pos = format!("{}.positional_embedding", prefix);
        if let Some(t) = state.get(&key_pos) {
            self.positional_embedding = t.clone();
        }
        for (i, l) in self.layers.iter_mut().enumerate() {
            l.load_state_dict(state, &format!("{}.transformer.resblocks.{}", prefix, i))?;
        }
        self.ln_final
            .load_state_dict(state, &format!("{}.ln_final", prefix))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// CLIP Model
#[derive(Clone)]
pub struct CLIP {
    pub visual: CLIPVisionTransformer,
    pub text: CLIPTextTransformer,
    pub visual_projection: Linear,
    pub text_projection: Linear,
    pub logit_scale: Tensor,
}

impl CLIP {
    pub fn new(config: CLIPConfig) -> Self {
        let visual = CLIPVisionTransformer::new(
            config.image_size,
            config.vision_patch_size,
            config.vision_width,
            config.vision_layers,
            config.vision_heads,
            config.embed_dim,
        );
        let text = CLIPTextTransformer::new(
            config.vocab_size,
            config.text_width,
            config.context_length,
            config.text_layers,
            config.text_heads,
        );
        let visual_projection = Linear::new(config.vision_width, config.embed_dim, false);
        let text_projection = Linear::new(config.text_width, config.embed_dim, false);
        let logit_scale = Tensor::new(Array::from_elem(IxDyn(&[1][..]), 0.07f32.ln()), true);
        CLIP {
            visual,
            text,
            visual_projection,
            text_projection,
            logit_scale,
        }
    }

    pub fn forward(&self, image: &Tensor, text: &Tensor) -> (Tensor, Tensor) {
        let image_features = self.visual_projection.forward(&self.visual.forward(image));
        let text_sequence = self.text.forward(text);
        let text_shape = text_sequence.shape();
        let text_last = Tensor::apply(
            Arc::new(crate::ops::Slice::new(1, text_shape[1] - 1, 1)),
            std::slice::from_ref(&text_sequence),
        )
        .reshape(vec![text_shape[0], text_shape[2]])
        .expect("CLIP text pooling reshape must succeed");
        let text_features = self.text_projection.forward(&text_last);

        // Normalize features
        let image_norm = self.l2_normalize(&image_features);
        let text_norm = self.l2_normalize(&text_features);

        (image_norm, text_norm)
    }

    fn l2_normalize(&self, x: &Tensor) -> Tensor {
        // x / (x^2.sum(-1, keepdim=True).sqrt() + eps)
        let sq = x.pow(2.0);
        let sum = sq.sum_axis(-1, true);

        // 1e-6 is safer for f32
        let eps = Tensor::new(Array::from_elem(IxDyn(&[1][..]), 1e-6), false);
        let sum_safe = sum.add(&eps);
        let norm = sum_safe.pow(0.5);
        let inv_norm = norm.pow(-1.0);
        x.mul(&inv_norm)
    }
}

impl Module for CLIP {
    fn forward(&self, input: &Tensor) -> Tensor {
        log::warn!("CLIP::forward(Tensor) called; use forward(image, text) instead.");
        input.clone()
    }

    fn parameters(&self) -> Vec<Tensor> {
        [
            self.visual.parameters(),
            self.text.parameters(),
            self.visual_projection.parameters(),
            self.text_projection.parameters(),
            vec![self.logit_scale.clone()],
        ]
        .concat()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut p = self.visual.named_parameters(&format!("{}.visual", prefix));
        p.extend(self.text.named_parameters(&format!("{}.text", prefix)));
        p.extend(
            self.visual_projection
                .named_parameters(&format!("{}.visual_projection", prefix)),
        );
        p.extend(
            self.text_projection
                .named_parameters(&format!("{}.text_projection", prefix)),
        );
        p.push((format!("{}.logit_scale", prefix), self.logit_scale.clone()));
        p
    }

    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        self.visual
            .load_state_dict(state, &format!("{}.visual", prefix))?;
        self.text
            .load_state_dict(state, &format!("{}.text", prefix))?;
        self.visual_projection
            .load_state_dict(state, &format!("{}.visual_projection", prefix))?;
        self.text_projection
            .load_state_dict(state, &format!("{}.text_projection", prefix))?;

        let key_scale = format!("{}.logit_scale", prefix);
        if let Some(t) = state.get(&key_scale) {
            self.logit_scale = t.clone();
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
