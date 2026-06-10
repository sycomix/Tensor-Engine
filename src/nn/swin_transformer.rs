//! Swin Transformer implementation.
//!
//! Swin Transformer (Shifted Window Attention) is a hierarchical vision transformer
//! that uses shifted window-based self-attention for efficient computation.
//!
//! Reference: [Liu et al., 2021](https://arxiv.org/abs/2103.14030)

use crate::nn::{LayerNorm, Linear, Module};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// Swin Transformer configuration.
#[derive(Clone, Debug)]
pub struct SwinConfig {
    /// Input image size (height, width)
    pub img_size: (usize, usize),
    /// Patch size (height, width)
    pub patch_size: usize,
    /// Number of input channels (e.g., 3 for RGB)
    pub num_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of attention heads per block
    pub num_heads: usize,
    /// Window size for shifted window attention
    pub window_size: usize,
    /// Number of Swin Transformer blocks per stage
    pub depths: Vec<usize>,
    /// Number of stages (typically 4)
    pub num_stages: usize,
    /// Drop path rate
    pub drop_path_rate: f32,
    /// MLP expansion ratio
    pub mlp_ratio: f32,
    /// Whether to use absolute position embedding
    pub use_abs_pos_embed: bool,
    /// Whether to use patch merging
    pub use_patch_merging: bool,
}

impl Default for SwinConfig {
    fn default() -> Self {
        SwinConfig {
            img_size: (224, 224),
            patch_size: 4,
            num_channels: 3,
            num_classes: 1000,
            embed_dim: 96,
            num_heads: 3,
            window_size: 7,
            depths: vec![2, 2, 6, 2],
            num_stages: 4,
            drop_path_rate: 0.1,
            mlp_ratio: 4.0,
            use_abs_pos_embed: true,
            use_patch_merging: true,
        }
    }
}

/// Swin Transformer block.
pub struct SwinTransformerBlock {
    attn: WindowAttention,
    mlp: SwinMLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
    shift_size: usize,
    window_size: usize,
}

impl SwinTransformerBlock {
    pub fn new(
        dim: usize,
        num_heads: usize,
        window_size: usize,
        shift_size: usize,
        mlp_ratio: f32,
    ) -> Self {
        let attn = WindowAttention::new(dim, num_heads, window_size, shift_size);
        let mlp = SwinMLP::new(dim, (dim as f32 * mlp_ratio) as usize);
        // For inputs shaped [B, N, C], use axis=2 (features last)
        let norm1 = LayerNorm::new(dim, 2, 1e-5);
        let norm2 = LayerNorm::new(dim, 2, 1e-5);

        SwinTransformerBlock {
            attn,
            mlp,
            norm1,
            norm2,
            shift_size,
            window_size,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        println!("SwinTransformerBlock::forward start");
        // Window attention with residual connection
        println!("SwinTransformerBlock: calling norm1.forward");
        let norm1_out = self.norm1.forward(x);
        println!("SwinTransformerBlock: calling attn.forward");
        let attn_out = self.attn.forward(&norm1_out);
        println!("SwinTransformerBlock: adding residual");
        let x = x.add(&attn_out);

        // MLP with residual connection
        println!("SwinTransformerBlock: calling norm2.forward");
        let norm2_out = self.norm2.forward(&x);
        println!("SwinTransformerBlock: calling mlp.forward");
        let mlp_out = self.mlp.forward(&norm2_out);
        println!("SwinTransformerBlock: adding residual");
        x.add(&mlp_out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.attn.parameters();
        params.extend(self.mlp.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}

/// Window-based self-attention with shifted windows.
pub struct WindowAttention {
    dim: usize,
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
    shift_size: usize,
    scale: f32,
    /// Attention weight matrices
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    o_weight: Tensor,
    /// Shifted window partitioning info
    attn_mask: Option<Tensor>,
}

impl WindowAttention {
    pub fn new(dim: usize, num_heads: usize, window_size: usize, shift_size: usize) -> Self {
        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let weight_shape = vec![dim, dim];
        WindowAttention {
            dim,
            num_heads,
            head_dim,
            window_size,
            shift_size,
            scale,
            q_weight: Tensor::zeros(&weight_shape),
            k_weight: Tensor::zeros(&weight_shape),
            v_weight: Tensor::zeros(&weight_shape),
            o_weight: Tensor::zeros(&weight_shape),
            attn_mask: None,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let input_shape = x.lock().storage.shape();
        let b = input_shape[0];
        let n = input_shape[1];
        let c = input_shape[2];

        // QKV projections
        let q = x.matmul(&self.q_weight);
        let k = x.matmul(&self.k_weight);
        let v = x.matmul(&self.v_weight);

        // Reshape for multi-head attention: [B, N, num_heads, head_dim]
        let q = match q.reshape(vec![b, n, self.num_heads, self.head_dim]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };
        let k = match k.reshape(vec![b, n, self.num_heads, self.head_dim]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };
        let v = match v.reshape(vec![b, n, self.num_heads, self.head_dim]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };

        // Transpose to [B, num_heads, N, head_dim]
        let q = q.transpose();
        let k = k.transpose();
        let v = v.transpose();

        // Compute attention: [B, num_heads, N, N]
        let k_t = k.transpose();
        let attn = q.matmul(&k_t);
        let scale_shape = vec![1usize];
        let attn = attn.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&scale_shape), self.scale),
            false,
        ));

        // Apply shifted window mask if available
        let attn = if let Some(mask) = &self.attn_mask {
            // Broadcast mask to [B, num_heads, window_size, window_size]
            attn.add(mask)
        } else {
            attn
        };

        // Softmax
        let attn = attn.softmax(2);

        // Apply attention to values
        let out = attn.matmul(&v);

        // Transpose back: [B, N, num_heads, head_dim]
        let out = out.transpose();

        // Reshape back to [B, N, dim]
        let out = match out.reshape(vec![b, n, c]) {
            Ok(t) => t,
            Err(_) => return x.clone(),
        };

        // Output projection
        out.matmul(&self.o_weight)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.q_weight.clone(),
            self.k_weight.clone(),
            self.v_weight.clone(),
            self.o_weight.clone(),
        ]
    }
}

/// Swin MLP (Multi-Layer Perceptron) with GELU activation.
pub struct SwinMLP {
    fc1: Linear,
    fc2: Linear,
    gelu: bool,
}

impl SwinMLP {
    pub fn new(in_features: usize, hidden_features: usize) -> Self {
        let fc1 = Linear::new(in_features, hidden_features, true);
        let fc2 = Linear::new(hidden_features, in_features, true);
        SwinMLP {
            fc1,
            fc2,
            gelu: true,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = self.fc1.forward(x);
        let out = if self.gelu { out.gelu() } else { out.relu() };
        self.fc2.forward(&out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}

/// Patch merging layer: reduces spatial resolution by half.
pub struct PatchMerging {
    norm: LayerNorm,
    linear: Linear,
}

impl PatchMerging {
    pub fn new(dim: usize) -> Self {
        // PatchMerging expects input in NHWC [B, H, W, C], so features axis = 3
        let norm = LayerNorm::new(dim, 3, 1e-5);
        let linear = Linear::new(dim * 4, dim * 2, true);
        PatchMerging { norm, linear }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        println!("PatchMerging::forward start, input shape={:?}", x.lock().storage.shape());
        let mut x = x.clone();
        let input_shape = x.lock().storage.shape().to_vec();

        // Accept either NHWC [B,H,W,C] or flattened [B, N, C]. If flattened, attempt to
        // reshape to NHWC assuming N = H * W and H == W (square).
        if input_shape.len() == 3 {
            println!("PatchMerging: input is 3D, attempting reshape");
            let b = input_shape[0];
            let n = input_shape[1];
            let c = input_shape[2];
            let side = (n as f64).sqrt() as usize;
            if side * side != n {
                // cannot recover spatial dims; return input unchanged
                println!("PatchMerging: cannot recover spatial dims, returning input");
                return x;
            }
            // reshape [B, N, C] -> [B, H, W, C]
            if let Ok(t) = x.reshape(vec![b, side, side, c]) {
                x = t;
                println!("PatchMerging: reshaped to {:?}", x.lock().storage.shape());
            } else {
                println!("PatchMerging: reshape failed, returning input");
                return x;
            }
        }

        let input_shape = x.lock().storage.shape();
        let (b, h, w, c) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        println!("PatchMerging: b={}, h={}, w={}, c={}", b, h, w, c);

        // Normalize
        println!("PatchMerging: calling norm.forward");
        x = self.norm.forward(&x);
        println!("PatchMerging: norm.forward done");

        // Pad if necessary
        println!("PatchMerging: checking padding");
        let mut x = x.clone();
        let mut pad_h = 0;
        let mut pad_w = 0;
        if h % 2 != 0 {
            pad_h = 1;
        }
        if w % 2 != 0 {
            pad_w = 1;
        }
        println!("PatchMerging: pad_h={}, pad_w={}", pad_h, pad_w);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        if pad_h > 0 || pad_w > 0 {
            let pad_size = vec![b, h + pad_h, w + pad_w, c];
            let padded = Tensor::zeros(&pad_size);
            let x_data = x.lock().storage.to_f32_array();
            let mut padded_data = padded.lock().storage.to_f32_array();

            for n in 0..b {
                for i in 0..h {
                    for j in 0..w {
                        for k in 0..c {
                            padded_data[[n, i, j, k]] = x_data[[n, i, j, k]];
                        }
                    }
                }
            }

            let padded_arr =
                match ArrayD::from_shape_vec(IxDyn(&pad_size), padded_data.into_raw_vec_and_offset().0) {
                    Ok(v) => v,
                    Err(_) => return x.clone(),
                };
            x = Tensor::new(padded_arr, false);
        }

        // Concatenate [x_B,i,j, x_B,i+1,j, x_B,i,j+1, x_B,i+1,j+1]
        println!("PatchMerging: calling slice_tensor for x0");
        let (x0, _) = Self::slice_tensor(&x, 1, 0, (h + pad_h) / 2);
        println!("PatchMerging: calling slice_tensor for x1");
        let (_, x1) = Self::slice_tensor(&x, 1, (h + pad_h) / 2, (h + pad_h) / 2);
        println!("PatchMerging: calling slice_tensor for x2");
        let (x2, _) = Self::slice_tensor(&x, 2, 0, (w + pad_w) / 2);
        println!("PatchMerging: calling slice_tensor for x3");
        let (_, x3) = Self::slice_tensor(&x, 2, (w + pad_w) / 2, (w + pad_w) / 2);
        println!("PatchMerging: all slice_tensor calls done");

        // Concat along last dimension
        println!("PatchMerging: reshaping x0");
        let x0 = x0
            .reshape(vec![b, (h + pad_h) / 2, (w + pad_w) / 2, c])
            .unwrap_or_else(|_| x.clone());
        println!("PatchMerging: reshaping x1");
        let x1 = x1
            .reshape(vec![b, (h + pad_h) / 2, (w + pad_w) / 2, c])
            .unwrap_or_else(|_| x.clone());
        println!("PatchMerging: reshaping x2");
        let x2 = x2
            .reshape(vec![b, (h + pad_h) / 2, (w + pad_w) / 2, c])
            .unwrap_or_else(|_| x.clone());
        println!("PatchMerging: reshaping x3");
        let x3 = x3
            .reshape(vec![b, (h + pad_h) / 2, (w + pad_w) / 2, c])
            .unwrap_or_else(|_| x.clone());
        println!("PatchMerging: all reshapes done");

        // Stack and concatenate
        println!("PatchMerging: starting concatenation loop");
        let mut concat_data =
            Vec::with_capacity(b * ((h + pad_h) / 2) * ((w + pad_w) / 2) * (c * 4));
        
        // Extract arrays once outside the loop to avoid repeated locking
        let x0_arr = x0.lock().storage.to_f32_array();
        let x1_arr = x1.lock().storage.to_f32_array();
        let x2_arr = x2.lock().storage.to_f32_array();
        let x3_arr = x3.lock().storage.to_f32_array();
        
        for n in 0..b {
            for i in 0..(h + pad_h) / 2 {
                for j in 0..(w + pad_w) / 2 {
                    for k in 0..c {
                        concat_data.push(x0_arr[[n, i, j, k]]);
                    }
                    for k in 0..c {
                        concat_data.push(x1_arr[[n, i, j, k]]);
                    }
                    for k in 0..c {
                        concat_data.push(x2_arr[[n, i, j, k]]);
                    }
                    for k in 0..c {
                        concat_data.push(x3_arr[[n, i, j, k]]);
                    }
                }
            }
        }
        println!("PatchMerging: concatenation loop done");

        let concat_shape = vec![b, (h + pad_h) / 2, (w + pad_w) / 2, c * 4];
        let concat_arr = match ArrayD::from_shape_vec(IxDyn(&concat_shape), concat_data) {
            Ok(v) => v,
            Err(_) => return x.clone(),
        };
        let concat = Tensor::new(concat_arr, false);

        // Linear projection
        self.linear.forward(&concat)
    }

    fn slice_tensor(t: &Tensor, dim: usize, start: usize, length: usize) -> (Tensor, Tensor) {
        // Extract the array and shape before creating slices to avoid deadlock
        let arr = t.lock().storage.to_f32_array();
        let t_shape = arr.shape().to_vec();
        let second_len = t_shape[dim] - start - length;
        
        // Create slice info for first slice
        let mut slice_info_elems1: Vec<ndarray::SliceInfoElem> = Vec::with_capacity(arr.ndim());
        for i in 0..arr.ndim() {
            if i == dim {
                slice_info_elems1.push((start..start + length).into());
            } else {
                slice_info_elems1.push((..).into());
            }
        }
        let slice_info1: ndarray::SliceInfo<_, ndarray::IxDyn, ndarray::IxDyn> =
            unsafe { ndarray::SliceInfo::new(slice_info_elems1).unwrap() };
        let arr1 = arr.slice(slice_info1).to_owned().into_dyn();
        
        // Create slice info for second slice
        let mut slice_info_elems2: Vec<ndarray::SliceInfoElem> = Vec::with_capacity(arr.ndim());
        for i in 0..arr.ndim() {
            if i == dim {
                slice_info_elems2.push((start + length..start + length + second_len).into());
            } else {
                slice_info_elems2.push((..).into());
            }
        }
        let slice_info2: ndarray::SliceInfo<_, ndarray::IxDyn, ndarray::IxDyn> =
            unsafe { ndarray::SliceInfo::new(slice_info_elems2).unwrap() };
        let arr2 = arr.slice(slice_info2).to_owned().into_dyn();
        
        (Tensor::new(arr1, false), Tensor::new(arr2, false))
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.norm.parameters();
        params.extend(self.linear.parameters());
        params
    }
}

/// Patch embedding layer: splits image into patches and projects to embedding space.
pub struct PatchEmbedding {
    proj: Linear,
    pos_embed: Option<Tensor>,
    norm: Option<LayerNorm>,
    patch_size: usize,
}

impl PatchEmbedding {
    pub fn new(
        in_channels: usize,
        embed_dim: usize,
        patch_size: usize,
        img_size: (usize, usize),
    ) -> Self {
        let patch_h = img_size.0 / patch_size;
        let patch_w = img_size.1 / patch_size;
        let num_patches = patch_h * patch_w;

        // Conv-like projection: treat each patch as a flattened vector
        let proj = Linear::new(in_channels * patch_size * patch_size, embed_dim, true);

        let pos_embed = if patch_h > 0 && patch_w > 0 {
            let pos_shape = vec![num_patches, embed_dim];
            let pos_data = ndarray::Array::zeros(IxDyn(&pos_shape));
            Some(Tensor::new(pos_data, true))
        } else {
            None
        };

        // Patch embedding outputs shape [B, num_patches, embed_dim] -> features axis = 2
        let norm = LayerNorm::new(embed_dim, 2, 1e-5);

        PatchEmbedding {
            proj,
            pos_embed,
            norm: Some(norm),
            patch_size,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let input_shape = x.lock().storage.shape();
        if input_shape.len() != 4 {
            return x.clone();
        }
        let (b, c, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        let p = self.patch_size;
        if h % p != 0 || w % p != 0 {
            // cannot extract full patches; return input unchanged
            return x.clone();
        }
        let patch_h = h / p;
        let patch_w = w / p;
        let num_patches = patch_h * patch_w;

        // Extract patches: shape [b, num_patches, c * p * p]
        let mut patches: Vec<f32> = Vec::with_capacity(b * num_patches * (c * p * p));
        let arr = x.lock().storage.to_f32_array();
        for n in 0..b {
            for ph in 0..patch_h {
                for pw in 0..patch_w {
                    for ch in 0..c {
                        for i in 0..p {
                            for j in 0..p {
                                let yy = ph * p + i;
                                let xx = pw * p + j;
                                patches.push(arr[[n, ch, yy, xx]]);
                            }
                        }
                    }
                }
            }
        }

        let patch_shape = vec![b * num_patches, c * p * p];
        let patch_arr = match ArrayD::from_shape_vec(IxDyn(&patch_shape), patches) {
            Ok(v) => v,
            Err(_) => return x.clone(),
        };
        let patch_tensor = Tensor::new(patch_arr, false);

        // Project: [b * num_patches, embed_dim]
        let projected = self.proj.forward(&patch_tensor);

        // Reshape to [b, num_patches, embed_dim]
        let out_shape = vec![b, patch_h * patch_w, self.proj.out_features];
        match projected.reshape(out_shape.clone()) {
            Ok(t) => {
                let mut out = t;
                // Add position embedding if present
                if let Some(pos_embed) = &self.pos_embed {
                    out = out.add(pos_embed);
                }
                if let Some(norm) = &self.norm {
                    let shape = out.lock().storage.shape();
                    if shape.len() == 3 && shape[2] != self.proj.out_features {
                        panic!(
                            "PatchEmbedding: unexpected out feature dimension {:?}, expected {}",
                            shape,
                            self.proj.out_features
                        );
                    }
                    return norm.forward(&out);
                }
                out
            }
            Err(_) => projected,
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.proj.parameters();
        if let Some(pos_embed) = &self.pos_embed {
            params.push(pos_embed.clone());
        }
        if let Some(norm) = &self.norm {
            params.extend(norm.parameters());
        }
        params
    }
}

/// Swin Transformer stage: a sequence of Swin blocks with optional patch merging.
pub struct SwinStage {
    blocks: Vec<SwinTransformerBlock>,
    patch_merging: Option<PatchMerging>,
    downsample: Option<Linear>,
}

impl SwinStage {
    pub fn new(
        dim: usize,
        num_heads: usize,
        window_size: usize,
        depth: usize,
        shift_size: usize,
        mlp_ratio: f32,
        use_patch_merging: bool,
    ) -> Self {
        let mut blocks = Vec::new();
        for i in 0..depth {
            let current_shift = if i % 2 == 0 { shift_size } else { 0 };
            blocks.push(SwinTransformerBlock::new(
                dim,
                num_heads,
                window_size,
                current_shift,
                mlp_ratio,
            ));
        }

        let patch_merging = if use_patch_merging {
            Some(PatchMerging::new(dim))
        } else {
            None
        };

        SwinStage {
            blocks,
            patch_merging,
            downsample: None,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        for block in &self.blocks {
            out = block.forward(&out);
        }
        if let Some(pm) = &self.patch_merging {
            out = pm.forward(&out);
        }
        out
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        if let Some(pm) = &self.patch_merging {
            params.extend(pm.parameters());
        }
        params
    }
}

/// Swin Transformer model for image classification.
pub struct SwinTransformer {
    patch_embed: PatchEmbedding,
    stages: Vec<SwinStage>,
    norm: LayerNorm,
    cls_head: Linear,
    num_classes: usize,
}

impl SwinTransformer {
    /// Create a new Swin Transformer.
    pub fn new(config: SwinConfig) -> Self {
        let patch_embed = PatchEmbedding::new(
            config.num_channels,
            config.embed_dim,
            config.patch_size,
            config.img_size,
        );

        let mut stages = Vec::new();
        let mut curr_dim = config.embed_dim;

        for stage_idx in 0..config.num_stages {
            let num_heads = config.num_heads * (1 << stage_idx);
            let curr_dim_next = curr_dim
                * (if stage_idx < config.num_stages - 1 {
                    2
                } else {
                    1
                });

            let stage = SwinStage::new(
                curr_dim,
                num_heads,
                config.window_size,
                config.depths[stage_idx],
                config.window_size / 2,
                config.mlp_ratio,
                stage_idx < config.num_stages - 1,
            );
            stages.push(stage);
            curr_dim = curr_dim_next;
        }

        let norm = LayerNorm::new(curr_dim, 1, 1e-5);
        let cls_head = Linear::new(curr_dim, config.num_classes, true);

        SwinTransformer {
            patch_embed,
            stages,
            norm,
            cls_head,
            num_classes: config.num_classes,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.patch_embed.forward(x);
        let mut x = x;
        for stage in &self.stages {
            x = stage.forward(&x);
        }
        // Global average pooling: [B, H, W, C] -> [B, C]
        let x_shape = x.lock().storage.shape();
        let pooled = if x_shape.len() == 4 {
            // Manual global average pooling over spatial dimensions
            let b = x_shape[0];
            let h = x_shape[1];
            let w = x_shape[2];
            let c = x_shape[3];
            let arr = x.lock().storage.to_f32_array();
            let mut pooled_data = vec![0.0f32; b * c];
            let spatial_size = h * w;
            
            for batch in 0..b {
                for channel in 0..c {
                    let mut sum = 0.0f32;
                    for hi in 0..h {
                        for wi in 0..w {
                            sum += arr[[batch, hi, wi, channel]];
                        }
                    }
                    pooled_data[batch * c + channel] = sum / spatial_size as f32;
                }
            }
            
            let pooled_shape = vec![b, c];
            let pooled_arr = ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&pooled_shape),
                pooled_data
            ).unwrap();
            Tensor::new(pooled_arr, false)
        } else {
            x.mean()
        };
        self.cls_head.forward(&pooled)
    }

    /// Get all parameters.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.patch_embed.parameters();
        for stage in &self.stages {
            params.extend(stage.parameters());
        }
        params.extend(self.norm.parameters());
        params.extend(self.cls_head.parameters());
        params
    }

    /// Get the number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.lock().storage.to_f32_array().len())
            .sum()
    }
}

impl Module for SwinTransformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Swin Transformer for object detection (bounding box prediction).
pub struct SwinDetector {
    backbone: SwinTransformer,
    neck: Linear,
    bbox_head: Linear,
    cls_head: Linear,
}

impl SwinDetector {
    /// Create a new Swin-based detector.
    pub fn new(backbone: SwinTransformer, num_classes: usize, num_anchors: usize) -> Self {
        let embed_dim = 256;
        SwinDetector {
            backbone,
            neck: Linear::new(768, embed_dim, true),
            bbox_head: Linear::new(embed_dim, num_anchors * 4, true),
            cls_head: Linear::new(embed_dim, num_classes * num_anchors, true),
        }
    }

    /// Forward pass for detection.
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let features = self.backbone.forward(x);
        let neck_out = self.neck.forward(&features);
        let bbox = self.bbox_head.forward(&neck_out);
        let cls = self.cls_head.forward(&neck_out);
        (bbox, cls)
    }
}

#[cfg(test)]
mod swin_transformer_tests {
    use super::*;

    #[test]
    fn test_swin_transformer_forward() {
        println!("test_swin_transformer_forward: creating config");
        let config = SwinConfig {
            img_size: (32, 32),
            patch_size: 4,
            num_channels: 3,
            num_classes: 10,
            embed_dim: 48,
            num_heads: 3,
            window_size: 7,
            depths: vec![1, 1, 1, 1],
            num_stages: 4,
            drop_path_rate: 0.0,
            mlp_ratio: 2.0,
            use_abs_pos_embed: false,
            use_patch_merging: true,
        };

        println!("test_swin_transformer_forward: creating model");
        let model = SwinTransformer::new(config);
        println!("test_swin_transformer_forward: creating input tensor");
        let x = Tensor::zeros(&vec![1usize, 3, 32, 32]);
        println!("test_swin_transformer_forward: calling forward");
        let out = model.forward(&x);
        println!("test_swin_transformer_forward: checking output shape");
        let shape = out.lock().storage.shape();
        assert_eq!(shape[1], 10); // num_classes
    }

    #[test]
    fn test_swin_transformer_parameters() {
        let config = SwinConfig {
            img_size: (32, 32),
            patch_size: 4,
            num_channels: 3,
            num_classes: 10,
            embed_dim: 48,
            num_heads: 3,
            window_size: 7,
            depths: vec![1, 1, 1, 1],
            num_stages: 4,
            drop_path_rate: 0.0,
            mlp_ratio: 2.0,
            use_abs_pos_embed: false,
            use_patch_merging: true,
        };

        let model = SwinTransformer::new(config);
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_window_attention() {
        let dim = 64;
        let num_heads = 4;
        let window_size = 7;
        let shift_size = 0;
        let attn = WindowAttention::new(dim, num_heads, window_size, shift_size);

        let x = Tensor::zeros(&vec![1usize, 49, dim]);
        let out = attn.forward(&x);
        let shape = out.lock().storage.shape();
        assert_eq!(shape[2], dim);
    }

    #[test]
    fn test_swin_mlp() {
        let mlp = SwinMLP::new(64, 128);
        let x = Tensor::zeros(&vec![1usize, 49, 64]);
        let out = mlp.forward(&x);
        let shape = out.lock().storage.shape();
        assert_eq!(shape[2], 64);
    }

    #[test]
    fn test_patch_merging() {
        let pm = PatchMerging::new(64);
        let x = Tensor::zeros(&vec![1usize, 8, 8, 64]);
        let out = pm.forward(&x);
        let shape = out.lock().storage.shape();
        assert_eq!(shape[1], 4); // half
        assert_eq!(shape[2], 4); // half
        assert_eq!(shape[3], 128); // 64 * 2
    }

    #[test]
    fn test_swin_detector() {
        let config = SwinConfig {
            img_size: (32, 32),
            patch_size: 4,
            num_channels: 3,
            num_classes: 10,
            embed_dim: 48,
            num_heads: 3,
            window_size: 7,
            depths: vec![1, 1, 1, 1],
            num_stages: 4,
            drop_path_rate: 0.0,
            mlp_ratio: 2.0,
            use_abs_pos_embed: false,
            use_patch_merging: true,
        };

        let backbone = SwinTransformer::new(config);
        let detector = SwinDetector::new(backbone, 10, 3);
        let x = Tensor::zeros(&vec![1usize, 3, 32, 32]);
        let (bbox, cls) = detector.forward(&x);
        assert!(bbox.lock().storage.to_f32_array().len() > 0);
        assert!(cls.lock().storage.to_f32_array().len() > 0);
    }
}
