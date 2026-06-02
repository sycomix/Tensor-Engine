use super::data_source::DataSource;
use super::embedding::Embedding;
use ndarray::{s, Array1, Array2, Array3, Axis};
use regex::Regex;
use std::io::Read;

#[derive(Clone)]
pub struct DataSettings {
    pub force_f16: bool,
}

impl DataSettings {
    pub fn new() -> Self {
        DataSettings { force_f16: false }
    }

    pub fn force_f16(mut self) -> Self {
        self.force_f16 = true;
        self
    }
}

impl Default for DataSettings {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple linear layer (weights and optional bias)
#[derive(Clone)]
pub struct Linear {
    pub weight: Array2<f32>, // shape (out_dim, in_dim)
    pub bias: Option<Array1<f32>>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        // Initialize weights with small values (zeros for deterministic tests)
        Linear {
            weight: Array2::zeros((out_dim, in_dim)),
            bias: Some(Array1::zeros(out_dim)),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // x: (seq, in_dim) -> out: (seq, out_dim) via x * W^T + b
        let mut out = x.dot(&self.weight.t());
        if let Some(b) = &self.bias {
            for mut row in out.axis_iter_mut(Axis(0)) {
                row += b;
            }
        }
        out
    }
}

pub struct AttentionCache {
    pub ks: Array2<f32>, // (seq, dim)
    pub vs: Array2<f32>,
}

impl AttentionCache {
    pub fn new(_max_seq_len: usize, dim: usize) -> Self {
        AttentionCache {
            ks: Array2::zeros((0, dim)),
            vs: Array2::zeros((0, dim)),
        }
    }

    pub fn append(&mut self, k: &Array2<f32>, v: &Array2<f32>) {
        // Stack along axis 0 (seq)
        if self.ks.is_empty() {
            self.ks = k.clone();
            self.vs = v.clone();
        } else {
            let mut newk = Array2::zeros((self.ks.nrows() + k.nrows(), self.ks.ncols()));
            newk.slice_mut(s![0..self.ks.nrows(), ..]).assign(&self.ks);
            newk.slice_mut(s![self.ks.nrows().., ..]).assign(k);
            self.ks = newk;
            let mut newv = Array2::zeros((self.vs.nrows() + v.nrows(), self.vs.ncols()));
            newv.slice_mut(s![0..self.vs.nrows(), ..]).assign(&self.vs);
            newv.slice_mut(s![self.vs.nrows().., ..]).assign(v);
            self.vs = newv;
        }
    }
}

pub struct TransformerCaches {
    pub layer_caches: Vec<AttentionCache>,
}

impl TransformerCaches {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize) -> Self {
        let mut lc = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            lc.push(AttentionCache::new(max_seq_len, dim));
        }
        TransformerCaches { layer_caches: lc }
    }
}

pub struct Attention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    pub fn new(dim: usize, n_heads: usize) -> Self {
        let head_dim = dim / n_heads;
        Attention {
            wq: Linear::new(dim, dim),
            wk: Linear::new(dim, dim),
            wv: Linear::new(dim, dim),
            wo: Linear::new(dim, dim),
            n_heads,
            head_dim,
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f32>, // (seq, dim)
        cache: &mut AttentionCache,
        causal: bool,
    ) -> Array2<f32> {
        let seq = x.nrows();
        let dim = x.ncols();
        // Project
        let q = self.wq.forward(x); // (seq, dim)
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        // Append to cache
        cache.append(&k, &v);
        let k_total = &cache.ks; // (seq_total, dim)
        let v_total = &cache.vs;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;

        // Reshape into (n_heads, seq, head_dim) for q, k_total and v_total
        let q_view = q.view();
        let q3 = q_view
            .to_shape((seq, n_heads, head_dim))
            .unwrap()
            .permuted_axes([1, 0, 2]); // (n_heads, seq, head_dim)
        let k_view = k_total.view();
        let k3 = k_view
            .to_shape((k_total.nrows(), n_heads, head_dim))
            .unwrap()
            .permuted_axes([1, 0, 2]); // (n_heads, seq_total, head_dim)
        let v_view = v_total.view();
        let v3 = v_view
            .to_shape((v_total.nrows(), n_heads, head_dim))
            .unwrap()
            .permuted_axes([1, 0, 2]); // (n_heads, seq_total, head_dim)

        // Prepare output buffer for heads: (n_heads, seq, head_dim)
        let mut out_heads = Array3::zeros((n_heads, seq, head_dim));

        // Compute per-head attention using better memory layout (fewer copies)
        for h in 0..n_heads {
            let qh = q3.slice(s![h, .., ..]); // (seq, head_dim)
            let kh = k3.slice(s![h, .., ..]); // (seq_total, head_dim)
            let vh = v3.slice(s![h, .., ..]); // (seq_total, head_dim)
            // scores = q * k^T -> (seq, seq_total)
            let mut scores = qh.dot(&kh.t());
            // scale
            let scale = (head_dim as f32).sqrt();
            for val in scores.iter_mut() {
                *val /= scale;
            }
            // causal mask
            if causal {
                for i in 0..scores.nrows() {
                    for j in 0..scores.ncols() {
                        if j > i {
                            scores[(i, j)] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            // softmax
            for mut row in scores.axis_iter_mut(Axis(0)) {
                let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
                for val in row.iter_mut() {
                    *val = (*val - maxv).exp();
                }
                let sum: f32 = row.iter().sum();
                if sum > 0.0 {
                    for val in row.iter_mut() {
                        *val /= sum;
                    }
                }
            }
            // context = scores * v -> (seq, head_dim)
            let context = scores.dot(&vh);
            // Write into out_heads
            out_heads.slice_mut(s![h, .., ..]).assign(&context);
        }

        // Recombine heads into (seq, dim)
        // out_heads is (n_heads, seq, head_dim) -> permute to (seq, n_heads, head_dim) then reshape
        let out_perm = out_heads.permuted_axes([1, 0, 2]); // (seq, n_heads, head_dim)
        let mut out = Array2::zeros((seq, dim));
        for i in 0..seq {
            for h in 0..n_heads {
                for j in 0..head_dim {
                    out[(i, h * head_dim + j)] = out_perm[(i, h, j)];
                }
            }
        }
        // final linear
        self.wo.forward(&out)
    }
}

pub struct FeedForward {
    pub w1: Linear,
    pub w2: Linear,
}

impl FeedForward {
    pub fn new(dim: usize, hidden: usize) -> Self {
        FeedForward {
            w1: Linear::new(dim, hidden),
            w2: Linear::new(hidden, dim),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut h = self.w1.forward(x);
        // GELU approximation
        for v in h.iter_mut() {
            let x = *v;
            *v = 0.5 * x * (1.0 + (x / (2.0f32).sqrt()).tanh());
        }
        self.w2.forward(&h)
    }
}

pub struct TransformerBlock {
    pub attn: Attention,
    pub ffn: FeedForward,
    // Optional layer-norm parameters (gamma=weight, beta=bias)
    pub ln1_weight: Option<Array1<f32>>,
    pub ln1_bias: Option<Array1<f32>>,
    pub ln2_weight: Option<Array1<f32>>,
    pub ln2_bias: Option<Array1<f32>>,
}

impl TransformerBlock {
    pub fn new(dim: usize, n_heads: usize) -> Self {
        TransformerBlock {
            attn: Attention::new(dim, n_heads),
            ffn: FeedForward::new(dim, dim * 4),
            ln1_weight: None,
            ln1_bias: None,
            ln2_weight: None,
            ln2_bias: None,
        }
    }

    pub fn forward(&self, x: &Array2<f32>, cache: &mut AttentionCache) -> Array2<f32> {
        let a = self.attn.forward(x, cache, true);
        let x = x + a; // residual
        let f = self.ffn.forward(&x);
        x + f
    }
}

pub struct Transformer {
    pub emb: Embedding,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub data_settings: DataSettings,
    pub blocks: Vec<TransformerBlock>,
    pub output: Linear,
}

impl Transformer {
    pub fn new(
        emb: Embedding,
        dim: usize,
        n_layers: usize,
        n_heads: usize,
        max_seq_len: usize,
        data_settings: DataSettings,
    ) -> Self {
        assert_eq!(dim % n_heads, 0);
        let head_dim = dim / n_heads;
        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            blocks.push(TransformerBlock::new(dim, n_heads));
        }
        Transformer {
            emb,
            dim,
            n_layers,
            n_heads,
            max_seq_len,
            head_dim,
            data_settings,
            blocks,
            output: Linear::new(dim, dim),
        }
    }

    pub fn make_caches(&self) -> TransformerCaches {
        TransformerCaches::new(self.n_layers, self.max_seq_len, self.dim)
    }

    /// Forward pass: given token ids, produce a vector for the last token (seq output)
    pub fn forward(&self, tokens: &[usize], caches: &mut TransformerCaches) -> Array2<f32> {
        let seq = tokens.len();
        // Build input matrix X (seq, dim)
        let mut x = Array2::zeros((seq, self.dim));
        for (i, &tok) in tokens.iter().enumerate() {
            if let Some(row) = self.emb.get_embedding(tok) {
                for j in 0..self.dim {
                    x[(i, j)] = row[j];
                }
            }
        }
        // Pass through layers
        for (i, block) in self.blocks.iter().enumerate() {
            let out = block.forward(&x, &mut caches.layer_caches[i]);
            x = out;
        }
        // Return last token vector as (1, dim)
        let last = x.slice(s![seq - 1..seq, ..]).to_owned();
        // Optional output projection
        self.output.forward(&last)
    }

    /// Apply a map of weight arrays into this transformer's linear layers.
    /// Keys should be the parameter names (e.g., "model.layers.0.self_attn.q_proj.weight").
    fn choose_matrix_for_expected(
        w: &Array2<f32>,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Option<Array2<f32>> {
        if w.nrows() == expected_rows && w.ncols() == expected_cols {
            return Some(w.clone());
        }
        if w.nrows() == expected_cols && w.ncols() == expected_rows {
            return Some(w.t().to_owned());
        }
        None
    }

    fn set_bias_from_matrix(target_len: usize, w: &Array2<f32>) -> Option<Array1<f32>> {
        if w.ncols() == 1 && w.nrows() == target_len {
            let mut out = Array1::zeros(target_len);
            for i in 0..target_len {
                out[i] = w[(i, 0)];
            }
            return Some(out);
        }
        if w.nrows() == 1 && w.ncols() == target_len {
            let mut out = Array1::zeros(target_len);
            for i in 0..target_len {
                out[i] = w[(0, i)];
            }
            return Some(out);
        }
        None
    }

    pub fn apply_weight_map(&mut self, weights: &std::collections::HashMap<String, Array2<f32>>) {
        // Flexible assignment: look for names and layer ids using heuristics
        let re_layers =
            Regex::new(r"(?:layers|h|blocks|transformer\.h|encoder\.layer|layer)\.(?P<id>\d+)")
                .unwrap();

        // Global / head weights
        for (k, w) in weights.iter() {
            if k.ends_with("lm_head.weight") || k.ends_with("output.weight") {
                if w.shape() == &[self.output.weight.nrows(), self.output.weight.ncols()] {
                    self.output.weight = w.clone();
                } else if w.shape() == &[self.output.weight.ncols(), self.output.weight.nrows()] {
                    self.output.weight = w.t().to_owned();
                }
                continue;
            }
            if k.ends_with("lm_head.bias") || k.ends_with("output.bias") {
                if let Some(b) = Self::set_bias_from_matrix(self.output.weight.nrows(), w) {
                    self.output.bias = Some(b);
                }
                continue;
            }

            // Try to extract layer id
            if let Some(cap) = re_layers.captures(k) {
                if let Some(m) = cap.name("id") {
                    if let Ok(layer_id) = m.as_str().parse::<usize>() {
                        if layer_id >= self.n_layers {
                            continue;
                        }
                        let ffn_h = self.ffn_hidden();
                        let block = &mut self.blocks[layer_id];
                        // Handle concatenated qkv (common in some HF checkpoints)
                        // QKV concatenated or c_attn cases
                        if k.contains("c_attn") || k.contains("qkv") {
                            let r = w.nrows();
                            let c = w.ncols();
                            // row concatenated
                            if r == 3 * self.dim && c == self.dim {
                                let q = w.slice(s![0..self.dim, ..]).to_owned();
                                let k_ = w.slice(s![self.dim..2 * self.dim, ..]).to_owned();
                                let v = w.slice(s![2 * self.dim..3 * self.dim, ..]).to_owned();
                                block.attn.wq.weight =
                                    Self::choose_matrix_for_expected(&q, self.dim, self.dim)
                                        .unwrap_or(q);
                                block.attn.wk.weight =
                                    Self::choose_matrix_for_expected(&k_, self.dim, self.dim)
                                        .unwrap_or(k_);
                                block.attn.wv.weight =
                                    Self::choose_matrix_for_expected(&v, self.dim, self.dim)
                                        .unwrap_or(v);
                                continue;
                            // column concatenated
                            } else if r == self.dim && c == 3 * self.dim {
                                let q = w.slice(s![.., 0..self.dim]).to_owned();
                                let k_ = w.slice(s![.., self.dim..2 * self.dim]).to_owned();
                                let v = w.slice(s![.., 2 * self.dim..3 * self.dim]).to_owned();
                                block.attn.wq.weight =
                                    Self::choose_matrix_for_expected(&q, self.dim, self.dim)
                                        .unwrap_or(q.t().to_owned());
                                block.attn.wk.weight =
                                    Self::choose_matrix_for_expected(&k_, self.dim, self.dim)
                                        .unwrap_or(k_.t().to_owned());
                                block.attn.wv.weight =
                                    Self::choose_matrix_for_expected(&v, self.dim, self.dim)
                                        .unwrap_or(v.t().to_owned());
                                continue;
                            }
                            // Bias for c_attn (common layout: (3*dim) x 1 or 1 x (3*dim))
                            if k.contains("bias") {
                                // try to assign split biases
                                if let Some(b) = Self::set_bias_from_matrix(self.dim, w) {
                                    block.attn.wo.bias = Some(b.clone());
                                }
                            }
                        }

                        // Individual named weights and biases
                        if k.contains("q_proj") || k.contains(".q.") || k.contains(".query") {
                            if let Some(m) = Self::choose_matrix_for_expected(w, self.dim, self.dim)
                            {
                                block.attn.wq.weight = m;
                            }
                        } else if k.contains("k_proj") || k.contains(".k.") || k.contains(".key") {
                            if let Some(m) = Self::choose_matrix_for_expected(w, self.dim, self.dim)
                            {
                                block.attn.wk.weight = m;
                            }
                        } else if k.contains("v_proj") || k.contains(".v.") || k.contains(".value")
                        {
                            if let Some(m) = Self::choose_matrix_for_expected(w, self.dim, self.dim)
                            {
                                block.attn.wv.weight = m;
                            }
                        } else if k.contains("o_proj")
                            || k.contains("out_proj")
                            || k.contains("wo")
                            || k.contains("o_proj")
                        {
                            if let Some(m) = Self::choose_matrix_for_expected(w, self.dim, self.dim)
                            {
                                block.attn.wo.weight = m;
                            }
                        } else if k.contains("mlp.fc1") || k.contains("fc1") {
                            if let Some(m) = Self::choose_matrix_for_expected(w, ffn_h, self.dim) {
                                block.ffn.w1.weight = m;
                            }
                        } else if k.contains("mlp.fc2") || k.contains("fc2") {
                            if let Some(m) = Self::choose_matrix_for_expected(w, self.dim, ffn_h) {
                                block.ffn.w2.weight = m;
                            }
                        }

                        // Bias handling patterns
                        if k.ends_with(".bias") || k.contains(".bias") || k.contains("bias") {
                            if k.contains("q_proj") || k.contains(".q.") || k.contains(".query") {
                                if let Some(bv) = Self::set_bias_from_matrix(self.dim, w) {
                                    block.attn.wq.bias = Some(bv);
                                }
                            } else if k.contains("k_proj")
                                || k.contains(".k.")
                                || k.contains(".key")
                            {
                                if let Some(bv) = Self::set_bias_from_matrix(self.dim, w) {
                                    block.attn.wk.bias = Some(bv);
                                }
                            } else if k.contains("v_proj")
                                || k.contains(".v.")
                                || k.contains(".value")
                            {
                                if let Some(bv) = Self::set_bias_from_matrix(self.dim, w) {
                                    block.attn.wv.bias = Some(bv);
                                }
                            } else if k.contains("o_proj")
                                || k.contains("out_proj")
                                || k.contains("wo")
                            {
                                if let Some(bv) = Self::set_bias_from_matrix(self.dim, w) {
                                    block.attn.wo.bias = Some(bv);
                                }
                            } else if k.contains("fc1") {
                                if let Some(bv) = Self::set_bias_from_matrix(ffn_h, w) {
                                    block.ffn.w1.bias = Some(bv);
                                }
                            } else if k.contains("fc2") {
                                if let Some(bv) = Self::set_bias_from_matrix(self.dim, w) {
                                    block.ffn.w2.bias = Some(bv);
                                }
                            } else if k.contains("norm")
                                || k.contains("ln")
                                || k.contains("layernorm")
                            {
                                // Heuristic: decide which norm (ln1/ln2) based on presence of ffn or attn in name
                                if k.contains("ffn") || k.contains("mlp") {
                                    if let Some(bv) = Self::set_bias_from_matrix(self.dim, w) {
                                        block.ln2_bias = Some(bv);
                                    }
                                } else {
                                    if let Some(bv) = Self::set_bias_from_matrix(self.dim, w) {
                                        block.ln1_bias = Some(bv);
                                    }
                                }
                            }
                        }

                        // Layer-norm weight handling
                        if k.ends_with(".weight")
                            && (k.contains("norm") || k.contains("ln") || k.contains("layernorm"))
                        {
                            if k.contains("ffn") || k.contains("mlp") {
                                if let Some(bv) = Self::set_bias_from_matrix(self.dim, w) {
                                    block.ln2_weight = Some(bv);
                                }
                            } else {
                                if let Some(bv) = Self::set_bias_from_matrix(self.dim, w) {
                                    block.ln1_weight = Some(bv);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn ffn_hidden(&self) -> usize {
        self.dim * 4
    }

    pub fn from_unpickled(
        emb: Embedding,
        data_settings: DataSettings,
        data_source: DataSource,
    ) -> Result<Self, String> {
        // Build a minimal transformer and try to load weights from data_source
        let dim = emb.cols;
        let n_heads = 1usize;
        let n_layers = 1usize;
        let max_seq_len = 1024usize;
        let mut t = Transformer::new(emb, dim, n_layers, n_heads, max_seq_len, data_settings);
        // Attempt to load weights; ignore errors to keep this progressive
        let _ = t.load_weights_from_datasource(data_source);
        Ok(t)
    }

    /// Attempt to load tensors from a `DataSource` by inspecting its unpickled values. This will
    /// read any tensors that can be converted to a `TensorBuilder` and fill a weight map, then
    /// apply that map to the transformer's layers.
    pub fn load_weights_from_datasource(&mut self, data_source: DataSource) -> Result<(), String> {
        // If the data source carries a HF model config, adopt those sizing parameters first
        match data_source {
            DataSource::VicunaSource(_, ref model, _) => {
                let cfg = &model.config;
                // Update dims
                self.dim = cfg.hidden_size;
                self.n_heads = cfg.num_attention_heads;
                self.head_dim = if self.n_heads > 0 {
                    self.dim / self.n_heads
                } else {
                    self.head_dim
                };
                self.n_layers = cfg.num_hidden_layers;
                self.max_seq_len = cfg.max_position_embeddings;
                // Rebuild blocks to match new sizes
                self.blocks = (0..self.n_layers)
                    .map(|_| TransformerBlock::new(self.dim, self.n_heads))
                    .collect();
                self.output = Linear::new(self.dim, self.dim);
            }
            _ => {
                // No-op for other sources for now
            }
        }

        // Two-pass approach: first read raw matrices (Float/BFloat -> F32 arrays, K4 -> packed bytes), then post-process packed ones using found scales/zeros in the raw map.
        let mut raw_map: std::collections::HashMap<String, RawRead> =
            std::collections::HashMap::new();
        let mut weight_map: std::collections::HashMap<String, Array2<f32>> =
            std::collections::HashMap::new();

        for unpickled in data_source.unpickled().iter() {
            for key in unpickled.keys().iter() {
                if let Some(val) = unpickled.get_str_key(key) {
                    if let Some(tb) = val.to_tensor_builder(key.clone()) {
                        // Read raw. For float types we get a matrix; for K4 we store packed bytes.
                        match read_raw_builder_matrix(&tb, data_source.clone()) {
                            Ok(RawRead::F32(mat)) => {
                                raw_map.insert(key.clone(), RawRead::F32(mat));
                            }
                            Ok(RawRead::PackedK4(bytes, rows, cols)) => {
                                raw_map.insert(key.clone(), RawRead::PackedK4(bytes, rows, cols));
                            }
                            Err(_e) => {
                                // skip unreadable
                            }
                        }
                    }
                }
            }
        }

        // Post-process packed K4 matrices by attempting to find companion scales/zeros and then dequantize
        for (key, value) in raw_map.iter() {
            match value {
                RawRead::F32(mat) => {
                    // directly usable
                    weight_map.insert(key.clone(), mat.clone());
                }
                RawRead::PackedK4(bytes, rows, cols) => {
                    // Find scales/zeros candidates
                    // Simple heuristics: look for keys containing 'scale' or 'scales' and containing the same prefix as key (split by '.')
                    let mut found_scales: Option<&Array2<f32>> = None;
                    let mut found_zeros: Option<&Array2<f32>> = None;
                    // Compute prefix heuristic (first two dot segments)
                    let prefix_segments: Vec<&str> = key.split('.').take(2).collect();
                    for (k2, v2) in raw_map.iter() {
                        let klower = k2.to_lowercase();
                        if klower.contains("scale") || klower.contains("scales") {
                            // try to extract Array2
                            if let RawRead::F32(arr) = v2 {
                                // quick heuristic: ensure k2 shares a prefix segment or same layer id
                                if k2.starts_with(&prefix_segments.join("."))
                                    || (!prefix_segments.is_empty()
                                    && k2.contains(prefix_segments.get(1).unwrap_or(&"")))
                                {
                                    found_scales = Some(arr);
                                }
                            }
                        }
                        if klower.contains("zero")
                            || klower.contains("qzeros")
                            || klower.contains("q_zero")
                        {
                            if let RawRead::F32(arr) = v2 {
                                if k2.starts_with(&prefix_segments.join("."))
                                    || (!prefix_segments.is_empty()
                                    && k2.contains(prefix_segments.get(1).unwrap_or(&"")))
                                {
                                    found_zeros = Some(arr);
                                }
                            }
                        }
                    }

                    // Dequantize
                    match decode_k4_buffer_to_matrix_with_scales(
                        &bytes,
                        *rows,
                        *cols,
                        found_scales,
                        found_zeros,
                    ) {
                        Ok(mat) => {
                            weight_map.insert(key.clone(), mat);
                        }
                        Err(_) => {
                            // fallback to signed nibble decode
                            match decode_k4_buffer_to_matrix(&bytes, *rows, *cols) {
                                Ok(mat) => {
                                    weight_map.insert(key.clone(), mat);
                                }
                                Err(_e) => { /* skip */ }
                            }
                        }
                    }
                }
            }
        }

        // Apply the map
        self.apply_weight_map(&weight_map);
        Ok(())
    }
}

// Read a raw builder and return either an F32 matrix or packed K4 bytes
fn read_raw_builder_matrix(
    builder: &crate::hf_compat::unpickler::TensorBuilder,
    data_source: crate::hf_compat::data_source::DataSource,
) -> Result<RawRead, crate::hf_compat::unpickler::UnpicklingError> {
    use std::io::Seek;
    let rows = builder.rows as usize;
    let cols = builder.cols as usize;

    let path = std::path::PathBuf::from("data").join(&builder.src_path);
    // Shard 0 for now
    let mut f = data_source
        .open(path.clone(), &builder.tensor_name, 0)
        .map_err(|e| {
            crate::hf_compat::unpickler::UnpicklingError::UnpicklingError(format!("IO: {}", e))
        })?;
    // Seek to offset
    let offset_bytes = (builder.offset as i64) * (builder.dtype.bytes_for_nvalues(1) as i64);
    f.seek(std::io::SeekFrom::Current(offset_bytes))
        .map_err(|e| {
            crate::hf_compat::unpickler::UnpicklingError::UnpicklingError(format!(
                "seek error: {}",
                e
            ))
        })?;

    let nbytes = builder.dtype.bytes_for_nvalues(builder.cols as usize);
    let mut buf: Vec<u8> = vec![0u8; nbytes * rows];
    // Read full buffer for all rows
    f.read_exact(&mut buf).map_err(|e| {
        crate::hf_compat::unpickler::UnpicklingError::UnpicklingError(format!("read error: {}", e))
    })?;

    match builder.dtype {
        crate::hf_compat::unpickler::TensorDType::Float16 => {
            let mut mat = Array2::zeros((rows, cols));
            let mut off = 0usize;
            for r in 0..rows {
                for c in 0..cols {
                    let v = half::f16::from_bits(u16::from_le_bytes([buf[off], buf[off + 1]]));
                    mat[(r, c)] = v.to_f32();
                    off += 2;
                }
            }
            Ok(RawRead::F32(mat))
        }
        crate::hf_compat::unpickler::TensorDType::Float32 => {
            let mut mat = Array2::zeros((rows, cols));
            let mut off = 0usize;
            for r in 0..rows {
                for c in 0..cols {
                    let v =
                        f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
                    mat[(r, c)] = v;
                    off += 4;
                }
            }
            Ok(RawRead::F32(mat))
        }
        crate::hf_compat::unpickler::TensorDType::K4BitQuantization => {
            Ok(RawRead::PackedK4(buf, rows, cols))
        }
    }
}

/// Basic fallback decoder: signed nibble -8..7 -> f32
pub fn decode_k4_buffer_to_matrix(
    buf: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Array2<f32>, crate::hf_compat::unpickler::UnpicklingError> {
    let mut mat = Array2::zeros((rows, cols));
    let mut off = 0usize;
    for r in 0..rows {
        let mut c = 0usize;
        while c < cols {
            if off >= buf.len() {
                return Err(crate::hf_compat::unpickler::UnpicklingError::InvalidTensorData);
            }
            let b = buf[off];
            let low = (b & 0x0F) as i8;
            let high = (b >> 4) as i8;
            let low_signed = if low >= 8 { low - 16 } else { low };
            mat[(r, c)] = low_signed as f32;
            c += 1;
            if c < cols {
                let high_signed = if high >= 8 { high - 16 } else { high };
                mat[(r, c)] = high_signed as f32;
                c += 1;
            }
            off += 1;
        }
    }
    Ok(mat)
}

/// Decode packed K4 with optional scales/zeros. If `scales` is Some, it is interpreted
/// as either (rows, groups), (rows,1), (1,groups) or (rows,cols) and handled accordingly.
pub fn decode_k4_buffer_to_matrix_with_scales(
    buf: &[u8],
    rows: usize,
    cols: usize,
    scales: Option<&Array2<f32>>,
    zeros: Option<&Array2<f32>>,
) -> Result<Array2<f32>, crate::hf_compat::unpickler::UnpicklingError> {
    // First unpack to u8 nibble matrix
    let mut q = ndarray::Array2::<u8>::zeros((rows, cols));
    let mut off = 0usize;
    for r in 0..rows {
        let mut c = 0usize;
        while c < cols {
            if off >= buf.len() {
                return Err(crate::hf_compat::unpickler::UnpicklingError::InvalidTensorData);
            }
            let b = buf[off];
            let low = (b & 0x0F) as u8;
            let high = (b >> 4) as u8;
            q[(r, c)] = low;
            c += 1;
            if c < cols {
                q[(r, c)] = high;
                c += 1;
            }
            off += 1;
        }
    }

    if let Some(sarr) = scales {
        // Determine if sarr is (rows, groups) or (rows, cols) etc.
        let sshape = sarr.shape();
        let mut out = Array2::zeros((rows, cols));
        if sshape == &[rows, cols] {
            // per-element scale
            let zarr_opt = zeros;
            for r in 0..rows {
                for c in 0..cols {
                    let qv = q[(r, c)] as f32;
                    let s = sarr[(r, c)];
                    let z = if let Some(zarr) = zarr_opt {
                        zarr[(r, c)]
                    } else {
                        0.0
                    };
                    out[(r, c)] = (qv - z) * s;
                }
            }
            return Ok(out);
        }
        // try groups
        if sshape.len() == 2 && sshape[0] == rows {
            let k_groups = sshape[1];
            if cols % k_groups == 0 {
                let group_size = cols / k_groups;
                for r in 0..rows {
                    for g in 0..k_groups {
                        let s = sarr[(r, g)];
                        let z = if let Some(zarr) = zeros {
                            zarr[(r, g)]
                        } else {
                            0.0
                        };
                        for i in 0..group_size {
                            let c = g * group_size + i;
                            out[(r, c)] = (q[(r, c)] as f32 - z) * s;
                        }
                    }
                }
                return Ok(out);
            }
        }
        // try (rows,1)
        if sshape == &[rows, 1] {
            for r in 0..rows {
                let s = sarr[(r, 0)];
                let z = if let Some(zarr) = zeros {
                    zarr[(r, 0)]
                } else {
                    0.0
                };
                for c in 0..cols {
                    out[(r, c)] = (q[(r, c)] as f32 - z) * s;
                }
            }
            return Ok(out);
        }
        // try (1,groups)
        if sshape.len() == 2 && sshape[0] == 1 {
            let k_groups = sshape[1];
            if cols % k_groups == 0 {
                let group_size = cols / k_groups;
                for r in 0..rows {
                    for g in 0..k_groups {
                        let s = sarr[(0, g)];
                        let z = if let Some(zarr) = zeros {
                            zarr[(0, g)]
                        } else {
                            0.0
                        };
                        for i in 0..group_size {
                            let c = g * group_size + i;
                            out[(r, c)] = (q[(r, c)] as f32 - z) * s;
                        }
                    }
                }
                return Ok(out);
            }
        }
        // try single scalar scale (1x1)
        if sshape == &[1, 1] {
            let s = sarr[(0, 0)];
            for r in 0..rows {
                for c in 0..cols {
                    out[(r, c)] = (q[(r, c)] as f32) * s;
                }
            }
            return Ok(out);
        }
    }

    // no scales available or couldn't interpret -> fallback to signed nibble
    let mut out = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let v = q[(r, c)] as i8;
            let signed = if v >= 8 { v - 16 } else { v };
            out[(r, c)] = signed as f32;
        }
    }
    Ok(out)
}

// Backwards-compatible read_builder_matrix that reads a full float matrix (no packed metadata detection)
pub fn read_builder_matrix(
    builder: &crate::hf_compat::unpickler::TensorBuilder,
    data_source: crate::hf_compat::data_source::DataSource,
) -> Result<Array2<f32>, crate::hf_compat::unpickler::UnpicklingError> {
    match read_raw_builder_matrix(builder, data_source)? {
        RawRead::F32(mat) => Ok(mat),
        RawRead::PackedK4(bytes, rows, cols) => {
            // fallback decode
            decode_k4_buffer_to_matrix(&bytes, rows, cols)
        }
    }
}

// Internal enum used by raw reader
#[derive(Debug)]
enum RawRead {
    F32(Array2<f32>),
    PackedK4(Vec<u8>, usize, usize),
}
