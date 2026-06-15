use super::embeddings::{EmbeddingError, TokenEmbedding};
use super::positional_embeddings::{PositionalEmbedding, PositionalEmbeddingError};
use super::transformer_block::{TransformerBlock, TransformerBlockError};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum GPTModelError {
    InvalidConfig(&'static str),
    EmptyInput,
    EmptySequenceInBatch {
        index: usize,
    },
    RaggedBatch,
    SequenceTooLong {
        seq_len: usize,
        max_seq_len: usize,
    },
    TokenEmbeddingFailed(EmbeddingError),
    PositionalEmbeddingFailed(PositionalEmbeddingError),
    TransformerLayerFailed {
        layer: usize,
        source: TransformerBlockError,
    },
    ProjectionFailed(EmbeddingError),
    DecodeCacheInvalid(&'static str),
}

impl Display for GPTModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GPTModelError::InvalidConfig(msg) => write!(f, "invalid model config: {}", msg),
            GPTModelError::EmptyInput => write!(f, "input token sequence is empty"),
            GPTModelError::EmptySequenceInBatch { index } => {
                write!(f, "batch sequence at index {} is empty", index)
            }
            GPTModelError::RaggedBatch => {
                write!(f, "all batch sequences must have the same length")
            }
            GPTModelError::SequenceTooLong {
                seq_len,
                max_seq_len,
            } => write!(
                f,
                "sequence length {} exceeds max_seq_len {}",
                seq_len, max_seq_len
            ),
            GPTModelError::TokenEmbeddingFailed(err) => {
                write!(f, "token embedding failed: {}", err)
            }
            GPTModelError::PositionalEmbeddingFailed(err) => {
                write!(f, "positional embedding failed: {}", err)
            }
            GPTModelError::TransformerLayerFailed { layer, source } => {
                write!(f, "transformer layer {} failed: {}", layer, source)
            }
            GPTModelError::ProjectionFailed(err) => {
                write!(f, "vocab projection failed: {}", err)
            }
            GPTModelError::DecodeCacheInvalid(msg) => {
                write!(f, "decode cache is invalid: {}", msg)
            }
        }
    }
}

impl Error for GPTModelError {}

#[derive(Debug, Clone, Copy)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub seed: u64,
    pub tie_weights: bool,
}

impl GPTConfig {
    pub fn validate(&self) -> Result<(), GPTModelError> {
        if self.vocab_size == 0 {
            return Err(GPTModelError::InvalidConfig("vocab_size must be > 0"));
        }
        if self.max_seq_len == 0 {
            return Err(GPTModelError::InvalidConfig("max_seq_len must be > 0"));
        }
        if self.embedding_dim == 0 {
            return Err(GPTModelError::InvalidConfig("embedding_dim must be > 0"));
        }
        if self.hidden_dim == 0 {
            return Err(GPTModelError::InvalidConfig("hidden_dim must be > 0"));
        }
        if self.num_heads == 0 {
            return Err(GPTModelError::InvalidConfig("num_heads must be > 0"));
        }
        if self.num_layers == 0 {
            return Err(GPTModelError::InvalidConfig("num_layers must be > 0"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GPTModel {
    token_embedding: TokenEmbedding,
    positional_embedding: PositionalEmbedding,
    layers: Vec<TransformerBlock>,
    lm_head: Option<Vec<Vec<f32>>>, // [vocab_size][embedding_dim] when untied
    tie_weights: bool,
    vocab_size: usize,
    embedding_dim: usize,
    max_seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct GPTDecodeCache {
    layer_inputs: Vec<DecodeLayerBuffer>,
    next_position: usize,
    embedding_dim: usize,
    max_seq_len: usize,
}

#[derive(Debug, Clone)]
struct DecodeLayerBuffer {
    data: Vec<f32>,
    len: usize,
    embedding_dim: usize,
    max_seq_len: usize,
}

impl DecodeLayerBuffer {
    fn new(max_seq_len: usize, embedding_dim: usize) -> Self {
        Self {
            data: Vec::with_capacity(max_seq_len.saturating_mul(embedding_dim)),
            len: 0,
            embedding_dim,
            max_seq_len,
        }
    }

    fn clear(&mut self) {
        self.data.clear();
        self.len = 0;
    }

    fn replace_rows(&mut self, rows: &[Vec<f32>]) -> Result<(), GPTModelError> {
        if rows.len() > self.max_seq_len {
            return Err(GPTModelError::SequenceTooLong {
                seq_len: rows.len(),
                max_seq_len: self.max_seq_len,
            });
        }
        self.data.clear();
        self.data
            .reserve(rows.len().saturating_mul(self.embedding_dim));
        for row in rows {
            if row.len() != self.embedding_dim {
                return Err(GPTModelError::DecodeCacheInvalid(
                    "prefill row has wrong embedding dimension",
                ));
            }
            self.data.extend_from_slice(row);
        }
        self.len = rows.len();
        Ok(())
    }

    fn push_row(&mut self, row: &[f32]) -> Result<(), GPTModelError> {
        if row.len() != self.embedding_dim {
            return Err(GPTModelError::DecodeCacheInvalid(
                "decoded row has wrong embedding dimension",
            ));
        }
        if self.len >= self.max_seq_len {
            return Err(GPTModelError::SequenceTooLong {
                seq_len: self.len + 1,
                max_seq_len: self.max_seq_len,
            });
        }
        self.data.extend_from_slice(row);
        self.len += 1;
        Ok(())
    }

    fn as_flat(&self) -> &[f32] {
        &self.data
    }
}

impl GPTDecodeCache {
    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn cached_layers(&self) -> usize {
        self.layer_inputs.len()
    }

    pub fn cached_tokens(&self) -> usize {
        self.layer_inputs
            .first()
            .map(|layer| layer.len)
            .unwrap_or(0)
    }

    pub fn cached_capacity_per_layer(&self) -> usize {
        self.layer_inputs
            .first()
            .map(|layer| layer.data.capacity())
            .unwrap_or(0)
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layer_inputs {
            layer.clear();
        }
        self.next_position = 0;
    }
}

impl GPTModel {
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        embedding_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_layers: usize,
        seed: u64,
    ) -> Result<Self, GPTModelError> {
        let cfg = GPTConfig {
            vocab_size,
            max_seq_len,
            embedding_dim,
            hidden_dim,
            num_heads,
            num_layers,
            seed,
            tie_weights: true,
        };
        Self::from_config(cfg)
    }

    pub fn from_config(cfg: GPTConfig) -> Result<Self, GPTModelError> {
        cfg.validate()?;

        let token_embedding = TokenEmbedding::random(cfg.vocab_size, cfg.embedding_dim, cfg.seed)
            .map_err(GPTModelError::TokenEmbeddingFailed)?;
        let positional_embedding =
            PositionalEmbedding::random(cfg.max_seq_len, cfg.embedding_dim, cfg.seed ^ 0xA5A5_5A5A)
                .map_err(GPTModelError::PositionalEmbeddingFailed)?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            layers.push(TransformerBlock::new(
                cfg.embedding_dim,
                cfg.hidden_dim,
                cfg.num_heads,
            ));
        }

        let lm_head = if cfg.tie_weights {
            None
        } else {
            let lm =
                TokenEmbedding::random(cfg.vocab_size, cfg.embedding_dim, cfg.seed ^ 0x5A5A_A5A5)
                    .map_err(GPTModelError::TokenEmbeddingFailed)?;
            let mut out = Vec::with_capacity(cfg.vocab_size);
            for token_id in 0..cfg.vocab_size {
                out.push(
                    lm.try_embedding_ref(token_id as u32)
                        .map_err(GPTModelError::TokenEmbeddingFailed)?
                        .to_vec(),
                );
            }
            Some(out)
        };

        Ok(Self {
            token_embedding,
            positional_embedding,
            layers,
            lm_head,
            tie_weights: cfg.tie_weights,
            vocab_size: cfg.vocab_size,
            embedding_dim: cfg.embedding_dim,
            max_seq_len: cfg.max_seq_len,
        })
    }

    pub fn forward(&self, token_ids: &[u32], training: bool) -> Vec<Vec<f32>> {
        match self.try_forward(token_ids, training) {
            Ok(v) => v,
            Err(_) => Vec::new(),
        }
    }

    pub fn try_forward(
        &self,
        token_ids: &[u32],
        training: bool,
    ) -> Result<Vec<Vec<f32>>, GPTModelError> {
        if token_ids.is_empty() {
            return Err(GPTModelError::EmptyInput);
        }
        if token_ids.len() > self.max_seq_len {
            return Err(GPTModelError::SequenceTooLong {
                seq_len: token_ids.len(),
                max_seq_len: self.max_seq_len,
            });
        }

        let token_emb = self
            .token_embedding
            .try_embed(token_ids)
            .map_err(GPTModelError::TokenEmbeddingFailed)?;

        let positions: Vec<usize> = (0..token_ids.len()).collect();
        let pos_emb = self
            .positional_embedding
            .try_embed_positions(&positions)
            .map_err(GPTModelError::PositionalEmbeddingFailed)?;

        let mut hidden = add_matrices(&token_emb, &pos_emb)
            .ok_or(GPTModelError::InvalidConfig("embedding shape mismatch"))?;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.try_forward(&hidden, training).map_err(|source| {
                GPTModelError::TransformerLayerFailed {
                    layer: layer_index,
                    source,
                }
            })?;
        }

        self.project_to_vocab(&hidden)
    }

    pub fn try_forward_last(
        &self,
        token_ids: &[u32],
        training: bool,
    ) -> Result<Vec<f32>, GPTModelError> {
        if training {
            let logits = self.try_forward(token_ids, true)?;
            return logits
                .last()
                .cloned()
                .ok_or(GPTModelError::InvalidConfig("model returned empty logits"));
        }

        let mut cache = self.new_decode_cache();
        self.try_prefill_decode_cache(token_ids, &mut cache)
    }

    pub fn forward_batch(
        &self,
        batch_token_ids: &[Vec<u32>],
        training: bool,
    ) -> Vec<Vec<Vec<f32>>> {
        match self.try_forward_batch(batch_token_ids, training) {
            Ok(v) => v,
            Err(_) => Vec::new(),
        }
    }

    pub fn try_forward_batch(
        &self,
        batch_token_ids: &[Vec<u32>],
        training: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>, GPTModelError> {
        if batch_token_ids.is_empty() {
            return Err(GPTModelError::EmptyInput);
        }

        let seq_len = batch_token_ids[0].len();
        if seq_len == 0 {
            return Err(GPTModelError::EmptySequenceInBatch { index: 0 });
        }
        if seq_len > self.max_seq_len {
            return Err(GPTModelError::SequenceTooLong {
                seq_len,
                max_seq_len: self.max_seq_len,
            });
        }

        for (index, seq) in batch_token_ids.iter().enumerate() {
            if seq.is_empty() {
                return Err(GPTModelError::EmptySequenceInBatch { index });
            }
            if seq.len() != seq_len {
                return Err(GPTModelError::RaggedBatch);
            }
        }

        let mut out = Vec::with_capacity(batch_token_ids.len());
        for seq in batch_token_ids {
            out.push(self.try_forward(seq, training)?);
        }
        Ok(out)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn tie_weights(&self) -> bool {
        self.tie_weights
    }

    pub fn new_decode_cache(&self) -> GPTDecodeCache {
        GPTDecodeCache {
            layer_inputs: (0..self.layers.len())
                .map(|_| DecodeLayerBuffer::new(self.max_seq_len, self.embedding_dim))
                .collect(),
            next_position: 0,
            embedding_dim: self.embedding_dim,
            max_seq_len: self.max_seq_len,
        }
    }

    pub fn try_prefill_decode_cache(
        &self,
        token_ids: &[u32],
        cache: &mut GPTDecodeCache,
    ) -> Result<Vec<f32>, GPTModelError> {
        if token_ids.is_empty() {
            return Err(GPTModelError::EmptyInput);
        }
        if token_ids.len() > self.max_seq_len {
            return Err(GPTModelError::SequenceTooLong {
                seq_len: token_ids.len(),
                max_seq_len: self.max_seq_len,
            });
        }

        self.validate_decode_cache_shape(cache)?;
        cache.clear();

        let token_emb = self
            .token_embedding
            .try_embed(token_ids)
            .map_err(GPTModelError::TokenEmbeddingFailed)?;
        let positions: Vec<usize> = (0..token_ids.len()).collect();
        let pos_emb = self
            .positional_embedding
            .try_embed_positions(&positions)
            .map_err(GPTModelError::PositionalEmbeddingFailed)?;
        let mut hidden = add_matrices(&token_emb, &pos_emb)
            .ok_or(GPTModelError::InvalidConfig("embedding shape mismatch"))?;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            cache.layer_inputs[layer_index].replace_rows(&hidden)?;
            hidden = layer.try_forward(&hidden, false).map_err(|source| {
                GPTModelError::TransformerLayerFailed {
                    layer: layer_index,
                    source,
                }
            })?;
        }

        cache.next_position = token_ids.len();
        let last_hidden = hidden.last().ok_or(GPTModelError::InvalidConfig(
            "model returned empty hidden states",
        ))?;
        self.project_row_to_vocab(last_hidden)
    }

    pub fn try_decode_next_logits(
        &self,
        token_id: u32,
        cache: &mut GPTDecodeCache,
    ) -> Result<Vec<f32>, GPTModelError> {
        self.validate_decode_cache_shape(cache)?;
        if cache.next_position >= self.max_seq_len {
            return Err(GPTModelError::SequenceTooLong {
                seq_len: cache.next_position + 1,
                max_seq_len: self.max_seq_len,
            });
        }

        let token_row = self
            .token_embedding
            .try_embedding_ref(token_id)
            .map_err(GPTModelError::TokenEmbeddingFailed)?;
        let pos_row = self
            .positional_embedding
            .try_embedding_ref(cache.next_position)
            .map_err(GPTModelError::PositionalEmbeddingFailed)?;

        let mut current = Vec::with_capacity(self.embedding_dim);
        for i in 0..self.embedding_dim {
            current.push(token_row[i] + pos_row[i]);
        }

        let mut layer_inputs_to_append = Vec::with_capacity(self.layers.len());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            if cache.layer_inputs[layer_index].len != cache.next_position {
                return Err(GPTModelError::DecodeCacheInvalid(
                    "cached layer length does not match next position",
                ));
            }

            layer_inputs_to_append.push(current.clone());
            current = layer
                .try_forward_last_flat(
                    cache.layer_inputs[layer_index].as_flat(),
                    cache.next_position,
                    &current,
                    false,
                )
                .map_err(|source| GPTModelError::TransformerLayerFailed {
                    layer: layer_index,
                    source,
                })?;
        }

        for (layer_index, row) in layer_inputs_to_append.into_iter().enumerate() {
            cache.layer_inputs[layer_index].push_row(&row)?;
        }
        cache.next_position += 1;

        self.project_row_to_vocab(&current)
    }

    fn validate_decode_cache_shape(&self, cache: &GPTDecodeCache) -> Result<(), GPTModelError> {
        if cache.layer_inputs.len() != self.layers.len() {
            return Err(GPTModelError::DecodeCacheInvalid(
                "layer count does not match model",
            ));
        }
        if cache.embedding_dim != self.embedding_dim {
            return Err(GPTModelError::DecodeCacheInvalid(
                "embedding dimension does not match model",
            ));
        }
        if cache.max_seq_len != self.max_seq_len {
            return Err(GPTModelError::DecodeCacheInvalid(
                "max sequence length does not match model",
            ));
        }
        if cache.next_position > self.max_seq_len {
            return Err(GPTModelError::SequenceTooLong {
                seq_len: cache.next_position,
                max_seq_len: self.max_seq_len,
            });
        }
        if !cache.layer_inputs.is_empty()
            && cache.layer_inputs.iter().any(|layer| {
                layer.len != cache.next_position
                    || layer.embedding_dim != self.embedding_dim
                    || layer.max_seq_len != self.max_seq_len
                    || layer.data.len() != layer.len.saturating_mul(self.embedding_dim)
            })
        {
            return Err(GPTModelError::DecodeCacheInvalid(
                "cached layer metadata does not match model",
            ));
        }
        Ok(())
    }

    fn project_to_vocab(&self, hidden_states: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, GPTModelError> {
        let mut logits = Vec::with_capacity(hidden_states.len());

        for state in hidden_states {
            logits.push(self.project_row_to_vocab(state)?);
        }

        Ok(logits)
    }

    fn project_row_to_vocab(&self, state: &[f32]) -> Result<Vec<f32>, GPTModelError> {
        if state.len() != self.embedding_dim {
            return Err(GPTModelError::InvalidConfig(
                "hidden state dimension does not match embedding_dim",
            ));
        }

        let mut row_logits = vec![0.0_f32; self.vocab_size];
        for token_id in 0..self.vocab_size {
            let head_row = if self.tie_weights {
                self.token_embedding
                    .try_embedding_ref(token_id as u32)
                    .map_err(GPTModelError::ProjectionFailed)?
            } else {
                self.lm_head
                    .as_ref()
                    .and_then(|head| head.get(token_id).map(Vec::as_slice))
                    .ok_or(GPTModelError::InvalidConfig("lm_head row missing"))?
            };

            let mut dot = 0.0_f32;
            for i in 0..self.embedding_dim {
                dot += state[i] * head_row[i];
            }
            row_logits[token_id] = dot;
        }
        Ok(row_logits)
    }
}

fn add_matrices(a: &[Vec<f32>], b: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
    if a.len() != b.len() {
        return None;
    }
    if a.iter().zip(b.iter()).any(|(x, y)| x.len() != y.len()) {
        return None;
    }

    let mut out = Vec::with_capacity(a.len());
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        let mut row_out = Vec::with_capacity(row_a.len());
        for (x, y) in row_a.iter().zip(row_b.iter()) {
            row_out.push(x + y);
        }
        out.push(row_out);
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{GPTConfig, GPTModel};

    #[test]
    fn default_constructor_uses_tied_weights() {
        let model = GPTModel::new(32, 16, 8, 16, 2, 2, 123).unwrap();
        assert!(model.tie_weights());
    }

    #[test]
    fn untied_config_runs_forward_with_expected_shape() {
        let model = GPTModel::from_config(GPTConfig {
            vocab_size: 20,
            max_seq_len: 12,
            embedding_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 2,
            seed: 99,
            tie_weights: false,
        })
        .unwrap();

        let logits = model.try_forward(&[1, 2, 3, 4], false).unwrap();
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), 20);
    }

    #[test]
    fn forward_last_matches_full_forward_last_row() {
        let model = GPTModel::from_config(GPTConfig {
            vocab_size: 24,
            max_seq_len: 16,
            embedding_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 3,
            seed: 77,
            tie_weights: true,
        })
        .unwrap();

        let tokens = [1, 2, 3, 4, 5];
        let full = model.try_forward(&tokens, false).unwrap();
        let last = model.try_forward_last(&tokens, false).unwrap();
        assert_eq!(last.len(), model.vocab_size());
        assert_eq!(full.last().unwrap().len(), last.len());
        for (a, b) in full.last().unwrap().iter().zip(last.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn decode_cache_matches_full_recompute_for_next_token() {
        let model = GPTModel::from_config(GPTConfig {
            vocab_size: 32,
            max_seq_len: 16,
            embedding_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 3,
            seed: 1234,
            tie_weights: false,
        })
        .unwrap();

        let prompt = [3, 4, 5, 6];
        let next_token = 7;
        let mut cache = model.new_decode_cache();
        let prompt_logits = model.try_prefill_decode_cache(&prompt, &mut cache).unwrap();
        let full_prompt = model.try_forward(&prompt, false).unwrap();
        for (a, b) in full_prompt.last().unwrap().iter().zip(prompt_logits.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        let cached_logits = model
            .try_decode_next_logits(next_token, &mut cache)
            .unwrap();
        let mut extended = prompt.to_vec();
        extended.push(next_token);
        let full_extended = model.try_forward(&extended, false).unwrap();
        for (a, b) in full_extended
            .last()
            .unwrap()
            .iter()
            .zip(cached_logits.iter())
        {
            assert!((a - b).abs() < 1e-5);
        }
        assert_eq!(cache.next_position(), extended.len());
        assert_eq!(cache.cached_tokens(), extended.len());
    }

    #[test]
    fn decode_cache_preallocates_contiguous_layer_buffers() {
        let model = GPTModel::from_config(GPTConfig {
            vocab_size: 16,
            max_seq_len: 10,
            embedding_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 2,
            seed: 55,
            tie_weights: true,
        })
        .unwrap();

        let cache = model.new_decode_cache();
        assert_eq!(cache.cached_layers(), 2);
        assert_eq!(cache.cached_tokens(), 0);
        assert!(cache.cached_capacity_per_layer() >= model.max_seq_len() * model.embedding_dim());
    }
}
