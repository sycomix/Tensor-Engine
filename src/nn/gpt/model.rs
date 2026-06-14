use super::embeddings::{EmbeddingError, TokenEmbedding};
use super::positional_embeddings::{PositionalEmbedding, PositionalEmbeddingError};
use super::transformer_block::{TransformerBlock, TransformerBlockError};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum GPTModelError {
    InvalidConfig(&'static str),
    EmptyInput,
    EmptySequenceInBatch { index: usize },
    RaggedBatch,
    SequenceTooLong { seq_len: usize, max_seq_len: usize },
    TokenEmbeddingFailed(EmbeddingError),
    PositionalEmbeddingFailed(PositionalEmbeddingError),
    TransformerLayerFailed { layer: usize, source: TransformerBlockError },
    ProjectionFailed(EmbeddingError),
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
        let positional_embedding = PositionalEmbedding::random(
            cfg.max_seq_len,
            cfg.embedding_dim,
            cfg.seed ^ 0xA5A5_5A5A,
        )
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
            let lm = TokenEmbedding::random(
                cfg.vocab_size,
                cfg.embedding_dim,
                cfg.seed ^ 0x5A5A_A5A5,
            )
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
            hidden = layer
                .try_forward(&hidden, training)
                .map_err(|source| GPTModelError::TransformerLayerFailed {
                    layer: layer_index,
                    source,
                })?;
        }

        self.project_to_vocab(&hidden)
    }

    pub fn forward_batch(&self, batch_token_ids: &[Vec<u32>], training: bool) -> Vec<Vec<Vec<f32>>> {
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

    fn project_to_vocab(&self, hidden_states: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, GPTModelError> {
        let mut logits = Vec::with_capacity(hidden_states.len());

        for state in hidden_states {
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
            logits.push(row_logits);
        }

        Ok(logits)
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
}
