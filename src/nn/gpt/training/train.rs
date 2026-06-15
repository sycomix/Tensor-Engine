use super::super::dataset::{overlapping_windows, BatchShard, FinalWindowPolicy};
use super::super::framework::backend::{BackendError, CpuAutogradBackend, TensorBackend};
use super::super::framework::nn::{
    Linear as FrameworkLinear, Module as FrameworkModule, Sgd as FrameworkSgd,
    TransformerBlock as FrameworkTransformerBlock,
};
use super::loss::{try_next_token_cross_entropy_batch, CrossEntropyError};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(feature = "tch-backend")]
use super::train_tch::{
    default_tch_device, train_transformer_tch, TchPrecision, TchTrainConfig, TchTrainError,
    TchTransformerConfig, TchTransformerLM,
};

#[derive(Debug, Clone, PartialEq)]
pub enum TrainError {
    InvalidConfig(&'static str),
    EmptyDataset,
    InvalidSequenceLength { index: usize, len: usize },
    RaggedBatch { expected: usize, found: usize },
    TokenOutOfRange { token: u32, vocab_size: usize },
    FrameworkBackend(BackendError),
    Loss(CrossEntropyError),
    Io(String),
    Serialization(String),
    Dataset(String),
}

impl Display for TrainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            TrainError::EmptyDataset => write!(f, "dataset is empty"),
            TrainError::InvalidSequenceLength { index, len } => write!(
                f,
                "sequence at index {} must have length >= 2, found {}",
                index, len
            ),
            TrainError::RaggedBatch { expected, found } => {
                write!(
                    f,
                    "ragged batch: expected len {}, found {}",
                    expected, found
                )
            }
            TrainError::TokenOutOfRange { token, vocab_size } => write!(
                f,
                "token {} out of range for vocab size {}",
                token, vocab_size
            ),
            TrainError::FrameworkBackend(err) => write!(f, "framework backend error: {:?}", err),
            TrainError::Loss(err) => write!(f, "loss computation failed: {}", err),
            TrainError::Io(msg) => write!(f, "io error: {}", msg),
            TrainError::Serialization(msg) => write!(f, "serialization error: {}", msg),
            TrainError::Dataset(msg) => write!(f, "dataset pipeline error: {}", msg),
        }
    }
}

impl Error for TrainError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            TrainError::Loss(err) => Some(err),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub max_grad_norm: f32,
    pub adamw: AdamWConfig,
    pub schedule: LrSchedule,
    pub seed: u64,
    #[serde(default)]
    pub initial_global_step: usize,
    #[serde(default)]
    pub checkpoint_interval: usize,
    #[serde(default = "default_max_checkpoints")]
    pub max_checkpoints: usize,
    #[serde(default)]
    pub checkpoint_dir: Option<String>,
}

fn default_max_checkpoints() -> usize {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedPackingConfig {
    pub window_size: usize,
    pub stride: usize,
    pub pad_token_id: u32,
    pub rank: usize,
    pub world_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftExample {
    pub prompt_tokens: Vec<u32>,
    pub response_tokens: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftFormatConfig {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub assistant_prefix_tokens: Vec<u32>,
    pub max_seq_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyEvalCase {
    pub prompt_tokens: Vec<u32>,
    pub disallowed_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyEvalSummary {
    pub total_cases: usize,
    pub disallowed_hits: usize,
    pub refusal_hits: usize,
    pub disallowed_hit_rate: f32,
    pub refusal_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentEvalReport {
    pub quality: EvalMetrics,
    pub safety: SafetyEvalSummary,
}

impl DistributedPackingConfig {
    pub fn validate(&self) -> Result<(), TrainError> {
        if self.window_size == 0 {
            return Err(TrainError::InvalidConfig("window_size must be > 0"));
        }
        if self.stride == 0 {
            return Err(TrainError::InvalidConfig("stride must be > 0"));
        }
        if self.world_size == 0 || self.rank >= self.world_size {
            return Err(TrainError::InvalidConfig(
                "rank/world_size must satisfy world_size > 0 and rank < world_size",
            ));
        }
        Ok(())
    }
}

pub fn packed_window_stream<'a>(
    token_ids: &'a [u32],
    cfg: &DistributedPackingConfig,
) -> Result<impl Iterator<Item = Vec<u32>> + 'a, TrainError> {
    cfg.validate()?;
    let shard = BatchShard::new(cfg.rank, cfg.world_size)
        .map_err(|_| TrainError::Dataset("invalid shard settings".to_string()))?;
    let iter = overlapping_windows(
        token_ids,
        cfg.window_size,
        cfg.stride,
        FinalWindowPolicy::Pad {
            pad_id: cfg.pad_token_id,
        },
    )
    .map_err(|_| TrainError::Dataset("failed to create overlapping windows".to_string()))?
    .with_shard(shard);

    Ok(iter.map(|window| window.iter_padded().collect::<Vec<u32>>()))
}

pub fn build_packed_dataset_from_corpus_tokens(
    token_ids: &[u32],
    cfg: &DistributedPackingConfig,
) -> Result<Vec<Vec<u32>>, TrainError> {
    let windows: Vec<Vec<u32>> = packed_window_stream(token_ids, cfg)?.collect();
    if windows.is_empty() {
        return Err(TrainError::Dataset(
            "packed dataset is empty; adjust window/stride or provide more tokens".to_string(),
        ));
    }
    Ok(windows)
}

impl TrainConfig {
    pub fn validate(&self) -> Result<(), TrainError> {
        if self.vocab_size == 0 {
            return Err(TrainError::InvalidConfig("vocab_size must be > 0"));
        }
        if self.embedding_dim == 0 {
            return Err(TrainError::InvalidConfig("embedding_dim must be > 0"));
        }
        if self.hidden_dim == 0 {
            return Err(TrainError::InvalidConfig("hidden_dim must be > 0"));
        }
        if self.epochs == 0 {
            return Err(TrainError::InvalidConfig("epochs must be > 0"));
        }
        if self.batch_size == 0 {
            return Err(TrainError::InvalidConfig("batch_size must be > 0"));
        }
        if self.max_grad_norm <= 0.0 {
            return Err(TrainError::InvalidConfig("max_grad_norm must be > 0"));
        }
        if self.checkpoint_interval > 0 && self.max_checkpoints == 0 {
            return Err(TrainError::InvalidConfig(
                "max_checkpoints must be > 0 when checkpoint_interval is enabled",
            ));
        }
        self.adamw.validate()?;
        self.schedule.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl AdamWConfig {
    pub fn validate(&self) -> Result<(), TrainError> {
        if self.lr <= 0.0 {
            return Err(TrainError::InvalidConfig("learning rate must be > 0"));
        }
        if !(0.0..1.0).contains(&self.beta1) {
            return Err(TrainError::InvalidConfig("beta1 must be in [0, 1)"));
        }
        if !(0.0..1.0).contains(&self.beta2) {
            return Err(TrainError::InvalidConfig("beta2 must be in [0, 1)"));
        }
        if self.eps <= 0.0 {
            return Err(TrainError::InvalidConfig("eps must be > 0"));
        }
        if self.weight_decay < 0.0 {
            return Err(TrainError::InvalidConfig("weight_decay must be >= 0"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LrSchedule {
    pub warmup_steps: usize,
    pub step_decay_every: usize,
    pub step_decay_gamma: f32,
    pub min_lr_scale: f32,
}

impl LrSchedule {
    pub fn validate(&self) -> Result<(), TrainError> {
        if self.step_decay_every == 0 {
            return Err(TrainError::InvalidConfig("step_decay_every must be > 0"));
        }
        if self.step_decay_gamma <= 0.0 || self.step_decay_gamma > 1.0 {
            return Err(TrainError::InvalidConfig(
                "step_decay_gamma must be in (0, 1]",
            ));
        }
        if self.min_lr_scale <= 0.0 || self.min_lr_scale > 1.0 {
            return Err(TrainError::InvalidConfig("min_lr_scale must be in (0, 1]"));
        }
        Ok(())
    }

    pub fn lr_scale(&self, step: usize) -> f32 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return (step as f32 + 1.0) / self.warmup_steps as f32;
        }

        let decay_steps = step / self.step_decay_every;
        let decayed = self.step_decay_gamma.powi(decay_steps as i32);
        decayed.max(self.min_lr_scale)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinySeqModel {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub embedding: Vec<Vec<f32>>, // [vocab][embed]
    pub w1: Vec<Vec<f32>>,        // [embed][hidden]
    pub b1: Vec<f32>,             // [hidden]
    pub w2: Vec<Vec<f32>>,        // [hidden][vocab]
    pub b2: Vec<f32>,             // [vocab]
}

impl TinySeqModel {
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        hidden_dim: usize,
        seed: u64,
    ) -> Result<Self, TrainError> {
        if vocab_size == 0 || embedding_dim == 0 || hidden_dim == 0 {
            return Err(TrainError::InvalidConfig(
                "vocab_size, embedding_dim, hidden_dim must be > 0",
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let emb_scale = (1.0_f32 / embedding_dim as f32).sqrt();
        let h_scale = (1.0_f32 / hidden_dim as f32).sqrt();

        let embedding = init_matrix_random(vocab_size, embedding_dim, emb_scale, &mut rng);
        let w1 = init_matrix_random(embedding_dim, hidden_dim, emb_scale, &mut rng);
        let w2 = init_matrix_random(hidden_dim, vocab_size, h_scale, &mut rng);

        Ok(Self {
            vocab_size,
            embedding_dim,
            hidden_dim,
            embedding,
            w1,
            b1: vec![0.0; hidden_dim],
            w2,
            b2: vec![0.0; vocab_size],
        })
    }

    pub fn forward(&self, input_tokens: &[u32]) -> Result<Vec<Vec<f32>>, TrainError> {
        let mut logits = Vec::with_capacity(input_tokens.len());

        for &token in input_tokens {
            let token_idx = token as usize;
            if token_idx >= self.vocab_size {
                return Err(TrainError::TokenOutOfRange {
                    token,
                    vocab_size: self.vocab_size,
                });
            }

            let x = &self.embedding[token_idx];
            let mut hidden = vec![0.0_f32; self.hidden_dim];
            for (j, h) in hidden.iter_mut().enumerate() {
                let mut sum = self.b1[j];
                for (i, &x_i) in x.iter().enumerate() {
                    sum += x_i * self.w1[i][j];
                }
                *h = sum;
            }

            let mut out = vec![0.0_f32; self.vocab_size];
            for (k, logit) in out.iter_mut().enumerate() {
                let mut sum = self.b2[k];
                for (j, &h_j) in hidden.iter().enumerate() {
                    sum += h_j * self.w2[j][k];
                }
                *logit = sum;
            }
            logits.push(out);
        }

        Ok(logits)
    }

    pub fn forward_batch(
        &self,
        batch_tokens: &[Vec<u32>],
    ) -> Result<Vec<Vec<Vec<f32>>>, TrainError> {
        let mut out = Vec::with_capacity(batch_tokens.len());
        for seq in batch_tokens {
            out.push(self.forward(seq)?);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerModelConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub seed: u64,
}

pub struct TransformerSeqModel {
    backend: CpuAutogradBackend,
    token_embedding: Vec<Vec<f32>>,      // [vocab][embed]
    positional_embedding: Vec<Vec<f32>>, // [max_seq][embed]
    blocks: Vec<FrameworkTransformerBlock<CpuAutogradBackend>>,
    lm_head: FrameworkLinear<CpuAutogradBackend>,
    model_config: TransformerModelConfig,
    vocab_size: usize,
    max_seq_len: usize,
    embedding_dim: usize,
}

#[derive(Debug, Clone)]
pub struct TransformerKvCache {
    keys: Vec<Vec<Vec<f32>>>,
    values: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeLatencyBenchmark {
    pub max_new_tokens: usize,
    pub full_recompute_ms: f64,
    pub kv_cache_ms: f64,
    pub speedup: f64,
}

#[derive(Debug, Clone)]
struct TransformerBlockInferenceWeights {
    q_weight: Vec<Vec<f32>>,
    q_bias: Vec<f32>,
    k_weight: Vec<Vec<f32>>,
    k_bias: Vec<f32>,
    v_weight: Vec<Vec<f32>>,
    v_bias: Vec<f32>,
    out_weight: Vec<Vec<f32>>,
    out_bias: Vec<f32>,
    ln1_gamma: Vec<f32>,
    ln1_beta: Vec<f32>,
    ln2_gamma: Vec<f32>,
    ln2_beta: Vec<f32>,
    ff1_weight: Vec<Vec<f32>>,
    ff1_bias: Vec<f32>,
    ff2_weight: Vec<Vec<f32>>,
    ff2_bias: Vec<f32>,
}

impl std::fmt::Debug for TransformerSeqModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerSeqModel")
            .field("vocab_size", &self.vocab_size)
            .field("max_seq_len", &self.max_seq_len)
            .field("embedding_dim", &self.embedding_dim)
            .field("num_layers", &self.blocks.len())
            .finish()
    }
}

impl TransformerSeqModel {
    pub fn new(cfg: TransformerModelConfig) -> Result<Self, TrainError> {
        if cfg.vocab_size == 0
            || cfg.max_seq_len == 0
            || cfg.embedding_dim == 0
            || cfg.hidden_dim == 0
            || cfg.num_heads == 0
            || cfg.num_layers == 0
        {
            return Err(TrainError::InvalidConfig(
                "transformer config dimensions and counts must be > 0",
            ));
        }

        let backend = CpuAutogradBackend;
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let scale = (1.0_f32 / cfg.embedding_dim as f32).sqrt();

        let token_embedding =
            init_matrix_random(cfg.vocab_size, cfg.embedding_dim, scale, &mut rng);
        let positional_embedding =
            init_matrix_random(cfg.max_seq_len, cfg.embedding_dim, scale, &mut rng);

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let block = FrameworkTransformerBlock::new_with_dropout_seeded(
                &backend,
                cfg.embedding_dim,
                cfg.num_heads,
                cfg.hidden_dim,
                0.0,
                cfg.seed.wrapping_add(i as u64),
                &format!("transformer.block{}", i),
            )
            .map_err(TrainError::FrameworkBackend)?;
            blocks.push(block);
        }

        let lm_head = FrameworkLinear::new(
            &backend,
            cfg.embedding_dim,
            cfg.vocab_size,
            "transformer.lm_head",
        )
        .map_err(TrainError::FrameworkBackend)?;

        Ok(Self {
            backend,
            token_embedding,
            positional_embedding,
            blocks,
            lm_head,
            model_config: cfg.clone(),
            vocab_size: cfg.vocab_size,
            max_seq_len: cfg.max_seq_len,
            embedding_dim: cfg.embedding_dim,
        })
    }

    pub fn config(&self) -> &TransformerModelConfig {
        &self.model_config
    }

    pub fn forward(
        &self,
        input_tokens: &[u32],
        training: bool,
    ) -> Result<Vec<Vec<f32>>, TrainError> {
        let logits = self.forward_logits_tensor(input_tokens, training)?;
        let data = self.backend.data(&logits);
        Ok(reshape_2d(data, input_tokens.len(), self.vocab_size))
    }

    pub fn forward_batch(
        &self,
        batch_tokens: &[Vec<u32>],
        training: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>, TrainError> {
        let mut out = Vec::with_capacity(batch_tokens.len());
        for seq in batch_tokens {
            out.push(self.forward(seq, training)?);
        }
        Ok(out)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn to_checkpoint(&self) -> Result<TransformerSeqCheckpoint, TrainError> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            blocks.push(TransformerBlockCheckpoint {
                q_weight: tensor_to_2d(&self.backend, &block.attention.q_proj.weight.tensor)?,
                q_bias: tensor_to_1d(&self.backend, &block.attention.q_proj.bias.tensor)?,
                k_weight: tensor_to_2d(&self.backend, &block.attention.k_proj.weight.tensor)?,
                k_bias: tensor_to_1d(&self.backend, &block.attention.k_proj.bias.tensor)?,
                v_weight: tensor_to_2d(&self.backend, &block.attention.v_proj.weight.tensor)?,
                v_bias: tensor_to_1d(&self.backend, &block.attention.v_proj.bias.tensor)?,
                out_weight: tensor_to_2d(&self.backend, &block.attention.out_proj.weight.tensor)?,
                out_bias: tensor_to_1d(&self.backend, &block.attention.out_proj.bias.tensor)?,
                ln1_gamma: tensor_to_1d(&self.backend, &block.ln1.gamma.tensor)?,
                ln1_beta: tensor_to_1d(&self.backend, &block.ln1.beta.tensor)?,
                ln2_gamma: tensor_to_1d(&self.backend, &block.ln2.gamma.tensor)?,
                ln2_beta: tensor_to_1d(&self.backend, &block.ln2.beta.tensor)?,
                ff1_weight: tensor_to_2d(&self.backend, &block.ff1.weight.tensor)?,
                ff1_bias: tensor_to_1d(&self.backend, &block.ff1.bias.tensor)?,
                ff2_weight: tensor_to_2d(&self.backend, &block.ff2.weight.tensor)?,
                ff2_bias: tensor_to_1d(&self.backend, &block.ff2.bias.tensor)?,
            });
        }

        Ok(TransformerSeqCheckpoint {
            model_config: self.model_config.clone(),
            token_embedding: self.token_embedding.clone(),
            positional_embedding: self.positional_embedding.clone(),
            blocks,
            lm_head_weight: tensor_to_2d(&self.backend, &self.lm_head.weight.tensor)?,
            lm_head_bias: tensor_to_1d(&self.backend, &self.lm_head.bias.tensor)?,
        })
    }

    pub fn from_checkpoint(checkpoint: &TransformerSeqCheckpoint) -> Result<Self, TrainError> {
        if checkpoint.blocks.len() != checkpoint.model_config.num_layers {
            return Err(TrainError::InvalidConfig(
                "checkpoint block count does not match model_config.num_layers",
            ));
        }

        let mut model = Self::new(checkpoint.model_config.clone())?;
        model.token_embedding = checkpoint.token_embedding.clone();
        model.positional_embedding = checkpoint.positional_embedding.clone();

        for (index, block_state) in checkpoint.blocks.iter().enumerate() {
            let block = &mut model.blocks[index];

            set_linear_from_state(
                &model.backend,
                &mut block.attention.q_proj,
                &block_state.q_weight,
                &block_state.q_bias,
            )?;
            set_linear_from_state(
                &model.backend,
                &mut block.attention.k_proj,
                &block_state.k_weight,
                &block_state.k_bias,
            )?;
            set_linear_from_state(
                &model.backend,
                &mut block.attention.v_proj,
                &block_state.v_weight,
                &block_state.v_bias,
            )?;
            set_linear_from_state(
                &model.backend,
                &mut block.attention.out_proj,
                &block_state.out_weight,
                &block_state.out_bias,
            )?;

            set_vector_param(
                &model.backend,
                &mut block.ln1.gamma.tensor,
                &block_state.ln1_gamma,
            )?;
            set_vector_param(
                &model.backend,
                &mut block.ln1.beta.tensor,
                &block_state.ln1_beta,
            )?;
            set_vector_param(
                &model.backend,
                &mut block.ln2.gamma.tensor,
                &block_state.ln2_gamma,
            )?;
            set_vector_param(
                &model.backend,
                &mut block.ln2.beta.tensor,
                &block_state.ln2_beta,
            )?;

            set_linear_from_state(
                &model.backend,
                &mut block.ff1,
                &block_state.ff1_weight,
                &block_state.ff1_bias,
            )?;
            set_linear_from_state(
                &model.backend,
                &mut block.ff2,
                &block_state.ff2_weight,
                &block_state.ff2_bias,
            )?;
        }

        set_linear_from_state(
            &model.backend,
            &mut model.lm_head,
            &checkpoint.lm_head_weight,
            &checkpoint.lm_head_bias,
        )?;

        Ok(model)
    }

    fn forward_logits_tensor(
        &self,
        input_tokens: &[u32],
        _training: bool,
    ) -> Result<<CpuAutogradBackend as TensorBackend>::Tensor, TrainError> {
        if input_tokens.is_empty() {
            return Err(TrainError::InvalidConfig(
                "transformer input must be non-empty",
            ));
        }
        if input_tokens.len() > self.max_seq_len {
            return Err(TrainError::InvalidConfig(
                "sequence length exceeds max_seq_len",
            ));
        }

        let input = self.build_input_tensor(input_tokens)?;

        let mut hidden = input;
        for block in &self.blocks {
            hidden = block
                .forward(&self.backend, &hidden)
                .map_err(TrainError::FrameworkBackend)?;
        }

        self.lm_head
            .forward(&self.backend, &hidden)
            .map_err(TrainError::FrameworkBackend)
    }

    fn build_input_tensor(
        &self,
        input_tokens: &[u32],
    ) -> Result<<CpuAutogradBackend as TensorBackend>::Tensor, TrainError> {
        let mut input_data = vec![0.0_f32; input_tokens.len() * self.embedding_dim];
        for (pos, &token) in input_tokens.iter().enumerate() {
            let token_idx = token as usize;
            if token_idx >= self.vocab_size {
                return Err(TrainError::TokenOutOfRange {
                    token,
                    vocab_size: self.vocab_size,
                });
            }
            for d in 0..self.embedding_dim {
                input_data[pos * self.embedding_dim + d] =
                    self.token_embedding[token_idx][d] + self.positional_embedding[pos][d];
            }
        }

        self.backend
            .from_data(
                input_data,
                vec![input_tokens.len(), self.embedding_dim],
                false,
            )
            .map_err(TrainError::FrameworkBackend)
    }

    fn grad_global_norm(&mut self) -> f32 {
        let backend = self.backend;
        let mut sum_sq = 0.0_f64;

        for block in &mut self.blocks {
            block.for_each_parameter_mut(&mut |parameter| {
                if let Some(grad) = backend.grad(&parameter.tensor) {
                    for g in grad {
                        let v = g as f64;
                        sum_sq += v * v;
                    }
                }
            });
        }

        self.lm_head.for_each_parameter_mut(&mut |parameter| {
            if let Some(grad) = backend.grad(&parameter.tensor) {
                for g in grad {
                    let v = g as f64;
                    sum_sq += v * v;
                }
            }
        });

        (sum_sq as f32).sqrt()
    }

    pub fn train_on_sequence(
        &mut self,
        input_tokens: &[u32],
        targets: &[u32],
        learning_rate: f32,
    ) -> Result<(f32, f32), TrainError> {
        if input_tokens.is_empty() || input_tokens.len() != targets.len() {
            return Err(TrainError::InvalidConfig(
                "transformer train sequence must be non-empty and align with targets",
            ));
        }

        let logits = self.forward_logits_tensor(input_tokens, true)?;

        let mut target_data = vec![0.0_f32; input_tokens.len() * self.vocab_size];
        for (t, &target) in targets.iter().enumerate() {
            let target_idx = target as usize;
            if target_idx >= self.vocab_size {
                return Err(TrainError::TokenOutOfRange {
                    token: target,
                    vocab_size: self.vocab_size,
                });
            }
            target_data[t * self.vocab_size + target_idx] = 1.0;
        }

        let target_tensor = self
            .backend
            .from_data(
                target_data,
                vec![input_tokens.len(), self.vocab_size],
                false,
            )
            .map_err(TrainError::FrameworkBackend)?;

        let probs = self
            .backend
            .softmax_last_dim(&logits)
            .map_err(TrainError::FrameworkBackend)?;
        let picked = self
            .backend
            .mul(&probs, &target_tensor)
            .map_err(TrainError::FrameworkBackend)?;
        let target_probs = self
            .backend
            .sum_last_dim(&picked)
            .map_err(TrainError::FrameworkBackend)?;

        let eps = self
            .backend
            .scalar(1.0e-9, false)
            .map_err(TrainError::FrameworkBackend)?;
        let safe_target_probs = self
            .backend
            .add(&target_probs, &eps)
            .map_err(TrainError::FrameworkBackend)?;
        let log_target_probs = self
            .backend
            .log(&safe_target_probs)
            .map_err(TrainError::FrameworkBackend)?;

        let minus_one = self
            .backend
            .scalar(-1.0, false)
            .map_err(TrainError::FrameworkBackend)?;
        let nll = self
            .backend
            .mul(&log_target_probs, &minus_one)
            .map_err(TrainError::FrameworkBackend)?;
        let loss = self
            .backend
            .mean(&nll)
            .map_err(TrainError::FrameworkBackend)?;

        self.backend
            .backward(&loss)
            .map_err(TrainError::FrameworkBackend)?;

        let grad_norm = self.grad_global_norm();

        let optimizer = FrameworkSgd::new(learning_rate);
        for block in &mut self.blocks {
            optimizer
                .step(&self.backend, block)
                .map_err(TrainError::FrameworkBackend)?;
        }
        optimizer
            .step(&self.backend, &mut self.lm_head)
            .map_err(TrainError::FrameworkBackend)?;

        let loss_data = self.backend.data(&loss);
        let loss = loss_data
            .first()
            .copied()
            .ok_or(TrainError::InvalidConfig("loss tensor was empty"))?;

        Ok((loss, grad_norm))
    }

    pub fn init_kv_cache(&self) -> TransformerKvCache {
        TransformerKvCache {
            keys: vec![Vec::new(); self.blocks.len()],
            values: vec![Vec::new(); self.blocks.len()],
        }
    }

    pub fn generate_greedy_with_kv_cache(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, TrainError> {
        self.generate_greedy_with_kv_cache_constrained(prompt, max_new_tokens, &[], None, 0)
    }

    pub fn generate_greedy_with_kv_cache_constrained(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        disallowed_token_ids: &[u32],
        eos_token_id: Option<u32>,
        min_new_tokens_before_eos: usize,
    ) -> Result<Vec<u32>, TrainError> {
        if prompt.is_empty() {
            return Err(TrainError::InvalidConfig("prompt must be non-empty"));
        }
        if prompt.len() + max_new_tokens > self.max_seq_len {
            return Err(TrainError::InvalidConfig(
                "prompt + max_new_tokens exceeds max_seq_len",
            ));
        }

        let weights = self.extract_inference_weights()?;
        let mut cache = self.init_kv_cache();
        let mut generated = prompt.to_vec();
        let mut last_logits = Vec::new();

        for (position, &token) in prompt.iter().enumerate() {
            last_logits =
                self.decode_next_logits_with_cache_internal(token, position, &weights, &mut cache)?;
        }

        for new_idx in 0..max_new_tokens {
            let mut masked = disallowed_token_ids.to_vec();
            if let Some(eos_id) = eos_token_id {
                if new_idx < min_new_tokens_before_eos {
                    masked.push(eos_id);
                }
            }

            let next = masked_argmax_index(&last_logits, &masked)
                .unwrap_or_else(|| argmax_index(&last_logits)) as u32;
            generated.push(next);
            let position = generated.len() - 1;
            last_logits =
                self.decode_next_logits_with_cache_internal(next, position, &weights, &mut cache)?;

            if eos_token_id == Some(next) {
                break;
            }
        }

        Ok(generated)
    }

    fn decode_next_logits_with_cache_internal(
        &self,
        token: u32,
        position: usize,
        weights: &[TransformerBlockInferenceWeights],
        cache: &mut TransformerKvCache,
    ) -> Result<Vec<f32>, TrainError> {
        if position >= self.max_seq_len {
            return Err(TrainError::InvalidConfig(
                "decode position exceeds max_seq_len",
            ));
        }
        let token_idx = token as usize;
        if token_idx >= self.vocab_size {
            return Err(TrainError::TokenOutOfRange {
                token,
                vocab_size: self.vocab_size,
            });
        }

        let mut x = vec![0.0_f32; self.embedding_dim];
        for (d, dst) in x.iter_mut().enumerate() {
            *dst = self.token_embedding[token_idx][d] + self.positional_embedding[position][d];
        }

        let head_dim = self.embedding_dim / self.model_config.num_heads;
        for (layer_idx, layer_w) in weights.iter().enumerate() {
            let norm1 = layer_norm_forward(&x, &layer_w.ln1_gamma, &layer_w.ln1_beta, 1e-5);
            let q = linear_forward(&norm1, &layer_w.q_weight, &layer_w.q_bias);
            let k = linear_forward(&norm1, &layer_w.k_weight, &layer_w.k_bias);
            let v = linear_forward(&norm1, &layer_w.v_weight, &layer_w.v_bias);

            cache.keys[layer_idx].push(k);
            cache.values[layer_idx].push(v);

            let context = attention_single_query(
                &q,
                &cache.keys[layer_idx],
                &cache.values[layer_idx],
                self.model_config.num_heads,
                head_dim,
            );
            let attn_out = linear_forward(&context, &layer_w.out_weight, &layer_w.out_bias);
            let res1 = add_vec(&x, &attn_out);

            let norm2 = layer_norm_forward(&res1, &layer_w.ln2_gamma, &layer_w.ln2_beta, 1e-5);
            let ff_hidden = relu_vec(&linear_forward(
                &norm2,
                &layer_w.ff1_weight,
                &layer_w.ff1_bias,
            ));
            let ff_out = linear_forward(&ff_hidden, &layer_w.ff2_weight, &layer_w.ff2_bias);
            x = add_vec(&res1, &ff_out);
        }

        Ok(linear_forward(
            &x,
            &tensor_to_2d(&self.backend, &self.lm_head.weight.tensor)?,
            &tensor_to_1d(&self.backend, &self.lm_head.bias.tensor)?,
        ))
    }

    fn extract_inference_weights(
        &self,
    ) -> Result<Vec<TransformerBlockInferenceWeights>, TrainError> {
        let mut out = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            out.push(TransformerBlockInferenceWeights {
                q_weight: tensor_to_2d(&self.backend, &block.attention.q_proj.weight.tensor)?,
                q_bias: tensor_to_1d(&self.backend, &block.attention.q_proj.bias.tensor)?,
                k_weight: tensor_to_2d(&self.backend, &block.attention.k_proj.weight.tensor)?,
                k_bias: tensor_to_1d(&self.backend, &block.attention.k_proj.bias.tensor)?,
                v_weight: tensor_to_2d(&self.backend, &block.attention.v_proj.weight.tensor)?,
                v_bias: tensor_to_1d(&self.backend, &block.attention.v_proj.bias.tensor)?,
                out_weight: tensor_to_2d(&self.backend, &block.attention.out_proj.weight.tensor)?,
                out_bias: tensor_to_1d(&self.backend, &block.attention.out_proj.bias.tensor)?,
                ln1_gamma: tensor_to_1d(&self.backend, &block.ln1.gamma.tensor)?,
                ln1_beta: tensor_to_1d(&self.backend, &block.ln1.beta.tensor)?,
                ln2_gamma: tensor_to_1d(&self.backend, &block.ln2.gamma.tensor)?,
                ln2_beta: tensor_to_1d(&self.backend, &block.ln2.beta.tensor)?,
                ff1_weight: tensor_to_2d(&self.backend, &block.ff1.weight.tensor)?,
                ff1_bias: tensor_to_1d(&self.backend, &block.ff1.bias.tensor)?,
                ff2_weight: tensor_to_2d(&self.backend, &block.ff2.weight.tensor)?,
                ff2_bias: tensor_to_1d(&self.backend, &block.ff2.bias.tensor)?,
            });
        }
        Ok(out)
    }
}

pub fn benchmark_transformer_decode_latency(
    model: &TransformerSeqModel,
    prompt: &[u32],
    max_new_tokens: usize,
) -> Result<DecodeLatencyBenchmark, TrainError> {
    if prompt.is_empty() {
        return Err(TrainError::InvalidConfig("prompt must be non-empty"));
    }

    let start_full = Instant::now();
    let mut seq = prompt.to_vec();
    for _ in 0..max_new_tokens {
        let logits = model.forward(&seq, false)?;
        let last = logits.last().ok_or(TrainError::InvalidConfig(
            "empty logits during full decode benchmark",
        ))?;
        seq.push(argmax_index(last) as u32);
    }
    let full_ms = start_full.elapsed().as_secs_f64() * 1000.0;

    let start_cache = Instant::now();
    let _ = model.generate_greedy_with_kv_cache(prompt, max_new_tokens)?;
    let cache_ms = start_cache.elapsed().as_secs_f64() * 1000.0;

    let speedup = if cache_ms > 0.0 {
        full_ms / cache_ms
    } else {
        0.0
    };
    Ok(DecodeLatencyBenchmark {
        max_new_tokens,
        full_recompute_ms: full_ms,
        kv_cache_ms: cache_ms,
        speedup,
    })
}

#[derive(Debug)]
pub enum SequenceModel {
    Tiny(TinySeqModel),
    Transformer(TransformerSeqModel),
}

impl SequenceModel {
    pub fn vocab_size(&self) -> usize {
        match self {
            SequenceModel::Tiny(model) => model.vocab_size,
            SequenceModel::Transformer(model) => model.vocab_size(),
        }
    }

    pub fn forward(
        &self,
        input_tokens: &[u32],
        training: bool,
    ) -> Result<Vec<Vec<f32>>, TrainError> {
        match self {
            SequenceModel::Tiny(model) => model.forward(input_tokens),
            SequenceModel::Transformer(model) => model.forward(input_tokens, training),
        }
    }

    pub fn forward_batch(
        &self,
        batch_tokens: &[Vec<u32>],
        training: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>, TrainError> {
        match self {
            SequenceModel::Tiny(model) => model.forward_batch(batch_tokens),
            SequenceModel::Transformer(model) => model.forward_batch(batch_tokens, training),
        }
    }
}

#[derive(Debug, Clone)]
struct TinySeqGrads {
    embedding: Vec<Vec<f32>>,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

impl TinySeqGrads {
    fn zeros_like(model: &TinySeqModel) -> Self {
        Self {
            embedding: vec![vec![0.0; model.embedding_dim]; model.vocab_size],
            w1: vec![vec![0.0; model.hidden_dim]; model.embedding_dim],
            b1: vec![0.0; model.hidden_dim],
            w2: vec![vec![0.0; model.vocab_size]; model.hidden_dim],
            b2: vec![0.0; model.vocab_size],
        }
    }

    fn global_norm(&self) -> f32 {
        let mut ss = 0.0_f32;
        acc_sum_squares_2d(&self.embedding, &mut ss);
        acc_sum_squares_2d(&self.w1, &mut ss);
        acc_sum_squares_1d(&self.b1, &mut ss);
        acc_sum_squares_2d(&self.w2, &mut ss);
        acc_sum_squares_1d(&self.b2, &mut ss);
        ss.sqrt()
    }

    fn scale_in_place(&mut self, scale: f32) {
        scale_2d(&mut self.embedding, scale);
        scale_2d(&mut self.w1, scale);
        scale_1d(&mut self.b1, scale);
        scale_2d(&mut self.w2, scale);
        scale_1d(&mut self.b2, scale);
    }
}

#[derive(Debug, Clone)]
struct AdamWState {
    step: usize,
    m_embedding: Vec<Vec<f32>>,
    v_embedding: Vec<Vec<f32>>,
    m_w1: Vec<Vec<f32>>,
    v_w1: Vec<Vec<f32>>,
    m_b1: Vec<f32>,
    v_b1: Vec<f32>,
    m_w2: Vec<Vec<f32>>,
    v_w2: Vec<Vec<f32>>,
    m_b2: Vec<f32>,
    v_b2: Vec<f32>,
}

impl AdamWState {
    fn new(model: &TinySeqModel) -> Self {
        Self {
            step: 0,
            m_embedding: vec![vec![0.0; model.embedding_dim]; model.vocab_size],
            v_embedding: vec![vec![0.0; model.embedding_dim]; model.vocab_size],
            m_w1: vec![vec![0.0; model.hidden_dim]; model.embedding_dim],
            v_w1: vec![vec![0.0; model.hidden_dim]; model.embedding_dim],
            m_b1: vec![0.0; model.hidden_dim],
            v_b1: vec![0.0; model.hidden_dim],
            m_w2: vec![vec![0.0; model.vocab_size]; model.hidden_dim],
            v_w2: vec![vec![0.0; model.vocab_size]; model.hidden_dim],
            m_b2: vec![0.0; model.vocab_size],
            v_b2: vec![0.0; model.vocab_size],
        }
    }

    fn step(&mut self, model: &mut TinySeqModel, grads: &TinySeqGrads, cfg: &AdamWConfig, lr: f32) {
        self.step += 1;
        let t = self.step as f32;

        let bias_c1 = 1.0 - cfg.beta1.powf(t);
        let bias_c2 = 1.0 - cfg.beta2.powf(t);

        adamw_update_2d(
            &mut model.embedding,
            &grads.embedding,
            &mut self.m_embedding,
            &mut self.v_embedding,
            cfg,
            lr,
            bias_c1,
            bias_c2,
        );
        adamw_update_2d(
            &mut model.w1,
            &grads.w1,
            &mut self.m_w1,
            &mut self.v_w1,
            cfg,
            lr,
            bias_c1,
            bias_c2,
        );
        adamw_update_1d(
            &mut model.b1,
            &grads.b1,
            &mut self.m_b1,
            &mut self.v_b1,
            cfg,
            lr,
            bias_c1,
            bias_c2,
        );
        adamw_update_2d(
            &mut model.w2,
            &grads.w2,
            &mut self.m_w2,
            &mut self.v_w2,
            cfg,
            lr,
            bias_c1,
            bias_c2,
        );
        adamw_update_1d(
            &mut model.b2,
            &grads.b2,
            &mut self.m_b2,
            &mut self.v_b2,
            cfg,
            lr,
            bias_c1,
            bias_c2,
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchLog {
    pub epoch: usize,
    pub batch_index: usize,
    pub global_step: usize,
    pub lr: f32,
    pub grad_norm: f32,
    pub loss: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainSummary {
    pub train_logs: Vec<BatchLog>,
    pub val_epoch_losses: Vec<f32>,
    pub val_epoch_metrics: Vec<EvalMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetrics {
    pub loss: f32,
    pub accuracy: f32,
    pub perplexity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinySeqCheckpoint {
    pub model: TinySeqModel,
    pub train_config: Option<TrainConfig>,
    pub global_step: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerBlockCheckpoint {
    pub q_weight: Vec<Vec<f32>>,
    pub q_bias: Vec<f32>,
    pub k_weight: Vec<Vec<f32>>,
    pub k_bias: Vec<f32>,
    pub v_weight: Vec<Vec<f32>>,
    pub v_bias: Vec<f32>,
    pub out_weight: Vec<Vec<f32>>,
    pub out_bias: Vec<f32>,
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
    pub ff1_weight: Vec<Vec<f32>>,
    pub ff1_bias: Vec<f32>,
    pub ff2_weight: Vec<Vec<f32>>,
    pub ff2_bias: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerSeqCheckpoint {
    pub model_config: TransformerModelConfig,
    pub token_embedding: Vec<Vec<f32>>,
    pub positional_embedding: Vec<Vec<f32>>,
    pub blocks: Vec<TransformerBlockCheckpoint>,
    pub lm_head_weight: Vec<Vec<f32>>,
    pub lm_head_bias: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerTrainingCheckpoint {
    #[serde(default = "default_transformer_checkpoint_version")]
    pub checkpoint_version: u32,
    pub model: TransformerSeqCheckpoint,
    pub train_config: Option<TrainConfig>,
    pub global_step: usize,
}

fn default_transformer_checkpoint_version() -> u32 {
    1
}

pub fn save_tiny_checkpoint<P: AsRef<Path>>(
    path: P,
    checkpoint: &TinySeqCheckpoint,
) -> Result<(), TrainError> {
    let payload = serde_json::to_string_pretty(checkpoint)
        .map_err(|e| TrainError::Serialization(e.to_string()))?;
    fs::write(path, payload).map_err(|e| TrainError::Io(e.to_string()))
}

pub fn load_tiny_checkpoint<P: AsRef<Path>>(path: P) -> Result<TinySeqCheckpoint, TrainError> {
    let payload = fs::read_to_string(path).map_err(|e| TrainError::Io(e.to_string()))?;
    serde_json::from_str(&payload).map_err(|e| TrainError::Serialization(e.to_string()))
}

pub fn save_transformer_checkpoint<P: AsRef<Path>>(
    path: P,
    checkpoint: &TransformerTrainingCheckpoint,
) -> Result<(), TrainError> {
    let payload = serde_json::to_string_pretty(checkpoint)
        .map_err(|e| TrainError::Serialization(e.to_string()))?;
    fs::write(path, payload).map_err(|e| TrainError::Io(e.to_string()))
}

pub fn load_transformer_checkpoint<P: AsRef<Path>>(
    path: P,
) -> Result<TransformerTrainingCheckpoint, TrainError> {
    let payload = fs::read_to_string(path).map_err(|e| TrainError::Io(e.to_string()))?;
    serde_json::from_str(&payload).map_err(|e| TrainError::Serialization(e.to_string()))
}

fn validate_transformer_checkpoint_vocab_layout(
    checkpoint: &TransformerTrainingCheckpoint,
) -> Result<(), TrainError> {
    let vocab_size = checkpoint.model.model_config.vocab_size;
    if vocab_size == 0 {
        return Err(TrainError::InvalidConfig(
            "checkpoint vocab_size must be greater than zero",
        ));
    }

    let embedding_dim = checkpoint.model.model_config.embedding_dim;
    if checkpoint.model.token_embedding.len() != vocab_size {
        return Err(TrainError::InvalidConfig(
            "checkpoint token_embedding row count must match vocab_size",
        ));
    }
    if checkpoint
        .model
        .token_embedding
        .iter()
        .any(|row| row.len() != embedding_dim)
    {
        return Err(TrainError::InvalidConfig(
            "checkpoint token_embedding row width must match embedding_dim",
        ));
    }
    if checkpoint.model.lm_head_weight.len() != embedding_dim {
        return Err(TrainError::InvalidConfig(
            "checkpoint lm_head_weight row count must match embedding_dim",
        ));
    }
    if checkpoint
        .model
        .lm_head_weight
        .iter()
        .any(|row| row.len() != vocab_size)
    {
        return Err(TrainError::InvalidConfig(
            "checkpoint lm_head_weight column count must match vocab_size",
        ));
    }
    if checkpoint.model.lm_head_bias.len() != vocab_size {
        return Err(TrainError::InvalidConfig(
            "checkpoint lm_head_bias length must match vocab_size",
        ));
    }

    if let Some(train_config) = &checkpoint.train_config {
        if train_config.vocab_size != vocab_size {
            return Err(TrainError::InvalidConfig(
                "checkpoint train_config vocab_size must match model_config.vocab_size",
            ));
        }
    }

    Ok(())
}

pub fn resize_transformer_checkpoint_vocab(
    checkpoint: &mut TransformerTrainingCheckpoint,
    new_vocab_size: usize,
    donor_token_id: Option<usize>,
) -> Result<(), TrainError> {
    validate_transformer_checkpoint_vocab_layout(checkpoint)?;

    let old_vocab_size = checkpoint.model.model_config.vocab_size;
    if new_vocab_size == old_vocab_size {
        return Ok(());
    }
    if new_vocab_size == 0 {
        return Err(TrainError::InvalidConfig(
            "checkpoint vocab_size must be greater than zero",
        ));
    }

    if new_vocab_size < old_vocab_size {
        checkpoint.model.token_embedding.truncate(new_vocab_size);
        checkpoint.model.lm_head_bias.truncate(new_vocab_size);
        for row in &mut checkpoint.model.lm_head_weight {
            row.truncate(new_vocab_size);
        }
    } else {
        let donor_token_id = donor_token_id.ok_or(TrainError::InvalidConfig(
            "growing checkpoint vocab_size requires donor token id",
        ))?;
        if donor_token_id >= old_vocab_size {
            return Err(TrainError::InvalidConfig(
                "donor token id must be within checkpoint vocab_size",
            ));
        }

        let donor_embedding = checkpoint.model.token_embedding[donor_token_id].clone();
        let donor_bias = checkpoint.model.lm_head_bias[donor_token_id];
        let donor_lm_head_column: Vec<f32> = checkpoint
            .model
            .lm_head_weight
            .iter()
            .map(|row| row[donor_token_id])
            .collect();
        let additional_tokens = new_vocab_size - old_vocab_size;

        checkpoint.model.token_embedding.reserve(additional_tokens);
        checkpoint.model.lm_head_bias.reserve(additional_tokens);
        for _ in 0..additional_tokens {
            checkpoint
                .model
                .token_embedding
                .push(donor_embedding.clone());
            checkpoint.model.lm_head_bias.push(donor_bias);
        }
        for (row, donor_value) in checkpoint
            .model
            .lm_head_weight
            .iter_mut()
            .zip(donor_lm_head_column.into_iter())
        {
            for _ in 0..additional_tokens {
                row.push(donor_value);
            }
        }
    }

    checkpoint.model.model_config.vocab_size = new_vocab_size;
    if let Some(train_config) = checkpoint.train_config.as_mut() {
        train_config.vocab_size = new_vocab_size;
    }

    validate_transformer_checkpoint_vocab_layout(checkpoint)?;
    Ok(())
}

pub fn save_train_summary_json<P: AsRef<Path>>(
    path: P,
    summary: &TrainSummary,
) -> Result<(), TrainError> {
    let payload = serde_json::to_string_pretty(summary)
        .map_err(|e| TrainError::Serialization(e.to_string()))?;
    fs::write(path, payload).map_err(|e| TrainError::Io(e.to_string()))
}

pub fn train(
    model: &mut TinySeqModel,
    dataset: &[Vec<u32>],
    cfg: &TrainConfig,
) -> Result<Vec<BatchLog>, TrainError> {
    Ok(train_with_validation(model, dataset, None, cfg)?.train_logs)
}

pub fn train_model(
    model: &mut SequenceModel,
    dataset: &[Vec<u32>],
    cfg: &TrainConfig,
) -> Result<Vec<BatchLog>, TrainError> {
    Ok(train_model_with_validation(model, dataset, None, cfg)?.train_logs)
}

pub fn train_model_from_corpus_tokens(
    model: &mut SequenceModel,
    token_ids: &[u32],
    pack_cfg: &DistributedPackingConfig,
    cfg: &TrainConfig,
) -> Result<TrainSummary, TrainError> {
    let dataset = build_packed_dataset_from_corpus_tokens(token_ids, pack_cfg)?;
    train_model_with_validation(model, &dataset, None, cfg)
}

pub fn build_sft_dataset(
    examples: &[SftExample],
    format_cfg: &SftFormatConfig,
) -> Result<Vec<Vec<u32>>, TrainError> {
    if format_cfg.max_seq_len < 2 {
        return Err(TrainError::InvalidConfig("sft max_seq_len must be >= 2"));
    }
    if examples.is_empty() {
        return Err(TrainError::EmptyDataset);
    }

    let mut dataset = Vec::with_capacity(examples.len());
    for (index, example) in examples.iter().enumerate() {
        if example.prompt_tokens.is_empty() || example.response_tokens.is_empty() {
            return Err(TrainError::InvalidSequenceLength { index, len: 0 });
        }

        let mut seq = Vec::new();
        if let Some(bos) = format_cfg.bos_token_id {
            seq.push(bos);
        }
        seq.extend_from_slice(&example.prompt_tokens);
        seq.extend_from_slice(&format_cfg.assistant_prefix_tokens);
        seq.extend_from_slice(&example.response_tokens);
        if let Some(eos) = format_cfg.eos_token_id {
            seq.push(eos);
        }

        if seq.len() > format_cfg.max_seq_len {
            seq.truncate(format_cfg.max_seq_len);
        }
        if seq.len() < 2 {
            continue;
        }
        dataset.push(seq);
    }

    if dataset.is_empty() {
        return Err(TrainError::EmptyDataset);
    }
    Ok(dataset)
}

pub fn train_sft(
    model: &mut SequenceModel,
    examples: &[SftExample],
    format_cfg: &SftFormatConfig,
    train_cfg: &TrainConfig,
) -> Result<TrainSummary, TrainError> {
    let dataset = build_sft_dataset(examples, format_cfg)?;
    train_model_with_validation(model, &dataset, None, train_cfg)
}

pub fn evaluate_alignment_harness(
    model: &SequenceModel,
    quality_eval_dataset: &[Vec<u32>],
    safety_cases: &[SafetyEvalCase],
    refusal_token_id: Option<u32>,
) -> Result<AlignmentEvalReport, TrainError> {
    let quality = evaluate_model_dataset_metrics(model, quality_eval_dataset, false)?;

    let mut disallowed_hits = 0usize;
    let mut refusal_hits = 0usize;

    for case in safety_cases {
        if case.prompt_tokens.is_empty() {
            continue;
        }
        let logits = model.forward(&case.prompt_tokens, false)?;
        let last = logits
            .last()
            .ok_or(TrainError::InvalidConfig("empty logits in safety eval"))?;
        let pred = argmax_index(last) as u32;

        if case.disallowed_token_ids.contains(&pred) {
            disallowed_hits += 1;
        }
        if refusal_token_id == Some(pred) {
            refusal_hits += 1;
        }
    }

    let total_cases = safety_cases.len();
    let disallowed_hit_rate = if total_cases == 0 {
        0.0
    } else {
        disallowed_hits as f32 / total_cases as f32
    };
    let refusal_rate = if total_cases == 0 {
        0.0
    } else {
        refusal_hits as f32 / total_cases as f32
    };

    Ok(AlignmentEvalReport {
        quality,
        safety: SafetyEvalSummary {
            total_cases,
            disallowed_hits,
            refusal_hits,
            disallowed_hit_rate,
            refusal_rate,
        },
    })
}

pub fn save_alignment_eval_report_json<P: AsRef<Path>>(
    path: P,
    report: &AlignmentEvalReport,
) -> Result<(), TrainError> {
    let payload = serde_json::to_string_pretty(report)
        .map_err(|e| TrainError::Serialization(e.to_string()))?;
    fs::write(path, payload).map_err(|e| TrainError::Io(e.to_string()))
}

#[cfg(feature = "tch-backend")]
pub fn train_transformer_with_tch_backend(
    transformer_cfg: TchTransformerConfig,
    train_dataset: &[Vec<u32>],
    val_dataset: Option<&[Vec<u32>]>,
    train_cfg: &TchTrainConfig,
) -> Result<super::train_tch::TchTrainSummary, TchTrainError> {
    let mut model = TchTransformerLM::new(transformer_cfg)?;
    train_transformer_tch(&mut model, train_dataset, val_dataset, train_cfg)
}

#[cfg(feature = "tch-backend")]
pub fn build_default_tch_transformer_config(
    vocab_size: i64,
    max_seq_len: i64,
    model_dim: i64,
    ff_dim: i64,
    num_heads: i64,
    num_layers: usize,
    seed: i64,
) -> TchTransformerConfig {
    TchTransformerConfig {
        vocab_size,
        max_seq_len,
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        dropout: 0.1,
        pad_token_id: 0,
        device: default_tch_device(),
        seed,
        precision: TchPrecision::Bf16,
    }
}

pub fn train_model_with_validation(
    model: &mut SequenceModel,
    dataset: &[Vec<u32>],
    validation_dataset: Option<&[Vec<u32>]>,
    cfg: &TrainConfig,
) -> Result<TrainSummary, TrainError> {
    match model {
        SequenceModel::Tiny(tiny) => train_with_validation(tiny, dataset, validation_dataset, cfg),
        SequenceModel::Transformer(transformer) => {
            cfg.validate()?;
            validate_dataset(dataset, transformer.vocab_size())?;
            if let Some(val) = validation_dataset {
                validate_dataset(val, transformer.vocab_size())?;
            }
            train_transformer_with_validation(transformer, dataset, validation_dataset, cfg)
        }
    }
}

fn train_transformer_with_validation(
    model: &mut TransformerSeqModel,
    dataset: &[Vec<u32>],
    validation_dataset: Option<&[Vec<u32>]>,
    cfg: &TrainConfig,
) -> Result<TrainSummary, TrainError> {
    let mut logs = Vec::new();
    let mut val_losses = Vec::new();
    let mut val_metrics = Vec::new();
    let mut global_step = cfg.initial_global_step;
    let mut epoch_rng = StdRng::seed_from_u64(cfg.seed ^ 0xD1CE_BAAD);
    let mut order: Vec<usize> = (0..dataset.len()).collect();
    let mut checkpoint_paths: VecDeque<PathBuf> = VecDeque::new();
    let checkpoint_dir = cfg
        .checkpoint_dir
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    if cfg.checkpoint_interval > 0 {
        if let Err(err) = fs::create_dir_all(&checkpoint_dir) {
            eprintln!(
                "warning: failed to create checkpoint directory {}: {}",
                checkpoint_dir.display(),
                err
            );
        }
    }

    for epoch in 0..cfg.epochs {
        shuffle_in_place(&mut order, &mut epoch_rng);

        let mut start = 0usize;
        let mut batch_index = 0usize;

        while start < order.len() {
            let end = (start + cfg.batch_size).min(order.len());
            let batch_indices = &order[start..end];

            let lr = cfg.adamw.lr * cfg.schedule.lr_scale(global_step);
            let mut batch_loss = 0.0_f32;
            let mut batch_grad_norm = 0.0_f32;

            for &idx in batch_indices {
                let seq = &dataset[idx];
                let input = &seq[..seq.len() - 1];
                let targets = &seq[1..];
                let (loss, grad_norm) = model.train_on_sequence(input, targets, lr)?;
                batch_loss += loss;
                batch_grad_norm += grad_norm;
            }
            batch_loss /= batch_indices.len() as f32;
            batch_grad_norm /= batch_indices.len() as f32;

            println!(
                "epoch={} batch={} step={} lr={:.6} grad_norm={:.4} loss={:.6}",
                epoch + 1,
                batch_index,
                global_step,
                lr,
                batch_grad_norm,
                batch_loss
            );

            logs.push(BatchLog {
                epoch: epoch + 1,
                batch_index,
                global_step,
                lr,
                grad_norm: batch_grad_norm,
                loss: batch_loss,
            });

            if cfg.checkpoint_interval > 0 && batch_index % cfg.checkpoint_interval == 0 {
                let checkpoint_path = checkpoint_dir.join(format!("checkpoint_{}.pt", batch_index));
                match model.to_checkpoint() {
                    Ok(model_checkpoint) => {
                        let checkpoint = TransformerTrainingCheckpoint {
                            checkpoint_version: 1,
                            model: model_checkpoint,
                            train_config: Some(cfg.clone()),
                            global_step,
                        };

                        if let Err(err) = save_transformer_checkpoint(&checkpoint_path, &checkpoint)
                        {
                            eprintln!(
                                "warning: failed to save checkpoint at batch {} to {}: {}",
                                batch_index,
                                checkpoint_path.display(),
                                err
                            );
                        } else {
                            checkpoint_paths.push_back(checkpoint_path);
                            while checkpoint_paths.len() > cfg.max_checkpoints {
                                if let Some(old_path) = checkpoint_paths.pop_front() {
                                    if let Err(err) = fs::remove_file(&old_path) {
                                        eprintln!(
                                            "warning: failed to remove old checkpoint {}: {}",
                                            old_path.display(),
                                            err
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!(
                            "warning: failed to prepare checkpoint at batch {}: {}",
                            batch_index, err
                        );
                    }
                }
            }

            global_step += 1;
            batch_index += 1;
            start = end;
        }

        if let Some(val) = validation_dataset {
            let metrics = evaluate_transformer_dataset_metrics(model, val)?;
            let val_loss = metrics.loss;
            println!(
                "epoch={} validation_loss={:.6} val_acc={:.4} val_ppl={:.4}",
                epoch + 1,
                val_loss,
                metrics.accuracy,
                metrics.perplexity,
            );
            val_losses.push(val_loss);
            val_metrics.push(metrics);
        }
    }

    Ok(TrainSummary {
        train_logs: logs,
        val_epoch_losses: val_losses,
        val_epoch_metrics: val_metrics,
    })
}

fn evaluate_transformer_dataset_metrics(
    model: &TransformerSeqModel,
    dataset: &[Vec<u32>],
) -> Result<EvalMetrics, TrainError> {
    let mut batch_logits = Vec::with_capacity(dataset.len());
    let mut batch_targets = Vec::with_capacity(dataset.len());

    let mut correct = 0usize;
    let mut total = 0usize;

    for seq in dataset {
        let input_tokens = &seq[..seq.len() - 1];
        let targets = seq[1..].to_vec();
        let logits = model.forward(input_tokens, false)?;

        for (row, &target) in logits.iter().zip(targets.iter()) {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &v) in row.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
            }
            if best_idx == target as usize {
                correct += 1;
            }
            total += 1;
        }

        batch_logits.push(logits);
        batch_targets.push(targets);
    }

    let loss = try_next_token_cross_entropy_batch(&batch_logits, &batch_targets)
        .map_err(TrainError::Loss)?;
    let accuracy = if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32
    };
    let perplexity = loss.exp();

    Ok(EvalMetrics {
        loss,
        accuracy,
        perplexity,
    })
}

pub fn train_with_validation(
    model: &mut TinySeqModel,
    dataset: &[Vec<u32>],
    validation_dataset: Option<&[Vec<u32>]>,
    cfg: &TrainConfig,
) -> Result<TrainSummary, TrainError> {
    cfg.validate()?;
    validate_dataset(dataset, model.vocab_size)?;

    if let Some(val) = validation_dataset {
        validate_dataset(val, model.vocab_size)?;
    }

    let mut optimizer = AdamWState::new(model);
    let mut logs = Vec::new();
    let mut val_losses = Vec::new();
    let mut val_metrics = Vec::new();
    let mut global_step = 0usize;
    let mut epoch_rng = StdRng::seed_from_u64(cfg.seed ^ 0xD1CE_BAAD);
    let mut order: Vec<usize> = (0..dataset.len()).collect();

    for epoch in 0..cfg.epochs {
        shuffle_in_place(&mut order, &mut epoch_rng);

        let mut start = 0usize;
        let mut batch_index = 0usize;

        while start < order.len() {
            let end = (start + cfg.batch_size).min(order.len());
            let batch: Vec<Vec<u32>> = order[start..end]
                .iter()
                .map(|&idx| dataset[idx].clone())
                .collect();

            let (loss, mut grads) = compute_batch_loss_and_grads(model, &batch)?;

            let grad_norm = grads.global_norm();
            if grad_norm.is_finite() && grad_norm > cfg.max_grad_norm {
                let clip_scale = cfg.max_grad_norm / (grad_norm + 1e-12);
                grads.scale_in_place(clip_scale);
            }

            let lr = cfg.adamw.lr * cfg.schedule.lr_scale(global_step);
            optimizer.step(model, &grads, &cfg.adamw, lr);

            println!(
                "epoch={} batch={} step={} lr={:.6} grad_norm={:.4} loss={:.6}",
                epoch + 1,
                batch_index,
                global_step,
                lr,
                grad_norm,
                loss
            );

            logs.push(BatchLog {
                epoch: epoch + 1,
                batch_index,
                global_step,
                lr,
                grad_norm,
                loss,
            });

            global_step += 1;
            batch_index += 1;
            start = end;
        }

        if let Some(val) = validation_dataset {
            let metrics = evaluate_dataset_metrics(model, val)?;
            let val_loss = metrics.loss;
            println!(
                "epoch={} validation_loss={:.6} val_acc={:.4} val_ppl={:.4}",
                epoch + 1,
                val_loss,
                metrics.accuracy,
                metrics.perplexity,
            );
            val_losses.push(val_loss);
            val_metrics.push(metrics);
        }
    }

    Ok(TrainSummary {
        train_logs: logs,
        val_epoch_losses: val_losses,
        val_epoch_metrics: val_metrics,
    })
}

pub fn evaluate_dataset_loss(
    model: &TinySeqModel,
    dataset: &[Vec<u32>],
) -> Result<f32, TrainError> {
    Ok(evaluate_dataset_metrics(model, dataset)?.loss)
}

pub fn evaluate_model_dataset_loss(
    model: &SequenceModel,
    dataset: &[Vec<u32>],
) -> Result<f32, TrainError> {
    Ok(evaluate_model_dataset_metrics(model, dataset, false)?.loss)
}

pub fn evaluate_dataset_metrics(
    model: &TinySeqModel,
    dataset: &[Vec<u32>],
) -> Result<EvalMetrics, TrainError> {
    let model = SequenceModel::Tiny(model.clone());
    evaluate_model_dataset_metrics(&model, dataset, false)
}

pub fn evaluate_model_dataset_metrics(
    model: &SequenceModel,
    dataset: &[Vec<u32>],
    training: bool,
) -> Result<EvalMetrics, TrainError> {
    validate_dataset(dataset, model.vocab_size())?;

    let mut batch_logits = Vec::with_capacity(dataset.len());
    let mut batch_targets = Vec::with_capacity(dataset.len());

    let mut correct = 0usize;
    let mut total = 0usize;

    for seq in dataset {
        let input_tokens = &seq[..seq.len() - 1];
        let targets = seq[1..].to_vec();
        let logits = model.forward(input_tokens, training)?;

        for (row, &target) in logits.iter().zip(targets.iter()) {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &v) in row.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
            }
            if best_idx == target as usize {
                correct += 1;
            }
            total += 1;
        }

        batch_logits.push(logits);
        batch_targets.push(targets);
    }

    let loss = try_next_token_cross_entropy_batch(&batch_logits, &batch_targets)
        .map_err(TrainError::Loss)?;
    let accuracy = if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32
    };
    let perplexity = loss.exp();

    Ok(EvalMetrics {
        loss,
        accuracy,
        perplexity,
    })
}

fn compute_batch_loss_and_grads(
    model: &TinySeqModel,
    batch: &[Vec<u32>],
) -> Result<(f32, TinySeqGrads), TrainError> {
    let mut batch_logits = Vec::with_capacity(batch.len());
    let mut batch_targets = Vec::with_capacity(batch.len());

    let mut cache_inputs: Vec<Vec<u32>> = Vec::with_capacity(batch.len());
    let mut cache_hidden: Vec<Vec<Vec<f32>>> = Vec::with_capacity(batch.len());
    let mut cache_logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(batch.len());

    let mut total_tokens = 0usize;

    for (index, seq) in batch.iter().enumerate() {
        debug_assert!(seq.len() >= 2, "invalid sequence length at index {}", index);

        let input_tokens = &seq[..seq.len() - 1];
        let targets = seq[1..].to_vec();

        let mut seq_hidden = Vec::with_capacity(input_tokens.len());
        let mut seq_logits = Vec::with_capacity(input_tokens.len());

        for &token in input_tokens {
            let token_idx = token as usize;
            debug_assert!(token_idx < model.vocab_size, "token out of range");

            let x = &model.embedding[token_idx];

            let mut h = vec![0.0_f32; model.hidden_dim];
            for j in 0..model.hidden_dim {
                let mut sum = model.b1[j];
                for (i, &x_i) in x.iter().enumerate() {
                    sum += x_i * model.w1[i][j];
                }
                h[j] = sum;
            }

            let mut logits = vec![0.0_f32; model.vocab_size];
            for (k, logit) in logits.iter_mut().enumerate() {
                let mut sum = model.b2[k];
                for (j, &h_j) in h.iter().enumerate() {
                    sum += h_j * model.w2[j][k];
                }
                *logit = sum;
            }

            seq_hidden.push(h);
            seq_logits.push(logits);
        }

        total_tokens += input_tokens.len();

        cache_inputs.push(input_tokens.to_vec());
        cache_hidden.push(seq_hidden.clone());
        cache_logits.push(seq_logits.clone());

        batch_logits.push(seq_logits);
        batch_targets.push(targets);
    }

    let loss = try_next_token_cross_entropy_batch(&batch_logits, &batch_targets)
        .map_err(TrainError::Loss)?;

    let mut grads = TinySeqGrads::zeros_like(model);
    let inv_tokens = 1.0_f32 / total_tokens as f32;

    for seq_idx in 0..batch.len() {
        let input_tokens = &cache_inputs[seq_idx];
        let seq_hidden = &cache_hidden[seq_idx];
        let seq_logits = &cache_logits[seq_idx];
        let seq_targets = &batch_targets[seq_idx];

        for t in 0..input_tokens.len() {
            let token = input_tokens[t] as usize;
            let target = seq_targets[t] as usize;
            let h = &seq_hidden[t];
            let logits = &seq_logits[t];

            let mut max_logit = f32::NEG_INFINITY;
            for &v in logits {
                if v > max_logit {
                    max_logit = v;
                }
            }

            let probs = softmax_grad_from_logits(logits, max_logit, target, inv_tokens);
            accumulate_output_layer_grads(h, &probs, &mut grads);

            let dh = backprop_to_hidden(&model.w2, &probs);

            for (j, &g) in dh.iter().enumerate() {
                grads.b1[j] += g;
            }

            let x = &model.embedding[token];
            accumulate_input_layer_grads(x, &dh, &mut grads.w1);

            let dx = backprop_to_hidden(&model.w1, &dh);
            for (i, &g) in dx.iter().enumerate() {
                grads.embedding[token][i] += g;
            }
        }
    }

    Ok((loss, grads))
}

fn adamw_update_1d(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    cfg: &AdamWConfig,
    lr: f32,
    bias_c1: f32,
    bias_c2: f32,
) {
    for i in 0..param.len() {
        m[i] = cfg.beta1 * m[i] + (1.0 - cfg.beta1) * grad[i];
        v[i] = cfg.beta2 * v[i] + (1.0 - cfg.beta2) * grad[i] * grad[i];

        let m_hat = m[i] / bias_c1;
        let v_hat = v[i] / bias_c2;

        let update = m_hat / (v_hat.sqrt() + cfg.eps);
        param[i] -= lr * (update + cfg.weight_decay * param[i]);
    }
}

fn adamw_update_2d(
    param: &mut [Vec<f32>],
    grad: &[Vec<f32>],
    m: &mut [Vec<f32>],
    v: &mut [Vec<f32>],
    cfg: &AdamWConfig,
    lr: f32,
    bias_c1: f32,
    bias_c2: f32,
) {
    for r in 0..param.len() {
        adamw_update_1d(
            &mut param[r],
            &grad[r],
            &mut m[r],
            &mut v[r],
            cfg,
            lr,
            bias_c1,
            bias_c2,
        );
    }
}

fn acc_sum_squares_1d(v: &[f32], sum: &mut f32) {
    for &x in v {
        *sum += x * x;
    }
}

fn acc_sum_squares_2d(v: &[Vec<f32>], sum: &mut f32) {
    for row in v {
        acc_sum_squares_1d(row, sum);
    }
}

fn scale_1d(v: &mut [f32], scale: f32) {
    for x in v {
        *x *= scale;
    }
}

fn scale_2d(v: &mut [Vec<f32>], scale: f32) {
    for row in v {
        scale_1d(row, scale);
    }
}

fn flatten_2d(values: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::new();
    for row in values {
        out.extend_from_slice(row);
    }
    out
}

fn tensor_to_2d(
    backend: &CpuAutogradBackend,
    tensor: &<CpuAutogradBackend as TensorBackend>::Tensor,
) -> Result<Vec<Vec<f32>>, TrainError> {
    let shape = backend.shape(tensor);
    if shape.len() != 2 {
        return Err(TrainError::InvalidConfig(
            "expected rank-2 tensor for checkpoint",
        ));
    }
    let data = backend.data(tensor);
    Ok(reshape_2d(data, shape[0], shape[1]))
}

fn tensor_to_1d(
    backend: &CpuAutogradBackend,
    tensor: &<CpuAutogradBackend as TensorBackend>::Tensor,
) -> Result<Vec<f32>, TrainError> {
    let shape = backend.shape(tensor);
    if shape.len() != 2 {
        return Err(TrainError::InvalidConfig(
            "expected rank-2 tensor for vector checkpoint",
        ));
    }
    if shape[0] != 1 {
        return Err(TrainError::InvalidConfig(
            "expected shape [1, N] tensor for vector checkpoint",
        ));
    }
    Ok(backend.data(tensor))
}

fn set_vector_param(
    backend: &CpuAutogradBackend,
    tensor: &mut <CpuAutogradBackend as TensorBackend>::Tensor,
    values: &[f32],
) -> Result<(), TrainError> {
    *tensor = backend
        .from_data(values.to_vec(), vec![1, values.len()], true)
        .map_err(TrainError::FrameworkBackend)?;
    Ok(())
}

fn set_linear_from_state(
    backend: &CpuAutogradBackend,
    linear: &mut FrameworkLinear<CpuAutogradBackend>,
    weight: &[Vec<f32>],
    bias: &[f32],
) -> Result<(), TrainError> {
    if weight.is_empty() || weight[0].is_empty() {
        return Err(TrainError::InvalidConfig(
            "linear weight matrix cannot be empty",
        ));
    }
    let in_features = weight.len();
    let out_features = weight[0].len();
    if weight.iter().any(|row| row.len() != out_features) {
        return Err(TrainError::InvalidConfig(
            "linear weight rows must be same length",
        ));
    }
    if bias.len() != out_features {
        return Err(TrainError::InvalidConfig("linear bias length mismatch"));
    }

    linear.weight.tensor = backend
        .from_data(flatten_2d(weight), vec![in_features, out_features], true)
        .map_err(TrainError::FrameworkBackend)?;
    linear.bias.tensor = backend
        .from_data(bias.to_vec(), vec![1, out_features], true)
        .map_err(TrainError::FrameworkBackend)?;
    Ok(())
}

fn argmax_index(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx
}

fn masked_argmax_index(values: &[f32], disallowed_token_ids: &[u32]) -> Option<usize> {
    let mut best_idx = None;
    let mut best_value = f32::NEG_INFINITY;

    for (idx, &value) in values.iter().enumerate() {
        if disallowed_token_ids.contains(&(idx as u32)) {
            continue;
        }
        if value > best_value {
            best_value = value;
            best_idx = Some(idx);
        }
    }

    best_idx
}

fn add_vec(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect()
}

fn relu_vec(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&v| if v > 0.0 { v } else { 0.0 })
        .collect()
}

fn linear_forward(input: &[f32], weight: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let out_features = bias.len();
    let mut out = vec![0.0_f32; out_features];
    for o in 0..out_features {
        let mut acc = bias[o];
        for i in 0..input.len() {
            acc += input[i] * weight[i][o];
        }
        out[o] = acc;
    }
    out
}

fn layer_norm_forward(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len() as f32;
    let mean = input.iter().sum::<f32>() / n;
    let variance = input
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    let denom = (variance + eps).sqrt();

    let mut out = vec![0.0_f32; input.len()];
    for i in 0..input.len() {
        let normalized = (input[i] - mean) / denom;
        out[i] = normalized * gamma[i] + beta[i];
    }
    out
}

fn attention_single_query(
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let model_dim = query.len();
    let seq_len = keys.len();
    let mut out = vec![0.0_f32; model_dim];

    if seq_len == 0 {
        return out;
    }

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Process all heads in parallel if rayon is available
    #[cfg(feature = "parallel")]
    let head_iter = (0..num_heads).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let head_iter = 0..num_heads;

    let head_results: Vec<(usize, Vec<f32>)> = head_iter
        .map(|h| {
            let offset = h * head_dim;
            let qh = &query[offset..offset + head_dim];

            // Compute attention scores using vectorized dot products
            let mut scores = Vec::with_capacity(seq_len);
            for key in keys.iter() {
                let kh = &key[offset..offset + head_dim];
                // Vectorized dot product - compiler will auto-SIMD this
                let dot: f32 = qh.iter().zip(kh.iter()).map(|(a, b)| a * b).sum();
                scores.push(dot * scale);
            }

            // Softmax with numerical stability
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut weights: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                weights.iter_mut().for_each(|w| *w *= inv_sum);
            }

            // Weighted sum of values - vectorized
            let mut head_out = vec![0.0_f32; head_dim];
            for (t, value) in values.iter().enumerate() {
                let vh = &value[offset..offset + head_dim];
                let w = weights[t];
                // Vectorized weighted sum
                head_out
                    .iter_mut()
                    .zip(vh.iter())
                    .for_each(|(o, v)| *o += w * v);
            }

            (offset, head_out)
        })
        .collect();

    // Assemble output from all heads
    for (offset, head_out) in head_results {
        out[offset..offset + head_dim].copy_from_slice(&head_out);
    }

    out
}

fn init_matrix_random(rows: usize, cols: usize, scale: f32, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0_f32; cols]; rows];
    for row in &mut out {
        for v in row {
            *v = rng.random_range(-scale..scale);
        }
    }
    out
}

fn reshape_2d(flat: Vec<f32>, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0_f32; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r][c] = flat[r * cols + c];
        }
    }
    out
}

fn backprop_to_hidden(weights: &[Vec<f32>], grad_out: &[f32]) -> Vec<f32> {
    let mut grad_in = vec![0.0_f32; weights.len()];
    for (i, grad_i) in grad_in.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        for (j, &g_j) in grad_out.iter().enumerate() {
            sum += weights[i][j] * g_j;
        }
        *grad_i = sum;
    }
    grad_in
}

fn shuffle_in_place(values: &mut [usize], rng: &mut StdRng) {
    if values.len() <= 1 {
        return;
    }
    for i in (1..values.len()).rev() {
        let j = rng.random_range(0..=i);
        values.swap(i, j);
    }
}

fn validate_dataset(dataset: &[Vec<u32>], vocab_size: usize) -> Result<(), TrainError> {
    if dataset.is_empty() {
        return Err(TrainError::EmptyDataset);
    }
    for (index, seq) in dataset.iter().enumerate() {
        if seq.len() < 2 {
            return Err(TrainError::InvalidSequenceLength {
                index,
                len: seq.len(),
            });
        }
        for &token in seq {
            if token as usize >= vocab_size {
                return Err(TrainError::TokenOutOfRange { token, vocab_size });
            }
        }
    }
    Ok(())
}

fn softmax_grad_from_logits(logits: &[f32], max_logit: f32, target: usize, scale: f32) -> Vec<f32> {
    let mut probs = vec![0.0_f32; logits.len()];
    let mut sum_exp = 0.0_f32;
    for (k, &v) in logits.iter().enumerate() {
        let e = (v - max_logit).exp();
        probs[k] = e;
        sum_exp += e;
    }
    for p in &mut probs {
        *p /= sum_exp;
    }
    probs[target] -= 1.0;
    for p in &mut probs {
        *p *= scale;
    }
    probs
}

fn accumulate_output_layer_grads(hidden: &[f32], grad_logits: &[f32], grads: &mut TinySeqGrads) {
    for (k, &g) in grad_logits.iter().enumerate() {
        grads.b2[k] += g;
    }
    for (j, &h_j) in hidden.iter().enumerate() {
        for (k, &g) in grad_logits.iter().enumerate() {
            grads.w2[j][k] += h_j * g;
        }
    }
}

fn accumulate_input_layer_grads(input: &[f32], grad_hidden: &[f32], w_grads: &mut [Vec<f32>]) {
    for (i, &x_i) in input.iter().enumerate() {
        for (j, &dh_j) in grad_hidden.iter().enumerate() {
            w_grads[i][j] += x_i * dh_j;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        benchmark_transformer_decode_latency, build_packed_dataset_from_corpus_tokens,
        build_sft_dataset, evaluate_alignment_harness, load_tiny_checkpoint,
        load_transformer_checkpoint, resize_transformer_checkpoint_vocab,
        save_alignment_eval_report_json, save_tiny_checkpoint, save_train_summary_json,
        save_transformer_checkpoint, train_model_from_corpus_tokens, train_model_with_validation,
        train_sft, AdamWConfig, BatchLog, DistributedPackingConfig, EvalMetrics, LrSchedule,
        SafetyEvalCase, SequenceModel, SftExample, SftFormatConfig, TinySeqCheckpoint,
        TinySeqModel, TrainConfig, TrainError, TrainSummary, TransformerModelConfig,
        TransformerSeqModel, TransformerTrainingCheckpoint,
    };
    use std::collections::HashSet;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_path(prefix: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("{}_{}.json", prefix, nanos));
        path
    }

    #[test]
    fn tiny_checkpoint_roundtrip_preserves_model_shape() {
        let model = TinySeqModel::new(16, 8, 12, 123).unwrap();
        let ckpt = TinySeqCheckpoint {
            model,
            train_config: None,
            global_step: 77,
        };

        let path = unique_path("tiny_ckpt");
        save_tiny_checkpoint(&path, &ckpt).unwrap();
        let loaded = load_tiny_checkpoint(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(loaded.global_step, 77);
        assert_eq!(loaded.model.vocab_size, 16);
        assert_eq!(loaded.model.embedding_dim, 8);
        assert_eq!(loaded.model.hidden_dim, 12);
        assert_eq!(loaded.model.embedding.len(), 16);
        assert_eq!(loaded.model.embedding[0].len(), 8);
    }

    #[test]
    fn train_summary_json_persists_logs() {
        let summary = TrainSummary {
            train_logs: vec![BatchLog {
                epoch: 1,
                batch_index: 0,
                global_step: 0,
                lr: 1e-3,
                grad_norm: 0.5,
                loss: 2.1,
            }],
            val_epoch_losses: vec![2.0],
            val_epoch_metrics: vec![EvalMetrics {
                loss: 2.0,
                accuracy: 0.25,
                perplexity: 7.39,
            }],
        };

        let path = unique_path("train_summary");
        save_train_summary_json(&path, &summary).unwrap();
        let content = fs::read_to_string(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert!(content.contains("\"train_logs\""));
        assert!(content.contains("\"val_epoch_metrics\""));
    }

    #[test]
    fn transformer_sequence_model_trains_through_unified_api() {
        let mut model = SequenceModel::Transformer(
            TransformerSeqModel::new(TransformerModelConfig {
                vocab_size: 16,
                max_seq_len: 8,
                embedding_dim: 8,
                hidden_dim: 16,
                num_heads: 2,
                num_layers: 2,
                seed: 123,
            })
            .unwrap(),
        );

        let dataset = vec![
            vec![1, 2, 3, 4],
            vec![2, 3, 4, 5],
            vec![3, 4, 5, 6],
            vec![4, 5, 6, 7],
        ];

        let cfg = TrainConfig {
            vocab_size: 16,
            embedding_dim: 8,
            hidden_dim: 16,
            epochs: 1,
            batch_size: 2,
            max_grad_norm: 1.0,
            adamw: AdamWConfig {
                lr: 1e-2,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
            schedule: LrSchedule {
                warmup_steps: 0,
                step_decay_every: 100,
                step_decay_gamma: 1.0,
                min_lr_scale: 1.0,
            },
            seed: 7,
            initial_global_step: 0,
            checkpoint_interval: 0,
            max_checkpoints: 5,
            checkpoint_dir: None,
        };

        let summary = train_model_with_validation(&mut model, &dataset, None, &cfg).unwrap();
        assert!(!summary.train_logs.is_empty());
    }

    #[test]
    fn transformer_checkpoint_roundtrip_restores_outputs() {
        let model = TransformerSeqModel::new(TransformerModelConfig {
            vocab_size: 16,
            max_seq_len: 8,
            embedding_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 2,
            seed: 321,
        })
        .unwrap();

        let input = vec![1, 2, 3, 4];
        let before = model.forward(&input, false).unwrap();

        let checkpoint = TransformerTrainingCheckpoint {
            checkpoint_version: 1,
            model: model.to_checkpoint().unwrap(),
            train_config: None,
            global_step: 5,
        };

        let path = unique_path("transformer_ckpt");
        save_transformer_checkpoint(&path, &checkpoint).unwrap();
        let loaded = load_transformer_checkpoint(&path).unwrap();
        let _ = fs::remove_file(&path);

        let restored = TransformerSeqModel::from_checkpoint(&loaded.model).unwrap();
        let after = restored.forward(&input, false).unwrap();

        assert_eq!(loaded.global_step, 5);
        assert_eq!(loaded.checkpoint_version, 1);
        assert_eq!(before.len(), after.len());
        assert_eq!(before[0].len(), after[0].len());
        for t in 0..before.len() {
            for v in 0..before[t].len() {
                assert!((before[t][v] - after[t][v]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn resize_transformer_checkpoint_vocab_grows_with_donor_token() {
        let model = TransformerSeqModel::new(TransformerModelConfig {
            vocab_size: 8,
            max_seq_len: 8,
            embedding_dim: 6,
            hidden_dim: 12,
            num_heads: 2,
            num_layers: 2,
            seed: 99,
        })
        .unwrap();

        let mut checkpoint = TransformerTrainingCheckpoint {
            checkpoint_version: 1,
            model: model.to_checkpoint().unwrap(),
            train_config: Some(TrainConfig {
                vocab_size: 8,
                embedding_dim: 6,
                hidden_dim: 12,
                epochs: 1,
                batch_size: 2,
                max_grad_norm: 1.0,
                adamw: AdamWConfig {
                    lr: 1e-2,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.0,
                },
                schedule: LrSchedule {
                    warmup_steps: 0,
                    step_decay_every: 100,
                    step_decay_gamma: 1.0,
                    min_lr_scale: 1.0,
                },
                seed: 99,
                initial_global_step: 0,
                checkpoint_interval: 0,
                max_checkpoints: 5,
                checkpoint_dir: None,
            }),
            global_step: 0,
        };

        let donor_token_id = 3usize;
        let donor_embedding = checkpoint.model.token_embedding[donor_token_id].clone();
        let donor_bias = checkpoint.model.lm_head_bias[donor_token_id];
        let donor_column: Vec<f32> = checkpoint
            .model
            .lm_head_weight
            .iter()
            .map(|row| row[donor_token_id])
            .collect();

        resize_transformer_checkpoint_vocab(&mut checkpoint, 11, Some(donor_token_id)).unwrap();

        assert_eq!(checkpoint.model.model_config.vocab_size, 11);
        assert_eq!(checkpoint.train_config.as_ref().unwrap().vocab_size, 11);
        assert_eq!(checkpoint.model.token_embedding.len(), 11);
        assert_eq!(checkpoint.model.lm_head_bias.len(), 11);
        for token_id in 8..11 {
            assert_eq!(checkpoint.model.token_embedding[token_id], donor_embedding);
            assert_eq!(checkpoint.model.lm_head_bias[token_id], donor_bias);
        }
        for (row, donor_value) in checkpoint
            .model
            .lm_head_weight
            .iter()
            .zip(donor_column.iter())
        {
            assert_eq!(row.len(), 11);
            for token_id in 8..11 {
                assert_eq!(row[token_id], *donor_value);
            }
        }

        let restored = TransformerSeqModel::from_checkpoint(&checkpoint.model).unwrap();
        assert_eq!(restored.config().vocab_size, 11);
        assert!(restored.forward(&[1, 2, 3], false).is_ok());
    }

    #[test]
    fn resize_transformer_checkpoint_vocab_requires_donor_when_growing() {
        let model = TransformerSeqModel::new(TransformerModelConfig {
            vocab_size: 6,
            max_seq_len: 8,
            embedding_dim: 4,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 1,
            seed: 11,
        })
        .unwrap();

        let mut checkpoint = TransformerTrainingCheckpoint {
            checkpoint_version: 1,
            model: model.to_checkpoint().unwrap(),
            train_config: None,
            global_step: 0,
        };

        let err = resize_transformer_checkpoint_vocab(&mut checkpoint, 7, None).unwrap_err();
        assert_eq!(
            err,
            TrainError::InvalidConfig("growing checkpoint vocab_size requires donor token id")
        );
    }

    #[test]
    fn kv_cache_greedy_matches_full_recompute() {
        let model = TransformerSeqModel::new(TransformerModelConfig {
            vocab_size: 20,
            max_seq_len: 12,
            embedding_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 2,
            seed: 55,
        })
        .unwrap();

        let prompt = vec![1, 2, 3];
        let max_new = 4usize;

        let mut full = prompt.clone();
        for _ in 0..max_new {
            let logits = model.forward(&full, false).unwrap();
            let last = logits.last().unwrap();
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &v) in last.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
            }
            full.push(best_idx as u32);
        }

        let cached = model
            .generate_greedy_with_kv_cache(&prompt, max_new)
            .unwrap();

        assert_eq!(full, cached);
    }

    #[test]
    fn decode_latency_benchmark_reports_metrics() {
        let model = TransformerSeqModel::new(TransformerModelConfig {
            vocab_size: 20,
            max_seq_len: 12,
            embedding_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 2,
            seed: 56,
        })
        .unwrap();

        let result = benchmark_transformer_decode_latency(&model, &[1, 2, 3], 3).unwrap();
        assert_eq!(result.max_new_tokens, 3);
        assert!(result.full_recompute_ms >= 0.0);
        assert!(result.kv_cache_ms >= 0.0);
        assert!(result.speedup >= 0.0);
    }

    #[test]
    fn packed_dataset_respects_shard_partitioning() {
        let token_ids: Vec<u32> = (0u32..30u32).collect();
        let cfg_rank0 = DistributedPackingConfig {
            window_size: 6,
            stride: 3,
            pad_token_id: 0,
            rank: 0,
            world_size: 2,
        };
        let cfg_rank1 = DistributedPackingConfig {
            window_size: 6,
            stride: 3,
            pad_token_id: 0,
            rank: 1,
            world_size: 2,
        };

        let d0 = build_packed_dataset_from_corpus_tokens(&token_ids, &cfg_rank0).unwrap();
        let d1 = build_packed_dataset_from_corpus_tokens(&token_ids, &cfg_rank1).unwrap();

        assert!(!d0.is_empty());
        assert!(!d1.is_empty());
        let s0: HashSet<Vec<u32>> = d0.into_iter().collect();
        let s1: HashSet<Vec<u32>> = d1.into_iter().collect();
        assert!(s0.intersection(&s1).count() == 0);
    }

    #[test]
    fn train_model_from_corpus_tokens_runs() {
        let mut model = SequenceModel::Tiny(TinySeqModel::new(16, 8, 12, 42).unwrap());
        let token_ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        let pack_cfg = DistributedPackingConfig {
            window_size: 5,
            stride: 2,
            pad_token_id: 0,
            rank: 0,
            world_size: 1,
        };

        let cfg = TrainConfig {
            vocab_size: 16,
            embedding_dim: 8,
            hidden_dim: 12,
            epochs: 1,
            batch_size: 2,
            max_grad_norm: 1.0,
            adamw: AdamWConfig {
                lr: 1e-2,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
            schedule: LrSchedule {
                warmup_steps: 0,
                step_decay_every: 100,
                step_decay_gamma: 1.0,
                min_lr_scale: 1.0,
            },
            seed: 1,
            initial_global_step: 0,
            checkpoint_interval: 0,
            max_checkpoints: 5,
            checkpoint_dir: None,
        };

        let summary =
            train_model_from_corpus_tokens(&mut model, &token_ids, &pack_cfg, &cfg).unwrap();
        assert!(!summary.train_logs.is_empty());
    }

    #[test]
    fn sft_dataset_builder_formats_sequences() {
        let examples = vec![SftExample {
            prompt_tokens: vec![10, 11],
            response_tokens: vec![12, 13],
        }];
        let fmt = SftFormatConfig {
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            assistant_prefix_tokens: vec![99],
            max_seq_len: 16,
        };

        let dataset = build_sft_dataset(&examples, &fmt).unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset[0], vec![1, 10, 11, 99, 12, 13, 2]);
    }

    #[test]
    fn train_sft_runs_on_tiny_model() {
        let mut model = SequenceModel::Tiny(TinySeqModel::new(64, 8, 12, 123).unwrap());
        let examples = vec![
            SftExample {
                prompt_tokens: vec![5, 6],
                response_tokens: vec![7, 8],
            },
            SftExample {
                prompt_tokens: vec![6, 7],
                response_tokens: vec![8, 9],
            },
        ];
        let fmt = SftFormatConfig {
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            assistant_prefix_tokens: vec![3],
            max_seq_len: 12,
        };

        let cfg = TrainConfig {
            vocab_size: 64,
            embedding_dim: 8,
            hidden_dim: 12,
            epochs: 1,
            batch_size: 2,
            max_grad_norm: 1.0,
            adamw: AdamWConfig {
                lr: 1e-2,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
            schedule: LrSchedule {
                warmup_steps: 0,
                step_decay_every: 100,
                step_decay_gamma: 1.0,
                min_lr_scale: 1.0,
            },
            seed: 9,
            initial_global_step: 0,
            checkpoint_interval: 0,
            max_checkpoints: 5,
            checkpoint_dir: None,
        };

        let summary = train_sft(&mut model, &examples, &fmt, &cfg).unwrap();
        assert!(!summary.train_logs.is_empty());
    }

    #[test]
    fn alignment_harness_generates_json_report() {
        let model = SequenceModel::Tiny(TinySeqModel::new(32, 8, 12, 17).unwrap());
        let quality_eval_dataset = vec![vec![1, 2, 3], vec![2, 3, 4]];
        let safety_cases = vec![SafetyEvalCase {
            prompt_tokens: vec![1, 2],
            disallowed_token_ids: vec![31],
        }];

        let report =
            evaluate_alignment_harness(&model, &quality_eval_dataset, &safety_cases, Some(0))
                .unwrap();
        let path = unique_path("alignment_report");
        save_alignment_eval_report_json(&path, &report).unwrap();
        let content = fs::read_to_string(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert!(content.contains("\"quality\""));
        assert!(content.contains("\"safety\""));
    }
}
