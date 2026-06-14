use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

/// Compute backend for embedding storage/execution.
///
/// This module currently implements CPU storage in `Vec<Vec<f32>>`.
/// The GPU variant is included to keep the API forward-compatible with
/// future tensor/device backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingBackend {
    Cpu,
    Gpu { device_id: usize },
}

/// Numeric storage type for embedding weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingDType {
    F32,
    F16,
    BF16,
}

/// Behavior when `embed()` sees an out-of-range token ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OovPolicy {
    /// Replace unknown token IDs with an all-zero embedding.
    Zero,
    /// Panic immediately with a descriptive error.
    Panic,
    /// Trigger debug assertion; fallback to zeros in release builds.
    DebugAssert,
}

/// Errors returned by embedding initialization and strict embedding APIs.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingError {
    InvalidVocabSize,
    InvalidEmbeddingDim,
    EmptyPretrainedWeights,
    RaggedPretrainedWeights,
    PadEmbeddingDimMismatch { expected: usize, found: usize },
    TokenIdOutOfRange { token_id: u32, vocab_size: usize },
    InvalidShardConfig { rank: usize, world_size: usize },
    UnsupportedZeroCopyForDType(EmbeddingDType),
    GpuBackendNotAvailable,
}

impl Display for EmbeddingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingError::InvalidVocabSize => write!(f, "vocab_size must be > 0"),
            EmbeddingError::InvalidEmbeddingDim => write!(f, "embedding_dim must be > 0"),
            EmbeddingError::EmptyPretrainedWeights => write!(f, "pretrained weight matrix is empty"),
            EmbeddingError::RaggedPretrainedWeights => {
                write!(f, "pretrained weight rows must all have the same length")
            }
            EmbeddingError::PadEmbeddingDimMismatch { expected, found } => write!(
                f,
                "pad embedding dimension mismatch: expected {}, found {}",
                expected, found
            ),
            EmbeddingError::TokenIdOutOfRange {
                token_id,
                vocab_size,
            } => write!(
                f,
                "token id {} out of range for vocab size {}",
                token_id, vocab_size
            ),
            EmbeddingError::InvalidShardConfig { rank, world_size } => write!(
                f,
                "invalid shard config: rank={} world_size={}",
                rank, world_size
            ),
            EmbeddingError::UnsupportedZeroCopyForDType(dtype) => write!(
                f,
                "zero-copy embedding references are only available for f32 storage; got {:?}",
                dtype
            ),
            EmbeddingError::GpuBackendNotAvailable => {
                write!(f, "GPU backend is not available in this build")
            }
        }
    }
}

impl Error for EmbeddingError {}

/// View of a row-shard of the embedding matrix (for distributed setups).
#[derive(Debug, Clone)]
pub struct EmbeddingShard<'a> {
    pub start_row: usize,
    pub end_row: usize,
    pub embedding_dim: usize,
    pub dtype: EmbeddingDType,
    pub data: EmbeddingShardData<'a>,
}

#[derive(Debug, Clone, Copy)]
pub enum EmbeddingShardData<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
    BF16(&'a [u16]),
}

#[derive(Debug, Clone)]
enum EmbeddingStorage {
    F32(Arc<Vec<f32>>),
    F16(Arc<Vec<u16>>),
    BF16(Arc<Vec<u16>>),
}

impl EmbeddingStorage {
    fn dtype(&self) -> EmbeddingDType {
        match self {
            EmbeddingStorage::F32(_) => EmbeddingDType::F32,
            EmbeddingStorage::F16(_) => EmbeddingDType::F16,
            EmbeddingStorage::BF16(_) => EmbeddingDType::BF16,
        }
    }
}

/// Reusable contiguous output buffer for embedding batches.
#[derive(Debug, Clone, Default)]
pub struct BatchEmbeddingBuffer {
    pub data: Vec<f32>,
}

impl BatchEmbeddingBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Token embedding table used by the model input pipeline.
///
/// - Stores an embedding matrix in row-major logical form (`weights[token_id]`).
/// - Supports random initialization and pretrained loading.
/// - Supports optional padding-token behavior.
/// - Exposes zero-copy reference APIs where possible.
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    storage: EmbeddingStorage,
    vocab_size: usize,
    embedding_dim: usize,
    backend: EmbeddingBackend,
    pad_token_id: Option<u32>,
    pad_embedding: Option<Vec<f32>>,
    oov_policy: OovPolicy,
}

impl TokenEmbedding {
    /// Initialize random embeddings with small uniform values.
    ///
    /// Values are sampled from approximately `[-0.02, 0.02]` using a lightweight
    /// deterministic PRNG so results are reproducible for a given `seed`.
    pub fn random(
        vocab_size: usize,
        embedding_dim: usize,
        seed: u64,
    ) -> Result<Self, EmbeddingError> {
        if vocab_size == 0 {
            return Err(EmbeddingError::InvalidVocabSize);
        }
        if embedding_dim == 0 {
            return Err(EmbeddingError::InvalidEmbeddingDim);
        }

        let mut state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state as f64) / (u64::MAX as f64);
            (unit as f32) * 0.04 - 0.02
        };

        let mut weights = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            for _ in 0..embedding_dim {
                weights.push(next_f32());
            }
        }

        Ok(Self {
            storage: EmbeddingStorage::F32(Arc::new(weights)),
            vocab_size,
            embedding_dim,
            backend: EmbeddingBackend::Cpu,
            pad_token_id: None,
            pad_embedding: None,
            oov_policy: OovPolicy::Zero,
        })
    }

    /// Initialize random embeddings using a caller-provided scalar generator.
    ///
    /// This enables plugging in external RNGs (e.g., `rand::rngs::StdRng`) without
    /// forcing a dependency in this module.
    pub fn random_with_generator<F>(
        vocab_size: usize,
        embedding_dim: usize,
        mut next_value: F,
    ) -> Result<Self, EmbeddingError>
    where
        F: FnMut() -> f32,
    {
        if vocab_size == 0 {
            return Err(EmbeddingError::InvalidVocabSize);
        }
        if embedding_dim == 0 {
            return Err(EmbeddingError::InvalidEmbeddingDim);
        }

        let mut weights = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            for _ in 0..embedding_dim {
                weights.push(next_value());
            }
        }

        Ok(Self {
            storage: EmbeddingStorage::F32(Arc::new(weights)),
            vocab_size,
            embedding_dim,
            backend: EmbeddingBackend::Cpu,
            pad_token_id: None,
            pad_embedding: None,
            oov_policy: OovPolicy::Zero,
        })
    }

    /// Initialize from pretrained embedding weights.
    ///
    /// `weights[token_id][dim]`
    pub fn from_pretrained(weights: Vec<Vec<f32>>) -> Result<Self, EmbeddingError> {
        Self::from_pretrained_with_dtype(weights, EmbeddingDType::F32)
    }

    /// Initialize from pretrained embedding weights with an explicit storage dtype.
    pub fn from_pretrained_with_dtype(
        weights: Vec<Vec<f32>>,
        dtype: EmbeddingDType,
    ) -> Result<Self, EmbeddingError> {
        if weights.is_empty() {
            return Err(EmbeddingError::EmptyPretrainedWeights);
        }
        let embedding_dim = weights[0].len();
        if embedding_dim == 0 {
            return Err(EmbeddingError::InvalidEmbeddingDim);
        }
        if weights.iter().any(|row| row.len() != embedding_dim) {
            return Err(EmbeddingError::RaggedPretrainedWeights);
        }

        let vocab_size = weights.len();
        let mut flat = Vec::with_capacity(vocab_size * embedding_dim);
        for row in &weights {
            flat.extend_from_slice(row);
        }

        let storage = match dtype {
            EmbeddingDType::F32 => EmbeddingStorage::F32(Arc::new(flat)),
            EmbeddingDType::F16 => {
                let mut out = Vec::with_capacity(flat.len());
                for v in flat {
                    out.push(f32_to_f16_bits(v));
                }
                EmbeddingStorage::F16(Arc::new(out))
            }
            EmbeddingDType::BF16 => {
                let mut out = Vec::with_capacity(flat.len());
                for v in flat {
                    out.push(f32_to_bf16_bits(v));
                }
                EmbeddingStorage::BF16(Arc::new(out))
            }
        };

        Ok(Self {
            vocab_size,
            embedding_dim,
            storage,
            backend: EmbeddingBackend::Cpu,
            pad_token_id: None,
            pad_embedding: None,
            oov_policy: OovPolicy::Zero,
        })
    }

    /// Set backend metadata (forward-compatible for device execution paths).
    pub fn with_backend(mut self, backend: EmbeddingBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Configure out-of-vocabulary behavior for the ergonomic `embed()` API.
    pub fn with_oov_policy(mut self, policy: OovPolicy) -> Self {
        self.oov_policy = policy;
        self
    }

    /// Convert storage dtype (useful for memory tuning: f32 -> f16/bf16).
    pub fn with_dtype(mut self, dtype: EmbeddingDType) -> Self {
        if self.dtype() == dtype {
            return self;
        }

        let n = self.vocab_size * self.embedding_dim;
        let mut as_f32 = Vec::with_capacity(n);
        for i in 0..n {
            as_f32.push(self.value_f32_at_flat(i));
        }

        self.storage = match dtype {
            EmbeddingDType::F32 => EmbeddingStorage::F32(Arc::new(as_f32)),
            EmbeddingDType::F16 => {
                let mut out = Vec::with_capacity(n);
                for v in as_f32 {
                    out.push(f32_to_f16_bits(v));
                }
                EmbeddingStorage::F16(Arc::new(out))
            }
            EmbeddingDType::BF16 => {
                let mut out = Vec::with_capacity(n);
                for v in as_f32 {
                    out.push(f32_to_bf16_bits(v));
                }
                EmbeddingStorage::BF16(Arc::new(out))
            }
        };

        self
    }

    /// Try to switch to GPU backend.
    ///
    /// This sets backend metadata only when GPU support is compiled in.
    #[allow(unused_mut)]
    pub fn to_gpu(mut self, device_id: usize) -> Result<Self, EmbeddingError> {
        #[cfg(feature = "gpu")]
        {
            self.backend = EmbeddingBackend::Gpu { device_id };
            Ok(self)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = device_id;
            Err(EmbeddingError::GpuBackendNotAvailable)
        }
    }

    /// Configure optional padding-token behavior.
    ///
    /// - `pad_token_id`: token that should use the padding embedding.
    /// - `pad_embedding`: custom vector. If `None`, a zero-vector is used.
    pub fn with_padding(
        mut self,
        pad_token_id: u32,
        pad_embedding: Option<Vec<f32>>,
    ) -> Result<Self, EmbeddingError> {
        if let Some(ref vec) = pad_embedding {
            if vec.len() != self.embedding_dim {
                return Err(EmbeddingError::PadEmbeddingDimMismatch {
                    expected: self.embedding_dim,
                    found: vec.len(),
                });
            }
        }

        self.pad_token_id = Some(pad_token_id);
        self.pad_embedding = Some(
            pad_embedding.unwrap_or_else(|| vec![0.0_f32; self.embedding_dim]),
        );
        Ok(self)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn dtype(&self) -> EmbeddingDType {
        self.storage.dtype()
    }

    pub fn backend(&self) -> &EmbeddingBackend {
        &self.backend
    }

    #[allow(dead_code)]
    fn flat_index(&self, token_id: u32, dim: usize) -> Result<usize, EmbeddingError> {
        if token_id as usize >= self.vocab_size {
            return Err(EmbeddingError::TokenIdOutOfRange {
                token_id,
                vocab_size: self.vocab_size,
            });
        }
        Ok((token_id as usize) * self.embedding_dim + dim)
    }

    fn value_f32_at_flat(&self, flat_idx: usize) -> f32 {
        match &self.storage {
            EmbeddingStorage::F32(v) => v[flat_idx],
            EmbeddingStorage::F16(v) => f16_bits_to_f32(v[flat_idx]),
            EmbeddingStorage::BF16(v) => bf16_bits_to_f32(v[flat_idx]),
        }
    }

    fn write_row_to(&self, token_id: u32, out: &mut Vec<f32>) -> Result<(), EmbeddingError> {
        if self.pad_token_id == Some(token_id) {
            if let Some(ref pad) = self.pad_embedding {
                out.extend_from_slice(pad);
                return Ok(());
            }
        }

        if token_id as usize >= self.vocab_size {
            return Err(EmbeddingError::TokenIdOutOfRange {
                token_id,
                vocab_size: self.vocab_size,
            });
        }

        let base = (token_id as usize) * self.embedding_dim;
        match &self.storage {
            EmbeddingStorage::F32(v) => {
                out.extend_from_slice(&v[base..base + self.embedding_dim]);
            }
            EmbeddingStorage::F16(v) => {
                for &bits in &v[base..base + self.embedding_dim] {
                    out.push(f16_bits_to_f32(bits));
                }
            }
            EmbeddingStorage::BF16(v) => {
                for &bits in &v[base..base + self.embedding_dim] {
                    out.push(bf16_bits_to_f32(bits));
                }
            }
        }
        Ok(())
    }

    /// Strict single-token lookup.
    pub fn try_embedding_ref(&self, token_id: u32) -> Result<&[f32], EmbeddingError> {
        if self.pad_token_id == Some(token_id) {
            if let Some(ref pad) = self.pad_embedding {
                return Ok(pad.as_slice());
            }
        }

        if token_id as usize >= self.vocab_size {
            return Err(EmbeddingError::TokenIdOutOfRange {
                token_id,
                vocab_size: self.vocab_size,
            });
        }

        match &self.storage {
            EmbeddingStorage::F32(v) => {
                let base = (token_id as usize) * self.embedding_dim;
                Ok(&v[base..base + self.embedding_dim])
            }
            _ => Err(EmbeddingError::UnsupportedZeroCopyForDType(self.dtype())),
        }
    }

    /// Strict embedding with explicit error handling.
    pub fn try_embed(&self, token_ids: &[u32]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut out = Vec::with_capacity(token_ids.len());
        for &id in token_ids {
            let mut row = Vec::with_capacity(self.embedding_dim);
            self.write_row_to(id, &mut row)?;
            out.push(row);
        }
        Ok(out)
    }

    /// Maps token IDs to embeddings.
    ///
    /// This method follows the requested signature and is ergonomic in training loops.
    /// If an out-of-range token is encountered and no pad behavior is configured,
    /// it falls back to a zero vector.
    pub fn embed(&self, token_ids: &[u32]) -> Vec<Vec<f32>> {
        let zero_fallback = vec![0.0_f32; self.embedding_dim];
        let mut out = Vec::with_capacity(token_ids.len());

        for &id in token_ids {
            let mut row = Vec::with_capacity(self.embedding_dim);
            match self.write_row_to(id, &mut row) {
                Ok(_) => out.push(row),
                Err(_) => match self.oov_policy {
                    OovPolicy::Zero => out.push(zero_fallback.clone()),
                    OovPolicy::Panic => {
                        panic!(
                            "token id {} out of range for vocab size {}",
                            id, self.vocab_size
                        )
                    }
                    OovPolicy::DebugAssert => {
                        debug_assert!(
                            false,
                            "token id {} out of range for vocab size {}",
                            id,
                            self.vocab_size
                        );
                        out.push(zero_fallback.clone())
                    }
                },
            }
        }

        out
    }

    /// Zero-copy embedding references.
    ///
    /// Useful when downstream code can consume borrowed rows directly.
    pub fn embed_refs<'a>(&'a self, token_ids: &[u32]) -> Result<Vec<&'a [f32]>, EmbeddingError> {
        let mut out = Vec::with_capacity(token_ids.len());
        for &id in token_ids {
            out.push(self.try_embedding_ref(id)?);
        }
        Ok(out)
    }

    /// Embeds any iterator of token IDs (works directly with `window.iter_padded()`).
    pub fn try_embed_from_iter<I>(&self, token_ids: I) -> Result<Vec<Vec<f32>>, EmbeddingError>
    where
        I: IntoIterator<Item = u32>,
    {
        let mut out = Vec::new();
        for id in token_ids {
            let mut row = Vec::with_capacity(self.embedding_dim);
            self.write_row_to(id, &mut row)?;
            out.push(row);
        }
        Ok(out)
    }

    /// Batch embedding API for batched training.
    ///
    /// Input shape: `[batch][seq]` token IDs.
    /// Output shape: `[batch][seq][embedding_dim]`.
    pub fn try_embed_batch<'a, I>(&self, batch: I) -> Result<Vec<Vec<Vec<f32>>>, EmbeddingError>
    where
        I: IntoIterator<Item = &'a [u32]>,
    {
        let mut out = Vec::new();
        for seq in batch {
            out.push(self.try_embed(seq)?);
        }
        Ok(out)
    }

    /// Batch embedding flattened into a contiguous row-major buffer.
    ///
    /// Returns `(buffer, batch_size, seq_len, embedding_dim)` to ease device transfer
    /// and multi-GPU pipeline integration.
    pub fn try_embed_batch_contiguous<'a, I>(
        &self,
        batch: I,
    ) -> Result<(Vec<f32>, usize, usize, usize), EmbeddingError>
    where
        I: IntoIterator<Item = &'a [u32]>,
    {
        let sequences: Vec<&[u32]> = batch.into_iter().collect();
        let batch_size = sequences.len();
        let seq_len = sequences.first().map_or(0, |s| s.len());

        let mut flat = Vec::with_capacity(batch_size * seq_len * self.embedding_dim);

        for seq in &sequences {
            for &id in *seq {
                self.write_row_to(id, &mut flat)?;
            }
        }

        Ok((flat, batch_size, seq_len, self.embedding_dim))
    }

    /// Batch embedding into a caller-owned contiguous buffer.
    ///
    /// This avoids repeated large allocations across training steps.
    ///
    /// Returns `(batch_size, seq_len, embedding_dim)`.
    pub fn try_embed_batch_contiguous_into<'a, I>(
        &self,
        batch: I,
        out: &mut Vec<f32>,
    ) -> Result<(usize, usize, usize), EmbeddingError>
    where
        I: IntoIterator<Item = &'a [u32]>,
    {
        let sequences: Vec<&[u32]> = batch.into_iter().collect();
        let batch_size = sequences.len();
        let seq_len = sequences.first().map_or(0, |s| s.len());

        let required = batch_size * seq_len * self.embedding_dim;
        out.clear();
        if out.capacity() < required {
            out.reserve(required - out.capacity());
        }

        for seq in &sequences {
            for &id in *seq {
                self.write_row_to(id, out)?;
            }
        }

        Ok((batch_size, seq_len, self.embedding_dim))
    }

    /// Batch embedding into a reusable buffer wrapper.
    pub fn try_embed_batch_into_buffer<'a, I>(
        &self,
        batch: I,
        buffer: &mut BatchEmbeddingBuffer,
    ) -> Result<(usize, usize, usize), EmbeddingError>
    where
        I: IntoIterator<Item = &'a [u32]>,
    {
        self.try_embed_batch_contiguous_into(batch, &mut buffer.data)
    }

    /// Returns a shard (row slice) of the embedding matrix for distributed training.
    pub fn shard_rows(
        &self,
        rank: usize,
        world_size: usize,
    ) -> Result<EmbeddingShard<'_>, EmbeddingError> {
        if world_size == 0 || rank >= world_size {
            return Err(EmbeddingError::InvalidShardConfig { rank, world_size });
        }

        let rows_per_shard = self.vocab_size / world_size;
        let remainder = self.vocab_size % world_size;

        let extra_before = remainder.min(rank);
        let start = rank * rows_per_shard + extra_before;
        let local_rows = rows_per_shard + usize::from(rank < remainder);
        let end = start + local_rows;

        let flat_start = start * self.embedding_dim;
        let flat_end = end * self.embedding_dim;

        let data = match &self.storage {
            EmbeddingStorage::F32(v) => EmbeddingShardData::F32(&v[flat_start..flat_end]),
            EmbeddingStorage::F16(v) => EmbeddingShardData::F16(&v[flat_start..flat_end]),
            EmbeddingStorage::BF16(v) => EmbeddingShardData::BF16(&v[flat_start..flat_end]),
        };

        Ok(EmbeddingShard {
            start_row: start,
            end_row: end,
            embedding_dim: self.embedding_dim,
            dtype: self.dtype(),
            data,
        })
    }
}

fn f32_to_bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn bf16_bits_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 255 {
        if mant == 0 {
            return sign | 0x7C00;
        }
        return sign | 0x7E00;
    }

    let half_exp = exp - 127 + 15;

    if half_exp >= 31 {
        return sign | 0x7C00;
    }

    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x80_0000;
        let shift = 14 - half_exp;
        let mut half_mant = (mantissa >> shift) as u16;
        if ((mantissa >> (shift - 1)) & 1) != 0 {
            half_mant = half_mant.wrapping_add(1);
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    if ((mant >> 12) & 1) != 0 {
        half = half.wrapping_add(1);
    }
    half
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x03FF) as u32;

    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = -14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            let exp32 = ((e + 127) as u32) << 23;
            sign | exp32 | (m << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        let exp32 = ((exp - 15 + 127) as u32) << 23;
        sign | exp32 | (mant << 13)
    };

    f32::from_bits(out)
}
