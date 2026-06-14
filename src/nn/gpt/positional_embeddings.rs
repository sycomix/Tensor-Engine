use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

/// Compute backend metadata for positional embeddings.
///
/// GPU execution is intentionally not implemented here, but this enum keeps
/// the API forward-compatible with future device backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionalBackend {
	Cpu,
	Gpu { device_id: usize },
}

/// Numeric storage type for embedding weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionalDType {
	F32,
	F16,
	BF16,
}

/// Behavior when ergonomic embedding APIs see an out-of-range position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionOobPolicy {
	Zero,
	Panic,
	DebugAssert,
}

/// Errors for initialization and strict positional embedding APIs.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionalEmbeddingError {
	InvalidMaxSeqLen,
	InvalidEmbeddingDim,
	EmptyPretrainedWeights,
	RaggedPretrainedWeights,
	PositionOutOfRange { position: usize, max_seq_len: usize },
	UnsupportedZeroCopyForDType(PositionalDType),
	GpuBackendNotAvailable,
}

impl Display for PositionalEmbeddingError {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		match self {
			PositionalEmbeddingError::InvalidMaxSeqLen => write!(f, "max_seq_len must be > 0"),
			PositionalEmbeddingError::InvalidEmbeddingDim => {
				write!(f, "embedding_dim must be > 0")
			}
			PositionalEmbeddingError::EmptyPretrainedWeights => {
				write!(f, "pretrained weight matrix is empty")
			}
			PositionalEmbeddingError::RaggedPretrainedWeights => {
				write!(f, "pretrained weight rows must all have the same length")
			}
			PositionalEmbeddingError::PositionOutOfRange {
				position,
				max_seq_len,
			} => write!(
				f,
				"position {} out of range for max_seq_len {}",
				position, max_seq_len
			),
			PositionalEmbeddingError::UnsupportedZeroCopyForDType(dtype) => write!(
				f,
				"zero-copy embedding references are only available for f32 storage; got {:?}",
				dtype
			),
			PositionalEmbeddingError::GpuBackendNotAvailable => {
				write!(f, "GPU backend is not available in this build")
			}
		}
	}
}

impl Error for PositionalEmbeddingError {}

#[derive(Debug, Clone)]
enum PositionalStorage {
	F32(Arc<Vec<f32>>),
	F16(Arc<Vec<u16>>),
	BF16(Arc<Vec<u16>>),
}

impl PositionalStorage {
	fn dtype(&self) -> PositionalDType {
		match self {
			PositionalStorage::F32(_) => PositionalDType::F32,
			PositionalStorage::F16(_) => PositionalDType::F16,
			PositionalStorage::BF16(_) => PositionalDType::BF16,
		}
	}
}

/// Positional embedding table used by transformer input pipelines.
///
/// - Row-major contiguous layout (`weights[position * embedding_dim + dim]`).
/// - Supports f32/f16/bf16 storage.
/// - Offers strict and ergonomic APIs.
/// - Exposes zero-copy refs for f32 storage when possible.
#[derive(Debug, Clone)]
pub struct PositionalEmbedding {
	storage: PositionalStorage,
	max_seq_len: usize,
	embedding_dim: usize,
	backend: PositionalBackend,
	oob_policy: PositionOobPolicy,
}

impl PositionalEmbedding {
	/// Initialize random positional embeddings with deterministic seed.
	pub fn random(
		max_seq_len: usize,
		embedding_dim: usize,
		seed: u64,
	) -> Result<Self, PositionalEmbeddingError> {
		if max_seq_len == 0 {
			return Err(PositionalEmbeddingError::InvalidMaxSeqLen);
		}
		if embedding_dim == 0 {
			return Err(PositionalEmbeddingError::InvalidEmbeddingDim);
		}

		let mut state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
		let mut next_f32 = || {
			state ^= state << 13;
			state ^= state >> 7;
			state ^= state << 17;
			let unit = (state as f64) / (u64::MAX as f64);
			(unit as f32) * 0.04 - 0.02
		};

		Self::random_with_generator(max_seq_len, embedding_dim, &mut next_f32)
	}

	/// Initialize random positional embeddings using a caller-provided generator.
	pub fn random_with_generator<F>(
		max_seq_len: usize,
		embedding_dim: usize,
		mut next_value: F,
	) -> Result<Self, PositionalEmbeddingError>
	where
		F: FnMut() -> f32,
	{
		if max_seq_len == 0 {
			return Err(PositionalEmbeddingError::InvalidMaxSeqLen);
		}
		if embedding_dim == 0 {
			return Err(PositionalEmbeddingError::InvalidEmbeddingDim);
		}

		let total = max_seq_len * embedding_dim;
		let mut flat = Vec::with_capacity(total);
		for _ in 0..total {
			flat.push(next_value());
		}

		Ok(Self {
			storage: PositionalStorage::F32(Arc::new(flat)),
			max_seq_len,
			embedding_dim,
			backend: PositionalBackend::Cpu,
			oob_policy: PositionOobPolicy::Zero,
		})
	}

	/// Initialize from pretrained positional embeddings (`weights[position][dim]`).
	pub fn from_pretrained(
		weights: Vec<Vec<f32>>,
	) -> Result<Self, PositionalEmbeddingError> {
		Self::from_pretrained_with_dtype(weights, PositionalDType::F32)
	}

	/// Initialize from pretrained positional embeddings with explicit dtype.
	pub fn from_pretrained_with_dtype(
		weights: Vec<Vec<f32>>,
		dtype: PositionalDType,
	) -> Result<Self, PositionalEmbeddingError> {
		if weights.is_empty() {
			return Err(PositionalEmbeddingError::EmptyPretrainedWeights);
		}

		let embedding_dim = weights[0].len();
		if embedding_dim == 0 {
			return Err(PositionalEmbeddingError::InvalidEmbeddingDim);
		}
		if weights.iter().any(|row| row.len() != embedding_dim) {
			return Err(PositionalEmbeddingError::RaggedPretrainedWeights);
		}

		let max_seq_len = weights.len();
		let mut flat = Vec::with_capacity(max_seq_len * embedding_dim);
		for row in &weights {
			flat.extend_from_slice(row);
		}

		let storage = match dtype {
			PositionalDType::F32 => PositionalStorage::F32(Arc::new(flat)),
			PositionalDType::F16 => {
				let mut out = Vec::with_capacity(flat.len());
				for v in flat {
					out.push(f32_to_f16_bits(v));
				}
				PositionalStorage::F16(Arc::new(out))
			}
			PositionalDType::BF16 => {
				let mut out = Vec::with_capacity(flat.len());
				for v in flat {
					out.push(f32_to_bf16_bits(v));
				}
				PositionalStorage::BF16(Arc::new(out))
			}
		};

		Ok(Self {
			storage,
			max_seq_len,
			embedding_dim,
			backend: PositionalBackend::Cpu,
			oob_policy: PositionOobPolicy::Zero,
		})
	}

	/// Set backend metadata for future device execution paths.
	pub fn with_backend(mut self, backend: PositionalBackend) -> Self {
		self.backend = backend;
		self
	}

	/// Configure out-of-range handling for ergonomic APIs.
	pub fn with_oob_policy(mut self, policy: PositionOobPolicy) -> Self {
		self.oob_policy = policy;
		self
	}

	/// Convert internal storage dtype.
	pub fn with_dtype(mut self, dtype: PositionalDType) -> Self {
		if self.dtype() == dtype {
			return self;
		}

		let total = self.max_seq_len * self.embedding_dim;
		let mut as_f32 = Vec::with_capacity(total);
		for idx in 0..total {
			as_f32.push(self.value_f32_at_flat(idx));
		}

		self.storage = match dtype {
			PositionalDType::F32 => PositionalStorage::F32(Arc::new(as_f32)),
			PositionalDType::F16 => {
				let mut out = Vec::with_capacity(total);
				for v in as_f32 {
					out.push(f32_to_f16_bits(v));
				}
				PositionalStorage::F16(Arc::new(out))
			}
			PositionalDType::BF16 => {
				let mut out = Vec::with_capacity(total);
				for v in as_f32 {
					out.push(f32_to_bf16_bits(v));
				}
				PositionalStorage::BF16(Arc::new(out))
			}
		};

		self
	}

	/// Try to switch to GPU backend metadata.
	#[allow(unused_mut)]
	pub fn to_gpu(mut self, device_id: usize) -> Result<Self, PositionalEmbeddingError> {
		#[cfg(feature = "gpu")]
		{
			self.backend = PositionalBackend::Gpu { device_id };
			Ok(self)
		}
		#[cfg(not(feature = "gpu"))]
		{
			let _ = device_id;
			Err(PositionalEmbeddingError::GpuBackendNotAvailable)
		}
	}

	pub fn max_seq_len(&self) -> usize {
		self.max_seq_len
	}

	pub fn embedding_dim(&self) -> usize {
		self.embedding_dim
	}

	pub fn backend(&self) -> &PositionalBackend {
		&self.backend
	}

	pub fn dtype(&self) -> PositionalDType {
		self.storage.dtype()
	}

	fn flat_index(&self, position: usize, dim: usize) -> Result<usize, PositionalEmbeddingError> {
		if position >= self.max_seq_len {
			return Err(PositionalEmbeddingError::PositionOutOfRange {
				position,
				max_seq_len: self.max_seq_len,
			});
		}
		Ok(position * self.embedding_dim + dim)
	}

	fn value_f32_at_flat(&self, flat_idx: usize) -> f32 {
		match &self.storage {
			PositionalStorage::F32(v) => v[flat_idx],
			PositionalStorage::F16(v) => f16_bits_to_f32(v[flat_idx]),
			PositionalStorage::BF16(v) => bf16_bits_to_f32(v[flat_idx]),
		}
	}

	fn write_row_to(
		&self,
		position: usize,
		out: &mut Vec<f32>,
	) -> Result<(), PositionalEmbeddingError> {
		if position >= self.max_seq_len {
			return Err(PositionalEmbeddingError::PositionOutOfRange {
				position,
				max_seq_len: self.max_seq_len,
			});
		}

		let base = self.flat_index(position, 0)?;
		match &self.storage {
			PositionalStorage::F32(v) => out.extend_from_slice(&v[base..base + self.embedding_dim]),
			PositionalStorage::F16(v) => {
				for &bits in &v[base..base + self.embedding_dim] {
					out.push(f16_bits_to_f32(bits));
				}
			}
			PositionalStorage::BF16(v) => {
				for &bits in &v[base..base + self.embedding_dim] {
					out.push(bf16_bits_to_f32(bits));
				}
			}
		}

		Ok(())
	}

	/// Strict single-position lookup (zero-copy, f32 storage only).
	pub fn try_embedding_ref(&self, position: usize) -> Result<&[f32], PositionalEmbeddingError> {
		if position >= self.max_seq_len {
			return Err(PositionalEmbeddingError::PositionOutOfRange {
				position,
				max_seq_len: self.max_seq_len,
			});
		}

		match &self.storage {
			PositionalStorage::F32(v) => {
				let base = position * self.embedding_dim;
				Ok(&v[base..base + self.embedding_dim])
			}
			_ => Err(PositionalEmbeddingError::UnsupportedZeroCopyForDType(self.dtype())),
		}
	}

	/// Strict positional embedding lookup with explicit errors.
	pub fn try_embed_positions(
		&self,
		positions: &[usize],
	) -> Result<Vec<Vec<f32>>, PositionalEmbeddingError> {
		let mut out = Vec::with_capacity(positions.len());
		for &pos in positions {
			let mut row = Vec::with_capacity(self.embedding_dim);
			self.write_row_to(pos, &mut row)?;
			out.push(row);
		}
		Ok(out)
	}

	/// Ergonomic positional lookup matching requested API.
	///
	/// Out-of-range handling is controlled by `PositionOobPolicy`.
	pub fn embed_positions(&self, positions: &[usize]) -> Vec<Vec<f32>> {
		let zero_fallback = vec![0.0_f32; self.embedding_dim];
		let mut out = Vec::with_capacity(positions.len());

		for &pos in positions {
			let mut row = Vec::with_capacity(self.embedding_dim);
			match self.write_row_to(pos, &mut row) {
				Ok(_) => out.push(row),
				Err(_) => match self.oob_policy {
					PositionOobPolicy::Zero => out.push(zero_fallback.clone()),
					PositionOobPolicy::Panic => {
						panic!(
							"position {} out of range for max_seq_len {}",
							pos, self.max_seq_len
						)
					}
					PositionOobPolicy::DebugAssert => {
						debug_assert!(
							false,
							"position {} out of range for max_seq_len {}",
							pos,
							self.max_seq_len
						);
						out.push(zero_fallback.clone())
					}
				},
			}
		}

		out
	}

	/// Zero-copy positional embedding references (f32 storage only).
	pub fn embed_refs<'a>(
		&'a self,
		positions: &[usize],
	) -> Result<Vec<&'a [f32]>, PositionalEmbeddingError> {
		let mut out = Vec::with_capacity(positions.len());
		for &pos in positions {
			out.push(self.try_embedding_ref(pos)?);
		}
		Ok(out)
	}

	/// Batch embedding API.
	///
	/// Input shape: `[batch][seq]` positions.
	/// Output shape: `[batch][seq][embedding_dim]`.
	pub fn embed_batch<'a, I>(&self, batch: I) -> Vec<Vec<Vec<f32>>>
	where
		I: IntoIterator<Item = &'a [usize]>,
	{
		let mut out = Vec::new();
		for seq in batch {
			out.push(self.embed_positions(seq));
		}
		out
	}

	/// Strict batch embedding variant.
	pub fn try_embed_batch<'a, I>(
		&self,
		batch: I,
	) -> Result<Vec<Vec<Vec<f32>>>, PositionalEmbeddingError>
	where
		I: IntoIterator<Item = &'a [usize]>,
	{
		let mut out = Vec::new();
		for seq in batch {
			out.push(self.try_embed_positions(seq)?);
		}
		Ok(out)
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
