use std::ops::{Add, Div, Mul, Sub};

/// Small trait to support LayerNorm math for `f32` and `f64` without external crates.
pub trait LayerNormFloat:
    Copy
    + Clone
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f32(v: f32) -> Self;
    fn from_usize(v: usize) -> Self;
    fn sqrt(self) -> Self;
}

impl LayerNormFloat for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_f32(v: f32) -> Self {
        v
    }

    fn from_usize(v: usize) -> Self {
        v as f32
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl LayerNormFloat for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_f32(v: f32) -> Self {
        v as f64
    }

    fn from_usize(v: usize) -> Self {
        v as f64
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

/// Parameterized Layer Normalization.
///
/// - `gamma`: trainable scale vector
/// - `beta`: trainable shift vector
/// - `epsilon`: numerical stability term
///
/// The struct is generic over floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LayerNorm<T: LayerNormFloat> {
    embedding_dim: usize,
    epsilon: T,
    gamma: Vec<T>,
    beta: Vec<T>,
}

impl<T: LayerNormFloat> LayerNorm<T> {
    /// Shared constructor used by typed wrappers.
    fn new_internal(embedding_dim: usize, epsilon: Option<f32>) -> Self {
        let eps = epsilon.unwrap_or(1e-5);

        Self {
            embedding_dim,
            epsilon: T::from_f32(eps),
            gamma: vec![T::one(); embedding_dim],
            beta: vec![T::zero(); embedding_dim],
        }
    }

    /// Core normalization logic for one embedding vector.
    fn normalize_row(&self, input: &[T]) -> Option<Vec<T>> {
        if input.len() != self.embedding_dim || self.embedding_dim == 0 {
            return None;
        }

        // 1) Compute mean.
        let sum = input.iter().copied().fold(T::zero(), |acc, x| acc + x);
        let mean = sum / T::from_usize(self.embedding_dim);

        // 2) Compute variance.
        let var_sum = input
            .iter()
            .copied()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .fold(T::zero(), |acc, x| acc + x);
        let variance = var_sum / T::from_usize(self.embedding_dim);

        // 3) Normalize with epsilon for numerical stability.
        let denom = (variance + self.epsilon).sqrt();

        // 4) Apply trainable affine transform: normalized * gamma + beta.
        let mut out = vec![T::zero(); self.embedding_dim];
        for i in 0..self.embedding_dim {
            let normalized = (input[i] - mean) / denom;
            out[i] = normalized * self.gamma[i] + self.beta[i];
        }

        Some(out)
    }

    /// Access immutable gamma vector.
    pub fn gamma(&self) -> &[T] {
        &self.gamma
    }

    /// Access immutable beta vector.
    pub fn beta(&self) -> &[T] {
        &self.beta
    }
}

impl LayerNorm<f32> {
    /// Create a new LayerNorm for `f32` embeddings.
    ///
    /// - `embedding_dim`: size of each embedding vector.
    /// - `epsilon`: optional stability constant; defaults to `1e-5`.
    pub fn new(embedding_dim: usize, epsilon: Option<f32>) -> Self {
        Self::new_internal(embedding_dim, epsilon)
    }

    /// Normalize a single embedding vector.
    ///
    /// Returns an empty vector if input size does not match `embedding_dim` or
    /// if `embedding_dim == 0`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.normalize_row(input).unwrap_or_default()
    }

    /// Normalize a batch of embedding vectors.
    ///
    /// Returns an empty batch if any row has mismatched dimension.
    pub fn forward_batch(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(input.len());
        for row in input {
            match self.normalize_row(row) {
                Some(v) => out.push(v),
                None => return Vec::new(),
            }
        }
        out
    }
}

impl LayerNorm<f64> {
    /// Create a new LayerNorm for `f64` embeddings.
    ///
    /// - `embedding_dim`: size of each embedding vector.
    /// - `epsilon`: optional stability constant; defaults to `1e-5`.
    pub fn new(embedding_dim: usize, epsilon: Option<f32>) -> Self {
        Self::new_internal(embedding_dim, epsilon)
    }

    /// Normalize a single `f64` embedding vector.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.normalize_row(input).unwrap_or_default()
    }

    /// Normalize a batch of `f64` embedding vectors.
    pub fn forward_batch(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(input.len());
        for row in input {
            match self.normalize_row(row) {
                Some(v) => out.push(v),
                None => return Vec::new(),
            }
        }
        out
    }
}
