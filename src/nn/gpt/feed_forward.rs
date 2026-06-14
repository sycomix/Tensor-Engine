use rand::Rng;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Numeric trait for `FeedForward` supporting `f32` and `f64`.
pub trait FeedForwardFloat:
    Copy
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + PartialOrd
{
    fn zero() -> Self;
    fn from_f32(v: f32) -> Self;
    fn from_usize(v: usize) -> Self;
    fn tanh(self) -> Self;
    fn sqrt(self) -> Self;
}

impl FeedForwardFloat for f32 {
    fn zero() -> Self {
        0.0
    }

    fn from_f32(v: f32) -> Self {
        v
    }

    fn from_usize(v: usize) -> Self {
        v as f32
    }

    fn tanh(self) -> Self {
        f32::tanh(self)
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl FeedForwardFloat for f64 {
    fn zero() -> Self {
        0.0
    }

    fn from_f32(v: f32) -> Self {
        v as f64
    }

    fn from_usize(v: usize) -> Self {
        v as f64
    }

    fn tanh(self) -> Self {
        f64::tanh(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

/// Parameterized two-layer feed-forward block for transformer models.
///
/// Architecture:
/// - Linear1: `input_dim -> hidden_dim`
/// - GELU
/// - Linear2: `hidden_dim -> input_dim`
#[derive(Debug, Clone)]
pub struct FeedForward<T: FeedForwardFloat> {
    input_dim: usize,
    hidden_dim: usize,
    dropout_rate: Option<f32>,
    w1: Vec<Vec<T>>, // [hidden_dim][input_dim]
    b1: Vec<T>,      // [hidden_dim]
    w2: Vec<Vec<T>>, // [input_dim][hidden_dim]
    b2: Vec<T>,      // [input_dim]
}

/// Reusable workspace buffers for allocation-efficient forward passes.
#[derive(Debug, Clone)]
pub struct FeedForwardWorkspace<T: FeedForwardFloat> {
    hidden: Vec<T>,
    output: Vec<T>,
}

impl<T: FeedForwardFloat> FeedForwardWorkspace<T> {
    pub fn new(hidden_dim: usize, input_dim: usize) -> Self {
        Self {
            hidden: vec![T::zero(); hidden_dim],
            output: vec![T::zero(); input_dim],
        }
    }

    fn ensure_dims(&mut self, hidden_dim: usize, input_dim: usize) {
        if self.hidden.len() != hidden_dim {
            self.hidden.resize(hidden_dim, T::zero());
        }
        if self.output.len() != input_dim {
            self.output.resize(input_dim, T::zero());
        }
    }
}

impl<T: FeedForwardFloat> FeedForward<T> {
    /// Construct a new feed-forward block with random uniform initialization.
    ///
    /// We use simple Xavier-style limits:
    /// - w1 in `[-sqrt(6/(input_dim+hidden_dim)), +sqrt(6/(input_dim+hidden_dim))]`
    /// - w2 in `[-sqrt(6/(hidden_dim+input_dim)), +sqrt(6/(hidden_dim+input_dim))]`
    /// Biases are initialized to zero.
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        assert!(input_dim > 0, "input_dim must be > 0");
        assert!(hidden_dim > 0, "hidden_dim must be > 0");

        let mut rng = rand::rng();

        let fan_sum = T::from_usize(input_dim + hidden_dim);
        let limit = (T::from_f32(6.0) / fan_sum).sqrt();

        let mut w1 = vec![vec![T::zero(); input_dim]; hidden_dim];
        for row in &mut w1 {
            for v in row {
                let r = rng.random_range(-1.0_f32..1.0_f32);
                *v = T::from_f32(r) * limit;
            }
        }

        let mut w2 = vec![vec![T::zero(); hidden_dim]; input_dim];
        for row in &mut w2 {
            for v in row {
                let r = rng.random_range(-1.0_f32..1.0_f32);
                *v = T::from_f32(r) * limit;
            }
        }

        Self {
            input_dim,
            hidden_dim,
            dropout_rate: None,
            w1,
            b1: vec![T::zero(); hidden_dim],
            w2,
            b2: vec![T::zero(); input_dim],
        }
    }

    /// Set dropout probability for the hidden activation.
    ///
    /// Values `<= 0.0` disable dropout. Values `>= 1.0` are ignored.
    pub fn set_dropout_rate(&mut self, dropout_rate: Option<f32>) {
        self.dropout_rate = match dropout_rate {
            Some(p) if p > 0.0 && p < 1.0 => Some(p),
            _ => None,
        };
    }

    /// GELU activation using tanh approximation:
    ///
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`
    fn gelu(x: T) -> T {
        let c0 = T::from_f32(0.5);
        let c1 = T::from_f32(1.0);
        let c2 = T::from_f32(0.044_715);
        let c3 = T::from_f32(0.797_884_6); // sqrt(2/pi)

        let x3 = x * x * x;
        let inner = c3 * (x + c2 * x3);
        c0 * x * (c1 + inner.tanh())
    }

    /// Forward pass for a single embedding vector of shape `[input_dim]`.
    ///
    /// Returns empty output if input dimensionality is invalid.
    pub fn forward_single(&self, input: &[T]) -> Vec<T> {
        self.forward_single_training(input, false)
    }

    /// Forward pass for a single embedding vector with optional training behavior.
    ///
    /// When `training == true` and dropout is enabled, dropout is applied to
    /// the hidden activation after GELU using inverted-dropout scaling.
    pub fn forward_single_training(&self, input: &[T], training: bool) -> Vec<T> {
        let mut workspace = FeedForwardWorkspace::new(self.hidden_dim, self.input_dim);
        self.forward_single_into_training(input, &mut workspace, training)
    }

    /// Forward pass for a single embedding vector using caller-provided workspace.
    ///
    /// This avoids repeated temporary allocations in tight loops.
    pub fn forward_single_into(
        &self,
        input: &[T],
        workspace: &mut FeedForwardWorkspace<T>,
    ) -> Vec<T> {
        self.forward_single_into_training(input, workspace, false)
    }

    /// Forward pass for a single embedding vector using caller-provided workspace
    /// with optional training behavior.
    pub fn forward_single_into_training(
        &self,
        input: &[T],
        workspace: &mut FeedForwardWorkspace<T>,
        training: bool,
    ) -> Vec<T> {
        if input.len() != self.input_dim {
            return Vec::new();
        }

        workspace.ensure_dims(self.hidden_dim, self.input_dim);

        // First linear layer + bias.
        for (h, hidden_val) in workspace.hidden.iter_mut().enumerate() {
            let mut sum = self.b1[h];
            for (i, &x) in input.iter().enumerate() {
                sum = sum + self.w1[h][i] * x;
            }
            *hidden_val = Self::gelu(sum);
        }

        if training {
            if let Some(p) = self.dropout_rate {
                let keep_prob = 1.0_f32 - p;
                let scale = T::from_f32(1.0_f32 / keep_prob);
                let mut rng = rand::rng();
                for hidden_val in &mut workspace.hidden {
                    if rng.random::<f32>() < p {
                        *hidden_val = T::zero();
                    } else {
                        *hidden_val = *hidden_val * scale;
                    }
                }
            }
        }

        // Second linear layer + bias.
        for (o, out_val) in workspace.output.iter_mut().enumerate() {
            let mut sum = self.b2[o];
            for (h, &hv) in workspace.hidden.iter().enumerate() {
                sum = sum + self.w2[o][h] * hv;
            }
            *out_val = sum;
        }

        workspace.output.clone()
    }

    /// Forward pass for a batch of embeddings.
    ///
    /// Input shape: `[batch][input_dim]`
    /// Output shape: `[batch][input_dim]`
    ///
    /// Returns empty output if any row has invalid dimensionality.
    pub fn forward(&self, input: &[Vec<T>]) -> Vec<Vec<T>> {
        self.forward_training(input, false)
    }

    /// Forward pass for a batch with optional training behavior.
    pub fn forward_training(&self, input: &[Vec<T>], training: bool) -> Vec<Vec<T>> {
        let mut workspace = FeedForwardWorkspace::new(self.hidden_dim, self.input_dim);
        self.forward_into_training(input, &mut workspace, training)
    }

    /// Forward pass for a batch using caller-provided workspace.
    ///
    /// This keeps hidden/output temporaries contiguous and reused across rows.
    pub fn forward_into(
        &self,
        input: &[Vec<T>],
        workspace: &mut FeedForwardWorkspace<T>,
    ) -> Vec<Vec<T>> {
        self.forward_into_training(input, workspace, false)
    }

    /// Forward pass for a batch using caller-provided workspace with optional
    /// training behavior.
    pub fn forward_into_training(
        &self,
        input: &[Vec<T>],
        workspace: &mut FeedForwardWorkspace<T>,
        training: bool,
    ) -> Vec<Vec<T>> {
        let mut out = Vec::with_capacity(input.len());

        for row in input {
            let y = self.forward_single_into_training(row, workspace, training);
            if y.is_empty() && self.input_dim != 0 {
                return Vec::new();
            }
            out.push(y);
        }

        out
    }
}
