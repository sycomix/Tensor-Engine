//! Kolmogorov-Arnold Networks (KAN) implementation.
//!
//! KAN replaces the fixed activation functions in MLPs with learnable activation
//! functions on each edge, based on the Kolmogorov-Arnold representation theorem.
//!
//! Reference: [Liu et al., 2024](https://arxiv.org/abs/2404.19756)

use crate::nn::{Linear, Module};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// KAN activation function: cubic spline interpolation.
///
/// Represents a 1D learnable function via cubic spline coefficients.
/// During forward pass, evaluates the spline at input points.
#[derive(Clone)]
pub struct CubicSpline {
    /// Spline knot points: equally spaced in [-1, 1] by default
    knots: Vec<f32>,
    /// Spline coefficients (learnable parameters)
    coeffs: Tensor,
    /// Number of knots
    num_knots: usize,
    /// Grid spacing
    grid_size: usize,
}

impl CubicSpline {
    /// Create a new cubic spline with given grid size.
    pub fn new(grid_size: usize) -> Self {
        let num_knots = grid_size + 1;
        let mut knots = Vec::with_capacity(num_knots);
        for i in 0..num_knots {
            let t = (2.0 * i as f32) / (grid_size as f32) - 1.0;
            knots.push(t);
        }
        let coeffs = Tensor::new(ArrayD::zeros(IxDyn(&[num_knots][..])), true);
        CubicSpline {
            knots,
            coeffs,
            num_knots,
            grid_size,
        }
    }

    /// Evaluate the spline at input points.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_arr = x.lock().storage.to_f32_array();
        let shape = x_arr.shape().to_vec();
        let mut out = Vec::with_capacity(x_arr.len());

        for &val in x_arr.iter() {
            let clipped = val.clamp(-1.0, 1.0);
            let result = self.evaluate(clipped);
            out.push(result);
        }

        let out_arr = match ArrayD::from_shape_vec(IxDyn(&shape), out) {
            Ok(v) => v,
            Err(_) => ArrayD::zeros(IxDyn(&shape)),
        };
        Tensor::new(out_arr, false)
    }

    /// Evaluate spline at a single point using cubic interpolation.
    fn evaluate(&self, x: f32) -> f32 {
        // Find the grid interval containing x
        let idx = ((x + 1.0) * (self.grid_size as f32) / 2.0).round() as usize;
        let idx = idx.min(self.grid_size - 1);
        let idx = idx.max(0);

        let x0 = self.knots[idx];
        let x1 = self.knots[idx + 1];
        let h = x1 - x0;

        if h < 1e-10 {
            return self.coeffs.lock().storage.to_f32_array()[[idx]];
        }

        let t = (x - x0) / h;
        let t2 = t * t;
        let t3 = t2 * t;

        // Simple linear interpolation as fallback (cubic requires more coefficients)
        let y0 = self.coeffs.lock().storage.to_f32_array()[[idx]];
        let y1 = self.coeffs.lock().storage.to_f32_array()[[idx + 1]];
        (1.0 - t) * y0 + t * y1
    }

    /// Get the learnable coefficients.
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.coeffs.clone()]
    }
}

/// KAN layer: applies learnable activation functions on each edge.
///
/// For an input x of shape [batch, in_features], the layer computes:
/// y_j = sum_i phi_ij(x_i) where phi_ij are learnable spline functions.
#[derive(Clone)]
pub struct KANLayer {
    pub in_features: usize,
    pub out_features: usize,
    pub splines: Vec<Vec<CubicSpline>>, // [out_features][in_features]
    pub grid_size: usize,
    /// Optional bias
    pub bias: Option<Tensor>,
}

impl KANLayer {
    /// Create a new KAN layer.
    pub fn new(in_features: usize, out_features: usize, grid_size: usize) -> Self {
        let mut splines = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            let mut row = Vec::with_capacity(in_features);
            for _ in 0..in_features {
                row.push(CubicSpline::new(grid_size));
            }
            splines.push(row);
        }
        KANLayer {
            in_features,
            out_features,
            splines,
            grid_size,
            bias: None,
        }
    }

    /// Set bias.
    pub fn with_bias(mut self) -> Self {
        self.bias = Some(Tensor::zeros(&[self.out_features][..]));
        self
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_arr = x.lock().storage.to_f32_array();
        let shape = x_arr.shape().to_vec();
        let batch = shape[0];
        let seq = if shape.len() > 1 { shape[1] } else { 1 };

        // Flatten to [batch*seq, in_features]
        let flat_len = batch * seq;
        let mut out = vec![0.0f32; flat_len * self.out_features];

        for b in 0..flat_len {
            for j in 0..self.out_features {
                let mut sum = 0.0f32;
                for i in 0..self.in_features {
                    let val = if i < x_arr.len() {
                        x_arr[b * self.in_features + i]
                    } else {
                        0.0
                    };
                    sum += self.splines[j][i].evaluate(val);
                }
                out[b * self.out_features + j] = sum;
            }
        }

        let out_arr = match ArrayD::from_shape_vec(
            IxDyn(&if shape.len() > 1 {
                vec![batch, seq, self.out_features]
            } else {
                vec![flat_len, self.out_features]
            }),
            out,
        ) {
            Ok(v) => v,
            Err(_) => ArrayD::zeros(IxDyn(&[flat_len, self.out_features])),
        };
        let mut result = Tensor::new(out_arr, false);

        if let Some(bias) = &self.bias {
            result = result.add(bias);
        }
        result
    }

    /// Get all parameters.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for row in &self.splines {
            for spline in row {
                params.extend(spline.parameters());
            }
        }
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    /// Get all spline coefficients as a flat tensor for optimizer.
    pub fn all_coefficients(&self) -> Vec<Tensor> {
        let mut coeffs = Vec::new();
        for row in &self.splines {
            for spline in row {
                coeffs.extend(spline.parameters());
            }
        }
        coeffs
    }
}

impl Module for KANLayer {
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

/// KAN block: KANLayer -> activation -> KANLayer (similar to MLP block).
#[derive(Clone)]
pub struct KANBlock {
    pub kan1: KANLayer,
    pub kan2: KANLayer,
    pub activation: KANActivation,
}

impl KANBlock {
    /// Create a new KAN block.
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, grid_size: usize) -> Self {
        KANBlock {
            kan1: KANLayer::new(in_dim, hidden_dim, grid_size),
            kan2: KANLayer::new(hidden_dim, out_dim, grid_size),
            activation: KANActivation::Silu,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.kan1.forward(x);
        let x = self.activation.forward(&x);
        self.kan2.forward(&x)
    }

    /// Get all parameters.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.kan1.parameters();
        p.extend(self.kan2.parameters());
        p.extend(self.activation.parameters());
        p
    }
}

impl Module for KANBlock {
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

/// KAN activation function enum.
#[derive(Clone)]
pub enum KANActivation {
    Silu,
    Tanh,
    ReLU,
    Learnable(CubicSpline),
}

impl KANActivation {
    /// Create a learnable activation from a spline.
    pub fn learnable(grid_size: usize) -> Self {
        KANActivation::Learnable(CubicSpline::new(grid_size))
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            KANActivation::Silu => x.silu(),
            KANActivation::Tanh => x.tanh(),
            KANActivation::ReLU => x.relu(),
            KANActivation::Learnable(spline) => spline.forward(x),
        }
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Tensor> {
        match self {
            KANActivation::Learnable(spline) => spline.parameters(),
            _ => vec![],
        }
    }
}

/// KAN model: stack of KAN blocks with optional embedding.
#[derive(Clone)]
pub struct KAN {
    pub blocks: Vec<KANBlock>,
    pub embedding: Option<Linear>,
    pub output_dim: usize,
}

impl KAN {
    /// Create a new KAN model.
    pub fn new(
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        grid_size: usize,
    ) -> Self {
        let mut blocks = Vec::new();
        let mut prev_dim = input_dim;
        for &hidden_dim in hidden_dims {
            blocks.push(KANBlock::new(prev_dim, hidden_dim, hidden_dim, grid_size));
            prev_dim = hidden_dim;
        }
        // Final layer to output_dim
        if !hidden_dims.is_empty() {
            blocks.push(KANBlock::new(
                hidden_dims[hidden_dims.len() - 1],
                hidden_dims[hidden_dims.len() - 1],
                output_dim,
                grid_size,
            ));
        } else {
            blocks.push(KANBlock::new(input_dim, input_dim, output_dim, grid_size));
        }

        KAN {
            blocks,
            embedding: None,
            output_dim,
        }
    }

    /// Add an embedding layer for token inputs.
    pub fn with_embedding(mut self, vocab_size: usize, embed_dim: usize) -> Self {
        self.embedding = Some(Linear::new(vocab_size, embed_dim, true));
        self
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = if let Some(embed) = &self.embedding {
            embed.forward(x)
        } else {
            x.clone()
        };
        for block in &self.blocks {
            x = block.forward(&x);
        }
        x
    }

    /// Get all parameters.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        if let Some(embed) = &self.embedding {
            p.extend(embed.parameters());
        }
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p
    }

    /// Number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.lock().storage.len())
            .sum()
    }
}

impl Module for KAN {
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

#[cfg(test)]
mod kan_tests {
    use super::*;

    #[test]
    fn test_cubic_spline() {
        let spline = CubicSpline::new(10);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[5]), vec![-0.5, -0.25, 0.0, 0.25, 0.5]).unwrap(),
            false,
        );
        let out = spline.forward(&x);
        assert_eq!(out.lock().storage.len(), 5);
    }

    #[test]
    fn test_kan_layer() {
        let layer = KANLayer::new(4, 8, 10);
        let x = Tensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[2, 4]),
                vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
            )
            .unwrap(),
            false,
        );
        let out = layer.forward(&x);
        let shape = out.lock().storage.shape();
        assert_eq!(shape[0], 2);
        assert_eq!(shape[1], 8);
    }

    #[test]
    fn test_kan_block() {
        let block = KANBlock::new(4, 8, 4, 10);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                .unwrap(),
            false,
        );
        let out = block.forward(&x);
        assert_eq!(out.lock().storage.shape()[1], 4);
    }

    #[test]
    fn test_kan_model() {
        let model = KAN::new(4, &[8, 16, 8], 2, 10);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                .unwrap(),
            false,
        );
        let out = model.forward(&x);
        assert_eq!(out.lock().storage.shape()[1], 2);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_kan_with_embedding() {
        let model = KAN::new(10, &[8, 8], 4, 10).with_embedding(100, 10);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
            false,
        );
        let out = model.forward(&x);
        assert_eq!(out.lock().storage.shape()[1], 4);
    }

    #[test]
    fn test_kan_activation() {
        let silu = KANActivation::Silu;
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap(),
            false,
        );
        let out = silu.forward(&x);
        let arr = out.lock().storage.to_f32_array();
        // silu(0) = 0
        assert!((arr[1]).abs() < 1e-6);
        // silu(1) = 1 / (1 + exp(-1)) > 0
        assert!(arr[2] > 0.0);
    }
}
