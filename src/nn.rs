use crate::labels::Labels;
use crate::ops::Conv2D as Conv2DOp;
use crate::ops::MaxPool2D as MaxPool2DOp;
use crate::tensor::Tensor;
use ndarray::{arr0, ArrayD, IxDyn};
use std::collections::HashMap;
use std::sync::Arc;

/// A trait for neural network modules.
pub trait Module {
    /// Performs a forward pass through the module.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Returns the parameters of the module.
    fn parameters(&self) -> Vec<Tensor>;
}

/// A linear (fully connected) layer.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Creates a new linear layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` - The number of input features.
    /// * `out_features` - The number of output features.
    /// * `bias` - Whether to include a bias term.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight_data = ArrayD::zeros(IxDyn(&[in_features, out_features]));
        let weight = Tensor::new(weight_data, true);

        let bias = if bias {
            let bias_data = ArrayD::zeros(IxDyn(&[out_features]));
            Some(Tensor::new(bias_data, true))
        } else {
            None
        };

        Linear { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let output = input.matmul(&self.weight);
        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }
}

/// A sequential container for modules.
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

/// Layer normalization module
///
/// - `gamma`: per-feature learnable gain; shape should be `[num_features]` or broadcastable to the normalized axis.
/// - `beta`: per-feature learnable bias; shape should be `[num_features]` or broadcastable to the normalized axis.
/// - `axis`: the axis (dimension index) to normalize over. Use `1` for per-row norm in standard 2D inputs or `-1` for last axis semantics.
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub axis: usize,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(num_features: usize, axis: usize, eps: f32) -> Self {
        let gamma = Tensor::new(
            ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[num_features]),
                vec![1.0; num_features],
            )
            .unwrap(),
            true,
        );
        let beta = Tensor::new(
            ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[num_features]),
                vec![0.0; num_features],
            )
            .unwrap(),
            true,
        );
        LayerNorm {
            gamma,
            beta,
            axis,
            eps,
        }
    }

    /// Inherent forward method to simplify usage in tests and API consumers.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        input.layer_norm(self.axis, self.eps, &self.gamma, &self.beta)
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.layer_norm(self.axis, self.eps, &self.gamma, &self.beta)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

impl Sequential {
    /// Creates a new sequential container.
    pub fn new() -> Self {
        Sequential {
            modules: Vec::new(),
        }
    }

    /// Adds a module to the container.
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    /// Returns all parameters from all modules.
    pub fn parameters(&self) -> Vec<Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut output = input.clone();
        for module in &self.modules {
            output = module.forward(&output);
        }
        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }
}

/// A trait for optimizers.
pub trait Optimizer {
    /// Performs a single optimization step.
    fn step(&mut self, parameters: &[Tensor]);

    /// Sets the gradients of all parameters to zero.
    fn zero_grad(&mut self, parameters: &[Tensor]);
}

/// Stochastic Gradient Descent optimizer.
pub struct SGD {
    lr: f32,
    momentum: f32,
    velocity: HashMap<Tensor, ArrayD<f32>>,
}

impl SGD {
    /// Creates a new SGD optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - The learning rate.
    /// * `momentum` - The momentum factor.
    pub fn new(lr: f32, momentum: f32) -> Self {
        SGD {
            lr,
            momentum,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut param_lock = param.lock();
            if let Some(grad) = &param_lock.grad {
                let velocity = self
                    .velocity
                    .entry(param.clone())
                    .or_insert_with(|| ArrayD::zeros(grad.dim()));
                *velocity = &*velocity * self.momentum + grad * (1.0 - self.momentum);
                let update = velocity.mapv(|v| v * self.lr);
                param_lock.data = &param_lock.data - &update;
            }
        }
    }

    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut param_lock = param.lock();
            param_lock.grad = None;
        }
    }
}

/// Adam optimizer.
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    m: HashMap<Tensor, ArrayD<f32>>,
    v: HashMap<Tensor, ArrayD<f32>>,
}

impl Adam {
    /// Creates a new Adam optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - The learning rate.
    /// * `beta1` - The exponential decay rate for the first moment estimates.
    /// * `beta2` - The exponential decay rate for the second moment estimates.
    /// * `eps` - A small constant for numerical stability.
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Adam {
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &[Tensor]) {
        self.t += 1;

        for param in parameters {
            let mut param_lock = param.lock();
            if let Some(grad) = &param_lock.grad {
                let m = self
                    .m
                    .entry(param.clone())
                    .or_insert_with(|| ArrayD::zeros(grad.dim()));
                let v = self
                    .v
                    .entry(param.clone())
                    .or_insert_with(|| ArrayD::zeros(grad.dim()));

                *m = &*m * self.beta1 + grad * (1.0 - self.beta1);
                *v = &*v * self.beta2 + &(grad * grad) * (1.0 - self.beta2);

                let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));

                let update = (m_hat / (v_hat.mapv(|x| x.sqrt()) + self.eps)) * self.lr;
                param_lock.data = &param_lock.data - &update;
            }
        }
    }

    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut param_lock = param.lock();
            param_lock.grad = None;
        }
    }
}

/// MaxPool2D layer.
pub struct MaxPool2D {
    kernel_size: usize,
    stride: usize,
}

impl MaxPool2D {
    /// Creates a new MaxPool2D layer.
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        MaxPool2D {
            kernel_size,
            stride,
        }
    }
}

impl Module for MaxPool2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(MaxPool2DOp {
                kernel_size: self.kernel_size,
                stride: self.stride,
            }),
            &[input.clone()],
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// 2D convolution layer (NCHW)
pub struct Conv2D {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl Conv2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let weight_data = ndarray::Array::zeros(IxDyn(&[
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        ]));
        let weight = Tensor::new(weight_data, true);
        let bias = if bias {
            let bias_data = ndarray::Array::zeros(IxDyn(&[out_channels]));
            Some(Tensor::new(bias_data, true))
        } else {
            None
        };
        Conv2D {
            weight,
            bias,
            stride,
            padding,
        }
    }
}

impl Module for Conv2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut inputs = vec![input.clone(), self.weight.clone()];
        if let Some(b) = &self.bias {
            inputs.push(b.clone());
        }
        Tensor::apply(Arc::new(Conv2DOp::new(self.stride, self.padding)), &inputs)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

/// Dropout layer.
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32, training: bool) -> Self {
        Dropout { p, training }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(crate::ops::Dropout::new(self.p, self.training)),
            &[input.clone()],
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// MSE Loss.
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        MSELoss
    }

    pub fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        pred.sub(target).pow(2.0).mean()
    }
}

/// Cross Entropy Loss (simplified).
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss
    }

    pub fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        // Expect pred to be probabilities (softmax applied) and target as one-hot vectors.
        // Loss = - 1/N * sum(target * log(pred)) where N is the number of samples
        let logp = pred.log();
        let tlogp = target.mul(&logp);
        let total = tlogp.sum();
        let n_samples = pred.lock().data.shape()[0] as f32;
        let neg_factor = Tensor::new(arr0(-1.0 / n_samples).into_dyn(), false);
        total.mul(&neg_factor)
    }
}

/// Cross entropy loss layer that accepts logits and labels/indexes or one-hot vectors
pub struct CrossEntropyLogitsLoss;

impl CrossEntropyLogitsLoss {
    pub fn new() -> Self {
        CrossEntropyLogitsLoss
    }
    pub fn forward(&self, logits: &Tensor, targets: &Tensor, axis: isize) -> Tensor {
        logits.softmax_cross_entropy_with_logits(targets, axis)
    }
    pub fn forward_from_labels(&self, logits: &Tensor, labels: &Labels, axis: isize) -> Tensor {
        let num_classes = logits.lock().data.shape()[axis as usize];
        let one_hot = labels.to_one_hot(num_classes);
        let t = Tensor::new(one_hot, false);
        logits.softmax_cross_entropy_with_logits(&t, axis)
    }
}

/// Negative Log Likelihood (NLLLoss) wrapper expecting log-probabilities and integer labels (as floats) or one-hot vectors
pub struct NLLLossLayer;

impl NLLLossLayer {
    pub fn new() -> Self {
        NLLLossLayer
    }
    pub fn forward(&self, log_probs: &Tensor, targets: &Tensor) -> Tensor {
        log_probs.nll_loss(targets)
    }
    pub fn forward_from_labels(&self, log_probs: &Tensor, labels: &Labels) -> Tensor {
        let num_classes = log_probs.lock().data.shape()[1];
        let one_hot = labels.to_one_hot(num_classes);
        let t = Tensor::new(one_hot, false);
        log_probs.nll_loss(&t)
    }
}

/// Simple DataLoader.
pub struct DataLoader {
    data: Vec<(Tensor, Tensor)>,
    batch_size: usize,
    index: usize,
}

impl DataLoader {
    pub fn new(data: Vec<(Tensor, Tensor)>, batch_size: usize) -> Self {
        DataLoader {
            data,
            batch_size,
            index: 0,
        }
    }

    /// Shuffle the dataset in-place.
    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.data.shuffle(&mut rng);
        self.reset();
    }

    pub fn next_batch(&mut self) -> Option<(Tensor, Tensor)> {
        if self.index >= self.data.len() {
            return None;
        }
        let end = (self.index + self.batch_size).min(self.data.len());
        let batch_x: Vec<Tensor> = self.data[self.index..end]
            .iter()
            .map(|(x, _)| x.clone())
            .collect();
        let batch_y: Vec<Tensor> = self.data[self.index..end]
            .iter()
            .map(|(_, y)| y.clone())
            .collect();
        self.index = end;
        // Assume stack works
        let bx = Tensor::stack(&batch_x, 0);
        let by = Tensor::stack(&batch_y, 0);
        Some((bx, by))
    }

    pub fn reset(&mut self) {
        self.index = 0;
    }
}
