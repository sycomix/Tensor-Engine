use crate::tensor::Tensor;
use ndarray::ArrayD;

/// A trait for optimizers.
pub trait Optimizer {
    /// Performs a single optimization step.
    fn step(&mut self);

    /// Sets the gradients of all parameters to zero.
    fn zero_grad(&self);
}

/// Stochastic Gradient Descent (SGD) optimizer.
pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    velocities: Vec<Option<ArrayD<f32>>>,
}

impl SGD {
    /// Creates a new SGD optimizer.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to optimize.
    /// * `lr` - The learning rate.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let len = params.len();
        SGD {
            params,
            lr,
            momentum: 0.0,
            velocities: vec![None; len],
        }
    }

    /// Sets the momentum factor.
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let grad_clone = {
                let lock = param.lock();
                lock.grad.as_ref().map(|g| g.clone())
            };
            if let Some(grad) = grad_clone {
                let mut update = grad.clone();
                if self.momentum != 0.0 {
                    if let Some(v) = &self.velocities[i] {
                        // v = momentum * v + grad
                        let mut new_v = v.clone();
                        new_v *= self.momentum;
                        new_v += &grad;
                        update = new_v.clone();
                        self.velocities[i] = Some(new_v);
                    } else {
                        self.velocities[i] = Some(grad.clone());
                    }
                }

                // param = param - lr * update
                // MVP: storage stored as f32 array regardless of dtype
                let mut lock = param.lock();
                match &mut lock.storage {
                    crate::dtype::TensorStorage::F32(arr) => {
                        // arr -= lr * update
                        // ndarray supports this: arr - (lr * update)
                        // But we want in-place mutation ideally.
                        // zip iteration or scaled_add would be best.
                        // To avoid unwrap/zip complexity, simple explicit loop:
                        arr.zip_mut_with(&update, |p, g| *p -= self.lr * *g);
                    }
                    // For quantized/other storages, we'd need to dequantize, update, re-quantize.
                    // For now, we assume training happens on F32 weights or emulated types.
                    _ => {
                        // Fallback: convert to f32, update, convert back.
                        let mut arr = lock.storage.to_f32_array();
                        arr.zip_mut_with(&update, |p, g| *p -= self.lr * *g);
                        lock.storage =
                            crate::dtype::TensorStorage::from_f32_array(&arr, lock.dtype);
                    }
                }
            }
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

/// Adagrad optimizer.
///
/// Adaptive learning rate optimizer that maintains a per-parameter cumulative
/// sum of squared gradients. Parameters with large gradients receive smaller
/// effective learning rates, and vice versa.
///
/// # Reference
/// [`Duchi, J., Hazan, E., & Singer, 2011`](https://jmlr.org/papers/v12/duchi11a.html)
pub struct Adagrad {
    params: Vec<Tensor>,
    lr: f32,
    eps: f32,
    accumulators: Vec<Option<ArrayD<f32>>>,
}

impl Adagrad {
    /// Creates a new Adagrad optimizer.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to optimize.
    /// * `lr` - The learning rate.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let len = params.len();
        Adagrad {
            params,
            lr,
            eps: 1e-10,
            accumulators: vec![None; len],
        }
    }

    /// Sets the epsilon term for numerical stability.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let grad_clone = {
                let lock = param.lock();
                lock.grad.as_ref().map(|g| g.clone())
            };
            if let Some(grad) = grad_clone {
                // Initialize accumulator if needed
                if self.accumulators[i].is_none() {
                    self.accumulators[i] = Some(ArrayD::zeros(grad.dim()));
                }

                let acc = self.accumulators[i].as_mut().unwrap();

                // acc += grad^2
                acc.zip_mut_with(&grad, |a, g| *a += g * g);

                // Update parameters: theta = theta - lr * grad / (sqrt(acc) + eps)
                let mut lock = param.lock();
                match &mut lock.storage {
                    crate::dtype::TensorStorage::F32(arr) => {
                        ndarray::Zip::from(arr).and(&grad).and(acc).for_each(|theta, g, a| {
                            *theta -= self.lr * g / (a.sqrt() + self.eps);
                        });
                    }
                    _ => {
                        let mut arr = lock.storage.to_f32_array();
                        ndarray::Zip::from(&mut arr)
                            .and(&grad)
                            .and(acc)
                            .for_each(|theta, g, a| {
                                *theta -= self.lr * g / (a.sqrt() + self.eps);
                            });
                        lock.storage =
                            crate::dtype::TensorStorage::from_f32_array(&arr, lock.dtype);
                    }
                }
            }
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

/// Lion optimizer.
///
/// Sign-based adaptive optimizer that uses the sign of the momentum estimate
/// rather than the momentum itself. This can provide better generalization
/// and stability compared to Adam in certain settings.
///
/// # Reference
/// [`Chen, et al. 2023`](https://arxiv.org/abs/2302.06675)
pub struct Lion {
    params: Vec<Tensor>,
    lr: f32,
    beta: f32,
    momentums: Vec<Option<ArrayD<f32>>>,
}

impl Lion {
    /// Creates a new Lion optimizer.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to optimize.
    /// * `lr` - The learning rate.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let len = params.len();
        Lion {
            params,
            lr,
            beta: 0.9,
            momentums: vec![None; len],
        }
    }

    /// Sets the momentum factor.
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }
}

impl Optimizer for Lion {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let grad_clone = {
                let lock = param.lock();
                lock.grad.as_ref().map(|g| g.clone())
            };
            if let Some(grad) = grad_clone {
                // Initialize momentum if needed
                if self.momentums[i].is_none() {
                    self.momentums[i] = Some(ArrayD::zeros(grad.dim()));
                }

                let m_prev = self.momentums[i].as_ref().unwrap();

                // Update momentum: m_t = beta * m_{t-1} + grad
                let mut m_t = m_prev.clone();
                m_t.mapv_inplace(|x| x * self.beta);
                m_t.zip_mut_with(&grad, |m, g| *m += g);

                self.momentums[i] = Some(m_t.clone());

                // Update parameters: theta = theta - lr * sign(m_t)
                let mut lock = param.lock();
                match &mut lock.storage {
                    crate::dtype::TensorStorage::F32(arr) => {
                        ndarray::Zip::from(arr).and(&m_t).for_each(|theta, m| {
                            *theta -= self.lr * m.signum();
                        });
                    }
                    _ => {
                        let mut arr = lock.storage.to_f32_array();
                        ndarray::Zip::from(&mut arr).and(&m_t).for_each(|theta, m| {
                            *theta -= self.lr * m.signum();
                        });
                        lock.storage =
                            crate::dtype::TensorStorage::from_f32_array(&arr, lock.dtype);
                    }
                }
            }
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}
/// AdamW optimizer (Adam with decoupled weight decay).
///
/// Unlike Adam, weight decay is applied directly to parameters rather than
/// being absorbed into the adaptive learning rate. This provides better
/// regularization properties.
///
/// # Reference
/// [`Loshchilov, Hutter 2019`](https://arxiv.org/abs/1711.05101)
pub struct AdamW {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Option<ArrayD<f32>>>,
    v: Vec<Option<ArrayD<f32>>>,
    t: usize,
}

impl AdamW {
    /// Creates a new AdamW optimizer.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to optimize.
    /// * `lr` - The learning rate.
    /// * `weight_decay` - L2 regularization coefficient (default 0.01).
    pub fn new(params: Vec<Tensor>, lr: f32, weight_decay: f32) -> Self {
        let len = params.len();
        AdamW {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            m: vec![None; len],
            v: vec![None; len],
            t: 0,
        }
    }

    /// Sets the beta1 parameter.
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Sets the beta2 parameter.
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Sets the epsilon parameter.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the weight decay coefficient.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        self.t += 1;
        let t = self.t as f32;

        for (i, param) in self.params.iter().enumerate() {
            let grad_clone = {
                let lock = param.lock();
                lock.grad.as_ref().map(|g| g.clone())
            };
            if let Some(grad) = grad_clone {
                // Initialize state if needed
                if self.m[i].is_none() {
                    self.m[i] = Some(ArrayD::zeros(grad.dim()));
                    self.v[i] = Some(ArrayD::zeros(grad.dim()));
                }

                let m_prev = self.m[i].as_ref().unwrap();
                let v_prev = self.v[i].as_ref().unwrap();

                // Update biased first moment estimate
                let mut m_t = m_prev.clone();
                m_t.mapv_inplace(|x| x * self.beta1);
                m_t.zip_mut_with(&grad, |m, g| *m += (1.0 - self.beta1) * g);

                // Update biased second raw moment estimate
                let mut v_t = v_prev.clone();
                v_t.mapv_inplace(|x| x * self.beta2);
                v_t.zip_mut_with(&grad, |v, g| *v += (1.0 - self.beta2) * g * g);

                self.m[i] = Some(m_t.clone());
                self.v[i] = Some(v_t.clone());

                // Compute bias-corrected estimates
                let bias_correction1 = 1.0 - self.beta1.powf(t);
                let m_hat = m_t.mapv(|x| x / bias_correction1);

                let bias_correction2 = 1.0 - self.beta2.powf(t);
                let v_hat = v_t.mapv(|x| x / bias_correction2);

                // Update parameters with decoupled weight decay
                // theta = theta - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta)
                let mut lock = param.lock();
                match &mut lock.storage {
                    crate::dtype::TensorStorage::F32(arr) => {
                        ndarray::Zip::from(arr).and(&m_hat).and(&v_hat).for_each(
                            |theta, mh, vh| {
                                *theta -= self.lr * (mh / (vh.sqrt() + self.eps) + self.weight_decay * *theta);
                            },
                        );
                    }
                    _ => {
                        let mut arr = lock.storage.to_f32_array();
                        ndarray::Zip::from(&mut arr)
                            .and(&m_hat)
                            .and(&v_hat)
                            .for_each(|theta, mh, vh| {
                                *theta -= self.lr * (mh / (vh.sqrt() + self.eps) + self.weight_decay * *theta);
                            });
                        lock.storage =
                            crate::dtype::TensorStorage::from_f32_array(&arr, lock.dtype);
                    }
                }
            }
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

/// RMSProp optimizer.
///
/// Maintains a running average of squared gradients and divides the gradient
/// by the square root of this average, similar to Adagrad but with exponential
/// moving average instead of cumulative sum.
///
/// # Reference
/// [`Hinton, 2012`](https://www.cs.toronto.edu/~hinton/coursera_lecture.html)
pub struct RMSProp {
    params: Vec<Tensor>,
    lr: f32,
    alpha: f32,
    eps: f32,
    accumulators: Vec<Option<ArrayD<f32>>>,
}

impl RMSProp {
    /// Creates a new RMSProp optimizer.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to optimize.
    /// * `lr` - The learning rate.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let len = params.len();
        RMSProp {
            params,
            lr,
            alpha: 0.99,
            eps: 1e-8,
            accumulators: vec![None; len],
        }
    }

    /// Sets the decay factor.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the epsilon term.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl Optimizer for RMSProp {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let mut lock = param.lock();
            if let Some(grad) = &lock.grad {
                // Initialize accumulator if needed
                if self.accumulators[i].is_none() {
                    self.accumulators[i] = Some(ArrayD::zeros(grad.dim()));
                }

                let acc = self.accumulators[i].as_mut().unwrap();

                // acc = alpha * acc + (1 - alpha) * grad^2
                acc.mapv_inplace(|x| x * self.alpha);
                acc.zip_mut_with(grad, |a, g| *a += (1.0 - self.alpha) * g * g);

                // Update parameters
                match &mut lock.storage {
                    crate::dtype::TensorStorage::F32(arr) => {
                        ndarray::Zip::from(arr).and(grad).and(acc).for_each(|theta, g, a| {
                            *theta -= self.lr * g / (a.sqrt() + self.eps);
                        });
                    }
                    _ => {
                        let mut arr = lock.storage.to_f32_array();
                        ndarray::Zip::from(&mut arr)
                            .and(grad)
                            .and(acc)
                            .for_each(|theta, g, a| {
                                *theta -= self.lr * g / (a.sqrt() + self.eps);
                            });
                        lock.storage =
                            crate::dtype::TensorStorage::from_f32_array(&arr, lock.dtype);
                    }
                }
            }
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}
