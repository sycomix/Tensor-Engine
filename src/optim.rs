use crate::dtype::DType;
use crate::tensor::Tensor;
use ndarray::ArrayD;
use std::collections::HashMap;

/// A trait for optimizers.
///
/// Parameters are passed at `step()` / `zero_grad()` time so the same
/// optimizer can be reused with different parameter sets (or the set can
/// grow/shrink between training stages).
pub trait Optimizer {
    /// Performs a single optimization step on the given parameters.
    fn step(&mut self, parameters: &[Tensor]);

    /// Sets the gradients of all given parameters to zero.
    fn zero_grad(&mut self, parameters: &[Tensor]);

    /// Clip gradients in-place using global norm.
    fn clip_gradients(&mut self, parameters: &[Tensor], max_norm: f32) {
        if max_norm <= 0.0 {
            return;
        }
        let mut total_sq = 0.0f32;
        for p in parameters {
            let lock = p.lock();
            if let Some(g) = &lock.grad {
                for v in g.iter() {
                    total_sq += (*v) * (*v);
                }
            }
        }
        let total_norm = total_sq.sqrt();
        if total_norm <= max_norm {
            return;
        }
        let scale = max_norm / (total_norm + 1e-12);
        for p in parameters {
            let mut lock = p.lock();
            if let Some(g) = &mut lock.grad {
                g.mapv_inplace(|v| v * scale);
            }
        }
    }

    /// Clip gradients in-place by absolute value.
    fn clip_grad_values(&mut self, parameters: &[Tensor], clip_value: f32) {
        if clip_value <= 0.0 || !clip_value.is_finite() {
            return;
        }
        let c = clip_value.abs();
        for p in parameters {
            let mut lock = p.lock();
            if let Some(g) = &mut lock.grad {
                g.mapv_inplace(|v| v.clamp(-c, c));
            }
        }
    }

    /// Scale gradients in-place by a constant factor.
    fn scale_gradients(&mut self, parameters: &[Tensor], scale: f32) {
        if !scale.is_finite() {
            return;
        }
        if (scale - 1.0).abs() <= f32::EPSILON {
            return;
        }
        for p in parameters {
            let mut lock = p.lock();
            if let Some(g) = &mut lock.grad {
                g.mapv_inplace(|v| v * scale);
            }
        }
    }

    /// Cast parameters to a storage dtype (round-trip conversion).
    fn cast_params(&mut self, parameters: &[Tensor], dtype: DType) {
        for p in parameters {
            let converted = p.astype(dtype);
            let mut lock = p.lock();
            lock.storage = converted.lock().storage.clone();
            lock.dtype = dtype;
        }
    }
}

/// Helper: apply an in-place update to a parameter's storage, preferring the
/// fast F32 path and falling back to a round-trip through `to_f32_array`.
fn update_storage<F>(param: &Tensor, f: F)
where
    F: FnOnce(&mut ArrayD<f32>),
{
    let mut lock = param.lock();
    match &mut lock.storage {
        crate::dtype::TensorStorage::F32(arr) => {
            f(arr);
        }
        _ => {
            let mut arr = lock.storage.to_f32_array();
            f(&mut arr);
            lock.storage = crate::dtype::TensorStorage::from_f32_array(&arr, lock.dtype);
        }
    }
}

/// Helper: read a gradient, handling both F32 and fallback paths.
fn read_grad(param: &Tensor) -> Option<ArrayD<f32>> {
    let lock = param.lock();
    lock.grad.clone()
}

// ────────────────────────────────────────────────────────────────────────────
// SGD
// ────────────────────────────────────────────────────────────────────────────

/// Stochastic Gradient Descent optimizer (with optional momentum).
pub struct SGD {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocities: HashMap<Tensor, ArrayD<f32>>,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32) -> Self {
        SGD {
            lr,
            momentum,
            weight_decay: 0.0,
            velocities: HashMap::new(),
        }
    }

    /// Creates a new SGD optimizer with decoupled weight decay.
    pub fn new_with_weight_decay(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        SGD {
            lr,
            momentum,
            weight_decay,
            velocities: HashMap::new(),
        }
    }

    pub fn lr(&self) -> f32 {
        self.lr
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let grad = match read_grad(param) {
                Some(g) => g,
                None => continue,
            };

            let velocity = self
                .velocities
                .entry(param.clone())
                .or_insert_with(|| ArrayD::zeros(grad.dim()));

            if self.momentum != 0.0 {
                *velocity = &*velocity * self.momentum + &grad;
            } else {
                *velocity = grad.clone();
            }

            let lr = self.lr;
            let wd = self.weight_decay * lr;
            update_storage(param, |arr| {
                if self.momentum != 0.0 {
                    arr.zip_mut_with(velocity, |p, v| *p -= lr * *v);
                } else {
                    arr.zip_mut_with(&grad, |p, g| *p -= lr * *g);
                }
                if wd != 0.0 && wd.is_finite() {
                    arr.mapv_inplace(|p| p - wd * p);
                }
            });
        }
    }

    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut lock = param.lock();
            lock.grad = None;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Adam
// ────────────────────────────────────────────────────────────────────────────

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

    pub fn lr(&self) -> f32 {
        self.lr
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &[Tensor]) {
        self.t += 1;
        let t = self.t as f32;
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);
        let lr = self.lr * bias_correction2.sqrt() / bias_correction1;

        for param in parameters {
            let grad = match read_grad(param) {
                Some(g) => g,
                None => continue,
            };

            let m = self
                .m
                .entry(param.clone())
                .or_insert_with(|| ArrayD::zeros(grad.dim()));
            let v = self
                .v
                .entry(param.clone())
                .or_insert_with(|| ArrayD::zeros(grad.dim()));

            *m = &*m * self.beta1 + &grad * (1.0 - self.beta1);
            *v = &*v * self.beta2 + &(&grad * &grad) * (1.0 - self.beta2);

            let eps = self.eps;
            let m_clone = m.clone();
            let v_clone = v.clone();
            update_storage(param, |arr| {
                ndarray::Zip::from(arr)
                    .and(&m_clone)
                    .and(&v_clone)
                    .for_each(|p, m_val, v_val| {
                        *p -= lr * m_val / (v_val.sqrt() + eps);
                    });
            });
        }
    }

    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut lock = param.lock();
            lock.grad = None;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AdamW
// ────────────────────────────────────────────────────────────────────────────

/// AdamW optimizer (Adam with decoupled weight decay).
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: usize,
    m: HashMap<Tensor, ArrayD<f32>>,
    v: HashMap<Tensor, ArrayD<f32>>,
}

impl AdamW {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        AdamW {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn lr(&self) -> f32 {
        self.lr
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, parameters: &[Tensor]) {
        self.t += 1;
        let t = self.t as f32;
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);

        for param in parameters {
            let grad = match read_grad(param) {
                Some(g) => g,
                None => continue,
            };

            let m = self
                .m
                .entry(param.clone())
                .or_insert_with(|| ArrayD::zeros(grad.dim()));
            let v = self
                .v
                .entry(param.clone())
                .or_insert_with(|| ArrayD::zeros(grad.dim()));

            *m = &*m * self.beta1 + &grad * (1.0 - self.beta1);
            *v = &*v * self.beta2 + &(&grad * &grad) * (1.0 - self.beta2);

            let m_hat = &*m / bias_correction1;
            let v_hat = &*v / bias_correction2;

            let lr = self.lr;
            let wd = self.weight_decay;
            let eps = self.eps;
            update_storage(param, |arr| {
                // theta = theta - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta)
                ndarray::Zip::from(arr)
                    .and(&m_hat)
                    .and(&v_hat)
                    .for_each(|theta, mh, vh| {
                        *theta -= lr * (mh / (vh.sqrt() + eps) + wd * *theta);
                    });
            });
        }
    }

    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut lock = param.lock();
            lock.grad = None;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// RMSProp
// ────────────────────────────────────────────────────────────────────────────

/// RMSProp optimizer.
pub struct RMSProp {
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    state: HashMap<Tensor, ArrayD<f32>>,
}

impl RMSProp {
    pub fn new(lr: f32, alpha: f32, eps: f32) -> Self {
        RMSProp {
            lr,
            alpha,
            eps,
            weight_decay: 0.0,
            state: HashMap::new(),
        }
    }

    /// Creates a new RMSProp optimizer with decoupled weight decay.
    pub fn new_with_weight_decay(lr: f32, alpha: f32, eps: f32, weight_decay: f32) -> Self {
        RMSProp {
            lr,
            alpha,
            eps,
            weight_decay,
            state: HashMap::new(),
        }
    }

    pub fn lr(&self) -> f32 {
        self.lr
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for RMSProp {
    fn step(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let grad = match read_grad(param) {
                Some(g) => g,
                None => continue,
            };

            let s = self
                .state
                .entry(param.clone())
                .or_insert_with(|| ArrayD::zeros(grad.dim()));

            *s = &*s * self.alpha + &(&grad * &grad) * (1.0 - self.alpha);

            let lr = self.lr;
            let eps = self.eps;
            let wd = self.weight_decay * lr;
            let s_clone = s.clone();
            update_storage(param, |arr| {
                ndarray::Zip::from(&mut *arr)
                    .and(&grad)
                    .and(&s_clone)
                    .for_each(|theta, g, s_val| {
                        *theta -= lr * g / (s_val.sqrt() + eps);
                    });
                if wd != 0.0 && wd.is_finite() {
                    arr.mapv_inplace(|p| p - wd * p);
                }
            });
        }
    }

    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut lock = param.lock();
            lock.grad = None;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Adagrad
// ────────────────────────────────────────────────────────────────────────────

/// Adagrad optimizer.
///
/// Maintains a per-parameter cumulative sum of squared gradients.
/// Parameters with large gradients receive smaller effective learning rates.
pub struct Adagrad {
    lr: f32,
    eps: f32,
    accumulators: HashMap<Tensor, ArrayD<f32>>,
}

impl Adagrad {
    pub fn new(lr: f32) -> Self {
        Adagrad {
            lr,
            eps: 1e-10,
            accumulators: HashMap::new(),
        }
    }

    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn lr(&self) -> f32 {
        self.lr
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let grad = match read_grad(param) {
                Some(g) => g,
                None => continue,
            };

            let acc = self
                .accumulators
                .entry(param.clone())
                .or_insert_with(|| ArrayD::zeros(grad.dim()));

            *acc = &*acc + &(&grad * &grad);

            let lr = self.lr;
            let eps = self.eps;
            update_storage(param, |arr| {
                ndarray::Zip::from(arr)
                    .and(&grad)
                    .and(acc)
                    .for_each(|theta, g, a| {
                        *theta -= lr * g / (a.sqrt() + eps);
                    });
            });
        }
    }

    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut lock = param.lock();
            lock.grad = None;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Lion
// ────────────────────────────────────────────────────────────────────────────

/// Lion optimizer (sign-based adaptive optimizer).
///
/// Uses the sign of the momentum estimate rather than the momentum
/// itself, which can provide better generalization.
pub struct Lion {
    lr: f32,
    beta: f32,
    momentums: HashMap<Tensor, ArrayD<f32>>,
}

impl Lion {
    pub fn new(lr: f32) -> Self {
        Lion {
            lr,
            beta: 0.9,
            momentums: HashMap::new(),
        }
    }

    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    pub fn lr(&self) -> f32 {
        self.lr
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for Lion {
    fn step(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let grad = match read_grad(param) {
                Some(g) => g,
                None => continue,
            };

            let m = self
                .momentums
                .entry(param.clone())
                .or_insert_with(|| ArrayD::zeros(grad.dim()));

            *m = &*m * self.beta + &grad;

            let lr = self.lr;
            update_storage(param, |arr| {
                ndarray::Zip::from(arr).and(m).for_each(|theta, m_val| {
                    *theta -= lr * m_val.signum();
                });
            });
        }
    }

    fn zero_grad(&mut self, parameters: &[Tensor]) {
        for param in parameters {
            let mut lock = param.lock();
            lock.grad = None;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn clip_gradients_scales_by_global_norm() {
        let p = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2][..]), vec![0.0, 0.0]).unwrap(),
            true,
        );
        {
            let mut lock = p.lock();
            lock.grad = Some(ArrayD::from_shape_vec(IxDyn(&[2][..]), vec![3.0, 4.0]).unwrap());
        }

        let mut opt = SGD::new(0.1, 0.0);
        opt.clip_gradients(std::slice::from_ref(&p), 1.0);

        let g = p.lock().grad.clone().unwrap();
        let gs = g.as_slice().unwrap();
        assert!((gs[0] - 0.6).abs() < 1e-6);
        assert!((gs[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn clip_grad_values_clamps_elementwise() {
        let p = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![0.0, 0.0, 0.0]).unwrap(),
            true,
        );
        {
            let mut lock = p.lock();
            lock.grad =
                Some(ArrayD::from_shape_vec(IxDyn(&[3][..]), vec![2.0, -3.0, 0.5]).unwrap());
        }

        let mut opt = SGD::new(0.1, 0.0);
        opt.clip_grad_values(std::slice::from_ref(&p), 1.0);

        let g = p.lock().grad.clone().unwrap();
        let gs = g.as_slice().unwrap();
        assert!((gs[0] - 1.0).abs() < 1e-6);
        assert!((gs[1] - (-1.0)).abs() < 1e-6);
        assert!((gs[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn adamw_applies_decoupled_weight_decay_when_grad_zero() {
        let p = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![1.0]).unwrap(),
            true,
        );
        {
            let mut lock = p.lock();
            lock.grad = Some(ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![0.0]).unwrap());
        }

        let mut opt = AdamW::new(0.1, 0.0, 0.0, 1e-8, 0.1);
        opt.step(std::slice::from_ref(&p));

        let v = p.to_f32_array();
        let s = v.as_slice().unwrap();
        assert!((s[0] - 0.99).abs() < 1e-6);
    }

    #[test]
    fn sgd_basic_step() {
        let p = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2][..]), vec![1.0, 2.0]).unwrap(),
            true,
        );
        {
            let mut lock = p.lock();
            lock.grad = Some(ArrayD::from_shape_vec(IxDyn(&[2][..]), vec![0.1, 0.2]).unwrap());
        }

        let mut opt = SGD::new(0.5, 0.0);
        opt.step(std::slice::from_ref(&p));

        let v = p.to_f32_array();
        let s = v.as_slice().unwrap();
        assert!((s[0] - 0.95).abs() < 1e-6);
        assert!((s[1] - 1.9).abs() < 1e-6);
    }

    #[test]
    fn sgd_with_momentum() {
        let p = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![1.0]).unwrap(),
            true,
        );
        {
            let mut lock = p.lock();
            lock.grad = Some(ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![1.0]).unwrap());
        }

        let mut opt = SGD::new(0.1, 0.9);
        // Step 1: v = 0*0.9 + 1 = 1, p = 1 - 0.1*1 = 0.9
        opt.step(std::slice::from_ref(&p));
        let v = p.to_f32_array();
        assert!((v.as_slice().unwrap()[0] - 0.9).abs() < 1e-6);

        // Step 2: grad is still 1.0, v = 1*0.9 + 1 = 1.9, p = 0.9 - 0.1*1.9 = 0.71
        opt.step(std::slice::from_ref(&p));
        let v2 = p.to_f32_array();
        assert!((v2.as_slice().unwrap()[0] - 0.71).abs() < 1e-6);
    }

    #[test]
    fn adam_basic_step() {
        let p = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![0.0]).unwrap(),
            true,
        );
        {
            let mut lock = p.lock();
            lock.grad = Some(ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![1.0]).unwrap());
        }

        let mut opt = Adam::new(0.1, 0.9, 0.999, 1e-8);
        opt.step(std::slice::from_ref(&p));

        let v = p.to_f32_array();
        // With lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8:
        // t=1 => m=0.1, v=0.001, m_hat=1.0, v_hat=1.0, lr_eff=0.1*1/1=0.1
        // update = 0.1*1.0/(1.0+1e-8) ≈ 0.1
        // p = 0 - 0.1 = -0.1
        let s = v.as_slice().unwrap();
        assert!((s[0] - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn adagrad_basic_step() {
        let p = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![1.0]).unwrap(),
            true,
        );
        {
            let mut lock = p.lock();
            lock.grad = Some(ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![0.5]).unwrap());
        }

        let mut opt = Adagrad::new(0.1);
        opt.step(std::slice::from_ref(&p));

        let v = p.to_f32_array();
        // acc = 0.25, update = 0.1 * 0.5 / (0.5 + 1e-10) = 0.1
        // p = 1.0 - 0.1 = 0.9
        assert!((v.as_slice().unwrap()[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn lion_basic_step() {
        let p = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![1.0]).unwrap(),
            true,
        );
        {
            let mut lock = p.lock();
            lock.grad = Some(ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![0.1]).unwrap());
        }

        let mut opt = Lion::new(0.01);
        opt.step(std::slice::from_ref(&p));

        let v = p.to_f32_array();
        // m = 0*0.9 + 0.1 = 0.1, sign(m) = 1.0
        // p = 1.0 - 0.01 * 1.0 = 0.99
        assert!((v.as_slice().unwrap()[0] - 0.99).abs() < 1e-6);
    }

    #[test]
    fn rmsprop_basic_step() {
        let p = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![1.0]).unwrap(),
            true,
        );
        {
            let mut lock = p.lock();
            lock.grad = Some(ArrayD::from_shape_vec(IxDyn(&[1][..]), vec![0.5]).unwrap());
        }

        let mut opt = RMSProp::new(0.1, 0.99, 1e-8);
        opt.step(std::slice::from_ref(&p));

        let v = p.to_f32_array();
        // s = 0*0.99 + 0.25*0.01 = 0.0025, sqrt=0.05
        // update = 0.1 * 0.5 / (0.05 + 1e-8) = 1.0
        // p = 1.0 - 1.0 = 0.0
        assert!((v.as_slice().unwrap()[0] - 0.0).abs() < 1e-6);
    }
}
