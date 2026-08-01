//! Training utilities and helpers for efficient model training.
//!
//! Provides gradient accumulation, mixed precision training helpers, and other
//! utilities to streamline the training loop.

// TensorError/TensorResult available if needed in future
use crate::tensor::Tensor;
use ndarray::ArrayD;
use std::collections::HashMap;

/// Gradient Accumulator for efficient training with effective larger batch sizes.
///
/// This utility accumulates gradients over multiple micro-batches before performing
/// an optimizer step, effectively simulating training with a larger batch size while
/// keeping memory usage low.
///
/// # Example
/// ```ignore
/// let mut accumulator = GradientAccumulator::new(4); // Accumulate over 4 micro-batches
/// for batch_idx in 0..4 {
///     let loss = model.forward(&batch);
///     loss.backward();
///     accumulator.accumulate(&model.parameters());
///     model.zero_grad();
/// }
/// optimizer.step(); // Update after accumulation
/// ```
#[derive(Clone)]
pub struct GradientAccumulator {
    accumulation_steps: usize,
    current_step: usize,
    accumulated_grads: HashMap<usize, ArrayD<f32>>,
    param_count: usize,
}

impl GradientAccumulator {
    /// Create a new gradient accumulator.
    ///
    /// # Arguments
    /// * `accumulation_steps` - Number of micro-batches to accumulate over
    pub fn new(accumulation_steps: usize) -> Self {
        assert!(accumulation_steps > 0, "accumulation_steps must be > 0");
        GradientAccumulator {
            accumulation_steps,
            current_step: 0,
            accumulated_grads: HashMap::new(),
            param_count: 0,
        }
    }

    /// Accumulate gradients from a set of parameters.
    ///
    /// This method sums the current gradients into the accumulator. After calling this,
    /// you should call `zero_grad()` on the parameters before the next forward pass.
    pub fn accumulate(&mut self, params: &[Tensor]) {
        for (idx, param) in params.iter().enumerate() {
            let grad = {
                let lock = param.lock();
                lock.grad.clone()
            };

            if let Some(g) = grad {
                self.accumulated_grads
                    .entry(idx)
                    .and_modify(|acc| {
                        *acc += &g;
                    })
                    .or_insert_with(|| g.clone());
            }
        }
        self.param_count = params.len();
        self.current_step += 1;
    }

    /// Check if accumulation is complete (reached accumulation_steps).
    pub fn is_accumulation_complete(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Get the current accumulation step (1-indexed).
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get the total accumulation steps.
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }

    /// Apply accumulated gradients to parameters and optionally average them.
    ///
    /// This transfers the accumulated gradients back to the parameter tensors.
    /// If `average` is true, gradients are divided by the accumulation count.
    pub fn apply_accumulated_grads(&mut self, params: &mut [Tensor], average: bool) {
        let divisor = if average {
            self.current_step as f32
        } else {
            1.0
        };

        for (idx, param) in params.iter_mut().enumerate() {
            if let Some(acc_grad) = self.accumulated_grads.get(&idx) {
                let grad_to_apply = if average {
                    acc_grad / divisor
                } else {
                    acc_grad.clone()
                };

                let mut lock = param.lock();
                lock.grad = Some(grad_to_apply);
            }
        }
    }

    /// Reset the accumulator for the next accumulation cycle.
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.accumulated_grads.clear();
    }

    /// Get accumulated gradients as a map (for inspection/logging).
    pub fn get_accumulated_grads(&self) -> &HashMap<usize, ArrayD<f32>> {
        &self.accumulated_grads
    }

    /// Get the effective batch size (original_batch_size * accumulation_steps).
    pub fn effective_batch_multiplier(&self) -> usize {
        self.accumulation_steps
    }
}

/// Mixed Precision Training Helper for automatic loss scaling.
///
/// Addresses gradient underflow in FP16 training by scaling gradients up before
/// the backward pass and down after optimizer steps. This prevents numerical
/// instability in low-precision arithmetic.
#[derive(Clone, Debug)]
pub struct MixedPrecisionScaler {
    loss_scale: f32,
    max_loss_scale: f32,
    scale_factor: f32,
    scale_window: usize,
    failed_steps: usize,
    successful_steps: usize,
}

impl MixedPrecisionScaler {
    /// Create a new mixed precision scaler.
    ///
    /// # Arguments
    /// * `initial_scale` - Initial loss scale (typically 65536 for FP16)
    /// * `max_scale` - Maximum allowed loss scale
    /// * `scale_factor` - Factor to multiply/divide loss scale by (typically 2.0)
    /// * `scale_window` - Number of successful steps before increasing scale
    pub fn new(initial_scale: f32, max_scale: f32, scale_factor: f32, scale_window: usize) -> Self {
        MixedPrecisionScaler {
            loss_scale: initial_scale,
            max_loss_scale: max_scale,
            scale_factor,
            scale_window,
            failed_steps: 0,
            successful_steps: 0,
        }
    }

    /// Scale loss for backward pass.
    pub fn scale_loss(&self, loss: &Tensor) -> Tensor {
        let scale_tensor = Tensor::from_scalar(self.loss_scale);
        loss.mul(&scale_tensor)
    }

    /// Unscale gradients after backward pass.
    pub fn unscale_grads(&mut self, params: &[Tensor]) {
        let divisor = 1.0 / self.loss_scale;
        for param in params {
            let mut lock = param.lock();
            if let Some(ref mut grad) = lock.grad {
                grad.mapv_inplace(|g| g * divisor);
            }
        }
    }

    /// Record a successful optimizer step and potentially increase loss scale.
    pub fn record_step(&mut self, overflow_occurred: bool) {
        if overflow_occurred {
            self.failed_steps += 1;
            self.successful_steps = 0;

            // Decrease loss scale on overflow
            self.loss_scale = (self.loss_scale / self.scale_factor).max(1.0);
        } else {
            self.successful_steps += 1;

            // Increase loss scale after N successful steps
            if self.successful_steps >= self.scale_window {
                self.loss_scale = (self.loss_scale * self.scale_factor).min(self.max_loss_scale);
                self.successful_steps = 0;
            }
        }
    }

    /// Get the current loss scale.
    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }

    /// Get the number of failed steps (overflows).
    pub fn failed_steps(&self) -> usize {
        self.failed_steps
    }

    /// Reset statistics (e.g., at end of epoch).
    pub fn reset_stats(&mut self) {
        self.failed_steps = 0;
        self.successful_steps = 0;
    }
}

/// Utility to detect gradient overflow in mixed precision training.
pub fn detect_grad_overflow(params: &[Tensor]) -> bool {
    for param in params {
        let lock = param.lock();
        if let Some(grad) = &lock.grad {
            for &val in grad.iter() {
                if !val.is_finite() {
                    return true;
                }
            }
        }
    }
    false
}

/// Utility to scale gradients by a constant factor.
///
/// Useful for gradient clipping or dynamic scaling.
pub fn scale_gradients(params: &mut [Tensor], scale_factor: f32) {
    for param in params {
        let mut lock = param.lock();
        if let Some(ref mut grad) = lock.grad {
            grad.mapv_inplace(|g| g * scale_factor);
        }
    }
}

/// Utility to compute gradient norms for logging and debugging.
pub fn compute_grad_norms(params: &[Tensor]) -> Vec<f32> {
    params
        .iter()
        .map(|param| {
            let lock = param.lock();
            if let Some(grad) = &lock.grad {
                let mut norm = 0.0f32;
                for &val in grad.iter() {
                    norm += val * val;
                }
                norm.sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod training_tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_gradient_accumulator_basic() {
        let accumulator = GradientAccumulator::new(4);
        assert_eq!(accumulator.accumulation_steps(), 4);
        assert_eq!(accumulator.current_step(), 0);
        assert!(!accumulator.is_accumulation_complete());
    }

    #[test]
    fn test_gradient_accumulator_completion() {
        let mut accumulator = GradientAccumulator::new(2);
        let params = vec![Tensor::ones(&[2, 3][..]), Tensor::ones(&[3, 4])];

        // Simulate gradients
        for _ in 0..2 {
            for param in &params {
                let mut lock = param.lock();
                lock.grad = Some(ArrayD::ones(lock.storage.shape().to_vec()));
            }
            accumulator.accumulate(&params);
        }

        assert!(accumulator.is_accumulation_complete());
        assert_eq!(accumulator.current_step(), 2);
    }

    #[test]
    fn test_gradient_accumulator_reset() {
        let mut accumulator = GradientAccumulator::new(4);
        accumulator.current_step = 2;
        accumulator.reset();

        assert_eq!(accumulator.current_step(), 0);
        assert!(accumulator.get_accumulated_grads().is_empty());
    }

    #[test]
    fn test_mixed_precision_scaler_creation() {
        let scaler = MixedPrecisionScaler::new(65536.0, 262144.0, 2.0, 2000);
        assert_eq!(scaler.loss_scale(), 65536.0);
        assert_eq!(scaler.failed_steps(), 0);
    }

    #[test]
    fn test_mixed_precision_scaler_overflow() {
        let mut scaler = MixedPrecisionScaler::new(65536.0, 262144.0, 2.0, 2000);
        scaler.record_step(true); // Simulate overflow

        assert_eq!(scaler.failed_steps(), 1);
        assert!(scaler.loss_scale() < 65536.0);
    }

    #[test]
    fn test_mixed_precision_scaler_success() {
        let mut scaler = MixedPrecisionScaler::new(65536.0, 262144.0, 2.0, 2);

        // Two successful steps should trigger scale increase
        scaler.record_step(false);
        scaler.record_step(false);

        assert!(scaler.loss_scale() > 65536.0);
    }

    #[test]
    fn test_gradient_overflow_detection() {
        let param = Tensor::ones(&[2, 3][..]);
        let mut lock = param.lock();
        lock.grad = Some(ArrayD::from_elem(
            ndarray::IxDyn(&[2, 3][..]),
            f32::INFINITY,
        ));
        drop(lock);

        assert!(detect_grad_overflow(&[param][..]));
    }

    #[test]
    fn test_scale_gradients() {
        let param = Tensor::ones(&[2, 3][..]);
        let mut lock = param.lock();
        lock.grad = Some(ArrayD::ones(ndarray::IxDyn(&[2, 3][..])));
        drop(lock);

        let mut params = vec![param.clone()];
        scale_gradients(&mut params, 0.5);

        let lock = param.lock();
        let grad = lock.grad.as_ref().unwrap();
        for &val in grad.iter() {
            assert!((val - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_grad_norms() {
        let param1 = Tensor::ones(&[2, 2][..]);
        let param2 = Tensor::ones(&[2, 2][..]);

        let mut lock1 = param1.lock();
        lock1.grad = Some(ArrayD::ones(ndarray::IxDyn(&[2, 2][..])));
        drop(lock1);

        let mut lock2 = param2.lock();
        lock2.grad = Some(ArrayD::from_elem(ndarray::IxDyn(&[2, 2][..]), 2.0));
        drop(lock2);

        let norms = compute_grad_norms(&[param1, param2][..]);
        assert_eq!(norms.len(), 2);
        assert!(norms[0] > 0.0);
        assert!(norms[1] > norms[0]); // Second param has larger gradients
    }

    #[test]
    fn test_gradient_accumulator_apply_accumulated_grads_average() {
        let mut acc = GradientAccumulator::new(2);

        let p1 = Tensor::ones(&[2, 2][..]);
        let p2 = Tensor::ones(&[2, 2][..]);

        for step in 0..2 {
            let scale = (step + 1) as f32;
            let mut lock1 = p1.lock();
            lock1.grad = Some(ArrayD::from_elem(ndarray::IxDyn(&[2, 2][..]), scale));
            drop(lock1);

            let mut lock2 = p2.lock();
            lock2.grad = Some(ArrayD::from_elem(ndarray::IxDyn(&[2, 2][..]), scale * 2.0));
            drop(lock2);

            acc.accumulate(&[p1.clone(), p2.clone()][..]);

            let params = vec![p1.clone(), p2.clone()];
            for p in &params {
                let mut lock = p.lock();
                lock.grad = None;
            }
        }

        let mut params = vec![p1.clone(), p2.clone()];
        acc.apply_accumulated_grads(&mut params, true);

        let lock1 = params[0].lock();
        let g1 = lock1.grad.as_ref().unwrap();
        let expected1 = (1.0 + 2.0) / 2.0;
        assert!((g1.iter().next().unwrap() - expected1).abs() < 1e-6);

        let lock2 = params[1].lock();
        let g2 = lock2.grad.as_ref().unwrap();
        let expected2 = (2.0 + 4.0) / 2.0;
        assert!((g2.iter().next().unwrap() - expected2).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator_apply_accumulated_grads_sum() {
        let mut acc = GradientAccumulator::new(2);

        let p1 = Tensor::ones(&[2, 2][..]);

        for step in 0..2 {
            let scale = (step + 1) as f32;
            let mut lock = p1.lock();
            lock.grad = Some(ArrayD::from_elem(ndarray::IxDyn(&[2, 2][..]), scale));
            drop(lock);
            acc.accumulate(&[p1.clone()][..]);

            let params = vec![p1.clone()];
            for p in &params {
                let mut lock = p.lock();
                lock.grad = None;
            }
        }

        let mut params = vec![p1.clone()];
        acc.apply_accumulated_grads(&mut params, false);

        let lock = params[0].lock();
        let g = lock.grad.as_ref().unwrap();
        let expected = 1.0 + 2.0;
        assert!((g.iter().next().unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator_effective_batch_multiplier() {
        let acc = GradientAccumulator::new(8);
        assert_eq!(acc.effective_batch_multiplier(), 8);
    }

    #[test]
    fn test_mixed_precision_scaler_scale_loss() {
        let scaler = MixedPrecisionScaler::new(65536.0, 262144.0, 2.0, 2000);
        let loss = Tensor::from_scalar(1.0);
        let scaled = scaler.scale_loss(&loss);
        let val = scaled
            .lock()
            .storage
            .to_f32_array()
            .into_raw_vec_and_offset()
            .0[0];
        assert!((val - 65536.0).abs() < 1.0);
    }

    #[test]
    fn test_mixed_precision_scaler_unscale_grads() {
        let mut scaler = MixedPrecisionScaler::new(100.0, 1000.0, 2.0, 2);

        let param = Tensor::ones(&[2, 2][..]);
        let mut lock = param.lock();
        lock.grad = Some(ArrayD::from_elem(ndarray::IxDyn(&[2, 2][..]), 200.0));
        drop(lock);

        scaler.unscale_grads(&[param.clone()][..]);

        let lock = param.lock();
        let g = lock.grad.as_ref().unwrap();
        assert!((g.iter().next().unwrap() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_mixed_precision_scaler_reset_stats() {
        let mut scaler = MixedPrecisionScaler::new(65536.0, 262144.0, 2.0, 2);
        scaler.record_step(true);
        scaler.record_step(true);
        assert_eq!(scaler.failed_steps(), 2);

        scaler.reset_stats();
        assert_eq!(scaler.failed_steps(), 0);
        assert_eq!(scaler.loss_scale(), 16384.0);
    }
}
