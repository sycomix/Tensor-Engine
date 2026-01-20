/// Learning rate schedulers for training optimization.
///
/// This module provides various learning rate scheduling strategies to improve
/// training convergence and stability.

/// Base trait for learning rate schedulers.
pub trait LRScheduler {
    /// Advance the scheduler by one step (typically one epoch).
    fn step(&mut self);

    /// Get the current learning rate.
    fn get_lr(&self) -> f32;

    /// Reset the scheduler to its initial state.
    fn reset(&mut self);
}

/// Exponential learning rate decay scheduler.
///
/// Reduces the learning rate by multiplying it by a constant factor (gamma) at each step.
///
/// Formula: `lr_t = lr_0 * gamma^t`
///
/// # Example
/// ```
/// use tensor_engine::lr_scheduler::{LRScheduler, ExponentialLR};
///
/// let mut scheduler = ExponentialLR::new(0.1, 0.95);
/// assert_eq!(scheduler.get_lr(), 0.1);
/// scheduler.step();
/// assert!((scheduler.get_lr() - 0.095).abs() < 1e-6);
/// ```
pub struct ExponentialLR {
    initial_lr: f32,
    gamma: f32,
    current_epoch: usize,
}

impl ExponentialLR {
    /// Creates a new exponential learning rate scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - The initial learning rate (must be > 0)
    /// * `gamma` - The multiplicative decay factor (typically 0 < gamma < 1)
    ///
    /// # Panics
    ///
    /// Panics if `initial_lr <= 0` or `gamma <= 0`.
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        assert!(initial_lr > 0.0, "initial_lr must be positive");
        assert!(gamma > 0.0, "gamma must be positive");
        ExponentialLR {
            initial_lr,
            gamma,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self) {
        self.current_epoch += 1;
    }

    fn get_lr(&self) -> f32 {
        self.initial_lr * self.gamma.powi(self.current_epoch as i32)
    }

    fn reset(&mut self) {
        self.current_epoch = 0;
    }
}

/// Step learning rate decay scheduler.
///
/// Reduces the learning rate by a factor (gamma) every `step_size` epochs.
///
/// Formula: `lr_t = lr_0 * gamma^(floor(t / step_size))`
///
/// # Example
/// ```
/// use tensor_engine::lr_scheduler::{LRScheduler, StepLR};
///
/// let mut scheduler = StepLR::new(0.1, 0.1, 10);
/// assert_eq!(scheduler.get_lr(), 0.1);
/// for _ in 0..9 {
///     scheduler.step();
/// }
/// assert_eq!(scheduler.get_lr(), 0.1); // Still at initial LR
/// scheduler.step(); // 10th step
/// assert!((scheduler.get_lr() - 0.01).abs() < 1e-6); // Decayed
/// ```
pub struct StepLR {
    initial_lr: f32,
    gamma: f32,
    step_size: usize,
    current_epoch: usize,
}

impl StepLR {
    /// Creates a new step learning rate scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - The initial learning rate (must be > 0)
    /// * `gamma` - The multiplicative decay factor (typically 0 < gamma < 1)
    /// * `step_size` - Number of epochs between each decay (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if `initial_lr <= 0`, `gamma <= 0`, or `step_size == 0`.
    pub fn new(initial_lr: f32, gamma: f32, step_size: usize) -> Self {
        assert!(initial_lr > 0.0, "initial_lr must be positive");
        assert!(gamma > 0.0, "gamma must be positive");
        assert!(step_size > 0, "step_size must be positive");
        StepLR {
            initial_lr,
            gamma,
            step_size,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self) {
        self.current_epoch += 1;
    }

    fn get_lr(&self) -> f32 {
        let steps = self.current_epoch / self.step_size;
        self.initial_lr * self.gamma.powi(steps as i32)
    }

    fn reset(&mut self) {
        self.current_epoch = 0;
    }
}

/// Polynomial learning rate decay scheduler.
///
/// Reduces the learning rate using a polynomial decay function.
///
/// Formula: `lr_t = (lr_0 - lr_final) * (1 - t/T)^power + lr_final`
///
/// # Example
/// ```
/// use tensor_engine::lr_scheduler::{LRScheduler, PolynomialLR};
///
/// let mut scheduler = PolynomialLR::new(0.1, 0.001, 100, 2.0);
/// assert_eq!(scheduler.get_lr(), 0.1);
/// for _ in 0..50 {
///     scheduler.step();
/// }
/// // At halfway point with power=2, lr should be ~0.026
/// let lr = scheduler.get_lr();
/// assert!(lr > 0.02 && lr < 0.03);
/// ```
pub struct PolynomialLR {
    initial_lr: f32,
    final_lr: f32,
    max_epochs: usize,
    power: f32,
    current_epoch: usize,
}

impl PolynomialLR {
    /// Creates a new polynomial learning rate scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - The initial learning rate (must be > 0)
    /// * `final_lr` - The final learning rate (must be >= 0 and < initial_lr)
    /// * `max_epochs` - Total number of epochs for decay (must be > 0)
    /// * `power` - The polynomial power (typically 1.0 for linear, 2.0 for quadratic)
    ///
    /// # Panics
    ///
    /// Panics if constraints are violated.
    pub fn new(initial_lr: f32, final_lr: f32, max_epochs: usize, power: f32) -> Self {
        assert!(initial_lr > 0.0, "initial_lr must be positive");
        assert!(final_lr >= 0.0, "final_lr must be non-negative");
        assert!(
            final_lr < initial_lr,
            "final_lr must be less than initial_lr"
        );
        assert!(max_epochs > 0, "max_epochs must be positive");
        assert!(power > 0.0, "power must be positive");
        PolynomialLR {
            initial_lr,
            final_lr,
            max_epochs,
            power,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for PolynomialLR {
    fn step(&mut self) {
        self.current_epoch += 1;
    }

    fn get_lr(&self) -> f32 {
        if self.current_epoch >= self.max_epochs {
            return self.final_lr;
        }
        let progress = self.current_epoch as f32 / self.max_epochs as f32;
        let decay = (1.0 - progress).powf(self.power);
        (self.initial_lr - self.final_lr) * decay + self.final_lr
    }

    fn reset(&mut self) {
        self.current_epoch = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_lr_initial() {
        let scheduler = ExponentialLR::new(0.1, 0.9);
        assert_eq!(scheduler.get_lr(), 0.1);
    }

    #[test]
    fn test_exponential_lr_decay() {
        let mut scheduler = ExponentialLR::new(0.1, 0.9);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.09).abs() < 1e-6);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.081).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_lr_reset() {
        let mut scheduler = ExponentialLR::new(0.1, 0.9);
        scheduler.step();
        scheduler.step();
        scheduler.reset();
        assert_eq!(scheduler.get_lr(), 0.1);
    }

    #[test]
    #[should_panic(expected = "initial_lr must be positive")]
    fn test_exponential_lr_invalid_lr() {
        ExponentialLR::new(0.0, 0.9);
    }

    #[test]
    fn test_step_lr_initial() {
        let scheduler = StepLR::new(0.1, 0.1, 10);
        assert_eq!(scheduler.get_lr(), 0.1);
    }

    #[test]
    fn test_step_lr_no_decay_before_step() {
        let mut scheduler = StepLR::new(0.1, 0.1, 10);
        for _ in 0..9 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_lr(), 0.1);
    }

    #[test]
    fn test_step_lr_decay_at_step() {
        let mut scheduler = StepLR::new(0.1, 0.1, 10);
        for _ in 0..10 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_step_lr_multiple_steps() {
        let mut scheduler = StepLR::new(0.1, 0.5, 5);
        for _ in 0..10 {
            scheduler.step();
        }
        // After 10 epochs with step_size=5: 2 steps, so 0.1 * 0.5^2 = 0.025
        assert!((scheduler.get_lr() - 0.025).abs() < 1e-6);
    }

    #[test]
    fn test_step_lr_reset() {
        let mut scheduler = StepLR::new(0.1, 0.1, 5);
        for _ in 0..10 {
            scheduler.step();
        }
        scheduler.reset();
        assert_eq!(scheduler.get_lr(), 0.1);
    }

    #[test]
    fn test_polynomial_lr_initial() {
        let scheduler = PolynomialLR::new(0.1, 0.001, 100, 2.0);
        assert_eq!(scheduler.get_lr(), 0.1);
    }

    #[test]
    fn test_polynomial_lr_halfway() {
        let mut scheduler = PolynomialLR::new(0.1, 0.001, 100, 2.0);
        for _ in 0..50 {
            scheduler.step();
        }
        // At t=50, T=100, power=2: (0.1 - 0.001) * (1 - 0.5)^2 + 0.001 = 0.099 * 0.25 + 0.001 = 0.02575
        let lr = scheduler.get_lr();
        assert!((lr - 0.02575).abs() < 1e-4);
    }

    #[test]
    fn test_polynomial_lr_final() {
        let mut scheduler = PolynomialLR::new(0.1, 0.001, 100, 2.0);
        for _ in 0..100 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_lr(), 0.001);
    }

    #[test]
    fn test_polynomial_lr_beyond_max() {
        let mut scheduler = PolynomialLR::new(0.1, 0.001, 100, 2.0);
        for _ in 0..150 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_lr(), 0.001);
    }

    #[test]
    fn test_polynomial_lr_linear() {
        let mut scheduler = PolynomialLR::new(1.0, 0.0, 10, 1.0);
        for _ in 0..5 {
            scheduler.step();
        }
        // Linear decay: at t=5, T=10: 1.0 * (1 - 0.5)^1 = 0.5
        assert!((scheduler.get_lr() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_polynomial_lr_reset() {
        let mut scheduler = PolynomialLR::new(0.1, 0.001, 100, 2.0);
        for _ in 0..50 {
            scheduler.step();
        }
        scheduler.reset();
        assert_eq!(scheduler.get_lr(), 0.1);
    }

    #[test]
    #[should_panic(expected = "final_lr must be less than initial_lr")]
    fn test_polynomial_lr_invalid_final() {
        PolynomialLR::new(0.1, 0.2, 100, 2.0);
    }
}
