use crate::dtype::DType;
use crate::tensor::Tensor;
use std::cell::RefCell;

thread_local! {
    /// Global thread-local state for whether autocast is enabled.
    static AUTOCAST_ENABLED: RefCell<bool> = RefCell::new(false);
    /// The target dtype for autocast (e.g., F16 or BF16). Default is F16.
    static AUTOCAST_DTYPE: RefCell<DType> = RefCell::new(DType::F16);
}

/// Execute a closure with autocast enabled (or disabled) for the given dtype.
/// Restores the previous state afterwards.
pub fn autocast<F, R>(enabled: bool, dtype: DType, f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev_enabled = AUTOCAST_ENABLED.with(|e| *e.borrow());
    let prev_dtype = AUTOCAST_DTYPE.with(|d| *d.borrow());

    AUTOCAST_ENABLED.with(|e| *e.borrow_mut() = enabled);
    AUTOCAST_DTYPE.with(|d| *d.borrow_mut() = dtype);

    let result = f();

    AUTOCAST_ENABLED.with(|e| *e.borrow_mut() = prev_enabled);
    AUTOCAST_DTYPE.with(|d| *d.borrow_mut() = prev_dtype);

    result
}

/// Check if autocast is currently enabled.
pub fn is_autocast_enabled() -> bool {
    AUTOCAST_ENABLED.with(|e| *e.borrow())
}

/// Get the current autocast target dtype.
pub fn get_autocast_dtype() -> DType {
    AUTOCAST_DTYPE.with(|d| *d.borrow())
}

/// Gradient Scaler for mixed precision training.
/// Prevents gradient underflow by scaling the loss before backward pass.
pub struct GradScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    growth_tracker: usize,
}

impl GradScaler {
    pub fn new() -> Self {
        Self {
            scale: 65536.0, // Default start scale 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_tracker: 0,
        }
    }

    /// Scale the loss tensor.
    pub fn scale(&self, loss: &Tensor) -> Tensor {
        let scale_tensor = Tensor::from_scalar(self.scale);
        loss.mul(&scale_tensor)
    }

    /// Unscale gradients of the given parameters by dividing them by the scale factor.
    /// Returns true if all gradients were finite (success), false if any Inf/NaN found (failure).
    /// If failure, the optimizer step should be skipped.
    pub fn unscale(&self, params: &[Tensor]) -> bool {
        let scale_inv = 1.0 / self.scale;
        let mut found_inf = false;

        for p in params {
            let lock = p.lock();
            if let Some(grad) = &lock.grad {
                if !found_inf {
                    if grad.iter().any(|x| !x.is_finite()) {
                        found_inf = true;
                    }
                }
            }
        }

        if found_inf {
            log::info!(
                "GradScaler: Infinite gradients detected. Skipping step and reducing scale."
            );
        }

        // Unscale gradients in-place regardless of found_inf; step() will skip if needed
        for p in params {
            let mut lock = p.lock();
            if let Some(grad) = &mut lock.grad {
                grad.mapv_inplace(|x| x * scale_inv);
            }
        }

        !found_inf
    }

    /// Performs `optimizer.step()` if gradients are finite.
    /// Otherwise, skips step.
    /// Returns true if step was taken.
    pub fn step(&mut self, optimizer: &mut dyn crate::optim::Optimizer, params: &[Tensor]) -> bool {
        // 1. Unscale gradients in-place
        let scale_inv = 1.0 / self.scale;
        let mut found_inf = false;

        for p in params {
            let mut lock = p.lock();
            if let Some(grad) = &mut lock.grad {
                // Check for inf/nan
                // This is expensive on CPU/GPU without fused kernels but necessary for stability logic
                if !found_inf {
                    if grad.iter().any(|x| !x.is_finite()) {
                        found_inf = true;
                    }
                }

                // Apply unscale
                // We can do this safely even if inf because we skip the step later
                grad.mapv_inplace(|x| x * scale_inv);
            }
        }

        if found_inf {
            log::info!(
                "GradScaler: Infinite gradients detected. Skipping step and reducing scale."
            );
            self.growth_tracker = 0;
            self.scale *= self.backoff_factor;
            return false;
        }

        // 2. Optimizer step
        optimizer.step();

        // 3. Update scale (update logic is usually separate in `update()`)
        true
    }

    /// Update the scale factor.
    pub fn update(&mut self, was_finite: bool) {
        if was_finite {
            self.growth_tracker += 1;
            if self.growth_tracker >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.growth_tracker = 0;
                log::info!("GradScaler: Increasing scale to {}", self.scale);
            }
        } else {
            // Backoff handled in step() typically, or here.
            // PyTorch usually does backoff immediately when Inf found.
            // If we handled it in step(), we don't need to do it here.
            // But if the user calls `update()` manually, we should respect protocol.
        }
    }
}
