//! Knowledge distillation for model compression.
//!
//! Knowledge distillation transfers knowledge from a large "teacher" model
//! to a smaller "student" model by matching their output distributions
//! rather than just the ground-truth labels.
//!
//! Reference: [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)

use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

/// Distillation loss function.
///
/// Combines standard cross-entropy loss with KL divergence between
/// teacher and student softened probability distributions.
///
/// Loss = α * CE(student, targets) + (1 - α) * T^2 * KL(teacher_soft || student_soft)
pub struct DistillationLoss {
    /// Weight for standard cross-entropy loss
    pub ce_weight: f32,
    /// Weight for distillation loss
    pub distill_weight: f32,
    /// Temperature for softmax softening
    pub temperature: f32,
}

impl DistillationLoss {
    /// Create a new distillation loss.
    ///
    /// # Arguments
    /// * `ce_weight` - Weight for cross-entropy component (typically 0.5-1.0)
    /// * `distill_weight` - Weight for distillation component (typically 0.0-0.5)
    /// * `temperature` - Softmax temperature (higher = softer distribution, typically 2-8)
    pub fn new(ce_weight: f32, distill_weight: f32, temperature: f32) -> Self {
        assert!(temperature > 0.0, "temperature must be positive");
        assert!(
            (ce_weight + distill_weight - 1.0).abs() < 1e-6,
            "ce_weight + distill_weight should equal 1.0"
        );
        DistillationLoss {
            ce_weight,
            distill_weight,
            temperature,
        }
    }

    /// Compute distillation loss.
    ///
    /// # Arguments
    /// * `student_logits` - Logits from student model [batch, num_classes]
    /// * `teacher_logits` - Logits from teacher model [batch, num_classes]
    /// * `targets` - Ground truth labels [batch] (class indices as floats)
    pub fn forward(&self, student_logits: &Tensor, teacher_logits: &Tensor, targets: &Tensor) -> Tensor {
        let temp = self.temperature;
        let temp_sq = temp * temp;

        // Softened student probabilities: softmax(logits / T)
        let student_scaled = student_logits.div(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), temp),
            false,
        ));
        let student_soft = student_scaled.softmax(1);

        // Softened teacher probabilities
        let teacher_scaled = teacher_logits.div(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), temp),
            false,
        ));
        let teacher_soft = teacher_scaled.softmax(1);

        // KL divergence: KL(teacher_soft || student_soft) = sum(teacher_soft * log(teacher_soft / student_soft))
        let log_student_soft = student_soft.log().clamp(-1e10, 1e10);
        let kl_num = &teacher_soft * (&teacher_soft.log().clamp(-1e10, 1e10) - &log_student_soft);
        let kl_loss = kl_num.sum() / (teacher_soft.lock().storage.shape()[0] as f32);

        // Cross-entropy loss
        let target_one_hot = self.targets_to_one_hot(targets, student_logits.lock().storage.shape()[1]);
        let target_tensor = Tensor::new(target_one_hot, false);
        let log_student = student_logits.softmax(1).log().clamp(-1e10, 1e10);
        let ce_loss = (&target_tensor * &log_student).sum() / (-target_tensor.lock().storage.shape()[0] as f32);

        // Combined loss
        let ce_component = ce_loss.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), self.ce_weight),
            false,
        ));
        let distill_component = kl_loss.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), self.distill_weight * temp_sq),
            false,
        ));

        ce_component.add(&distill_component)
    }

    /// Convert class indices to one-hot encoding.
    fn targets_to_one_hot(&self, targets: &Tensor, num_classes: usize) -> ArrayD<f32> {
        let targets_arr = targets.lock().storage.to_f32_array();
        let batch_size = targets_arr.shape()[0];
        let mut one_hot = ArrayD::zeros(IxDyn(&[batch_size, num_classes][..]));

        for i in 0..batch_size {
            let class_idx = targets_arr[[i]] as usize;
            if class_idx < num_classes {
                one_hot[[i, class_idx]] = 1.0;
            }
        }

        one_hot
    }
}

/// Logits distillation: match student logits to teacher logits directly.
///
/// Uses MSE loss between softened teacher and student logits.
pub struct LogitsDistillation {
    /// Temperature for softening
    pub temperature: f32,
    /// Weight for logits loss
    pub weight: f32,
}

impl LogitsDistillation {
    /// Create a new logits distillation loss.
    pub fn new(temperature: f32, weight: f32) -> Self {
        assert!(temperature > 0.0, "temperature must be positive");
        LogitsDistillation {
            temperature,
            weight,
        }
    }

    /// Compute logits distillation loss (MSE between softened logits).
    pub fn forward(&self, student_logits: &Tensor, teacher_logits: &Tensor) -> Tensor {
        let temp = self.temperature;

        let student_scaled = student_logits.div(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), temp),
            false,
        ));
        let teacher_scaled = teacher_logits.div(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), temp),
            false,
        ));

        let diff = &student_scaled - &teacher_scaled;
        let mse = diff.pow(2.0).mean();

        mse.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), self.weight),
            false,
        ))
    }
}

/// Feature distillation: match intermediate feature representations.
///
/// Uses MSE loss between teacher and student intermediate features.
pub struct FeatureDistillation {
    /// Weight for feature distillation loss
    pub weight: f32,
}

impl FeatureDistillation {
    /// Create a new feature distillation loss.
    pub fn new(weight: f32) -> Self {
        FeatureDistillation { weight }
    }

    /// Compute feature distillation loss (MSE between features).
    pub fn forward(&self, student_features: &Tensor, teacher_features: &Tensor) -> Tensor {
        let diff = student_features.sub(teacher_features);
        let mse = diff.pow(2.0).mean();

        mse.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), self.weight),
            false,
        ))
    }
}

/// Attention distillation: match attention maps between teacher and student.
///
 /// Uses MSE loss between teacher and student attention matrices.
pub struct AttentionDistillation {
    /// Temperature for softening attention
    pub temperature: f32,
    /// Weight for attention distillation loss
    pub weight: f32,
}

impl AttentionDistillation {
    /// Create a new attention distillation loss.
    pub fn new(temperature: f32, weight: f32) -> Self {
        assert!(temperature > 0.0, "temperature must be positive");
        AttentionDistillation {
            temperature,
            weight,
        }
    }

    /// Compute attention distillation loss.
    ///
    /// # Arguments
    /// * `student_attn` - Student attention maps [batch, heads, seq, seq]
    /// * `teacher_attn` - Teacher attention maps [batch, heads, seq, seq]
    pub fn forward(&self, student_attn: &Tensor, teacher_attn: &Tensor) -> Tensor {
        let temp = self.temperature;

        // Soften attention maps
        let student_soft = student_attn.div(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), temp),
            false,
        )).softmax(3); // Softmax over last dimension

        let teacher_soft = teacher_attn.div(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), temp),
            false,
        )).softmax(3);

        let diff = &student_soft - &teacher_soft;
        let mse = diff.pow(2.0).mean();

        mse.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), self.weight),
            false,
        ))
    }
}

/// Distillation trainer helper.
///
 /// Manages the training loop for knowledge distillation, handling
 /// both teacher inference and student training.
pub struct DistillationTrainer {
    /// Cross-entropy loss weight
    pub ce_weight: f32,
    /// Distillation loss weight
    pub distill_weight: f32,
    /// Temperature for softening
    pub temperature: f32,
    /// Whether to use logit distillation
    pub use_logit_distill: bool,
    /// Logit distillation weight
    pub logit_distill_weight: f32,
}

impl DistillationTrainer {
    /// Create a new distillation trainer.
    pub fn new(ce_weight: f32, distill_weight: f32, temperature: f32) -> Self {
        DistillationTrainer {
            ce_weight,
            distill_weight,
            temperature,
            use_logit_distill: false,
            logit_distill_weight: 0.0,
        }
    }

    /// Enable logit distillation.
    pub fn with_logit_distill(mut self, weight: f32) -> Self {
        self.use_logit_distill = true;
        self.logit_distill_weight = weight;
        self
    }

    /// Compute total distillation loss.
    ///
    /// # Arguments
    /// * `student_logits` - Student model output [batch, num_classes]
    /// * `teacher_logits` - Teacher model output [batch, num_classes]
    /// * `targets` - Ground truth labels [batch]
    pub fn compute_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: &Tensor,
    ) -> Tensor {
        let distill_loss = DistillationLoss::new(self.ce_weight, self.distill_weight, self.temperature);
        let total = distill_loss.forward(student_logits, teacher_logits, targets);

        if self.use_logit_distill && self.logit_distill_weight > 0.0 {
            let logit_loss = LogitsDistillation::new(self.temperature, self.logit_distill_weight);
            total.add(&logit_loss.forward(student_logits, teacher_logits))
        } else {
            total
        }
    }
}

/// Student model wrapper for distillation.
///
 /// Holds both the student model and optionally a teacher model reference.
pub struct DistillationModel<S, T> {
    student: S,
    teacher: Option<T>,
}

impl<S, T> DistillationModel<S, T> {
    /// Create a new distillation model.
    pub fn new(student: S, teacher: Option<T>) -> Self {
        DistillationModel { student, teacher }
    }

    /// Get reference to student model.
    pub fn student(&self) -> &S {
        &self.student
    }

    /// Get reference to teacher model.
    pub fn teacher(&self) -> Option<&T> {
        self.teacher.as_ref()
    }
}

#[cfg(test)]
mod distillation_tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_distillation_loss_forward() {
        let distill = DistillationLoss::new(0.5, 0.5, 4.0);

        // Simple 2-class case
        let student_logits = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![1.0, -1.0, 0.5, 0.5]).unwrap(),
            true,
        );
        let teacher_logits = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2][..]), vec![2.0, -2.0, 1.0, -1.0]).unwrap(),
            false,
        );
        let targets = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2][..]), vec![0.0, 0.0]).unwrap(),
            false,
        );

        let loss = distill.forward(&student_logits, &teacher_logits, &targets);
        let loss_val = *loss.lock().storage.to_f32_array().iter().next().unwrap();

        assert!(loss_val > 0.0);
        assert!(loss_val.is_finite());
    }

    #[test]
    fn test_logits_distillation_forward() {
        let logit_distill = LogitsDistillation::new(4.0, 1.0);

        let student_logits = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 0.5, 0.0, 0.0, 0.5, 1.0]).unwrap(),
            true,
        );
        let teacher_logits = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0]).unwrap(),
            false,
        );

        let loss = logit_distill.forward(&student_logits, &teacher_logits);
        let loss_val = *loss.lock().storage.to_f32_array().iter().next().unwrap();

        assert!(loss_val >= 0.0);
    }

    #[test]
    fn test_feature_distillation_forward() {
        let feature_distill = FeatureDistillation::new(1.0);

        let student_features = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 4][..]), vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]).unwrap(),
            true,
        );
        let teacher_features = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 4][..]), vec![1.1, 2.1, 3.1, 4.1, 0.6, 1.6, 2.6, 3.6]).unwrap(),
            false,
        );

        let loss = feature_distill.forward(&student_features, &teacher_features);
        let loss_val = *loss.lock().storage.to_f32_array().iter().next().unwrap();

        // Loss should be small since features are close
        assert!(loss_val > 0.0);
        assert!(loss_val < 1.0);
    }

    #[test]
    fn test_distillation_trainer() {
        let trainer = DistillationTrainer::new(0.5, 0.5, 4.0);

        let student_logits = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 0.0, -1.0, 0.0, 1.0, -1.0]).unwrap(),
            true,
        );
        let teacher_logits = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![2.0, 0.0, -2.0, -2.0, 0.0, 2.0]).unwrap(),
            false,
        );
        let targets = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2][..]), vec![0.0, 1.0]).unwrap(),
            false,
        );

        let loss = trainer.compute_loss(&student_logits, &teacher_logits, &targets);
        let loss_val = *loss.lock().storage.to_f32_array().iter().next().unwrap();

        assert!(loss_val > 0.0);
        assert!(loss_val.is_finite());
    }

    #[test]
    fn test_distillation_trainer_with_logit_distill() {
        let trainer = DistillationTrainer::new(0.5, 0.5, 4.0)
            .with_logit_distill(0.5);

        let student_logits = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![1.0, 0.0, -1.0, 0.0, 1.0, -1.0]).unwrap(),
            true,
        );
        let teacher_logits = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3][..]), vec![2.0, 0.0, -2.0, -2.0, 0.0, 2.0]).unwrap(),
            false,
        );
        let targets = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2][..]), vec![0.0, 1.0]).unwrap(),
            false,
        );

        let loss = trainer.compute_loss(&student_logits, &teacher_logits, &targets);
        let loss_val = *loss.lock().storage.to_f32_array().iter().next().unwrap();

        assert!(loss_val > 0.0);
    }

    #[test]
    #[should_panic(expected = "temperature must be positive")]
    fn test_distillation_loss_invalid_temperature() {
        DistillationLoss::new(0.5, 0.5, 0.0);
    }

    #[test]
    fn test_distillation_model() {
        struct DummyStudent;
        impl crate::nn::Module for DummyStudent {
            fn forward(&self, _input: &Tensor) -> Tensor { Tensor::zeros(&[1]) }
            fn parameters(&self) -> Vec<Tensor> { Vec::new() }
            fn as_any(&self) -> &dyn std::any::Any { self }
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        }

        struct DummyTeacher;
        impl crate::nn::Module for DummyTeacher {
            fn forward(&self, _input: &Tensor) -> Tensor { Tensor::zeros(&[1]) }
            fn parameters(&self) -> Vec<Tensor> { Vec::new() }
            fn as_any(&self) -> &dyn std::any::Any { self }
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        }

        let model = DistillationModel::new(DummyStudent, Some(DummyTeacher));
        assert!(model.student() != &DummyStudent);
        assert!(model.teacher().is_some());
    }
}
