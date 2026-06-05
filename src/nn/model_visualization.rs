//! Model visualization utilities.
//!
//! Provides tools for inspecting and visualizing model architecture,
//! parameter distributions, and activation statistics.

use crate::tensor::Tensor;
use std::collections::HashMap;

/// Summary statistics for a tensor.
#[derive(Clone, Debug)]
pub struct TensorStats {
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Number of zero elements
    pub zero_count: usize,
    /// Total number of elements
    pub total_count: usize,
    /// Sparsity (fraction of zeros)
    pub sparsity: f32,
}

impl TensorStats {
    /// Compute statistics for a tensor.
    pub fn from_tensor(t: &Tensor) -> Self {
        let arr = t.lock().storage.to_f32_array();
        let total = arr.len();
        if total == 0 {
            return TensorStats {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                zero_count: 0,
                total_count: 0,
                sparsity: 0.0,
            };
        }

        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut zero_count = 0usize;

        for &v in arr.iter() {
            sum += v;
            sum_sq += v * v;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            if v.abs() < 1e-7 {
                zero_count += 1;
            }
        }

        let mean = sum / total as f32;
        let variance = (sum_sq / total as f32) - (mean * mean);
        let std = variance.max(0.0).sqrt();

        TensorStats {
            mean,
            std,
            min,
            max,
            zero_count,
            total_count: total,
            sparsity: zero_count as f32 / total as f32,
        }
    }

    /// Format statistics as a human-readable string.
    pub fn format(&self) -> String {
        format!(
            "mean={:.4e}, std={:.4e}, min={:.4e}, max={:.4e}, sparsity={:.2}%",
            self.mean,
            self.std,
            self.min,
            self.max,
            self.sparsity * 100.0
        )
    }
}

/// Parameter info for a named parameter.
#[derive(Clone, Debug)]
pub struct ParamInfo {
    /// Parameter name
    pub name: String,
    /// Shape as string
    pub shape: String,
    /// Number of elements
    pub num_elements: usize,
    /// Statistics
    pub stats: TensorStats,
}

impl ParamInfo {
    /// Create parameter info from a named tensor.
    pub fn from_named_param(name: &str, tensor: &Tensor) -> Self {
        let shape = tensor.lock().storage.shape();
        let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        let num_elements = shape.iter().product();

        ParamInfo {
            name: name.to_string(),
            shape: shape_str.join("x"),
            num_elements,
            stats: TensorStats::from_tensor(tensor),
        }
    }
}

/// Model summary containing parameter information.
#[derive(Clone, Debug)]
pub struct ModelSummary {
    /// Total number of parameters
    pub total_params: usize,
    /// Number of trainable parameters
    pub trainable_params: usize,
    /// Parameter info for each layer
    pub param_info: Vec<ParamInfo>,
    /// Layer-by-layer parameter count
    pub layer_params: Vec<(String, usize)>,
}

impl ModelSummary {
    /// Generate a summary for a module.
    pub fn from_module(module: &dyn std::any::Any) -> Self {
        // This is a simplified summary - in practice, you'd use the Module trait
        ModelSummary {
            total_params: 0,
            trainable_params: 0,
            param_info: Vec::new(),
            layer_params: Vec::new(),
        }
    }

    /// Format summary as a human-readable string.
    pub fn format(&self) -> String {
        let mut lines = Vec::new();
        lines.push("=".repeat(60));
        lines.push("Model Summary".to_string());
        lines.push("=".repeat(60));
        lines.push(format!("Total parameters: {}", self.total_params));
        lines.push(format!("Trainable parameters: {}", self.trainable_params));
        lines.push("=".repeat(60));

        for (name, count) in &self.layer_params {
            let pct = if self.total_params > 0 {
                (*count as f32 / self.total_params as f32) * 100.0
            } else {
                0.0
            };
            let bar = "#".repeat((pct / 2.0) as usize);
            lines.push(format!(
                "  {:<30} {:>10} {:>6.1}% {}",
                name, count, pct, bar
            ));
        }

        lines.push("=".repeat(60));
        lines.join("\n")
    }
}

/// Activation statistics tracker.
pub struct ActivationTracker {
    /// Collected activation stats by layer name
    pub activations: HashMap<String, TensorStats>,
    /// Maximum number of activations to store
    pub max_samples: usize,
    /// Current sample count
    pub sample_count: usize,
}

impl ActivationTracker {
    /// Create a new activation tracker.
    pub fn new(max_samples: usize) -> Self {
        ActivationTracker {
            activations: HashMap::new(),
            max_samples,
            sample_count: 0,
        }
    }

    /// Record activation statistics for a layer.
    pub fn record(&mut self, layer_name: &str, activation: &Tensor) {
        if self.sample_count >= self.max_samples {
            return;
        }

        let stats = TensorStats::from_tensor(activation);
        self.activations.insert(layer_name.to_string(), stats);
        self.sample_count += 1;
    }

    /// Get statistics for a specific layer.
    pub fn get_stats(&self, layer_name: &str) -> Option<&TensorStats> {
        self.activations.get(layer_name)
    }

    /// Get all recorded statistics.
    pub fn all_stats(&self) -> &HashMap<String, TensorStats> {
        &self.activations
    }

    /// Format all recorded activations.
    pub fn format_all(&self) -> String {
        let mut lines = Vec::new();
        for (name, stats) in &self.activations {
            lines.push(format!("  {}: {}", name, stats.format()));
        }
        lines.join("\n")
    }

    /// Clear all recorded activations.
    pub fn clear(&mut self) {
        self.activations.clear();
        self.sample_count = 0;
    }
}

/// Gradient statistics tracker for debugging gradient flow.
pub struct GradientTracker {
    /// Collected gradient stats by parameter name
    pub gradients: HashMap<String, TensorStats>,
    /// Maximum number of steps to store
    pub max_steps: usize,
    /// Current step
    pub step: usize,
}

impl GradientTracker {
    /// Create a new gradient tracker.
    pub fn new(max_steps: usize) -> Self {
        GradientTracker {
            gradients: HashMap::new(),
            max_steps,
            step: 0,
        }
    }

    /// Record gradient statistics for all parameters.
    pub fn record(&mut self, named_params: &[(String, Tensor)]) {
        if self.step >= self.max_steps {
            return;
        }

        for (name, param) in named_params {
            if let Some(grad) = &param.lock().grad {
                let stats = TensorStats::from_tensor(&Tensor::new(grad.clone().into_dyn(), false));
                self.gradients
                    .insert(format!("step_{}_{}", self.step, name), stats);
            }
        }

        self.step += 1;
    }

    /// Get gradient stats for a specific parameter at a specific step.
    pub fn get_at_step(&self, step: usize, name: &str) -> Option<&TensorStats> {
        self.gradients.get(&format!("step_{}_{}", step, name))
    }

    /// Get the latest gradient stats for a parameter.
    pub fn latest_for(&self, name: &str) -> Option<&TensorStats> {
        if self.step == 0 {
            return None;
        }
        self.gradients
            .get(&format!("step_{}_{}", self.step - 1, name))
    }

    /// Format all recorded gradients.
    pub fn format_all(&self) -> String {
        let mut lines = Vec::new();
        for (key, stats) in &self.gradients {
            lines.push(format!("  {}: {}", key, stats.format()));
        }
        lines.join("\n")
    }

    /// Clear all recorded gradients.
    pub fn clear(&mut self) {
        self.gradients.clear();
        self.step = 0;
    }
}

/// Gradient flow analysis for debugging vanishing/exploding gradients.
pub struct GradientFlowAnalyzer {
    /// Gradient norms over training steps
    pub gradient_norms: Vec<(usize, f32)>,
    /// Parameter norms over training steps
    pub parameter_norms: Vec<(usize, f32)>,
    /// Update norms over training steps
    pub update_norms: Vec<(usize, f32)>,
}

impl GradientFlowAnalyzer {
    /// Create a new gradient flow analyzer.
    pub fn new() -> Self {
        GradientFlowAnalyzer {
            gradient_norms: Vec::new(),
            parameter_norms: Vec::new(),
            update_norms: Vec::new(),
        }
    }

    /// Record gradient norm at a training step.
    pub fn record_gradient_norm(&mut self, step: usize, norm: f32) {
        self.gradient_norms.push((step, norm));
    }

    /// Record parameter norm at a training step.
    pub fn record_parameter_norm(&mut self, step: usize, norm: f32) {
        self.parameter_norms.push((step, norm));
    }

    /// Record update norm at a training step.
    pub fn record_update_norm(&mut self, step: usize, norm: f32) {
        self.update_norms.push((step, norm));
    }

    /// Compute gradient norm from named parameters with gradients.
    pub fn compute_gradient_norm(named_params: &[(String, Tensor)]) -> f32 {
        let mut sum_sq = 0.0f32;
        for (name, param) in named_params {
            if let Some(grad) = &param.lock().grad {
                for &v in grad.iter() {
                    sum_sq += v * v;
                }
            }
        }
        sum_sq.sqrt()
    }

    /// Compute parameter norm from named parameters.
    pub fn compute_parameter_norm(named_params: &[(String, Tensor)]) -> f32 {
        let mut sum_sq = 0.0f32;
        for (_, param) in named_params {
            let arr = param.lock().storage.to_f32_array();
            for &v in arr.iter() {
                sum_sq += v * v;
            }
        }
        sum_sq.sqrt()
    }

    /// Check for gradient issues (vanishing/exploding).
    pub fn check_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.gradient_norms.len() < 2 {
            return issues;
        }

        let latest = self.gradient_norms.last().unwrap();
        let prev = self.gradient_norms[self.gradient_norms.len() - 2];

        // Check for vanishing gradients
        if latest.1 < 1e-7 {
            issues.push(format!(
                "WARNING: Vanishing gradients at step {}: norm = {:.2e}",
                latest.0, latest.1
            ));
        }

        // Check for exploding gradients
        if latest.1 > 1e7 {
            issues.push(format!(
                "WARNING: Exploding gradients at step {}: norm = {:.2e}",
                latest.0, latest.1
            ));
        }

        // Check for sudden spikes
        if prev.1 > 0.0 {
            let ratio = latest.1 / prev.1;
            if ratio > 100.0 {
                issues.push(format!(
                    "WARNING: Gradient spike at step {}: ratio = {:.1}x",
                    latest.0, ratio
                ));
            }
        }

        // Check parameter norms
        if !self.parameter_norms.is_empty() {
            let latest_param = self.parameter_norms.last().unwrap();
            if latest_param.1 > 1e6 {
                issues.push(format!(
                    "WARNING: Exploding parameters at step {}: norm = {:.2e}",
                    latest_param.0, latest_param.1
                ));
            }
        }

        issues
    }

    /// Format analysis results.
    pub fn format(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Gradient Flow Analysis".to_string());
        lines.push("-".repeat(40));

        if !self.gradient_norms.is_empty() {
            let latest = self.gradient_norms.last().unwrap();
            lines.push(format!(
                "Latest gradient norm: {:.6e} (step {})",
                latest.1, latest.0
            ));

            if self.gradient_norms.len() > 1 {
                let first = self.gradient_norms.first().unwrap();
                lines.push(format!(
                    "First gradient norm: {:.6e} (step {})",
                    first.1, first.0
                ));
            }
        }

        if !self.parameter_norms.is_empty() {
            let latest = self.parameter_norms.last().unwrap();
            lines.push(format!(
                "Latest parameter norm: {:.6e} (step {})",
                latest.1, latest.0
            ));
        }

        let issues = self.check_issues();
        if !issues.is_empty() {
            lines.push("".to_string());
            lines.push("Issues detected:".to_string());
            for issue in &issues {
                lines.push(format!("  {}", issue));
            }
        }

        lines.join("\n")
    }
}

/// Weight histogram for visualization.
#[derive(Clone, Debug)]
pub struct WeightHistogram {
    /// Bin edges
    pub bins: Vec<f32>,
    /// Bin counts
    pub counts: Vec<usize>,
    /// Number of bins
    pub num_bins: usize,
}

impl WeightHistogram {
    /// Create a histogram from tensor data.
    pub fn from_tensor(t: &Tensor, num_bins: usize) -> Self {
        let arr = t.lock().storage.to_f32_array();
        if arr.len() == 0 {
            return WeightHistogram {
                bins: vec![],
                counts: vec![],
                num_bins,
            };
        }

        let min = arr.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let bin_width = if max > min {
            (max - min) / num_bins as f32
        } else {
            1.0
        };

        let mut bins = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins {
            bins.push(min + i as f32 * bin_width);
        }

        let mut counts = vec![0usize; num_bins];
        for &v in arr.iter() {
            let bin_idx = ((v - min) / bin_width) as usize;
            if bin_idx < num_bins {
                counts[bin_idx] += 1;
            } else if bin_idx == num_bins {
                counts[num_bins - 1] += 1;
            }
        }

        WeightHistogram {
            bins,
            counts,
            num_bins,
        }
    }

    /// Format histogram as ASCII art.
    pub fn format_ascii(&self, max_width: usize) -> String {
        if self.counts.is_empty() {
            return "No data".to_string();
        }

        let max_count = *self.counts.iter().max().unwrap_or(&1);
        let mut lines = Vec::new();

        for (i, &count) in self.counts.iter().enumerate() {
            let bar_width = if max_count > 0 {
                ((count as f32 / max_count as f32) * max_width as f32) as usize
            } else {
                0
            };
            let bar = "#".repeat(bar_width.max(1));
            lines.push(format!("{:8.3} |{}", self.bins[i], bar));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod visualization_tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn test_tensor_stats() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[5]), data).unwrap(), false);
        let stats = TensorStats::from_tensor(&tensor);

        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!(stats.min == 1.0);
        assert!(stats.max == 5.0);
        assert!(stats.zero_count == 0);
        assert!(stats.total_count == 5);
    }

    #[test]
    fn test_tensor_stats_with_zeros() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 2.0, 3.0];
        let tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[5]), data).unwrap(), false);
        let stats = TensorStats::from_tensor(&tensor);

        assert_eq!(stats.zero_count, 2);
        assert!((stats.sparsity - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_activation_tracker() {
        let mut tracker = ActivationTracker::new(10);

        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[3]), data).unwrap(), false);

        tracker.record("layer1", &tensor);
        tracker.record("layer2", &tensor);

        assert!(tracker.get_stats("layer1").is_some());
        assert!(tracker.get_stats("layer2").is_some());
        assert!(tracker.get_stats("layer3").is_none());
    }

    #[test]
    fn test_gradient_tracker() {
        let mut tracker = GradientTracker::new(5);

        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[3]), data).unwrap(), true);
        let named = vec![("weight".to_string(), tensor)];

        tracker.record(&named);

        assert_eq!(tracker.step, 1);
    }

    #[test]
    fn test_gradient_flow_analyzer() {
        let mut analyzer = GradientFlowAnalyzer::new();

        analyzer.record_gradient_norm(0, 1.0);
        analyzer.record_gradient_norm(1, 0.5);
        analyzer.record_gradient_norm(2, 0.25);

        let issues = analyzer.check_issues();
        // Should not detect issues with normal gradients
        assert!(!issues.iter().any(|s| s.contains("WARNING")));
    }

    #[test]
    fn test_gradient_flow_detect_vanishing() {
        let mut analyzer = GradientFlowAnalyzer::new();

        analyzer.record_gradient_norm(0, 1.0);
        analyzer.record_gradient_norm(1, 1e-8);

        let issues = analyzer.check_issues();
        assert!(issues.iter().any(|s| s.contains("Vanishing")));
    }

    #[test]
    fn test_gradient_flow_detect_exploding() {
        let mut analyzer = GradientFlowAnalyzer::new();

        analyzer.record_gradient_norm(0, 1.0);
        analyzer.record_gradient_norm(1, 1e8);

        let issues = analyzer.check_issues();
        assert!(issues.iter().any(|s| s.contains("Exploding")));
    }

    #[test]
    fn test_weight_histogram() {
        let data: Vec<f32> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[5]), data).unwrap(), false);
        let hist = WeightHistogram::from_tensor(&tensor, 5);

        assert_eq!(hist.num_bins, 5);
        assert_eq!(hist.counts.len(), 5);
    }

    #[test]
    fn test_weight_histogram_ascii() {
        let data: Vec<f32> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[5]), data).unwrap(), false);
        let hist = WeightHistogram::from_tensor(&tensor, 5);

        let ascii = hist.format_ascii(20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_param_info() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[2, 2]), data).unwrap(), false);
        let info = ParamInfo::from_named_param("weight", &tensor);

        assert_eq!(info.name, "weight");
        assert_eq!(info.shape, "2x2");
        assert_eq!(info.num_elements, 4);
    }
}
