use super::backend::{BackendError, CpuAutogradBackend};
use crate::nn::Module as CanonicalModule;
use crate::tensor::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::any::Any;
use std::cell::RefCell;

pub trait CheckedModule: CanonicalModule {
    fn try_forward(
        &self,
        backend: &CpuAutogradBackend,
        input: &Tensor,
    ) -> Result<Tensor, BackendError>;
}

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
    weight_name: String,
    bias_name: String,
}

impl Linear {
    pub fn new(
        backend: &CpuAutogradBackend,
        in_features: usize,
        out_features: usize,
        name_prefix: &str,
    ) -> Result<Self, BackendError> {
        let scale = 1.0_f32 / (in_features.max(1) as f32).sqrt();
        let mut weight_data = vec![0.0; in_features * out_features];
        for (idx, weight) in weight_data.iter_mut().enumerate() {
            let centered = (idx % 11) as f32 - 5.0;
            *weight = scale * 0.002 * centered;
        }
        let bias_data = vec![0.0; out_features];

        let weight = backend.from_data(weight_data, vec![in_features, out_features], true)?;
        let bias = backend.from_data(bias_data, vec![1, out_features], true)?;

        Ok(Self {
            weight,
            bias,
            weight_name: format!("{name_prefix}.weight"),
            bias_name: format!("{name_prefix}.bias"),
        })
    }

    #[cfg(test)]
    pub fn in_features(&self) -> usize {
        self.weight.shape()[0]
    }

    #[cfg(test)]
    pub fn out_features(&self) -> usize {
        self.weight.shape()[1]
    }
}

impl CheckedModule for Linear {
    fn try_forward(
        &self,
        backend: &CpuAutogradBackend,
        input: &Tensor,
    ) -> Result<Tensor, BackendError> {
        let output = backend.matmul(input, &self.weight)?;
        backend.add(&output, &self.bias)
    }
}

impl CanonicalModule for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.try_forward(&CpuAutogradBackend, input)
            .expect("GPT Linear forward validation failed")
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (self.weight_name.clone(), self.weight.clone()),
            (self.bias_name.clone(), self.bias.clone()),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct MultiHeadSelfAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    head_dim: usize,
}

impl MultiHeadSelfAttention {
    pub fn new(
        backend: &CpuAutogradBackend,
        model_dim: usize,
        num_heads: usize,
        name_prefix: &str,
    ) -> Result<Self, BackendError> {
        if num_heads == 0 {
            return Err(BackendError::Unsupported("num_heads must be > 0"));
        }
        if model_dim % num_heads != 0 {
            return Err(BackendError::Unsupported(
                "model_dim must be divisible by num_heads",
            ));
        }

        Ok(Self {
            q_proj: Linear::new(backend, model_dim, model_dim, &format!("{name_prefix}.q"))?,
            k_proj: Linear::new(backend, model_dim, model_dim, &format!("{name_prefix}.k"))?,
            v_proj: Linear::new(backend, model_dim, model_dim, &format!("{name_prefix}.v"))?,
            out_proj: Linear::new(backend, model_dim, model_dim, &format!("{name_prefix}.out"))?,
            head_dim: model_dim / num_heads,
        })
    }

    #[cfg(test)]
    pub fn model_dim(&self) -> usize {
        self.q_proj.in_features()
    }

    #[cfg(test)]
    pub fn num_heads(&self) -> usize {
        self.model_dim() / self.head_dim
    }

    #[cfg(test)]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl CheckedModule for MultiHeadSelfAttention {
    fn try_forward(
        &self,
        backend: &CpuAutogradBackend,
        input: &Tensor,
    ) -> Result<Tensor, BackendError> {
        let q = self.q_proj.try_forward(backend, input)?;
        let k = self.k_proj.try_forward(backend, input)?;
        let v = self.v_proj.try_forward(backend, input)?;

        let k_t = backend.transpose2d(&k)?;
        let scores = backend.matmul(&q, &k_t)?;

        let scale = backend.scalar(1.0 / (self.head_dim as f32).sqrt(), false)?;
        let scaled = backend.mul(&scores, &scale)?;
        let masked = backend.causal_mask_upper(&scaled, -1.0e9)?;

        let weights = backend.softmax_last_dim(&masked)?;

        let context = backend.matmul(&weights, &v)?;
        self.out_proj.try_forward(backend, &context)
    }
}

impl CanonicalModule for MultiHeadSelfAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.try_forward(&CpuAutogradBackend, input)
            .expect("GPT attention forward validation failed")
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut parameters = self.q_proj.parameters();
        parameters.extend(self.k_proj.parameters());
        parameters.extend(self.v_proj.parameters());
        parameters.extend(self.out_proj.parameters());
        parameters
    }

    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Tensor)> {
        let mut parameters = self.q_proj.named_parameters("");
        parameters.extend(self.k_proj.named_parameters(""));
        parameters.extend(self.v_proj.named_parameters(""));
        parameters.extend(self.out_proj.named_parameters(""));
        parameters
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    gamma_name: String,
    beta_name: String,
    feature_dim: usize,
    eps: f32,
}

impl LayerNorm {
    pub fn new(
        backend: &CpuAutogradBackend,
        feature_dim: usize,
        eps: f32,
        name_prefix: &str,
    ) -> Result<Self, BackendError> {
        let gamma = backend.from_data(vec![1.0; feature_dim], vec![1, feature_dim], true)?;
        let beta = backend.from_data(vec![0.0; feature_dim], vec![1, feature_dim], true)?;

        Ok(Self {
            gamma,
            beta,
            gamma_name: format!("{name_prefix}.gamma"),
            beta_name: format!("{name_prefix}.beta"),
            feature_dim,
            eps,
        })
    }
}

impl CheckedModule for LayerNorm {
    fn try_forward(
        &self,
        backend: &CpuAutogradBackend,
        input: &Tensor,
    ) -> Result<Tensor, BackendError> {
        let input_shape = backend.shape(input);
        if input_shape.len() != 2 {
            return Err(BackendError::InvalidShape {
                context: "LayerNorm::forward",
                expected_rank: 2,
                found_rank: input_shape.len(),
            });
        }
        if input_shape[1] != self.feature_dim {
            return Err(BackendError::Unsupported(
                "LayerNorm feature dim mismatch with input last dimension",
            ));
        }

        let inv_n = backend.scalar(1.0 / self.feature_dim as f32, false)?;
        let mean_sum = backend.sum_last_dim(input)?;
        let mean = backend.mul(&mean_sum, &inv_n)?;

        let neg_one = backend.scalar(-1.0, false)?;
        let neg_mean = backend.mul(&mean, &neg_one)?;
        let centered = backend.add(input, &neg_mean)?;

        let centered_sq = backend.mul(&centered, &centered)?;
        let var_sum = backend.sum_last_dim(&centered_sq)?;
        let var = backend.mul(&var_sum, &inv_n)?;
        let eps = backend.scalar(self.eps, false)?;
        let var_eps = backend.add(&var, &eps)?;
        let std = backend.sqrt(&var_eps)?;

        let normalized = backend.div(&centered, &std)?;
        let scaled = backend.mul(&normalized, &self.gamma)?;
        backend.add(&scaled, &self.beta)
    }
}

impl CanonicalModule for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.try_forward(&CpuAutogradBackend, input)
            .expect("GPT LayerNorm forward validation failed")
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (self.gamma_name.clone(), self.gamma.clone()),
            (self.beta_name.clone(), self.beta.clone()),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct Dropout {
    p: f32,
    training: bool,
    rng: RefCell<StdRng>,
}

impl Dropout {
    pub fn new_with_seed(p: f32, seed: u64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            training: true,
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    pub fn forward(
        &self,
        backend: &CpuAutogradBackend,
        input: &Tensor,
    ) -> Result<Tensor, BackendError> {
        if !self.training || self.p <= 0.0 {
            let zero = backend.scalar(0.0, false)?;
            return backend.add(input, &zero);
        }

        if self.p >= 1.0 {
            let zero = backend.scalar(0.0, false)?;
            return backend.mul(input, &zero);
        }

        let shape = backend.shape(input);
        let input_data = backend.data(input);
        let keep_prob = 1.0 - self.p;
        let inv_keep = 1.0 / keep_prob;

        let mut mask_data = vec![0.0; input_data.len()];
        {
            let mut rng = self.rng.borrow_mut();
            for mask_value in &mut mask_data {
                let keep = rng.random::<f32>() < keep_prob;
                *mask_value = if keep { inv_keep } else { 0.0 };
            }
        }

        let mask = backend.from_data(mask_data, shape, false)?;
        backend.mul(input, &mask)
    }
}

pub struct TransformerBlock {
    pub attention: MultiHeadSelfAttention,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub ff1: Linear,
    pub ff2: Linear,
    dropout_attn: Dropout,
    dropout_ff: Dropout,
}

impl TransformerBlock {
    #[cfg(test)]
    pub fn new(
        backend: &CpuAutogradBackend,
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        name_prefix: &str,
    ) -> Result<Self, BackendError> {
        Self::new_with_dropout(backend, model_dim, num_heads, ff_dim, 0.0, name_prefix)
    }

    #[cfg(test)]
    pub fn new_with_dropout(
        backend: &CpuAutogradBackend,
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        dropout_p: f32,
        name_prefix: &str,
    ) -> Result<Self, BackendError> {
        Self::new_with_dropout_seeded(
            backend,
            model_dim,
            num_heads,
            ff_dim,
            dropout_p,
            0,
            name_prefix,
        )
    }

    pub fn new_with_dropout_seeded(
        backend: &CpuAutogradBackend,
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        dropout_p: f32,
        dropout_seed: u64,
        name_prefix: &str,
    ) -> Result<Self, BackendError> {
        Ok(Self {
            attention: MultiHeadSelfAttention::new(
                backend,
                model_dim,
                num_heads,
                &format!("{name_prefix}.attn"),
            )?,
            ln1: LayerNorm::new(backend, model_dim, 1e-5, &format!("{name_prefix}.ln1"))?,
            ln2: LayerNorm::new(backend, model_dim, 1e-5, &format!("{name_prefix}.ln2"))?,
            ff1: Linear::new(backend, model_dim, ff_dim, &format!("{name_prefix}.ff1"))?,
            ff2: Linear::new(backend, ff_dim, model_dim, &format!("{name_prefix}.ff2"))?,
            dropout_attn: Dropout::new_with_seed(dropout_p, dropout_seed),
            dropout_ff: Dropout::new_with_seed(dropout_p, dropout_seed.wrapping_add(1)),
        })
    }

    #[cfg(test)]
    pub fn model_dim(&self) -> usize {
        self.attention.model_dim()
    }

    #[cfg(test)]
    pub fn ff_dim(&self) -> usize {
        self.ff1.out_features()
    }

    #[cfg(test)]
    pub fn num_heads(&self) -> usize {
        self.attention.num_heads()
    }

    pub fn set_training(&mut self, training: bool) {
        self.dropout_attn.set_training(training);
        self.dropout_ff.set_training(training);
    }
}

impl CheckedModule for TransformerBlock {
    fn try_forward(
        &self,
        backend: &CpuAutogradBackend,
        input: &Tensor,
    ) -> Result<Tensor, BackendError> {
        let normed_attn_in = self.ln1.try_forward(backend, input)?;
        let attn_out = self.attention.try_forward(backend, &normed_attn_in)?;
        let attn_out = self.dropout_attn.forward(backend, &attn_out)?;
        let attn_residual = backend.add(input, &attn_out)?;

        let normed_ff_in = self.ln2.try_forward(backend, &attn_residual)?;
        let ff_hidden = self.ff1.try_forward(backend, &normed_ff_in)?;
        let ff_activated = backend.relu(&ff_hidden)?;
        let ff_out = self.ff2.try_forward(backend, &ff_activated)?;
        let ff_out = self.dropout_ff.forward(backend, &ff_out)?;
        backend.add(&attn_residual, &ff_out)
    }
}

impl CanonicalModule for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.try_forward(&CpuAutogradBackend, input)
            .expect("GPT TransformerBlock forward validation failed")
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut parameters = self.attention.parameters();
        parameters.extend(self.ln1.parameters());
        parameters.extend(self.ln2.parameters());
        parameters.extend(self.ff1.parameters());
        parameters.extend(self.ff2.parameters());
        parameters
    }

    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Tensor)> {
        let mut parameters = self.attention.named_parameters("");
        parameters.extend(self.ln1.named_parameters(""));
        parameters.extend(self.ln2.named_parameters(""));
        parameters.extend(self.ff1.named_parameters(""));
        parameters.extend(self.ff2.named_parameters(""));
        parameters
    }

    fn set_training(&mut self, training: bool) {
        TransformerBlock::set_training(self, training);
        for parameter in self.parameters() {
            parameter.set_requires_grad(training);
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct Sgd {
    pub learning_rate: f32,
}

impl Sgd {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    pub fn step<M: CheckedModule>(
        &self,
        backend: &CpuAutogradBackend,
        module: &mut M,
    ) -> Result<(), BackendError> {
        let lr = self.learning_rate;
        let mut update_result: Result<(), BackendError> = Ok(());

        for parameter in module.parameters() {
            if update_result.is_err() {
                break;
            }

            let shape = backend.shape(&parameter);
            let data = backend.data(&parameter);
            let grad = backend
                .grad(&parameter)
                .unwrap_or_else(|| vec![0.0; data.len()]);

            if data.len() != grad.len() {
                update_result = Err(BackendError::Unsupported(
                    "parameter grad/data length mismatch",
                ));
                break;
            }

            let mut updated = vec![0.0; data.len()];
            for (i, dst) in updated.iter_mut().enumerate() {
                *dst = data[i] - lr * grad[i];
            }

            match backend.from_data(updated, shape, true) {
                Ok(new_tensor) => {
                    let new_storage = new_tensor.lock().storage.clone();
                    let mut parameter_data = parameter.lock();
                    parameter_data.storage = new_storage;
                    parameter_data.grad = None;
                    parameter_data.creator = None;
                    parameter_data.inputs.clear();
                }
                Err(e) => {
                    update_result = Err(e);
                }
            }
        }

        update_result
    }
}

#[cfg(test)]
fn train_step_mse<M: CheckedModule>(
    backend: &CpuAutogradBackend,
    model: &mut M,
    optimizer: &Sgd,
    input: &Tensor,
    target: &Tensor,
) -> Result<f32, BackendError> {
    let prediction = model.try_forward(backend, input)?;

    let minus_one = backend.scalar(-1.0, false)?;
    let neg_target = backend.mul(target, &minus_one)?;
    let diff = backend.add(&prediction, &neg_target)?;
    let sq = backend.mul(&diff, &diff)?;
    let loss = backend.mean(&sq)?;

    backend.backward(&loss)?;
    optimizer.step(backend, model)?;

    let loss_values = backend.data(&loss);
    let first = loss_values
        .first()
        .copied()
        .ok_or(BackendError::EmptyTensor {
            context: "train_step_mse(loss)",
        })?;

    Ok(first)
}

#[cfg(test)]
mod tests {
    use super::{
        train_step_mse, CheckedModule, Dropout, LayerNorm, Linear, MultiHeadSelfAttention, Sgd,
        TransformerBlock,
    };
    use crate::nn::gpt::framework::backend::CpuAutogradBackend;

    #[test]
    fn linear_train_step_updates_parameters() {
        let backend = CpuAutogradBackend;
        let mut linear = Linear::new(&backend, 3, 2, "linear").unwrap();
        let optimizer = Sgd::new(0.1);

        let input = backend
            .from_data(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .unwrap();
        let target = backend
            .from_data(vec![0.5, -0.25], vec![1, 2], false)
            .unwrap();

        let before = backend.data(&linear.weight);
        let loss = train_step_mse(&backend, &mut linear, &optimizer, &input, &target).unwrap();
        let after = backend.data(&linear.weight);

        assert!(loss.is_finite());
        assert_ne!(before, after);

        let out = linear.try_forward(&backend, &input).unwrap();
        let out_shape = backend.shape(&out);
        assert_eq!(out_shape, vec![1, 2]);
    }

    #[test]
    fn transformer_block_forward_preserves_model_dim() {
        let backend = CpuAutogradBackend;
        let block = TransformerBlock::new(&backend, 4, 2, 8, "block").unwrap();

        let input = backend
            .from_data(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4], false)
            .unwrap();
        let output = block.try_forward(&backend, &input).unwrap();

        assert_eq!(block.model_dim(), 4);
        assert_eq!(block.num_heads(), 2);
        assert_eq!(block.ff_dim(), 8);
        assert_eq!(backend.shape(&output), vec![1, 4]);
    }

    #[test]
    fn gpt_layers_implement_canonical_module_contract() {
        use crate::nn::Module as CanonicalModule;

        let backend = CpuAutogradBackend;
        let block = TransformerBlock::new(&backend, 4, 2, 8, "block").unwrap();
        let input = backend.from_data(vec![0.1; 12], vec![3, 4], false).unwrap();

        let output = CanonicalModule::forward(&block, &input);
        assert_eq!(output.shape(), vec![3, 4]);

        let parameters = CanonicalModule::parameters(&block);
        let named = CanonicalModule::named_parameters(&block, "");
        assert_eq!(parameters.len(), named.len());
        assert!(!parameters.is_empty());
        assert!(named.iter().all(|(name, _)| !name.is_empty()));
    }

    #[test]
    fn transformer_block_train_step_updates_parameters() {
        let backend = CpuAutogradBackend;
        let mut block = TransformerBlock::new(&backend, 4, 2, 8, "block").unwrap();
        block.set_training(false);
        let optimizer = Sgd::new(0.05);

        let input = backend
            .from_data(vec![0.5, -0.25, 0.75, 1.0], vec![1, 4], false)
            .unwrap();
        let target = backend
            .from_data(vec![0.1, 0.0, -0.2, 0.3], vec![1, 4], false)
            .unwrap();

        let before = backend.data(&block.attention.out_proj.weight);
        let loss = train_step_mse(&backend, &mut block, &optimizer, &input, &target).unwrap();
        let after = backend.data(&block.attention.out_proj.weight);

        assert!(loss.is_finite());
        assert_ne!(before, after);
    }

    #[test]
    fn transformer_block_seeded_dropout_is_reproducible() {
        let backend = CpuAutogradBackend;
        let mut block1 =
            TransformerBlock::new_with_dropout_seeded(&backend, 4, 2, 8, 0.3, 42, "block").unwrap();
        let mut block2 =
            TransformerBlock::new_with_dropout_seeded(&backend, 4, 2, 8, 0.3, 42, "block").unwrap();

        block1.set_training(true);
        block2.set_training(true);

        let input = backend
            .from_data(
                vec![
                    0.5, -0.25, 0.75, 1.0, // token 0
                    -0.1, 0.2, -0.3, 0.4, // token 1
                ],
                vec![2, 4],
                false,
            )
            .unwrap();

        let y1 = backend.data(&block1.try_forward(&backend, &input).unwrap());
        let y2 = backend.data(&block2.try_forward(&backend, &input).unwrap());
        assert_eq!(y1, y2);
    }

    #[test]
    fn layer_norm_outputs_zero_mean_per_row() {
        let backend = CpuAutogradBackend;
        let ln = LayerNorm::new(&backend, 4, 1e-5, "ln").unwrap();
        let x = backend
            .from_data(
                vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, -2.0, -4.0],
                vec![2, 4],
                false,
            )
            .unwrap();

        let y = ln.try_forward(&backend, &x).unwrap();
        let out = backend.data(&y);

        let row0_mean = (out[0] + out[1] + out[2] + out[3]) / 4.0;
        let row1_mean = (out[4] + out[5] + out[6] + out[7]) / 4.0;

        assert!(row0_mean.abs() < 1e-4);
        assert!(row1_mean.abs() < 1e-4);
    }

    #[test]
    fn dropout_seeded_is_reproducible_across_instances() {
        let backend = CpuAutogradBackend;
        let d1 = Dropout::new_with_seed(0.25, 1234);
        let d2 = Dropout::new_with_seed(0.25, 1234);

        let input = backend
            .from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .unwrap();

        let y1 = backend.data(&d1.forward(&backend, &input).unwrap());
        let y2 = backend.data(&d2.forward(&backend, &input).unwrap());
        assert_eq!(y1, y2);
    }

    #[test]
    fn dropout_eval_mode_is_identity() {
        let backend = CpuAutogradBackend;
        let mut dropout = Dropout::new_with_seed(0.5, 7);
        dropout.set_training(false);

        let input = backend
            .from_data(vec![0.5, -1.0, 2.0, 3.5], vec![2, 2], false)
            .unwrap();
        let out = backend.data(&dropout.forward(&backend, &input).unwrap());
        assert_eq!(out, vec![0.5, -1.0, 2.0, 3.5]);
    }

    #[test]
    fn dropout_inverted_scaling_preserves_mean_in_expectation() {
        let backend = CpuAutogradBackend;
        let dropout = Dropout::new_with_seed(0.2, 99);

        let n = 10_000usize;
        let input = backend.from_data(vec![1.0; n], vec![1, n], false).unwrap();
        let out = backend.data(&dropout.forward(&backend, &input).unwrap());
        let mean = out.iter().sum::<f32>() / n as f32;

        assert!((mean - 1.0).abs() < 0.08);
    }

    #[test]
    fn multihead_attention_scaffold_preserves_shape() {
        let backend = CpuAutogradBackend;
        let attention = MultiHeadSelfAttention::new(&backend, 6, 3, "attn").unwrap();
        let input = backend
            .from_data(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![1, 6], false)
            .unwrap();

        let output = attention.try_forward(&backend, &input).unwrap();
        assert_eq!(attention.model_dim(), 6);
        assert_eq!(attention.num_heads(), 3);
        assert_eq!(attention.head_dim(), 2);
        assert_eq!(backend.shape(&output), vec![1, 6]);
    }

    #[test]
    fn multihead_attention_is_causal_for_prefix_positions() {
        let backend = CpuAutogradBackend;
        let attention = MultiHeadSelfAttention::new(&backend, 4, 2, "attn").unwrap();

        let base = backend
            .from_data(
                vec![
                    0.1, 0.2, 0.3, 0.4, // token 0
                    0.2, 0.1, 0.0, -0.1, // token 1
                    -0.2, 0.3, 0.4, 0.5, // token 2
                ],
                vec![3, 4],
                false,
            )
            .unwrap();
        let changed_future = backend
            .from_data(
                vec![
                    0.1, 0.2, 0.3, 0.4, // token 0 unchanged
                    0.2, 0.1, 0.0, -0.1, // token 1 unchanged
                    9.0, 9.0, 9.0, 9.0, // token 2 changed (future for token 0/1)
                ],
                vec![3, 4],
                false,
            )
            .unwrap();

        let out_base = backend.data(&attention.try_forward(&backend, &base).unwrap());
        let out_changed = backend.data(&attention.try_forward(&backend, &changed_future).unwrap());

        for i in 0..8 {
            assert!((out_base[i] - out_changed[i]).abs() < 1e-5);
        }
    }
}
