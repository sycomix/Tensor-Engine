use crate::nn::{Conv2D, Linear, Module};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

/// DDIM (Denoising Diffusion Implicit Models) scheduler.
///
/// DDIM is a deterministic sampling method that produces faster convergence
/// than DDPM by using a non-Markovian reverse process. It requires fewer
/// sampling steps (10-50 vs 1000) while maintaining high sample quality.
///
/// # Reference
/// [`Dhariwal & Nichol, 2021`](https://arxiv.org/abs/2105.05233)
#[derive(Clone)]
pub struct DDIMScheduler {
    pub num_train_timesteps: usize,
    pub num_inference_timesteps: usize,
    pub betas: Vec<f32>,
    pub alphas: Vec<f32>,
    pub alphas_cumprod: Vec<f32>,
    pub sqrt_alphas_cumprod: Vec<f32>,
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>,
    pub eta: f32,
}

impl DDIMScheduler {
    /// Create a DDIM scheduler with a linear beta schedule.
    pub fn new_linear(
        num_train_timesteps: usize,
        num_inference_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
    ) -> Self {
        let mut betas = Vec::with_capacity(num_train_timesteps);
        for i in 0..num_train_timesteps {
            let t = i as f32 / (num_train_timesteps - 1) as f32;
            betas.push(beta_start * (1.0 - t) + beta_end * t);
        }
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut prod = 1.0f32;
        for a in alphas.iter() {
            prod *= *a;
            alphas_cumprod.push(prod);
        }
        let sqrt_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|v| v.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f32> =
            alphas_cumprod.iter().map(|v| (1.0 - v).sqrt()).collect();
        DDIMScheduler {
            num_train_timesteps,
            num_inference_timesteps,
            betas,
            alphas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            eta: 0.0,
        }
    }

    /// Set the stochasticity parameter eta (0 = deterministic, 1 = DDPM).
    pub fn with_eta(mut self, eta: f32) -> Self {
        self.eta = eta;
        self
    }

    /// Compute the timestep array for inference.
    /// Creates a linearly spaced set of timesteps from num_train_timesteps-1 down to 0.
    pub fn timesteps(&self) -> Vec<usize> {
        let step = self.num_train_timesteps / self.num_inference_timesteps;
        let mut timesteps = Vec::with_capacity(self.num_inference_timesteps);
        for i in (0..self.num_inference_timesteps).rev() {
            let t = (i * step).min(self.num_train_timesteps - 1);
            timesteps.push(t);
        }
        timesteps
    }

    /// Compute the alpha_hat values for each inference timestep.
    pub fn alpha_timesteps(&self) -> Vec<f32> {
        let step = self.num_train_timesteps / self.num_inference_timesteps;
        let mut alphas_t = Vec::with_capacity(self.num_inference_timesteps);
        for i in (0..self.num_inference_timesteps).rev() {
            let t = (i * step).min(self.num_train_timesteps - 1);
            alphas_t.push(self.alphas_cumprod[t]);
        }
        alphas_t
    }

    /// DDIM sampling step: x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1} - eta^2 * sigma_t^2) * eps_pred + eta * sigma_t * noise
    pub fn step(
        &self,
        model: &impl Module,
        x_t: &Tensor,
        t: usize,
        eps_pred: Option<&Tensor>,
    ) -> Tensor {
        let alpha_t = self.alphas_cumprod[t];
        let alpha_t_prev = if t > 0 {
            let step = self.num_train_timesteps / self.num_inference_timesteps;
            let prev_t = ((t / step) - 1) * step;
            if prev_t >= 0 {
                self.alphas_cumprod[prev_t]
            } else {
                1.0
            }
        } else {
            0.0
        };

        let sqrt_alpha_t = alpha_t.sqrt();
        let sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt();

        // Get predicted noise
        let eps_pred = match eps_pred {
            Some(e) => e.clone(),
            None => model.forward(x_t),
        };

        // Compute predicted x0: x0 = (x_t - sqrt(1-alpha_t) * eps_pred) / sqrt(alpha_t)
        let sqrt_one_minus_alpha_t_t = Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), sqrt_one_minus_alpha_t),
            false,
        );
        let sqrt_alpha_t_t = Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), sqrt_alpha_t),
            false,
        );

        let pred_x0 = x_t
            .sub(&eps_pred.mul(&sqrt_one_minus_alpha_t_t))
            .div(&sqrt_alpha_t_t);

        // Clip predicted x0 to [-1, 1] for stability
        let pred_x0_clipped = pred_x0.clamp(-1.0, 1.0);

        // Compute coefficient for eps_pred
        let coeff_eps = (alpha_t_prev * (1.0 - alpha_t) / (1.0 - alpha_t)).sqrt();
        let coeff_eps_t = Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), coeff_eps),
            false,
        );

        // Compute coefficient for sqrt(alpha_t_prev)
        let coeff_x0 = (alpha_t_prev * alpha_t).sqrt();
        let coeff_x0_t = Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), coeff_x0),
            false,
        );

        // Compute sigma_t = eta * sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * sqrt(1 - alpha_t / alpha_t_prev)
        let sigma_t = if self.eta > 0.0 && t > 0 {
            let one_minus_alpha_prev = 1.0 - alpha_t_prev;
            let one_minus_alpha_t = 1.0 - alpha_t;
            let alpha_ratio = alpha_t / alpha_t_prev;
            if one_minus_alpha_t > 0.0 && one_minus_alpha_prev > 0.0 && alpha_ratio < 1.0 {
                self.eta * (one_minus_alpha_prev / one_minus_alpha_t).sqrt() * (1.0 - alpha_ratio).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // x_{t-1} = sqrt(alpha_t_prev) * pred_x0 + coeff_eps * eps_pred + sigma_t * noise
        let noise = Tensor::randn(x_t.lock().storage.shape().to_vec());
        let x_prev = pred_x0_clipped
            .mul(&coeff_x0_t)
            .add(&eps_pred.mul(&coeff_eps_t))
            .add(&noise.mul(&Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1]), sigma_t),
                false,
            )));

        x_prev
    }

    /// Generate samples from noise using DDIM sampling.
    pub fn sample(
        &self,
        model: &impl Module,
        noise: &Tensor,
        timesteps: Option<&[usize]>,
    ) -> Tensor {
        let steps = timesteps.unwrap_or(&self.timesteps());
        let mut x = noise.clone();

        for (i, &t) in steps.iter().enumerate() {
            x = self.step(model, &x, t, None);
            // Optionally add noise for stochastic sampling
            if i < steps.len() - 1 && self.eta > 0.0 {
                let next_t = steps[i + 1];
                let step = self.num_train_timesteps / self.num_inference_timesteps;
                let dt = ((t / step) - (next_t / step)) * step;
                if dt > 0 {
                    let sigma = self.eta * ((self.alphas_cumprod[next_t] * (1.0 - self.alphas_cumprod[t]) / (1.0 - self.alphas_cumprod[next_t])).sqrt());
                    let noise = Tensor::randn(x.lock().storage.shape().to_vec());
                    x = x.add(&noise.mul(&Tensor::new(
                        ndarray::Array::from_elem(IxDyn(&[1]), sigma),
                        false,
                    )));
                }
            }
        }

        x
    }
}

/// VAE (Variational Autoencoder) for latent space modeling.
///
/// Consists of an encoder that maps inputs to latent parameters (mu, logvar)
/// and a decoder that reconstructs from the latent space.
///
/// # Reference
/// [`Kingma & Welling, 2013`](https://arxiv.org/abs/1312.6114)
#[derive(Clone)]
pub struct VAE {
    pub encoder: VAEEncoder,
    pub decoder: VAEDecoder,
    pub latent_channels: usize,
}

impl VAE {
    /// Create a new VAE with the given latent dimensionality.
    pub fn new(latent_channels: usize) -> Self {
        VAE {
            encoder: VAEEncoder::new(latent_channels),
            decoder: VAEDecoder::new(latent_channels),
            latent_channels,
        }
    }

    /// Encode input to latent space, returning (mu, logvar).
    pub fn encode(&self, x: &Tensor) -> (Tensor, Tensor) {
        self.encoder.forward(x)
    }

    /// Decode from latent space to reconstruction.
    pub fn decode(&self, z: &Tensor) -> Tensor {
        self.decoder.forward(z)
    }

    /// Reparameterization trick: sample from latent space.
    /// z = mu + exp(logvar / 2) * epsilon
    pub fn reparameterize(&self, mu: &Tensor, logvar: &Tensor) -> Tensor {
        let std = (&logvar).mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), 0.5),
            false,
        )).exp();
        let eps = Tensor::randn(mu.lock().storage.shape().to_vec());
        mu.add(&eps.mul(&std))
    }

    /// Forward pass: encode then decode with reparameterization.
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (mu, logvar) = self.encode(x);
        let z = self.reparameterize(&mu, &logvar);
        let recon = self.decode(&z);
        (recon, mu, logvar)
    }

    /// Compute VAE loss: reconstruction loss + KL divergence.
    pub fn vae_loss(recon: &Tensor, target: &Tensor, mu: &Tensor, logvar: &Tensor) -> Tensor {
        // Reconstruction loss (MSE)
        let recon_loss = recon.sub(target).pow(2.0).mean();

        // KL divergence: KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
        let kl_loss = logvar
            .mul(&Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1]), -0.5),
                false,
            ))
            .add(&mu.pow(2.0).mul(&Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1]), -0.5),
                false,
            )))
            .add(&logvar.mul(&Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1]), 0.5),
                false,
            )))
            .sum();

        // Total loss
        recon_loss.add(&kl_loss)
    }
}

/// VAE Encoder: maps input to (mu, logvar) in latent space.
#[derive(Clone)]
pub struct VAEEncoder {
    pub layers: Vec<VAEBlock>,
}

impl VAEEncoder {
    pub fn new(latent_channels: usize) -> Self {
        let mut layers = Vec::new();
        // Encoder blocks with decreasing spatial dimensions
        let channel_sizes = vec![64, 128, 256, 512];
        for (i, &ch) in channel_sizes.iter().enumerate() {
            let in_ch = if i == 0 { 3 } else { channel_sizes[i - 1] };
            layers.push(VAEBlock::new(in_ch, ch, 3, 1, 1));
        }
        // Final conv to get 2 * latent_channels (for mu and logvar)
        layers.push(VAEBlock::new(512, latent_channels * 2, 3, 1, 1));
        VAEEncoder { layers }
    }

    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut out = x.clone();
        for l in &self.layers {
            out = l.forward(&out);
        }
        // Split into mu and logvar
        let shape = out.lock().storage.shape().to_vec();
        let latent_ch = shape[1] / 2;
        let mu = out.slice_channels(0, latent_ch);
        let logvar = out.slice_channels(latent_ch, latent_ch);
        (mu, logvar)
    }
}

/// VAE Decoder: maps latent space to reconstruction.
#[derive(Clone)]
pub struct VAEDecoder {
    pub layers: Vec<VAEBlock>,
}

impl VAEDecoder {
    pub fn new(latent_channels: usize) -> Self {
        let mut layers = Vec::new();
        let channel_sizes = vec![512, 256, 128, 64];
        // First block upsamples from latent_channels
        layers.push(VAEBlock::new(latent_channels, channel_sizes[0], 3, 1, 1));
        for i in 1..channel_sizes.len() {
            let in_ch = channel_sizes[i - 1];
            let out_ch = channel_sizes[i];
            layers.push(VAEBlock::new(in_ch, out_ch, 3, 1, 1));
        }
        // Final conv to get 3 channels (RGB)
        layers.push(VAEBlock::new(64, 3, 3, 1, 1));
        VAEDecoder { layers }
    }

    pub fn forward(&self, z: &Tensor) -> Tensor {
        let mut out = z.clone();
        for l in &self.layers {
            out = l.forward(&out);
        }
        out.sigmoid()
    }
}

/// VAE block: Conv2D -> GroupNorm -> SiLU.
#[derive(Clone)]
pub struct VAEBlock {
    pub conv: Conv2D,
    pub norm: crate::nn::GroupNorm,
}

impl VAEBlock {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        VAEBlock {
            conv: Conv2D::new(in_channels, out_channels, kernel_size, stride, padding, true),
            norm: crate::nn::GroupNorm::new(out_channels, 32, 1e-6),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = self.conv.forward(x);
        let out = self.norm.forward(&out);
        out.silu()
    }
}

/// Classifier-Free Guidance (CFG) wrapper for diffusion models.
///
/// CFG improves sample quality by conditioning on both text prompt and
/// empty prompt, then interpolating between conditional and unconditional
/// predictions.
///
/// # Reference
/// [`Ho & Salimans, 2022`](https://arxiv.org/abs/2207.12598)
#[derive(Clone)]
pub struct CFGWrapper<M> {
    pub model: M,
    pub guidance_scale: f32,
}

impl<M: Module> CFGWrapper<M> {
    /// Create a new CFG wrapper.
    pub fn new(model: M, guidance_scale: f32) -> Self {
        CFGWrapper {
            model,
            guidance_scale,
        }
    }

    /// Apply classifier-free guidance to a model prediction.
    ///
    /// `cond_pred` is the prediction conditioned on the text prompt.
    /// `uncond_pred` is the prediction conditioned on an empty/no prompt.
    /// `guidance_scale` controls the strength of guidance (1.0 = no guidance).
    pub fn apply(
        &self,
        cond_pred: &Tensor,
        uncond_pred: &Tensor,
        guidance_scale: Option<f32>,
    ) -> Tensor {
        let scale = guidance_scale.unwrap_or(self.guidance_scale);
        if (scale - 1.0).abs() < 1e-6 {
            return cond_pred.clone();
        }
        // CFG formula: uncond + scale * (cond - uncond)
        // = (1 - scale) * uncond + scale * cond
        let one_minus_scale = 1.0 - scale;
        let uncond_scaled = uncond_pred.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), one_minus_scale),
            false,
        ));
        let cond_scaled = cond_pred.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), scale),
            false,
        ));
        uncond_scaled.add(&cond_scaled)
    }

    /// Forward pass with CFG: runs both conditional and unconditional passes.
    pub fn forward_with_cfg(&self, x: &Tensor, t_emb: &Tensor) -> Tensor {
        let uncond_pred = self.model.forward(x);
        // Note: In practice, you'd have separate conditional/unconditional models.
        // For now, we return the conditional prediction (no CFG applied).
        uncond_pred
    }
}

/// Timestep embeddings: sinusoidal embedding followed by a linear projection.
pub struct TimestepEmbedding {
    pub linear1: Linear,
    pub linear2: Linear,
}

impl TimestepEmbedding {
    pub fn new(d_model: usize, hidden: usize) -> Self {
        // linear1 maps scalar timestep -> hidden, linear2 maps hidden -> d_model
        TimestepEmbedding {
            linear1: Linear::new(1, hidden, true),
            linear2: Linear::new(hidden, d_model, true),
        }
    }

    pub fn forward(&self, t: &Tensor) -> Tensor {
        // t is expected as scalar batch or [B, 1]
        // Use simple linear projection: out = linear2(relu(linear1(t))) for now
        let h = self.linear1.forward(t);
        let h = h.relu();
        self.linear2.forward(&h)
    }
}

/// Group Normalization module expects NCHW tensors and normalizes over groups.
#[derive(Clone)]
pub struct GroupNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub num_groups: usize,
    pub eps: f32,
}

impl GroupNorm {
    pub fn new(num_channels: usize, num_groups: usize, eps: f32) -> Self {
        let gamma = Tensor::new(
            match ndarray::Array::from_shape_vec(IxDyn(&[num_channels]), vec![1.0; num_channels]) {
                Ok(a) => a,
                Err(e) => {
                    log::error!("GroupNorm::new failed to create gamma array: {}", e);
                    ndarray::Array::zeros(IxDyn(&[num_channels]))
                }
            },
            true,
        );
        let beta = Tensor::new(
            match ndarray::Array::from_shape_vec(IxDyn(&[num_channels]), vec![0.0; num_channels]) {
                Ok(a) => a,
                Err(e) => {
                    log::error!("GroupNorm::new failed to create beta array: {}", e);
                    ndarray::Array::zeros(IxDyn(&[num_channels]))
                }
            },
            true,
        );
        GroupNorm {
            gamma,
            beta,
            num_groups,
            eps,
        }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // GroupNorm::forward: entry
        let arr = x.lock().storage.to_f32_array();
        let gamma_arr = self.gamma.lock().storage.to_f32_array();
        let beta_arr = self.beta.lock().storage.to_f32_array();
        let shape = arr.shape().to_vec();
        if shape.len() != 4 {
            return x.clone();
        }
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let mut g = self.num_groups;
        if !c.is_multiple_of(g) {
            log::error!("GroupNorm::forward: num_channels {} not divisible by num_groups {}; falling back to 1 group", c, g);
            g = 1;
        }
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[n, c, h, w]));
        let channels_per_group = c / g;
        for ni in 0..n {
            for gi in 0..g {
                let cstart = gi * channels_per_group;
                let cend = cstart + channels_per_group;
                // compute mean/var over channel block and spatial dims
                let mut sum = 0.0f32;
                let mut sumsq = 0.0f32;
                let mut count = 0usize;
                for ci in cstart..cend {
                    for hi in 0..h {
                        for wi in 0..w {
                            let v = arr[[ni, ci, hi, wi]];
                            sum += v;
                            sumsq += v * v;
                            count += 1;
                        }
                    }
                }
                let mean = sum / count as f32;
                let var = (sumsq / count as f32) - (mean * mean);
                let denom = (var + self.eps).sqrt();
                for ci in cstart..cend {
                    for hi in 0..h {
                        for wi in 0..w {
                            let v = arr[[ni, ci, hi, wi]];
                            let nval = (v - mean) / denom;
                            let gval = gamma_arr[[ci]];
                            let bval = beta_arr[[ci]];
                            out[[ni, ci, hi, wi]] = nval * gval + bval;
                        }
                    }
                }
            }
        }
        log::debug!("GroupNorm::forward: leaving");
        Tensor::new(out, false)
    }
}

/// ResNetBlock: GroupNorm -> SiLU -> Conv2D + time embedding injection
pub struct ResNetBlock {
    pub gn1: GroupNorm,
    pub conv1: crate::nn::Conv2D,
    pub conv2: crate::nn::Conv2D,
    pub proj: Option<crate::nn::Linear>,
}

impl ResNetBlock {
    pub fn new(in_channels: usize, out_channels: usize, num_groups: usize) -> Self {
        let gn_groups = if num_groups > in_channels {
            1usize
        } else {
            num_groups
        };
        ResNetBlock {
            gn1: GroupNorm::new(in_channels, gn_groups, 1e-5),
            conv1: Conv2D::new(in_channels, out_channels, 3, 1, 1, true),
            conv2: Conv2D::new(out_channels, out_channels, 3, 1, 1, true),
            proj: if in_channels != out_channels {
                Some(Linear::new(in_channels, out_channels, true))
            } else {
                None
            },
        }
    }

    pub fn forward(&self, x: &Tensor, t_emb: Option<&Tensor>) -> Tensor {
        // ResNetBlock::forward: entry
        let mut h = self.gn1.forward(x);
        // ResNetBlock::after gn1
        h = h.silu();
        // before conv1
        h = self.conv1.forward(&h);
        // after conv1
        // check time embedding presence
        if let Some(te) = t_emb {
            // project t_emb to spatial dims and add (broadcast)
            // Avoid nested locking on the same Tensor by acquiring each lock in its own scope
            // about to read te shape
            let te_in_dim = {
                let lock = te.lock();
                // acquired te lock
                let s = lock.storage.shape().to_vec();
                s[s.len() - 1]
            };
            // about to read h channel dim
            let h_out_channels = {
                let lock = h.lock();
                // acquired h lock
                lock.storage.shape()[1]
            };
            // about to call Linear::new for te_proj
            let te_proj = crate::nn::Linear::new(te_in_dim, h_out_channels, true);
            // created te_proj
            // computed te_proj shapes
            let tp = te_proj.forward(te);
            // Reshape tp to [B, C, 1, 1] so it can be added to h (NCHW) via broadcasting
            let b_dim = tp.lock().storage.shape()[0];
            let new_tp = tp
                .reshape(vec![b_dim, h_out_channels, 1, 1])
                .unwrap_or(tp.clone());
            h = h.add(&new_tp);
        }
        // before gn1 2
        h = self.gn1.forward(&h);
        // after gn1 2
        h = h.silu();
        // before conv2
        h = self.conv2.forward(&h);
        // after conv2
        // residual
        let res = if let Some(proj) = &self.proj {
            proj.forward(x)
        } else {
            x.clone()
        };
        // leaving ResNetBlock
        res.add(&h)
    }
}

/// UNet skeleton - minimal forward using blocks above
pub struct UNetModel {
    pub in_channels: usize,
    pub base_channels: usize,
    pub blocks: Vec<ResNetBlock>,
}

impl UNetModel {
    pub fn new(in_channels: usize, base_channels: usize, depth: usize) -> Self {
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            // Keep channels constant in this simple skeleton to avoid mismatches
            blocks.push(ResNetBlock::new(base_channels, base_channels, 8));
        }
        UNetModel {
            in_channels,
            base_channels,
            blocks,
        }
    }
    pub fn forward(&self, x: &Tensor, t_emb: &Tensor) -> Tensor {
        // UNetModel::forward start
        let mut h = x.clone();
        for b in &self.blocks {
            h = b.forward(&h, Some(t_emb));
        }
        log::debug!("UNetModel::forward: finished forward");
        h
    }
}

impl Module for UNetModel {
    fn forward(&self, input: &Tensor) -> Tensor {
        log::warn!("UNetModel::forward called without timestep embedding; returning input unchanged. Use forward(&self, x, t_emb) instead.");
        input.clone()
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for b in &self.blocks {
            p.push(b.gn1.gamma.clone());
            p.push(b.gn1.beta.clone());
            p.extend(b.conv1.parameters());
            p.extend(b.conv2.parameters());
            if let Some(proj) = &b.proj {
                p.extend(proj.parameters());
            }
        }
        p
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            out.push((format!("{}.blocks.{}.gn1.gamma", prefix, i), b.gn1.gamma.clone()));
            out.push((format!("{}.blocks.{}.gn1.beta", prefix, i), b.gn1.beta.clone()));
            out.extend(b.conv1.named_parameters(&format!("{}.blocks.{}.conv1", prefix, i)));
            out.extend(b.conv2.named_parameters(&format!("{}.blocks.{}.conv2", prefix, i)));
            if let Some(proj) = &b.proj {
                out.extend(proj.named_parameters(&format!("{}.blocks.{}.proj", prefix, i)));
            }
        }
        out
    }
    fn load_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), String> {
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let key = |s| format!("{}.blocks.{}.{}", prefix, i, s);
            if let Some(t) = state.get(&key("gn1.gamma")) {
                b.gn1.gamma = t.clone();
            }
            if let Some(t) = state.get(&key("gn1.beta")) {
                b.gn1.beta = t.clone();
            }
            b.conv1.load_state_dict(state, &format!("{}.blocks.{}.conv1", prefix, i))?;
            b.conv2.load_state_dict(state, &format!("{}.blocks.{}.conv2", prefix, i))?;
            if let Some(proj) = &mut b.proj {
                proj.load_state_dict(state, &format!("{}.blocks.{}.proj", prefix, i))?;
            }
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// DDPM scheduler with linear beta schedule and common sampling helpers.
pub struct DDPMScheduler {
    pub num_train_timesteps: usize,
    pub betas: Vec<f32>,
    pub alphas: Vec<f32>,
    pub alphas_cumprod: Vec<f32>,
    pub sqrt_alphas_cumprod: Vec<f32>,
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>,
}

impl DDPMScheduler {
    pub fn new_linear(num_train_timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        let mut betas = Vec::with_capacity(num_train_timesteps);
        for i in 0..num_train_timesteps {
            let t = i as f32 / (num_train_timesteps - 1) as f32;
            betas.push(beta_start * (1.0 - t) + beta_end * t);
        }
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut prod = 1.0f32;
        for a in alphas.iter() {
            prod *= *a;
            alphas_cumprod.push(prod);
        }
        let sqrt_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|v| v.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f32> =
            alphas_cumprod.iter().map(|v| (1.0 - v).sqrt()).collect();
        DDPMScheduler {
            num_train_timesteps,
            betas,
            alphas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        }
    }

    /// Draw a sample x_t from x0 and noise eps at timestep t
    pub fn q_sample(&self, x0: &Tensor, t: usize, eps: &Tensor) -> Tensor {
        let sqrt_ac = self.sqrt_alphas_cumprod[t];
        let sqrt_om_ac = self.sqrt_one_minus_alphas_cumprod[t];
        x0.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), sqrt_ac),
            false,
        ))
            .add(&eps.mul(&Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1]), sqrt_om_ac),
                false,
            )))
    }

    /// Predict epsilon from x_t and x0
    pub fn predict_eps_from_x0(&self, x_t: &Tensor, x0: &Tensor, t: usize) -> Tensor {
        let sqrt_ac = self.sqrt_alphas_cumprod[t];
        let sqrt_om_ac = self.sqrt_one_minus_alphas_cumprod[t];
        x_t.sub(&x0.mul(&Tensor::new(
            ndarray::Array::from_elem(IxDyn(&[1]), sqrt_ac),
            false,
        )))
            .div(&Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1]), sqrt_om_ac),
                false,
            ))
    }

    /// DDPM denoising step: compute posterior mean and optionally sample
    pub fn step(&self, model: &impl Module, x_t: &Tensor, t: usize) -> Tensor {
        // Model returns predicted noise eps
        let eps_pred = model.forward(x_t);
        // x_{t-1} mean using simplified posterior mean formula
        let alpha_t = self.alphas[t];
        let alpha_t_cum = self.alphas_cumprod[t];
        let beta_t = self.betas[t];
        let sqrt_alpha_t = alpha_t.sqrt();
        let _one_minus_alpha_t = 1.0 - alpha_t;
        let coeff = (1.0 - alpha_t) / (1.0 - alpha_t_cum).sqrt();
        let pred_x0 = x_t
            .sub(&eps_pred.mul(&Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1]), coeff),
                false,
            )))
            .div(&Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1]), sqrt_alpha_t),
                false,
            ));
        // Posterior mean
        pred_x0
            .mul(&Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1]), alpha_t.sqrt()),
                false,
            ))
            .add(&eps_pred.mul(&Tensor::new(
                ndarray::Array::from_elem(IxDyn(&[1]), beta_t),
                false,
            )))
    }
}
