//! Production Model Server for Tensor Engine
//!
//! This module provides a high-performance, production-ready inference server
//! with the following capabilities:
//!
//! - HTTP/gRPC API endpoints
//! - Dynamic request batching
//! - Model loading and versioning
//! - Token streaming support (SSE)
//! - Health checks and monitoring
//! - SSL/TLS termination
//! - Request timeout and cancellation

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// Detect the state dict key prefix that contains the language model weights
/// (text model) for multimodal models. This handles cases like Qwen3.5 where
/// the text model parameters are nested under `model.language_model.*` instead of
/// directly under `model.*`.
fn detect_lm_prefix(state: &HashMap<String, Tensor>) -> Option<String> {
    // Common nested prefixes used by multimodal HuggingFace models
    let candidate_prefixes = [
        "model.language_model",
        "model.text_model",
        "model.decoder",
        "language_model",
        "text_model",
    ];
    // Look for known text-model weight keys to confirm the prefix
    let indicator_keys = ["embed_tokens.weight", "lm_head.weight", "norm.weight"];
    for prefix in &candidate_prefixes {
        for indicator in &indicator_keys {
            let key = format!("{}.{}", prefix, indicator);
            if state.contains_key(&key) {
                return Some(prefix.to_string());
            }
        }
    }
    None
}

/// Remap state dict keys from a nested multimodal prefix (e.g.
/// `model.language_model.*`) to the flat prefix expected by the model
/// implementation (e.g. `model.*`). Keys outside the detected prefix
/// (e.g. vision, mtp modules) are left untouched; `apply_state_dict` only
/// reads the keys it needs, so extra keys are harmless.
fn align_state_dict_prefix(state: &mut HashMap<String, Tensor>, target_root: &str) {
    let Some(lm_prefix) = detect_lm_prefix(state) else {
        return;
    };
    if lm_prefix == target_root {
        return;
    }
    let prefix_dot = format!("{}.", lm_prefix);
    let target_dot = format!("{}.", target_root.trim_end_matches('.'));
    let keys: Vec<String> = state.keys().cloned().collect();
    for k in keys {
        if k.starts_with(&prefix_dot) {
            let new_key = k.replacen(&prefix_dot, &target_dot, 1);
            if let Some(t) = state.remove(&k) {
                state.insert(new_key, t);
            }
        }
    }
    log::info!(
        "Aligned state dict prefix: '{}' -> '{}'",
        lm_prefix,
        target_root
    );
}

const REQUEST_TIMEOUT_ERROR: &str = "request generation timed out";

/// Configuration for the inference server
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server listening address
    pub host: String,
    /// Server port
    pub port: u16,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Request timeout in seconds
    pub request_timeout: Duration,
    /// Maximum prompt cache size
    pub prompt_cache_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Enable SSL/TLS
    pub enable_tls: bool,
    /// Model registry path
    pub model_registry_path: Option<String>,
    /// Allowed CORS origins (hostnames only, no IPs). Empty = deny all cross-origin.
    pub allowed_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: crate::config::server::DEFAULT_HOST.to_string(),
            port: crate::config::server::DEFAULT_PORT,
            max_concurrent_requests: 10,
            request_timeout: Duration::from_secs(30),
            prompt_cache_size: 1000,
            max_sequence_length: 0, // 0 = use per-model max_position_embeddings
            enable_tls: true,
            model_registry_path: None,
            allowed_origins: vec![],
        }
    }
}

use std::sync::RwLock;

#[derive(Debug, Clone, serde::Deserialize)]
struct RegistryModelConfig {
    vocab_size: usize,
    #[serde(alias = "d_model", alias = "n_embd")]
    hidden_size: usize,
    #[serde(alias = "d_ff", alias = "n_inner")]
    intermediate_size: usize,
    #[serde(alias = "num_layers", alias = "n_layer")]
    num_hidden_layers: usize,
    #[serde(alias = "num_heads", alias = "n_head")]
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    model_type: String,
}

/// Supported LLM architectures. This set must stay consistent with the
/// supported-model matrix published in `conformance/hf_model_matrix.json`:
/// unknown `model_type` values are rejected instead of silently defaulting to
/// another architecture.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ModelArch {
    Llama,
    Mistral,
    Phi,
    Qwen,
    Qwen3_5,
    Gemma,
}

impl RegistryModelConfig {
    fn detect_architecture(&self) -> Result<ModelArch, String> {
        match self.model_type.as_str() {
            "llama" => Ok(ModelArch::Llama),
            "mistral" => Ok(ModelArch::Mistral),
            "phi" | "phi-msft" | "phi3" => Ok(ModelArch::Phi),
            "qwen" | "qwen2" | "qwen3" | "qwen3_vl" | "qwen3_vl_text" => Ok(ModelArch::Qwen),
            "qwen3_5" | "qwen3_5_text" => Ok(ModelArch::Qwen3_5),
            "gemma" | "gemma2" => Ok(ModelArch::Gemma),
            other => Err(format!("unsupported model_type '{}'", other)),
        }
    }
}

/// Flatten a multimodal model config JSON so that fields nested inside a
/// `text_config` key (common in models like Qwen3-VL / Qwen3.5) are also
/// visible at the top level. This lets shared code (e.g.
/// `build_model_by_architecture`) read `hidden_size`, `head_dim`,
/// `layer_types`, etc. directly from the config value.
fn flatten_config_value(config_bytes: &[u8]) -> Result<serde_json::Value, serde_json::Error> {
    let mut value: serde_json::Value = serde_json::from_slice(config_bytes)?;
    if value.get("vocab_size").is_none() {
        let text_config = value
            .get("text_config")
            .and_then(|v| v.as_object())
            .cloned();
        if let Some(text_cfg) = text_config {
            if let Some(obj) = value.as_object_mut() {
                for (k, v) in text_cfg {
                    obj.entry(k).or_insert(v);
                }
            }
        }
    }
    Ok(value)
}

/// Deserialize a model config JSON that may have its fields nested
/// inside a `text_config` key (common in multimodal models like Qwen3-VL).
fn parse_registry_config(config_bytes: &[u8]) -> Result<RegistryModelConfig, serde_json::Error> {
    let value = flatten_config_value(config_bytes)?;
    serde_json::from_value(value)
}

/// Build the appropriate model struct based on detected architecture.
fn build_model_by_architecture(
    arch: ModelArch,
    vocab_size: usize,
    d_model: usize,
    num_layers: usize,
    d_ff: usize,
    num_heads: usize,
    kv_heads: usize,
    extra: &serde_json::Value,
) -> Result<Box<dyn crate::nn::LlamaStyleModel>, String> {
    match arch {
        ModelArch::Llama => {
            let m =
                crate::nn::Llama::new(vocab_size, d_model, num_layers, d_ff, num_heads, kv_heads)?;
            Ok(Box::new(m))
        }
        ModelArch::Mistral => {
            let sliding_window = extra
                .get("sliding_window")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096) as usize;
            let m = crate::nn::Mistral::new(
                vocab_size,
                d_model,
                num_layers,
                d_ff,
                num_heads,
                kv_heads,
                sliding_window,
            )?;
            Ok(Box::new(m))
        }
        ModelArch::Phi => {
            let final_bias = extra
                .get("final_bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let m = crate::nn::Phi::new(
                vocab_size, d_model, num_layers, d_ff, num_heads, kv_heads, final_bias,
            )?;
            Ok(Box::new(m))
        }
        ModelArch::Qwen => {
            let head_dim = d_model / num_heads;
            let rotary_dim = extra
                .get("rotary_dim")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .or_else(|| {
                    extra
                        .get("rope_scaling")
                        .and_then(|rs| rs.get("rotary_dim"))
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                })
                .unwrap_or(head_dim);
            let m = crate::nn::Qwen::new(
                vocab_size, d_model, num_layers, d_ff, num_heads, kv_heads, rotary_dim,
            )?;
            Ok(Box::new(m))
        }
        ModelArch::Qwen3_5 => {
            let head_dim = extra
                .get("head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as usize;
            let partial_rotary = extra
                .get("rope_parameters")
                .and_then(|rp| rp.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.25);
            let rotary_dim = extra
                .get("rotary_dim")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or((head_dim as f64 * partial_rotary) as usize);
            let rope_theta = extra
                .get("rope_theta")
                .and_then(|v| v.as_f64())
                .or_else(|| {
                    extra
                        .get("rope_parameters")
                        .and_then(|rp| rp.get("rope_theta"))
                        .and_then(|v| v.as_f64())
                })
                .unwrap_or(10000.0) as f32;
            let num_k_heads = extra
                .get("linear_num_key_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(16) as usize;
            let num_v_heads = extra
                .get("linear_num_value_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(16) as usize;
            let head_k_dim = extra
                .get("linear_key_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize;
            let head_v_dim = extra
                .get("linear_value_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize;
            let conv_k = extra
                .get("linear_conv_kernel_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(4) as usize;
            // Parse layer_types array or use default interval
            let layer_types: Vec<String> = extra
                .get("layer_types")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|s| s.as_str().unwrap_or("linear_attention").to_string())
                        .collect()
                })
                .unwrap_or_else(|| {
                    let interval = extra
                        .get("full_attention_interval")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4) as usize;
                    (0..num_layers)
                        .map(|i| {
                            if (i + 1) % interval == 0 {
                                "full_attention".to_string()
                            } else {
                                "linear_attention".to_string()
                            }
                        })
                        .collect()
                });
            let m = crate::nn::Qwen3_5TextModel::new(
                vocab_size,
                d_model,
                num_layers,
                d_ff,
                num_heads,
                kv_heads,
                head_dim,
                rotary_dim,
                rope_theta,
                &layer_types,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_k,
            )?;
            Ok(Box::new(m))
        }
        ModelArch::Gemma => {
            let embedding_multiplier = extra
                .get("embedding_multiplier")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            let m = crate::nn::Gemma::new(
                vocab_size,
                d_model,
                num_layers,
                d_ff,
                num_heads,
                kv_heads,
                embedding_multiplier,
            )?;
            Ok(Box::new(m))
        }
    }
}

impl RegistryModelConfig {
    fn validate(&self) -> Result<(), String> {
        if self.model_type.trim().is_empty() {
            return Err("missing required 'model_type' in config.json".to_string());
        }
        self.detect_architecture()?;
        let dimensions = [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
        ];
        if let Some((name, _)) = dimensions.iter().find(|(_, value)| *value == 0) {
            return Err(format!("{} must be greater than zero", name));
        }
        let kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        if kv_heads == 0 {
            return Err("num_key_value_heads must be greater than zero".to_string());
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err("hidden_size must be divisible by num_attention_heads".to_string());
        }
        if !self.num_attention_heads.is_multiple_of(kv_heads) {
            return Err("num_attention_heads must be divisible by num_key_value_heads".to_string());
        }
        Ok(())
    }
}

/// A discovered model entry — stores paths only, no weights in memory.
#[derive(Debug, Clone)]
struct ModelEntry {
    config_path: PathBuf,
    single_weight: Option<PathBuf>, // present for single-file safetensors
    shards: Vec<PathBuf>,           // non-empty for sharded safetensors
    tokenizer_path: PathBuf,
    max_seq_len: usize,
}

/// Inference server instance
pub struct InferenceServer {
    config: ServerConfig,
    entries: HashMap<String, ModelEntry>,
    loaded_models: RwLock<HashMap<String, Arc<LoadedModel>>>,
    request_count: std::sync::atomic::AtomicU64,
    active_requests: std::sync::atomic::AtomicUsize,
}

struct LoadedModel {
    model: Box<dyn crate::nn::LlamaStyleModel>,
    tokenizer: crate::tokenizer::Tokenizer,
    max_seq_len: usize,
}

struct ActiveRequestGuard {
    server: Arc<InferenceServer>,
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        self.server
            .active_requests
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
        self.server
            .request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl InferenceServer {
    /// Create a new inference server
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            loaded_models: RwLock::new(HashMap::new()),
            request_count: std::sync::atomic::AtomicU64::new(0),
            active_requests: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Start the inference server
    pub async fn start(mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use actix_web::{middleware, web, App, HttpServer};

        // Eagerly initialize the compute backend so the user sees which
        // backend is selected and any init failures surface at startup
        // rather than during the first inference request.
        {
            let backend = crate::backend::get_global_backend();
            log::info!("Compute backend: {}", backend.name());
        }

        let server_addr = format!("{}:{}", self.config.host, self.config.port);
        log::info!("Starting Tensor Engine Inference Server at {}", server_addr);

        let allowed_origins = self.config.allowed_origins.clone();

        if let Some(ref path) = self.config.model_registry_path.clone() {
            self.load_model_registry(path).await?;
        }

        let server_data = web::Data::new(Arc::new(self));
        let app = move || {
            let ao = allowed_origins.clone();
            let cors = actix_cors::Cors::default()
                .allowed_origin_fn(move |origin, _req_head| {
                    let origin_str = match origin.to_str() {
                        Ok(s) if !s.is_empty() => s,
                        _ => return false,
                    };
                    if origin_str.parse::<std::net::SocketAddr>().is_ok() {
                        return false;
                    }
                    ao.contains(&origin_str.to_string())
                })
                .allowed_methods(vec!["POST", "GET", "OPTIONS"])
                .allowed_headers(vec![
                    actix_web::http::header::CONTENT_TYPE,
                    actix_web::http::header::AUTHORIZATION,
                ])
                .max_age(3600);

            App::new()
                .wrap(middleware::Logger::default())
                .wrap(cors)
                .app_data(server_data.clone())
                .route("/health", actix_web::web::get().to(Self::handle_health))
                .route(
                    "/models",
                    actix_web::web::get().to(Self::handle_list_models),
                )
                .route(
                    "/models/{id}",
                    actix_web::web::get().to(Self::handle_get_model),
                )
                .route(
                    "/inference",
                    actix_web::web::post().to(Self::handle_inference),
                )
                .route(
                    "/inference/stream",
                    actix_web::web::post().to(Self::handle_stream_inference),
                )
                .route(
                    "/v1/models",
                    actix_web::web::get().to(Self::handle_openai_models),
                )
                .route(
                    "/v1/models/matrix",
                    actix_web::web::get().to(Self::handle_hf_model_matrix),
                )
                .route(
                    "/v1/completions",
                    actix_web::web::post().to(Self::handle_openai_completion),
                )
                .route(
                    "/v1/chat/completions",
                    actix_web::web::post().to(Self::handle_openai_chat_completion),
                )
        };

        HttpServer::new(app)
            .bind(server_addr)?
            .run()
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        log::info!("Tensor Engine Inference Server started successfully");
        Ok(())
    }

    /// Scan the model registry directory — discovers available models and
    /// validates directory structure, but does **not** load weights into memory.
    /// Actual model loading is deferred to `get_or_load_model`.
    ///
    /// If `path` itself contains a `config.json`, it is treated as a single
    /// model directory. Otherwise `path` is scanned for model subdirectories.
    async fn load_model_registry(
        &mut self,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("Scanning model registry: {}", path);

        let registry_path = std::path::Path::new(path);
        if !registry_path.exists() {
            log::warn!("Model registry path does not exist: {}", path);
            return Ok(());
        }
        if !registry_path.is_dir() {
            return Err(format!("Model registry path is not a directory: {}", path).into());
        }

        // If the registry path itself contains a config.json, treat it as a
        // single model directory.
        if registry_path
            .join(crate::config::filenames::CONFIG_JSON)
            .exists()
        {
            let model_id = registry_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("model")
                .to_string();
            let dir_path = registry_path.to_path_buf();
            self.register_model_entry(&model_id, &dir_path)?;
        } else {
            let dir_entries =
                std::fs::read_dir(registry_path)?.collect::<Result<Vec<_>, std::io::Error>>()?;
            for entry in dir_entries {
                let entry_path = entry.path();
                if !entry_path.is_dir() {
                    continue;
                }
                let model_id = entry_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let config_path = entry_path.join(crate::config::filenames::CONFIG_JSON);
                if !config_path.exists() {
                    log::debug!("Skipping directory without config.json: {:?}", entry_path);
                    continue;
                }
                self.register_model_entry(&model_id, &entry_path)?;
            }
        }

        log::info!(
            "Model registry scan complete: {} model(s) discovered",
            self.entries.len()
        );
        Ok(())
    }

    /// Register a single model directory. The directory must contain
    /// `config.json`, `tokenizer.json`, and safetensors weight files.
    fn register_model_entry(
        &mut self,
        model_id: &str,
        dir: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config_path = dir.join(crate::config::filenames::CONFIG_JSON);
        let tokenizer_path = dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(format!("model '{}' has no tokenizer.json", model_id).into());
        }

        let single = dir.join("model.safetensors");
        let (single_weight, shards) = if single.exists() {
            (Some(single), vec![])
        } else {
            let mut found: Vec<_> = std::fs::read_dir(dir)?
                .flatten()
                .filter(|e| {
                    let fname = e.file_name();
                    let n = fname.to_string_lossy();
                    n.ends_with(".safetensors")
                        && n != "model.safetensors"
                        && (n.contains("model-") || n.starts_with("model.safetensors-"))
                })
                .map(|e| e.path())
                .collect();
            found.sort();
            if found.is_empty() {
                return Err(format!(
                    "model '{}' has config.json but no .safetensors weights",
                    model_id
                )
                .into());
            }
            (None, found)
        };

        // Validate the config up front so unsupported or malformed
        // repositories fail at registry scan time, before any allocation.
        let config_bytes = std::fs::read(&config_path).map_err(|error| {
            format!(
                "model '{}' has an unreadable config.json: {}",
                model_id, error
            )
        })?;
        let config: RegistryModelConfig =
            parse_registry_config(&config_bytes).map_err(|error| {
                format!("model '{}' has an invalid config.json: {}", model_id, error)
            })?;
        config.validate().map_err(|message| {
            format!(
                "model '{}' has an unsupported config.json: {}",
                model_id, message
            )
        })?;

        let max_seq_len = config.max_position_embeddings.unwrap_or(2048);

        log::info!(
            "Discovered model '{}' (max_seq_len={})",
            model_id,
            max_seq_len
        );
        self.entries.insert(
            model_id.to_string(),
            ModelEntry {
                config_path,
                single_weight,
                shards,
                tokenizer_path,
                max_seq_len,
            },
        );
        Ok(())
    }

    /// Return a loaded model, loading it on-demand if this is the first request.
    fn get_or_load_model(
        &self,
        model_id: &str,
    ) -> Result<Arc<LoadedModel>, Box<dyn std::error::Error + Send + Sync>> {
        // Fast path: already loaded.
        if let Some(model) = self
            .loaded_models
            .read()
            .map_err(|_| "model registry lock poisoned")?
            .get(model_id)
        {
            return Ok(Arc::clone(model));
        }

        let entry = self
            .entries
            .get(model_id)
            .ok_or_else(|| format!("model '{}' not found in registry", model_id))?;

        log::info!("Loading model '{}' on demand", model_id);

        let loaded = if let Some(weight_path) = &entry.single_weight {
            Self::load_registry_model(&entry.config_path, weight_path, &entry.tokenizer_path)?
        } else {
            // Sharded — merge all shards into one state dict.
            let mut merged_state: HashMap<String, Tensor> = HashMap::new();
            for shard_path in &entry.shards {
                let shard_bytes = std::fs::read(shard_path)
                    .map_err(|e| format!("failed to read '{}': {}", shard_path.display(), e))?;
                let state =
                    crate::io::safetensors_loader::load_safetensors_from_bytes(&shard_bytes, false)
                        .map_err(|e| {
                            format!("failed to parse '{}': {}", shard_path.display(), e)
                        })?;
                merged_state.extend(state);
            }
            let model = Self::build_model_from_state(&entry.config_path, &merged_state)?;
            let tokenizer_path_text = entry
                .tokenizer_path
                .to_str()
                .ok_or_else(|| "tokenizer path is not UTF-8".to_string())?;
            let tokenizer = crate::tokenizer::Tokenizer::from_json(tokenizer_path_text)
                .map_err(|e| format!("failed to load tokenizer: {}", e))?;
            let max_seq_len = {
                let config_bytes = std::fs::read(&entry.config_path)
                    .map_err(|e| format!("failed to read config: {}", e))?;
                let config: RegistryModelConfig = parse_registry_config(&config_bytes)
                    .map_err(|e| format!("failed to parse config: {}", e))?;
                config.max_position_embeddings.unwrap_or(2048)
            };
            LoadedModel {
                model,
                tokenizer,
                max_seq_len,
            }
        };

        let loaded = Arc::new(loaded);
        self.loaded_models
            .write()
            .map_err(|_| "model registry lock poisoned")?
            .insert(model_id.to_string(), Arc::clone(&loaded));
        Ok(loaded)
    }

    fn load_registry_model(
        config_path: &std::path::Path,
        weights_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
    ) -> Result<LoadedModel, Box<dyn std::error::Error + Send + Sync>> {
        let config_bytes = std::fs::read(config_path)?;
        let config: RegistryModelConfig = parse_registry_config(&config_bytes)?;
        config.validate().map_err(|message| {
            format!(
                "invalid model config '{}': {}",
                config_path.display(),
                message
            )
        })?;

        let full_config = flatten_config_value(&config_bytes)?;
        let arch = config.detect_architecture().map_err(|message| {
            format!(
                "unsupported model config '{}': {}",
                config_path.display(),
                message
            )
        })?;
        let kv_heads = config
            .num_key_value_heads
            .unwrap_or(config.num_attention_heads);

        log::info!(
            "Detected architecture '{:?}' (model_type='{}') for {}",
            arch,
            config.model_type,
            config_path.display()
        );

        let mut model = build_model_by_architecture(
            arch,
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.intermediate_size,
            config.num_attention_heads,
            kv_heads,
            &full_config,
        )
        .map_err(|message| {
            format!(
                "failed to construct model from '{}': {}",
                config_path.display(),
                message
            )
        })?;

        let weights = std::fs::read(weights_path)?;
        let mut state = crate::io::safetensors_loader::load_safetensors_from_bytes(&weights, false)
            .map_err(|message| {
                format!(
                    "failed to load weights '{}': {}",
                    weights_path.display(),
                    message
                )
            })?;
        align_state_dict_prefix(&mut state, "model");
        model.apply_state_dict(&state, "model").map_err(|message| {
            format!(
                "failed to apply weights '{}': {}",
                weights_path.display(),
                message
            )
        })?;
        let tokenizer_path_text = tokenizer_path
            .to_str()
            .ok_or_else(|| format!("tokenizer path is not UTF-8: {}", tokenizer_path.display()))?;
        let tokenizer =
            crate::tokenizer::Tokenizer::from_json(tokenizer_path_text).map_err(|message| {
                format!(
                    "failed to load tokenizer '{}': {}",
                    tokenizer_path.display(),
                    message
                )
            })?;
        let max_seq_len = config.max_position_embeddings.unwrap_or(2048);
        log::info!(
            "Model '{}' loaded (max_seq_len={})",
            config_path.display(),
            max_seq_len
        );
        Ok(LoadedModel {
            model,
            tokenizer,
            max_seq_len,
        })
    }

    /// Build a model from a pre-merged state dict (for sharded safetensors).
    /// Tokenizer is loaded separately by the caller.
    fn build_model_from_state(
        config_path: &std::path::Path,
        state: &HashMap<String, Tensor>,
    ) -> Result<Box<dyn crate::nn::LlamaStyleModel>, Box<dyn std::error::Error + Send + Sync>> {
        let config_bytes = std::fs::read(config_path)?;
        let config: RegistryModelConfig = parse_registry_config(&config_bytes)?;
        config.validate().map_err(|message| {
            format!(
                "invalid model config '{}': {}",
                config_path.display(),
                message
            )
        })?;

        let full_config = flatten_config_value(&config_bytes)?;
        let arch = config.detect_architecture().map_err(|message| {
            format!(
                "unsupported model config '{}': {}",
                config_path.display(),
                message
            )
        })?;
        let kv_heads = config
            .num_key_value_heads
            .unwrap_or(config.num_attention_heads);

        log::info!(
            "Detected architecture '{:?}' (model_type='{}') for {}",
            arch,
            config.model_type,
            config_path.display()
        );

        let mut model = build_model_by_architecture(
            arch,
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.intermediate_size,
            config.num_attention_heads,
            kv_heads,
            &full_config,
        )
        .map_err(|message| {
            format!(
                "failed to construct model from '{}': {}",
                config_path.display(),
                message
            )
        })?;

        let mut aligned_state = state.clone();
        align_state_dict_prefix(&mut aligned_state, "model");
        model
            .apply_state_dict(&aligned_state, "model")
            .map_err(|message| format!("failed to apply merged state: {}", message))?;
        Ok(model)
    }

    fn try_acquire_request(
        state: &Arc<InferenceServer>,
    ) -> Result<ActiveRequestGuard, actix_web::HttpResponse> {
        loop {
            let current = state
                .active_requests
                .load(std::sync::atomic::Ordering::Acquire);
            if current >= state.config.max_concurrent_requests {
                state
                    .request_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(Self::openai_error(
                    actix_web::http::StatusCode::TOO_MANY_REQUESTS,
                    "server is at maximum concurrent inference capacity",
                ));
            }
            if state
                .active_requests
                .compare_exchange_weak(
                    current,
                    current + 1,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(ActiveRequestGuard {
                    server: Arc::clone(state),
                });
            }
        }
    }

    /// Health check endpoint
    async fn handle_health() -> &'static str {
        "OK"
    }

    /// List available models
    async fn handle_list_models(
        state: actix_web::web::Data<Arc<InferenceServer>>,
    ) -> actix_web::HttpResponse {
        let loaded = state.loaded_models.read().unwrap();
        let mut model_list: Vec<serde_json::Value> = state
            .entries
            .keys()
            .map(|id| {
                let status = if loaded.contains_key(id) {
                    "loaded"
                } else {
                    "available"
                };
                serde_json::json!({
                    "id": id,
                    "status": status
                })
            })
            .collect();
        model_list.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));

        let matrix = match crate::hf_matrix::HfModelMatrix::parse() {
            Ok(matrix) => {
                let _ = matrix.validate();
                serde_json::to_value(&matrix).unwrap_or_else(|_| serde_json::json!(null))
            }
            Err(message) => serde_json::json!({
                "error": message
            }),
        };

        actix_web::HttpResponse::Ok()
            .content_type("application/json")
            .json(serde_json::json!({
                "models": model_list,
                "count": model_list.len(),
                "supported_model_matrix": matrix
            }))
    }

    /// Get specific model metadata
    async fn handle_get_model(
        state: actix_web::web::Data<Arc<InferenceServer>>,
        path: actix_web::web::Path<String>,
    ) -> actix_web::HttpResponse {
        let model_id = path.into_inner();
        if !state.entries.contains_key(&model_id) {
            return actix_web::HttpResponse::NotFound()
                .content_type("application/json")
                .json(serde_json::json!({
                    "error": "Model not found",
                    "requested_id": model_id
                }));
        }
        let loaded = state.loaded_models.read().unwrap();
        let status = if loaded.contains_key(&model_id) {
            "loaded"
        } else {
            "available"
        };
        let model_type = if let Some(entry) = state.entries.get(&model_id) {
            let config_bytes = std::fs::read(&entry.config_path).ok();
            config_bytes
                .and_then(|b| parse_registry_config(&b).ok())
                .map(|cfg| cfg.model_type.clone())
                .unwrap_or_else(|| "unknown".to_string())
        } else {
            "unknown".to_string()
        };
        actix_web::HttpResponse::Ok()
            .content_type("application/json")
            .json(serde_json::json!({
                "id": model_id,
                "status": status,
                "type": model_type
            }))
    }

    async fn handle_openai_models(
        state: actix_web::web::Data<Arc<InferenceServer>>,
    ) -> actix_web::HttpResponse {
        let loaded = state.loaded_models.read().unwrap();
        let mut ids: Vec<String> = state.entries.keys().cloned().collect();
        ids.sort();
        let data = ids
            .into_iter()
            .map(|id| {
                let max_ctx = state
                    .entries
                    .get(&id)
                    .map(|e| e.max_seq_len)
                    .unwrap_or(2048);
                let is_loaded = loaded.contains_key(&id);
                serde_json::json!({
                    "id": id,
                    "object": "model",
                    "owned_by": "tensor-engine",
                    "max_context_length": max_ctx,
                    "status": if is_loaded { "loaded" } else { "available" }
                })
            })
            .collect::<Vec<_>>();
        let matrix = match crate::hf_matrix::HfModelMatrix::parse() {
            Ok(matrix) => {
                let _ = matrix.validate();
                serde_json::to_value(&matrix).unwrap_or_else(|_| serde_json::json!(null))
            }
            Err(message) => serde_json::json!({
                "error": message
            }),
        };
        actix_web::HttpResponse::Ok().json(serde_json::json!({
            "object": "list",
            "data": data,
            "meta": {
                "supported_model_matrix": matrix
            }
        }))
    }

    /// Serve the supported-model matrix as its own endpoint.
    async fn handle_hf_model_matrix(
        _state: actix_web::web::Data<Arc<InferenceServer>>,
    ) -> actix_web::HttpResponse {
        let value = match crate::hf_matrix::HfModelMatrix::parse() {
            Ok(matrix) => {
                if let Err(message) = matrix.validate() {
                    return actix_web::HttpResponse::InternalServerError().json(
                        serde_json::json!({
                            "error": message
                        }),
                    );
                }
                serde_json::to_value(&matrix).unwrap_or_else(|_| serde_json::json!(null))
            }
            Err(message) => {
                return actix_web::HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": message
                }));
            }
        };
        actix_web::HttpResponse::Ok().json(value)
    }

    async fn handle_openai_completion(
        state: actix_web::web::Data<Arc<InferenceServer>>,
        req: actix_web::web::Json<OpenAICompletionRequest>,
    ) -> actix_web::HttpResponse {
        if req.stream {
            return Self::stream_openai_text(&state, &req.model, &req.prompt, req.options(), false)
                .await;
        }
        Self::process_openai_text(&state, &req.model, &req.prompt, req.options(), false)
    }

    async fn handle_openai_chat_completion(
        state: actix_web::web::Data<Arc<InferenceServer>>,
        req: actix_web::web::Json<OpenAIChatRequest>,
    ) -> actix_web::HttpResponse {
        if req.messages.is_empty() {
            return Self::openai_error(
                actix_web::http::StatusCode::BAD_REQUEST,
                "messages must not be empty",
            );
        }
        let mut prompt = String::new();
        for message in &req.messages {
            if message.role.trim().is_empty() || message.content.trim().is_empty() {
                return Self::openai_error(
                    actix_web::http::StatusCode::BAD_REQUEST,
                    "each message requires a non-empty role and content",
                );
            }
            prompt.push_str(&message.role);
            prompt.push_str(": ");
            prompt.push_str(&message.content);
            prompt.push('\n');
        }
        prompt.push_str("assistant: ");
        if req.stream {
            return Self::stream_openai_text(&state, &req.model, &prompt, req.options(), true)
                .await;
        }
        Self::process_openai_text(&state, &req.model, &prompt, req.options(), true)
    }

    fn process_openai_text(
        state: &Arc<InferenceServer>,
        model_id: &str,
        prompt: &str,
        options: OpenAIOptions,
        chat: bool,
    ) -> actix_web::HttpResponse {
        if prompt.is_empty() {
            return Self::openai_error(
                actix_web::http::StatusCode::BAD_REQUEST,
                "prompt must not be empty",
            );
        }
        let loaded = match state.get_or_load_model(model_id) {
            Ok(model) => model,
            Err(e) => {
                return Self::openai_error(
                    actix_web::http::StatusCode::NOT_FOUND,
                    &format!("model '{}' not found: {}", model_id, e),
                )
            }
        };
        let prompt_ids = loaded
            .tokenizer
            .encode(prompt)
            .into_iter()
            .map(|id| id as u32)
            .collect::<Vec<_>>();
        if prompt_ids.is_empty() {
            return Self::openai_error(
                actix_web::http::StatusCode::BAD_REQUEST,
                "tokenizer produced no prompt tokens",
            );
        }
        let effective_max = if state.config.max_sequence_length > 0 {
            state.config.max_sequence_length.min(loaded.max_seq_len)
        } else {
            loaded.max_seq_len
        };
        if prompt_ids.len() > effective_max {
            return Self::openai_error(
                actix_web::http::StatusCode::BAD_REQUEST,
                &format!(
                    "prompt length {} exceeds maximum context length {}",
                    prompt_ids.len(),
                    effective_max
                ),
            );
        }
        let _guard = match Self::try_acquire_request(state) {
            Ok(guard) => guard,
            Err(response) => return response,
        };
        let generated = match Self::generate_tokens(
            &*loaded.model,
            &prompt_ids,
            options.max_tokens,
            options.temperature,
            options.top_p,
            options.seed,
            Some(std::time::Instant::now() + state.config.request_timeout),
        ) {
            Ok(tokens) => tokens,
            Err(message) => {
                let status = if message == REQUEST_TIMEOUT_ERROR {
                    actix_web::http::StatusCode::REQUEST_TIMEOUT
                } else {
                    actix_web::http::StatusCode::INTERNAL_SERVER_ERROR
                };
                return Self::openai_error(status, &message);
            }
        };
        let generated_ids = generated
            .iter()
            .map(|token| *token as usize)
            .collect::<Vec<_>>();
        let text = loaded.tokenizer.decode(&generated_ids);
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        let id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let choice = if chat {
            serde_json::json!({
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length"
            })
        } else {
            serde_json::json!({
                "index": 0,
                "text": text,
                "finish_reason": "length"
            })
        };
        actix_web::HttpResponse::Ok().json(serde_json::json!({
            "id": id,
            "object": if chat { "chat.completion" } else { "text_completion" },
            "created": created,
            "model": model_id,
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_ids.len(),
                "completion_tokens": generated_ids.len(),
                "total_tokens": prompt_ids.len() + generated_ids.len()
            }
        }))
    }

    async fn stream_openai_text(
        state: &Arc<InferenceServer>,
        model_id: &str,
        prompt: &str,
        options: OpenAIOptions,
        chat: bool,
    ) -> actix_web::HttpResponse {
        if prompt.is_empty() {
            return Self::openai_error(
                actix_web::http::StatusCode::BAD_REQUEST,
                "prompt must not be empty",
            );
        }
        let loaded = match state.get_or_load_model(model_id) {
            Ok(model) => model,
            Err(e) => {
                return Self::openai_error(
                    actix_web::http::StatusCode::NOT_FOUND,
                    &format!("model '{}' not found: {}", model_id, e),
                )
            }
        };
        let prompt_ids = loaded
            .tokenizer
            .encode(prompt)
            .into_iter()
            .map(|id| id as u32)
            .collect::<Vec<_>>();
        if prompt_ids.is_empty() {
            return Self::openai_error(
                actix_web::http::StatusCode::BAD_REQUEST,
                "tokenizer produced no prompt tokens",
            );
        }
        let effective_max = if state.config.max_sequence_length > 0 {
            state.config.max_sequence_length.min(loaded.max_seq_len)
        } else {
            loaded.max_seq_len
        };
        if prompt_ids.len() > effective_max {
            return Self::openai_error(
                actix_web::http::StatusCode::BAD_REQUEST,
                &format!(
                    "prompt length {} exceeds maximum context length {}",
                    prompt_ids.len(),
                    effective_max
                ),
            );
        }
        let guard = match Self::try_acquire_request(state) {
            Ok(guard) => guard,
            Err(response) => return response,
        };

        let (tx, rx) =
            futures::channel::mpsc::channel::<Result<actix_web::web::Bytes, std::io::Error>>(32);
        let stream_id = format!(
            "{}-{}",
            if chat { "chatcmpl" } else { "cmpl" },
            uuid::Uuid::new_v4()
        );
        let model_id = model_id.to_string();
        let deadline = std::time::Instant::now() + state.config.request_timeout;
        actix_web::rt::spawn(async move {
            let _guard = guard;
            let mut tx = tx;
            if let Err(message) = Self::generate_openai_sse(
                &loaded,
                &prompt_ids,
                options,
                &stream_id,
                &model_id,
                chat,
                deadline,
                &mut tx,
            )
            .await
            {
                use futures::SinkExt;
                let error_type = if message == REQUEST_TIMEOUT_ERROR {
                    "timeout_error"
                } else {
                    "server_error"
                };
                let payload = serde_json::json!({
                    "error": {"message": message, "type": error_type}
                });
                let _ = tx
                    .send(Ok(actix_web::web::Bytes::from(format!(
                        "event: error\ndata: {}\n\n",
                        payload
                    ))))
                    .await;
            }
        });

        actix_web::HttpResponse::Ok()
            .content_type("text/event-stream")
            .append_header(("Cache-Control", "no-cache"))
            .append_header(("Connection", "keep-alive"))
            .body(actix_web::body::BodyStream::new(rx))
    }

    async fn generate_openai_sse(
        loaded: &LoadedModel,
        prompt_ids: &[u32],
        options: OpenAIOptions,
        stream_id: &str,
        model_id: &str,
        chat: bool,
        deadline: std::time::Instant,
        tx: &mut futures::channel::mpsc::Sender<Result<actix_web::web::Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        use crate::generation::sampling::Sampler;
        use futures::SinkExt;

        let mut model = loaded.model.clone_model();
        ensure_before_deadline(deadline)?;
        model.init_kv_caches(prompt_ids.len() + options.max_tokens)?;
        for &token_id in &prompt_ids[..prompt_ids.len().saturating_sub(1)] {
            ensure_before_deadline(deadline)?;
            let token = Tensor::new(
                ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1usize]), vec![token_id as f32])
                    .map_err(|error| format!("tensor shape error: {}", error))?,
                false,
            );
            model.forward_single_token(&token, None)?;
        }

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        if chat {
            let role_chunk = openai_stream_chunk(
                stream_id,
                model_id,
                created,
                true,
                Some("assistant"),
                "",
                None,
            );
            send_sse_json(tx, &role_chunk).await?;
        }

        let mut sampler = Sampler::new(options.temperature, 0, options.top_p, options.seed);
        let mut last_token = *prompt_ids
            .last()
            .ok_or_else(|| "prompt must contain at least one token".to_string())?;
        for _ in 0..options.max_tokens {
            ensure_before_deadline(deadline)?;
            let token = Tensor::new(
                ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1usize]), vec![last_token as f32])
                    .map_err(|error| format!("tensor shape error: {}", error))?,
                false,
            );
            let logits = model.forward_single_token(&token, None)?;
            let sampled = sampler.sample(&logits);
            last_token = sampled.token as u32;
            let text = loaded.tokenizer.decode(&[last_token as usize]);
            let chunk = openai_stream_chunk(stream_id, model_id, created, chat, None, &text, None);
            send_sse_json(tx, &chunk).await?;
        }

        let final_chunk =
            openai_stream_chunk(stream_id, model_id, created, chat, None, "", Some("length"));
        send_sse_json(tx, &final_chunk).await?;
        tx.send(Ok(actix_web::web::Bytes::from("data: [DONE]\n\n")))
            .await
            .map_err(|error| format!("SSE send error: {}", error))
    }

    fn openai_error(status: actix_web::http::StatusCode, message: &str) -> actix_web::HttpResponse {
        actix_web::HttpResponse::build(status).json(serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error"
            }
        }))
    }

    /// Handle inference request
    ///
    async fn handle_inference(
        req: actix_web::web::Json<InferenceRequest>,
        state: actix_web::web::Data<Arc<InferenceServer>>,
    ) -> Result<actix_web::HttpResponse, actix_web::Error> {
        let _guard = match Self::try_acquire_request(state.get_ref()) {
            Ok(guard) => guard,
            Err(response) => return Ok(response),
        };

        Self::process_request(&state, req.into_inner())
            .await
            .map(|res| actix_web::HttpResponse::Ok().json(res))
            .map_err(|error| {
                let message = error.to_string();
                if message.contains(REQUEST_TIMEOUT_ERROR) {
                    actix_web::error::ErrorRequestTimeout(message)
                } else {
                    actix_web::error::ErrorInternalServerError(message)
                }
            })
    }

    /// Handle streaming inference via Server-Sent Events (SSE).
    ///
    /// Generates tokens one at a time and streams each as an SSE event.
    /// The response has `Content-Type: text/event-stream`.
    async fn handle_stream_inference(
        req: actix_web::web::Json<InferenceRequest>,
        state: actix_web::web::Data<Arc<InferenceServer>>,
    ) -> Result<actix_web::HttpResponse, actix_web::Error> {
        let req_inner = req.into_inner();

        // Validate before streaming
        if let Err(e) = Self::validate_request(&state, &req_inner) {
            return Err(actix_web::error::ErrorBadRequest(e.to_string()));
        }

        // Get model
        let model = state.get_or_load_model(&req_inner.model_id).map_err(|_| {
            actix_web::error::ErrorNotFound(format!("Model '{}' not found", req_inner.model_id))
        })?;

        let max_tokens = req_inner.max_tokens.unwrap_or(32) as usize;
        let temperature = req_inner.temperature.unwrap_or(1.0);
        let top_p = req_inner.top_p.unwrap_or(1.0);
        let seed = req_inner.seed.unwrap_or(42);
        let input_tokens = req_inner.input.clone();
        let model_id = req_inner.model_id.clone();
        let deadline = std::time::Instant::now() + state.config.request_timeout;
        let guard = match Self::try_acquire_request(state.get_ref()) {
            Ok(guard) => guard,
            Err(response) => return Ok(response),
        };

        // Create a futures MPSC channel for streaming bytes.
        // futures::channel::mpsc::Receiver implements Stream directly,
        // so it can be wrapped in BodyStream without extra adapters.
        let (tx, rx) =
            futures::channel::mpsc::channel::<Result<actix_web::web::Bytes, std::io::Error>>(32);
        let body = actix_web::body::BodyStream::new(rx);

        // Spawn generation task
        actix_web::rt::spawn(async move {
            let _guard = guard;
            let mut tx = tx;
            let result = Self::generate_streaming(
                &*model.model,
                &input_tokens,
                max_tokens,
                temperature,
                top_p,
                seed,
                Some(deadline),
                &mut tx,
            )
            .await;

            if let Err(e) = result {
                log::error!("Streaming generation error for model '{}': {}", model_id, e);
                use futures::SinkExt;
                let error_type = if e == REQUEST_TIMEOUT_ERROR {
                    "timeout_error"
                } else {
                    "server_error"
                };
                let _ = tx
                    .send(Ok(actix_web::web::Bytes::from(format!(
                        "event: error\ndata: {}\n\n",
                        serde_json::json!({"error": {"message": e, "type": error_type}})
                    ))))
                    .await;
            }
        });

        Ok(actix_web::HttpResponse::Ok()
            .content_type("text/event-stream")
            .append_header(("Cache-Control", "no-cache"))
            .append_header(("Connection", "keep-alive"))
            .body(body))
    }

    /// Process a single inference request: run autoregressive generation
    /// and return the full output.
    async fn process_request(
        state: &Arc<InferenceServer>,
        req: InferenceRequest,
    ) -> Result<InferenceResponse, crate::error::TensorError> {
        Self::validate_request(state, &req)?;

        let model = state.get_or_load_model(&req.model_id).map_err(|_| {
            crate::error::TensorError::Generic {
                message: format!("Model '{}' not found", req.model_id),
            }
        })?;

        let max_tokens = req.max_tokens.unwrap_or(32) as usize;
        let temperature = req.temperature.unwrap_or(1.0);
        let top_p = req.top_p.unwrap_or(1.0);
        let seed = req.seed.unwrap_or(42);

        let start_time = std::time::Instant::now();

        let generated = Self::generate_tokens(
            &*model.model,
            &req.input,
            max_tokens,
            temperature,
            top_p,
            seed,
            Some(std::time::Instant::now() + state.config.request_timeout),
        )
        .map_err(|e| crate::error::TensorError::Generic {
            message: format!("Generation failed: {}", e),
        })?;

        let inference_time = start_time.elapsed();
        let tokens_generated = generated.len();

        log::info!(
            "Inference completed in {:?} ({} tokens generated)",
            inference_time,
            tokens_generated
        );

        Ok(InferenceResponse {
            output: generated,
            inference_time_ms: inference_time.as_millis() as u64,
            tokens_generated,
            model_id: req.model_id.clone(),
        })
    }

    /// Validate an inference request.
    fn validate_request(
        state: &Arc<InferenceServer>,
        req: &InferenceRequest,
    ) -> Result<(), crate::error::TensorError> {
        let entry =
            state
                .entries
                .get(&req.model_id)
                .ok_or_else(|| crate::error::TensorError::Generic {
                    message: format!("Model '{}' not found in registry", req.model_id),
                })?;
        if req.input.is_empty() {
            return Err(crate::error::TensorError::ValidationError {
                field: "input".to_string(),
                value: "[]".to_string(),
                constraint: "non-empty input required".to_string(),
            });
        }
        let effective_max = if state.config.max_sequence_length > 0 {
            state.config.max_sequence_length.min(entry.max_seq_len)
        } else {
            entry.max_seq_len
        };
        if req.input.len() > effective_max {
            return Err(crate::error::TensorError::ValidationError {
                field: "input".to_string(),
                value: req.input.len().to_string(),
                constraint: format!("max {} tokens", effective_max),
            });
        }
        Ok(())
    }

    /// Run autoregressive generation on a Llama-style model.
    ///
    /// Uses `forward_single_token` with KV cache for efficient incremental decoding.
    /// Returns the generated token IDs (excluding the prompt).
    fn generate_tokens(
        model: &dyn crate::nn::LlamaStyleModel,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: u64,
        deadline: Option<std::time::Instant>,
    ) -> Result<Vec<f32>, String> {
        use crate::generation::sampling::Sampler;

        let mut sampler = Sampler::new(temperature, 0, top_p, seed);

        // Clone the model's internal state for this generation call.
        // We need mutable access to run forward_single_token, but we only have Arc.
        ensure_optional_deadline(deadline)?;
        let mut model = model.clone_model();

        let max_seq_len = prompt_ids.len() + max_new_tokens;
        model.init_kv_caches(max_seq_len)?;

        // Process prompt tokens one at a time to populate KV cache
        // (all except the last, which we use for the first generation step)
        let prompt_len = prompt_ids.len();
        if prompt_len == 0 {
            return Err("prompt must contain at least one token".to_string());
        }

        // Feed all prompt tokens except the last through the cache
        for &token_id in &prompt_ids[..prompt_len.saturating_sub(1)] {
            ensure_optional_deadline(deadline)?;
            let token_tensor = Tensor::new(
                ndarray::Array::from_shape_vec(
                    ndarray::IxDyn(&[1usize][..]),
                    vec![token_id as f32],
                )
                .map_err(|e| format!("tensor shape error: {}", e))?,
                false,
            );
            model.forward_single_token(&token_tensor, None)?;
        }

        // Generate tokens
        let mut generated: Vec<f32> = Vec::with_capacity(max_new_tokens);
        let mut last_token = prompt_ids[prompt_len - 1];

        for _ in 0..max_new_tokens {
            ensure_optional_deadline(deadline)?;
            let token_tensor = Tensor::new(
                ndarray::Array::from_shape_vec(
                    ndarray::IxDyn(&[1usize][..]),
                    vec![last_token as f32],
                )
                .map_err(|e| format!("tensor shape error: {}", e))?,
                false,
            );

            let logits = model.forward_single_token(&token_tensor, None)?;
            let result = sampler.sample(&logits);
            let next_token = result.token as u32;
            generated.push(next_token as f32);
            last_token = next_token;

            log::debug!(
                "Generated token {} (prob={:.4}), total: {}",
                next_token,
                result.prob,
                generated.len()
            );
        }

        Ok(generated)
    }

    /// Stream generated tokens via SSE.
    ///
    /// Sends each token as a `data:` SSE event containing a JSON object
    /// with the token ID. Terminates with a `data: [DONE]` event.
    async fn generate_streaming(
        model: &dyn crate::nn::LlamaStyleModel,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: u64,
        deadline: Option<std::time::Instant>,
        tx: &mut futures::channel::mpsc::Sender<Result<actix_web::web::Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        use crate::generation::sampling::Sampler;
        use futures::SinkExt;

        let mut sampler = Sampler::new(temperature, 0, top_p, seed);
        ensure_optional_deadline(deadline)?;
        let mut model = model.clone_model();

        let max_seq_len = prompt_ids.len() + max_new_tokens;
        model.init_kv_caches(max_seq_len)?;

        let prompt_len = prompt_ids.len();
        if prompt_len == 0 {
            return Err("prompt must contain at least one token".to_string());
        }

        for &token_id in &prompt_ids[..prompt_len.saturating_sub(1)] {
            ensure_optional_deadline(deadline)?;
            let token_tensor = Tensor::new(
                ndarray::Array::from_shape_vec(
                    ndarray::IxDyn(&[1usize][..]),
                    vec![token_id as f32],
                )
                .map_err(|e| format!("tensor shape error: {}", e))?,
                false,
            );
            model.forward_single_token(&token_tensor, None)?;
        }

        let mut last_token = prompt_ids[prompt_len - 1];

        for _ in 0..max_new_tokens {
            ensure_optional_deadline(deadline)?;
            let token_tensor = Tensor::new(
                ndarray::Array::from_shape_vec(
                    ndarray::IxDyn(&[1usize][..]),
                    vec![last_token as f32],
                )
                .map_err(|e| format!("tensor shape error: {}", e))?,
                false,
            );

            let logits = model.forward_single_token(&token_tensor, None)?;
            let result = sampler.sample(&logits);
            let next_token = result.token as u32;
            last_token = next_token;

            let event_data = serde_json::json!({
                "token": next_token,
                "prob": result.prob,
            });
            let sse_line = format!("data: {}\n\n", event_data);
            let bytes = actix_web::web::Bytes::from(sse_line);

            tx.send(Ok(bytes))
                .await
                .map_err(|e| format!("SSE send error: {}", e))?;
        }

        // Send termination event
        let done_bytes = actix_web::web::Bytes::from("data: [DONE]\n\n");
        tx.send(Ok(done_bytes))
            .await
            .map_err(|e| format!("SSE done send error: {}", e))?;

        Ok(())
    }
}

/// Start the canonical Tensor Engine server with an explicit configuration.
///
/// Models are discovered from `model_registry_path`; this launcher never
/// constructs placeholder models or falls back to a compatibility runtime.
pub async fn serve(config: ServerConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    InferenceServer::new(config).start().await
}

#[derive(Debug, Clone, serde::Deserialize)]
struct OpenAICompletionRequest {
    model: String,
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
}

impl OpenAICompletionRequest {
    fn options(&self) -> OpenAIOptions {
        OpenAIOptions {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed.unwrap_or(42),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
}

impl OpenAIChatRequest {
    fn options(&self) -> OpenAIOptions {
        OpenAIOptions {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed.unwrap_or(42),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Copy)]
struct OpenAIOptions {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: u64,
}

fn default_max_tokens() -> usize {
    32
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn ensure_before_deadline(deadline: std::time::Instant) -> Result<(), String> {
    if std::time::Instant::now() >= deadline {
        Err(REQUEST_TIMEOUT_ERROR.to_string())
    } else {
        Ok(())
    }
}

fn ensure_optional_deadline(deadline: Option<std::time::Instant>) -> Result<(), String> {
    match deadline {
        Some(deadline) => ensure_before_deadline(deadline),
        None => Ok(()),
    }
}

fn openai_stream_chunk(
    id: &str,
    model: &str,
    created: u64,
    chat: bool,
    role: Option<&str>,
    text: &str,
    finish_reason: Option<&str>,
) -> serde_json::Value {
    let choice = if chat {
        let mut delta = serde_json::Map::new();
        if let Some(role) = role {
            delta.insert("role".to_string(), serde_json::json!(role));
        }
        if !text.is_empty() {
            delta.insert("content".to_string(), serde_json::json!(text));
        }
        serde_json::json!({
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        })
    } else {
        serde_json::json!({
            "index": 0,
            "text": text,
            "finish_reason": finish_reason
        })
    };
    serde_json::json!({
        "id": id,
        "object": if chat { "chat.completion.chunk" } else { "text_completion" },
        "created": created,
        "model": model,
        "choices": [choice]
    })
}

async fn send_sse_json(
    tx: &mut futures::channel::mpsc::Sender<Result<actix_web::web::Bytes, std::io::Error>>,
    value: &serde_json::Value,
) -> Result<(), String> {
    use futures::SinkExt;
    tx.send(Ok(actix_web::web::Bytes::from(format!(
        "data: {}\n\n",
        value
    ))))
    .await
    .map_err(|error| format!("SSE send error: {}", error))
}

/// Request structure for inference
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct InferenceRequest {
    /// Model identifier
    pub model_id: String,
    /// Input text tokens
    pub input: Vec<u32>,
    /// Generation parameters
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

/// Response structure for inference
#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceResponse {
    /// Output token IDs (flattened as f32 for compatibility)
    pub output: Vec<f32>,
    /// Time taken for inference in milliseconds
    pub inference_time_ms: u64,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Model ID used
    pub model_id: String,
}

/// CLI configuration for inference server
#[derive(Debug, Clone)]
pub struct InferenceCli {
    /// Enable inference server
    pub enable_inference_server: bool,
    /// Server host
    pub inference_server_host: Option<String>,
    /// Server port
    pub inference_server_port: Option<u16>,
    /// Maximum concurrent requests
    pub inference_server_max_concurrent_inferences: Option<usize>,
    /// API path for inference server
    pub inference_server_api_path: Option<String>,
    /// Prompt cache size
    pub inference_server_prompt_cache_size: Option<usize>,
    /// Exit after one query
    pub inference_server_exit_after_one_query: Option<bool>,
    /// Maximum sequence length (default: 0 = use per-model max_position_embeddings)
    pub max_sequence_length: Option<usize>,
}

/// Main entry point for inference server
#[cfg(feature = "server")]
pub async fn server_inference(
    cli: InferenceCli,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if cli.enable_inference_server {
        let config = ServerConfig {
            host: cli
                .inference_server_host
                .unwrap_or_else(|| crate::config::server::DEFAULT_HOST.to_string()),
            port: cli
                .inference_server_port
                .unwrap_or(crate::config::server::DEFAULT_PORT),
            max_concurrent_requests: cli
                .inference_server_max_concurrent_inferences
                .unwrap_or(crate::config::server::DEFAULT_MAX_CONCURRENT_INFERENCES),
            request_timeout: Duration::from_secs(30),
            prompt_cache_size: cli
                .inference_server_prompt_cache_size
                .unwrap_or(crate::config::server::DEFAULT_PROMPT_CACHE_SIZE),
            max_sequence_length: cli.max_sequence_length.unwrap_or(0), // 0 = use per-model max_position_embeddings
            enable_tls: false,
            model_registry_path: cli.inference_server_api_path,
            allowed_origins: vec![],
        };

        println!("--- Starting Tensor Engine Inference Server ---");
        println!("Configuration: {:#?}", config);
        serve(config).await.map_err(|e| {
            log::error!("Server failed: {}", e);
            e
        })?;
    }

    Ok(())
}

#[cfg(not(feature = "server"))]
pub async fn server_inference(
    _cli: InferenceCli,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    eprintln!("Inference server is not enabled in this build.");
    eprintln!("Please enable it with the \"server\" feature.");
    eprintln!("Example: cargo run --features server -- [args]");
    Err("Inference server not enabled".into())
}

#[cfg(test)]
mod tests {
    use super::RegistryModelConfig;

    #[test]
    fn registry_config_accepts_hugging_face_names() {
        let config: RegistryModelConfig = serde_json::from_str(
            r#"{
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "model_type": "llama"
            }"#,
        )
        .unwrap();

        assert_eq!(config.hidden_size, 32);
        assert_eq!(config.num_key_value_heads, Some(2));
        assert!(config.validate().is_ok());
        assert_eq!(config.detect_architecture(), Ok(super::ModelArch::Llama));
    }

    #[test]
    fn registry_config_accepts_tensor_engine_aliases() {
        let config: RegistryModelConfig = serde_json::from_str(
            r#"{
                "vocab_size": 128,
                "d_model": 32,
                "d_ff": 64,
                "num_layers": 2,
                "num_heads": 4,
                "model_type": "llama"
            }"#,
        )
        .unwrap();

        assert_eq!(config.intermediate_size, 64);
        assert_eq!(config.num_key_value_heads, None);
        assert!(config.validate().is_ok());
        assert_eq!(config.detect_architecture(), Ok(super::ModelArch::Llama));
    }

    #[test]
    fn registry_config_accepts_nested_text_config() {
        let config: RegistryModelConfig = super::parse_registry_config(
            br#"{
                "model_type": "qwen3_vl",
                "text_config": {
                    "vocab_size": 151936,
                    "hidden_size": 4096,
                    "intermediate_size": 12288,
                    "num_hidden_layers": 36,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8
                }
            }"#,
        )
        .unwrap();

        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_key_value_heads, Some(8));
        assert!(config.validate().is_ok());
        assert_eq!(config.detect_architecture(), Ok(super::ModelArch::Qwen));
    }

    #[test]
    fn registry_config_rejects_unknown_model_type() {
        let config: RegistryModelConfig = serde_json::from_str(
            r#"{
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "model_type": "bert"
            }"#,
        )
        .unwrap();

        assert_eq!(
            config.detect_architecture(),
            Err("unsupported model_type 'bert'".to_string())
        );
        assert_eq!(
            config.validate().unwrap_err(),
            "unsupported model_type 'bert'"
        );
    }

    #[test]
    fn registry_config_rejects_missing_model_type() {
        let result = serde_json::from_str::<RegistryModelConfig>(
            r#"{
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4
            }"#,
        );

        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("model_type"),
            "missing model_type must be reported as a structured load error, got: {}",
            error
        );
    }

    #[test]
    fn registry_config_rejects_empty_model_type() {
        let config: RegistryModelConfig = serde_json::from_str(
            r#"{
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "model_type": ""
            }"#,
        )
        .unwrap();

        assert_eq!(
            config.validate().unwrap_err(),
            "missing required 'model_type' in config.json"
        );
    }

    #[test]
    fn registry_config_accepts_phi3_model_type() {
        let config: RegistryModelConfig = serde_json::from_str(
            r#"{
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "model_type": "phi3"
            }"#,
        )
        .unwrap();

        assert!(config.validate().is_ok());
        assert_eq!(config.detect_architecture(), Ok(super::ModelArch::Phi));
    }

    #[test]
    fn registry_config_rejects_invalid_head_dimensions() {
        let config: RegistryModelConfig = serde_json::from_str(
            r#"{
                "vocab_size": 128,
                "hidden_size": 30,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "model_type": "llama"
            }"#,
        )
        .unwrap();

        assert_eq!(
            config.validate().unwrap_err(),
            "hidden_size must be divisible by num_attention_heads"
        );
    }

    #[test]
    fn openai_completion_defaults_are_stable() {
        let request: super::OpenAICompletionRequest =
            serde_json::from_str(r#"{"model":"tiny","prompt":"hello"}"#).unwrap();
        let options = request.options();

        assert_eq!(options.max_tokens, 32);
        assert_eq!(options.temperature, 1.0);
        assert_eq!(options.top_p, 1.0);
        assert_eq!(options.seed, 42);
        assert!(!request.stream);
    }

    #[test]
    fn openai_chat_requires_structured_messages() {
        let request: super::OpenAIChatRequest = serde_json::from_str(
            r#"{
                "model":"tiny",
                "messages":[{"role":"user","content":"hello"}],
                "max_tokens":4
            }"#,
        )
        .unwrap();

        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, "user");
        assert_eq!(request.options().max_tokens, 4);
    }

    #[test]
    fn chat_stream_chunk_uses_delta_schema() {
        let chunk = super::openai_stream_chunk(
            "chatcmpl-test",
            "tiny",
            1,
            true,
            Some("assistant"),
            "hello",
            None,
        );

        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunk["choices"][0]["delta"]["content"], "hello");
        assert!(chunk["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn completion_stream_finish_chunk_has_length_reason() {
        let chunk =
            super::openai_stream_chunk("cmpl-test", "tiny", 1, false, None, "", Some("length"));

        assert_eq!(chunk["object"], "text_completion");
        assert_eq!(chunk["choices"][0]["text"], "");
        assert_eq!(chunk["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn request_guard_enforces_limit_and_releases_on_drop() {
        let server = std::sync::Arc::new(super::InferenceServer::new(super::ServerConfig {
            max_concurrent_requests: 1,
            ..super::ServerConfig::default()
        }));

        let guard = super::InferenceServer::try_acquire_request(&server).unwrap();
        assert_eq!(
            server
                .active_requests
                .load(std::sync::atomic::Ordering::Acquire),
            1
        );

        let rejected = match super::InferenceServer::try_acquire_request(&server) {
            Ok(_) => panic!("request above the configured limit was accepted"),
            Err(response) => response,
        };
        assert_eq!(
            rejected.status(),
            actix_web::http::StatusCode::TOO_MANY_REQUESTS
        );

        drop(guard);
        assert_eq!(
            server
                .active_requests
                .load(std::sync::atomic::Ordering::Acquire),
            0
        );
        assert!(super::InferenceServer::try_acquire_request(&server).is_ok());
    }

    #[test]
    fn expired_deadline_returns_stable_timeout_error() {
        let expired = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(1))
            .unwrap();

        assert_eq!(
            super::ensure_before_deadline(expired).unwrap_err(),
            super::REQUEST_TIMEOUT_ERROR
        );
    }

    #[test]
    fn future_and_optional_deadlines_are_accepted() {
        let future = std::time::Instant::now() + std::time::Duration::from_secs(1);

        assert!(super::ensure_before_deadline(future).is_ok());
        assert!(super::ensure_optional_deadline(Some(future)).is_ok());
        assert!(super::ensure_optional_deadline(None).is_ok());
    }
}
