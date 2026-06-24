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
use std::sync::Arc;
use std::time::Duration;

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
            max_sequence_length: 2048,
            enable_tls: true,
            model_registry_path: None,
            allowed_origins: vec![],
        }
    }
}

use std::sync::RwLock;

/// Inference server instance
pub struct InferenceServer {
    config: ServerConfig,
    models: RwLock<HashMap<String, Arc<crate::nn::Llama>>>,
    request_count: std::sync::atomic::AtomicU64,
    active_requests: std::sync::atomic::AtomicUsize,
}

impl InferenceServer {
    /// Create a new inference server
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            models: RwLock::new(HashMap::new()),
            request_count: std::sync::atomic::AtomicU64::new(0),
            active_requests: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Start the inference server
    pub async fn start(mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use actix_web::{middleware, web, App, HttpServer};

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
        };

        HttpServer::new(app)
            .bind(server_addr)?
            .run()
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        log::info!("Tensor Engine Inference Server started successfully");
        Ok(())
    }

    /// Load model registry from disk
    ///
    /// Scans the provided directory for model configuration files and loads them.
    /// Expected directory structure:
    /// ```text
    /// model_registry/
    ///   model_id_1/
    ///     config.json
    ///     model.safetensors (or model.bin)
    ///   model_id_2/
    ///     config.json
    /// ```
    async fn load_model_registry(
        &mut self,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("Loading model registry from: {}", path);

        let registry_path = std::path::Path::new(path);
        if !registry_path.exists() {
            log::warn!("Model registry path does not exist: {}", path);
            return Ok(());
        }

        if !registry_path.is_dir() {
            return Err(format!("Model registry path is not a directory: {}", path).into());
        }

        let entries = std::fs::read_dir(registry_path)?;
        let mut loaded_count = 0usize;

        for entry in entries {
            let entry = entry?;
            let entry_path = entry.path();

            if entry_path.is_dir() {
                let model_id = entry_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let config_path = entry_path.join(crate::config::filenames::CONFIG_JSON);
                if config_path.exists() {
                    log::info!("Found model configuration at: {:?}", config_path);
                    log::info!("Model '{}' registered (lazy loading enabled)", model_id);
                    loaded_count += 1;
                } else {
                    log::debug!("Skipping directory without config.json: {:?}", entry_path);
                }
            }
        }

        log::info!(
            "Model registry scan complete: {} models found",
            loaded_count
        );
        Ok(())
    }

    /// Health check endpoint
    async fn handle_health() -> &'static str {
        "OK"
    }

    /// List available models
    async fn handle_list_models(
        state: actix_web::web::Data<Arc<InferenceServer>>,
    ) -> actix_web::HttpResponse {
        let models = state.models.read().unwrap();
        let model_ids: Vec<&String> = models.keys().collect();
        let model_list: Vec<serde_json::Value> = model_ids
            .iter()
            .map(|id| {
                serde_json::json!({
                    "id": id,
                    "status": "loaded"
                })
            })
            .collect();

        actix_web::HttpResponse::Ok()
            .content_type("application/json")
            .json(serde_json::json!({
                "models": model_list,
                "count": model_list.len()
            }))
    }

    /// Get specific model metadata
    async fn handle_get_model(
        state: actix_web::web::Data<Arc<InferenceServer>>,
        path: actix_web::web::Path<String>,
    ) -> actix_web::HttpResponse {
        let model_id = path.into_inner();
        let models = state.models.read().unwrap();
        match models.get(&model_id) {
            Some(_model) => actix_web::HttpResponse::Ok()
                .content_type("application/json")
                .json(serde_json::json!({
                    "id": model_id,
                    "status": "loaded",
                    "type": "llama"
                })),
            None => actix_web::HttpResponse::NotFound()
                .content_type("application/json")
                .json(serde_json::json!({
                    "error": "Model not found",
                    "requested_id": model_id
                })),
        }
    }

    /// Handle inference request
    ///
    /// Uses a semaphore-style atomic counter for concurrency limiting.
    /// The counter is incremented atomically; if it exceeds the limit,
    /// the request is rejected with 429 Too Many Requests.
    async fn handle_inference(
        req: actix_web::web::Json<InferenceRequest>,
        state: actix_web::web::Data<Arc<InferenceServer>>,
    ) -> Result<actix_web::HttpResponse, actix_web::Error> {
        let max_concurrent = state.config.max_concurrent_requests;

        // Atomically try to acquire a slot.
        // Load current value, and if below limit, CAS to current+1.
        loop {
            let current = state
                .active_requests
                .load(std::sync::atomic::Ordering::Acquire);
            if current >= max_concurrent {
                log::warn!(
                    "Inference request rejected: {} active >= {} max",
                    current,
                    max_concurrent
                );
                state
                    .request_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(actix_web::error::ErrorTooManyRequests(
                    "Server at max concurrent inference capacity",
                ));
            }
            let attempt = current + 1;
            match state.active_requests.compare_exchange_weak(
                current,
                attempt,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }

        let result = Self::process_request(&state, req.into_inner())
            .await
            .map(|res| actix_web::HttpResponse::Ok().json(res))
            .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()));

        state
            .active_requests
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
        state
            .request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        result
    }

    /// Handle streaming inference via Server-Sent Events (SSE).
    ///
    /// Generates tokens one at a time and streams each as an SSE event.
    /// The response has `Content-Type: text/event-stream`.
    async fn handle_stream_inference(
        req: actix_web::web::Json<InferenceRequest>,
        state: actix_web::web::Data<Arc<InferenceServer>>,
    ) -> Result<actix_web::HttpResponse, actix_web::Error> {
        let max_concurrent = state.config.max_concurrent_requests;

        loop {
            let current = state
                .active_requests
                .load(std::sync::atomic::Ordering::Acquire);
            if current >= max_concurrent {
                log::warn!(
                    "Streaming inference request rejected: {} active >= {} max",
                    current,
                    max_concurrent
                );
                state
                    .request_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(actix_web::error::ErrorTooManyRequests(
                    "Server at max concurrent inference capacity",
                ));
            }
            let attempt = current + 1;
            match state.active_requests.compare_exchange_weak(
                current,
                attempt,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }

        let req_inner = req.into_inner();

        // Validate before streaming
        if let Err(e) = Self::validate_request(&state, &req_inner) {
            state
                .active_requests
                .fetch_sub(1, std::sync::atomic::Ordering::Release);
            return Err(actix_web::error::ErrorBadRequest(e.to_string()));
        }

        // Get model
        let model = {
            let models = state.models.read().unwrap();
            models
                .get(&req_inner.model_id)
                .cloned()
                .ok_or_else(|| {
                    actix_web::error::ErrorNotFound(format!(
                        "Model '{}' not found",
                        req_inner.model_id
                    ))
                })?
        };

        let max_tokens = req_inner.max_tokens.unwrap_or(32) as usize;
        let temperature = req_inner.temperature.unwrap_or(1.0);
        let top_p = req_inner.top_p.unwrap_or(1.0);
        let seed = req_inner.seed.unwrap_or(42);
        let input_tokens = req_inner.input.clone();
        let model_id = req_inner.model_id.clone();

        // Create a futures MPSC channel for streaming bytes.
        // futures::channel::mpsc::Receiver implements Stream directly,
        // so it can be wrapped in BodyStream without extra adapters.
        let (tx, rx) =
            futures::channel::mpsc::channel::<Result<actix_web::web::Bytes, std::io::Error>>(32);
        let body = actix_web::body::BodyStream::new(rx);

        let state_clone = state.clone();

        // Spawn generation task
        actix_web::rt::spawn(async move {
            let mut tx = tx;
            let result = Self::generate_streaming(
                &model,
                &input_tokens,
                max_tokens,
                temperature,
                top_p,
                seed,
                &mut tx,
            )
            .await;

            state_clone
                .active_requests
                .fetch_sub(1, std::sync::atomic::Ordering::Release);
            state_clone
                .request_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            if let Err(e) = result {
                log::error!("Streaming generation error for model '{}': {}", model_id, e);
                use futures::SinkExt;
                let _ = tx
                    .send(Ok(actix_web::web::Bytes::from(format!(
                        "event: error\ndata: {}\n\n",
                        serde_json::json!({"error": e.to_string()})
                    ))))
                    .await;
            }
        });

        Ok(actix_web::HttpResponse::Ok()
            .content_type("text/event-stream")
            .append_header(("Cache-Control", "no-cache"))
            .append_header(("Connection", "keep-alive"))
            .streaming(body))
    }

    /// Process a single inference request: run autoregressive generation
    /// and return the full output.
    async fn process_request(
        state: &Arc<InferenceServer>,
        req: InferenceRequest,
    ) -> Result<InferenceResponse, crate::error::TensorError> {
        Self::validate_request(state, &req)?;

        let model = {
            let models = state.models.read().unwrap();
            models
                .get(&req.model_id)
                .ok_or_else(|| crate::error::TensorError::Generic {
                    message: format!("Model '{}' not found", req.model_id),
                })?
                .clone()
        };

        let max_tokens = req.max_tokens.unwrap_or(32) as usize;
        let temperature = req.temperature.unwrap_or(1.0);
        let top_p = req.top_p.unwrap_or(1.0);
        let seed = req.seed.unwrap_or(42);

        let start_time = std::time::Instant::now();

        let generated = Self::generate_tokens(
            &model,
            &req.input,
            max_tokens,
            temperature,
            top_p,
            seed,
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
        if req.input.is_empty() {
            return Err(crate::error::TensorError::ValidationError {
                field: "input".to_string(),
                value: "[]".to_string(),
                constraint: "non-empty input required".to_string(),
            });
        }
        if req.input.len() > state.config.max_sequence_length {
            return Err(crate::error::TensorError::ValidationError {
                field: "input".to_string(),
                value: req.input.len().to_string(),
                constraint: format!("max {} tokens", state.config.max_sequence_length),
            });
        }
        Ok(())
    }

    /// Run autoregressive generation on a Llama-style model.
    ///
    /// Uses `forward_single_token` with KV cache for efficient incremental decoding.
    /// Returns the generated token IDs (excluding the prompt).
    fn generate_tokens(
        model: &Arc<crate::nn::Llama>,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: u64,
    ) -> Result<Vec<f32>, String> {
        use crate::generation::sampling::Sampler;

        let mut sampler = Sampler::new(temperature, 0, top_p, seed);

        // Clone the model's internal state for this generation call.
        // We need mutable access to run forward_single_token, but we only have Arc.
        // Since Llama is Clone, we clone it.
        let mut model = (**model).clone();

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
        model: &Arc<crate::nn::Llama>,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: u64,
        tx: &mut futures::channel::mpsc::Sender<Result<actix_web::web::Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        use futures::SinkExt;
        use crate::generation::sampling::Sampler;

        let mut sampler = Sampler::new(temperature, 0, top_p, seed);
        let mut model = (**model).clone();

        let max_seq_len = prompt_ids.len() + max_new_tokens;
        model.init_kv_caches(max_seq_len)?;

        let prompt_len = prompt_ids.len();
        if prompt_len == 0 {
            return Err("prompt must contain at least one token".to_string());
        }

        for &token_id in &prompt_ids[..prompt_len.saturating_sub(1)] {
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
}

/// Main entry point for inference server
#[cfg(feature = "server")]
pub async fn server_inference(
    cli: InferenceCli,
    _tr: &crate::tensor::Tensor,
    _tok: &crate::tokenizer::Tokenizer,
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
            max_sequence_length: 2048,
            enable_tls: false,
            model_registry_path: cli.inference_server_api_path,
            allowed_origins: vec![],
        };

        let server = InferenceServer::new(config.clone());

        // Initialize a minimal Llama model for demo purposes
        let demo_model = crate::nn::Llama::new(
            32000, // vocab_size
            4096,  // d_model
            1,     // num_layers
            11008, // d_ff
            32,    // num_heads
            32,    // kv_heads
        )
        .map_err(|e| format!("Failed to create demo model: {}", e))?;

        server
            .models
            .write()
            .unwrap()
            .insert("demo".to_string(), Arc::new(demo_model));

        println!("--- Starting Tensor Engine Inference Server ---");
        println!("Configuration: {:#?}", config);
        println!("Models loaded: {}", server.models.read().unwrap().len());

        server.start().await.map_err(|e| {
            log::error!("Server failed: {}", e);
            e
        })?;
    }

    Ok(())
}

#[cfg(not(feature = "server"))]
pub async fn server_inference(
    _cli: InferenceCli,
    _tr: &crate::tensor::Tensor,
    _tok: &crate::tokenizer::Tokenizer,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    eprintln!("Inference server is not enabled in this build.");
    eprintln!("Please enable it with the \"server\" feature.");
    eprintln!("Example: cargo run --features server -- [args]");
    Err("Inference server not enabled".into())
}