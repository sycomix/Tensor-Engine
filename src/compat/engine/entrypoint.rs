use crate::compat::engine::data_source::DataSource;
use crate::compat::engine::embedding::Embedding;
use crate::compat::engine::model_params::ModelParams;

#[cfg(feature = "opencl")]
use crate::compat::engine::tensor_opencl_support::OpenCL;
use crate::compat::engine::token_sampler::TokenSampler;
use crate::compat::engine::tokenizer::{TokenId, Tokenizer};
use crate::compat::engine::transformer::{DataSettings, Transformer};

#[cfg(feature = "rocket")]
use crate::compat::engine::transformer::TransformerCaches;
use clap::Parser;
use colored::Colorize;
#[cfg(feature = "rocket")]
use rocket::data::ToByteUnit;
#[cfg(feature = "rocket")]
use rocket::response::Responder;
#[cfg(feature = "rocket")]
use rocket::tokio::io::AsyncReadExt;
#[cfg(feature = "rocket")]
use rocket::serde::json::Json;
use rocket::{http::ContentType, response, response::status, Data, Request, Response, State};
use serde::{Deserialize, Serialize};
#[cfg(feature = "rocket")]
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
#[cfg(feature = "rocket")]
use std::sync::RwLock;

const DEFAULT_CONFIG_NAME: &str = "engine.toml";

#[derive(Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
struct EngineConfig {
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    param_path: Option<String>,
    prompt: Option<String>,
    prompt_file: Option<String>,
    interactive_system_prompt: Option<String>,
    interactive_stop: Option<Vec<String>>,
    interactive_prompt_postfix: Option<String>,
    interactive_prompt_prefix: Option<String>,
    start_interactive: Option<bool>,
    chatml: Option<bool>,
    system_prompt: Option<String>,
    max_seq_len: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    repetition_penalty: Option<f32>,
    max_threads: Option<usize>,
    f16: Option<bool>,
    quiet: Option<bool>,
    cli_mode: Option<bool>,
    inference_server_port: Option<u16>,
    inference_server_host: Option<String>,
    inference_server_max_concurrent_inferences: Option<usize>,
    inference_server_prompt_cache_size: Option<usize>,
    inference_server_exit_after_one_query: Option<bool>,
    models_dir: Option<String>,

    #[cfg(feature = "opencl")]
    opencl_device: Option<usize>,
    #[cfg(feature = "opencl")]
    percentage_to_gpu: Option<f32>,
}

fn load_config(path: &str) -> Result<EngineConfig, Box<dyn std::error::Error>> {
    let mut f = std::fs::File::open(path)?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(toml::from_str(&s)?)
}

fn merge_config(
    cli: &Cli,
    config: Option<&EngineConfig>,
) -> EngineConfig {
    let cfg = config.cloned().unwrap_or_default();
    EngineConfig {
        model_path: cli.model_path.clone().or(cfg.model_path),
        tokenizer_path: cli.tokenizer_path.clone().or(cfg.tokenizer_path),
        param_path: cli.param_path.clone().or(cfg.param_path),
        prompt: cli.prompt.clone().or(cfg.prompt),
        prompt_file: cli.prompt_file.clone().or(cfg.prompt_file),
        interactive_system_prompt: cli.interactive_system_prompt.clone().or(cfg.interactive_system_prompt),
        interactive_stop: Some(
            if !cli.interactive_stop.is_empty() {
                cli.interactive_stop.clone()
            } else if cfg.interactive_stop.is_some() {
                cfg.interactive_stop.unwrap_or_default()
            } else {
                vec![]
            }
        ),
        interactive_prompt_postfix: cli.interactive_prompt_postfix.clone().or(cfg.interactive_prompt_postfix),
        interactive_prompt_prefix: cli.interactive_prompt_prefix.clone().or(cfg.interactive_prompt_prefix),
        start_interactive: cli.start_interactive.or(cfg.start_interactive),
        chatml: cli.chatml.or(cfg.chatml),
        system_prompt: cli.system_prompt.clone().or(cfg.system_prompt),
        max_seq_len: cli.max_seq_len.or(cfg.max_seq_len),
        temperature: cli.temperature.or(cfg.temperature),
        top_p: cli.top_p.or(cfg.top_p),
        top_k: cli.top_k.or(cfg.top_k),
        repetition_penalty: cli.repetition_penalty.or(cfg.repetition_penalty),
        max_threads: cli.max_threads.or(cfg.max_threads),
        f16: cli.f16.or(cfg.f16),
        quiet: cli.quiet.or(cfg.quiet),
        cli_mode: cli.cli_mode.or(cfg.cli_mode),
        inference_server_port: cli.inference_server_port.or(cfg.inference_server_port),
        inference_server_host: cli.inference_server_host.clone().or(cfg.inference_server_host),
        inference_server_max_concurrent_inferences: cli.inference_server_max_concurrent_inferences.or(cfg.inference_server_max_concurrent_inferences),
        inference_server_prompt_cache_size: cli.inference_server_prompt_cache_size.or(cfg.inference_server_prompt_cache_size),
        inference_server_exit_after_one_query: cli.inference_server_exit_after_one_query.or(cfg.inference_server_exit_after_one_query),
        models_dir: cli.models_dir.clone().or(cfg.models_dir),

        #[cfg(feature = "opencl")]
        opencl_device: cli.opencl_device.or(cfg.opencl_device),
        #[cfg(feature = "opencl")]
        percentage_to_gpu: cli.percentage_to_gpu.or(cfg.percentage_to_gpu),
    }
}

const INIT_CONFIG_TEMPLATE: &str = r#"# == Tensor Engine Configuration ==
# Generated by: engine.exe --init-config
# Usage: engine.exe --config engine.toml

# === Required Paths ===
# Path to the model directory containing config.json and weight files
# model_path = "/path/to/model"
# Path to the tokenizer file (sentencepiece .model)
# tokenizer_path = "/path/to/tokenizer.model"
# Path to config.json (optional: defaults to model_path/config.json, then model_path/params.json)
# param_path = "/path/to/config.json"

# === Inference Sampling ===
# Prompt text for one-shot generation
# prompt = "Once upon a time"
# Path to a text file containing the prompt
# prompt_file = "/path/to/prompt.txt"
# Maximum sequence length (context + generation)
# max_seq_len = 2048
# Sampling temperature (higher = more random)
# temperature = 0.8
# Top-p nucleus sampling threshold
# top_p = 0.9
# Top-k sampling (0 = disabled)
# top_k = 40
# Repetition penalty (>1.0 discourages repeats)
# repetition_penalty = 1.1

# === Server Mode (default) ===
# Port for the HTTP inference server
# inference_server_port = 8080
# Bind address for the server
# inference_server_host = "0.0.0.0"
# Maximum concurrent inference requests
# inference_server_max_concurrent_inferences = 4
# Prompt cache size (number of cached attention states)
# inference_server_prompt_cache_size = 128
# Exit after handling one query (useful for benchmarks)
# inference_server_exit_after_one_query = false

# === CLI Mode ===
# Run in CLI mode instead of server mode (requires prompt or start_interactive)
# cli_mode = true
# System prompt for interactive chat mode
# interactive_system_prompt = ""
# Stop sequences for interactive mode
# interactive_stop = []
# Prefix added to each user input in interactive mode
# interactive_prompt_prefix = ""
# Postfix added to each user input in interactive mode
# interactive_prompt_postfix = ""
# Start in interactive chat mode (CLI mode only)
# start_interactive = false

# === System ===
# Thread pool size (default: CPU count)
# max_threads = 4
# Use half-precision (f16) storage for weights
# f16 = false
# Suppress startup output
# quiet = false
"#;

fn onboarding() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("{}", "╔══════════════════════════════════════════════════╗".cyan());
    eprintln!("{}", "║        Tensor Engine — First Run Setup            ║".cyan());
    eprintln!("{}", "╚══════════════════════════════════════════════════╝".cyan());
    eprintln!();
    eprintln!("{}", "No configuration found.".yellow());
    eprintln!();
    eprintln!("  To get started, generate a config file:");
    eprintln!();
    eprintln!("    {}", "engine.exe --init-config".cyan());
    eprintln!();
    eprintln!("  This creates {} in the current directory.", DEFAULT_CONFIG_NAME.bold());
    eprintln!("  Edit it with your model paths, then run:");
    eprintln!();
    eprintln!("    {}", "engine.exe --config engine.toml".cyan());
    eprintln!();
    eprintln!("  Or pass everything on the command line:");
    eprintln!();
    eprintln!("    {} {} {}",
        "engine.exe".cyan(),
        "--model-path /path/to/model".bold(),
        "--tokenizer-path /path/to/tokenizer.model"
    );
    eprintln!();
    eprintln!("  Run {} for all available options.", "engine.exe --help".green());
    eprintln!();
    Err("No configuration provided. Run --init-config to create a config file.".into())
}

// Refer to README.md to see what all these options mean.
#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    config: Option<String>,

    #[arg(long)]
    init_config: bool,

    #[arg(long)]
    model_path: Option<String>,
    #[arg(long)]
    tokenizer_path: Option<String>,
    #[arg(long)]
    param_path: Option<String>,

    #[arg(short, long)]
    quiet: Option<bool>,

    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<String>,

    #[arg(long)]
    interactive_system_prompt: Option<String>,
    #[arg(long)]
    interactive_stop: Vec<String>,
    #[arg(long)]
    interactive_prompt_postfix: Option<String>,
    #[arg(long)]
    interactive_prompt_prefix: Option<String>,
    #[arg(long)]
    start_interactive: Option<bool>,

    #[arg(long)]
    chatml: Option<bool>,

    #[arg(long)]
    system_prompt: Option<String>,

    #[arg(long)]
    max_seq_len: Option<usize>,

    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    top_k: Option<i32>,
    #[arg(long)]
    repetition_penalty: Option<f32>,

    #[arg(long)]
    max_threads: Option<usize>,

    #[arg(long)]
    f16: Option<bool>,

    #[cfg(feature = "opencl")]
    #[arg(long)]
    opencl_device: Option<usize>,

    #[cfg(feature = "opencl")]
    #[arg(long)]
    percentage_to_gpu: Option<f32>,

    #[arg(long)]
    cli_mode: Option<bool>,

    #[arg(long)]
    inference_server_port: Option<u16>,

    #[arg(long)]
    inference_server_host: Option<String>,

    #[arg(long)]
    inference_server_max_concurrent_inferences: Option<usize>,

    #[arg(long)]
    inference_server_prompt_cache_size: Option<usize>,

    #[arg(long)]
    inference_server_exit_after_one_query: Option<bool>,

    #[arg(long)]
    models_dir: Option<String>,
}

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if cli.init_config {
        let path = cli.config.as_deref().unwrap_or(DEFAULT_CONFIG_NAME);
        if Path::new(path).exists() {
            eprintln!("{} already exists. Remove it first or use a different path.", path);
            return Err("Config file already exists.".into());
        }
        std::fs::write(path, INIT_CONFIG_TEMPLATE)?;
        eprintln!("Created example config: {}", path);
        eprintln!("Edit it with your model paths, then run:");
        eprintln!("  engine.exe --config {}", path);
        return Ok(());
    }

    let config = match &cli.config {
        Some(path) => Some(load_config(path)?),
        None => None,
    };

    let cfg = merge_config(&cli, config.as_ref());

    let model_path = cfg.model_path.clone().ok_or_else(|| {
        onboarding().ok();
        "Missing model_path"
    })?;
    let tokenizer_path = cfg.tokenizer_path.clone().ok_or_else(|| {
        onboarding().ok();
        "Missing tokenizer_path"
    })?;
    let param_path = match cfg.param_path.clone() {
        Some(p) => p,
        None => {
            let config_json = Path::new(&model_path).join("config.json");
            let params_json = Path::new(&model_path).join("params.json");
            if config_json.exists() {
                config_json.to_string_lossy().to_string()
            } else if params_json.exists() {
                params_json.to_string_lossy().to_string()
            } else {
                onboarding().ok();
                return Err("No param_path provided and neither config.json nor params.json found in model_path.".into());
            }
        }
    };

    let interactive_system_prompt = cfg
        .interactive_system_prompt
        .unwrap_or(crate::config::prompts::DEFAULT_SYSTEM_PROMPT.to_string());
    let mut interactive_stop = cfg.interactive_stop.clone().unwrap_or_default();
    if interactive_stop.is_empty() {
        interactive_stop = crate::config::prompts::default_stop_tokens();
    }
    let interactive_prompt_prefix = cfg
        .interactive_prompt_prefix
        .unwrap_or(crate::config::prompts::DEFAULT_INTERACTIVE_PREFIX.to_string());
    let interactive_prompt_postfix = cfg
        .interactive_prompt_postfix
        .unwrap_or(crate::config::prompts::DEFAULT_INTERACTIVE_POSTFIX.to_string());
    let start_interactive = cfg.start_interactive.unwrap_or(false);
    let chatml = cfg.chatml.unwrap_or(false);
    let system_prompt = cfg
        .system_prompt
        .unwrap_or_else(|| "You are a helpful assistant.".to_string());
    let cli_mode = cfg.cli_mode.unwrap_or(false);
    #[cfg(not(feature = "rocket"))]
    if !cli_mode {
        eprintln!("Inference server mode requires the 'rocket' feature.");
        return Err("Inference server mode is not available in this build.".into());
    }

    let max_threads: usize = match cfg.max_threads {
        None => rayon::current_num_threads(),
        Some(n) => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global()
                .unwrap();
            n
        }
    };

    #[cfg(feature = "opencl")]
    let percentage_to_gpu: f32 = cfg
        .percentage_to_gpu
        .unwrap_or(crate::config::opencl::DEFAULT_GPU_PERCENTAGE);

    let mut be_quiet: bool = false;
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        be_quiet = true;
    }
    if cfg.quiet.unwrap_or(false) {
        be_quiet = true;
    }
    if be_quiet {
        colored::control::SHOULD_COLORIZE.set_override(false);
    }

    macro_rules! pln {
        ($($arg:tt)*) => {
            if !be_quiet {
                std::println!($($arg)*);
            }
        };
    }

    let mut fs = std::fs::File::open(&param_path)?;
    let mut bs = Vec::new();
    fs.read_to_end(&mut bs)?;
    std::mem::drop(fs);

    let prompt: String = match (&cfg.prompt, &cfg.prompt_file, start_interactive) {
        (Some(ref p), None, _) => {
            pln!("Using prompt: {}", p);
            p.clone()
        }
        (None, Some(ref pf), _) => {
            pln!("Using prompt file: {}", pf);
            let mut fs = std::fs::File::open(pf)?;
            let mut bs = Vec::new();
            fs.read_to_end(&mut bs)?;
            std::mem::drop(fs);
            String::from_utf8(bs)?
        }
        (_, _, false) => {
            if cli_mode {
                eprintln!("Please provide either a prompt or a prompt file.");
                return Err("Please provide either a prompt or a prompt file.".into());
            } else {
                "".to_string()
            }
        }
        (None, None, true) => "".to_string(),
        (_, _, true) => {
            eprintln!("Please provide either a prompt or a prompt file.");
            return Err("Please provide either a prompt or a prompt file.".into());
        }
    };

    pln!("Starting up. Loading tokenizer from {}", tokenizer_path);
    let tok = Tokenizer::load(tokenizer_path.as_str())?;
    pln!("Tokenizer loaded. Loading model from {}", model_path);

    let model_data_source = DataSource::from_inferred_source(model_path.clone())?;

    let params: ModelParams = {
        let mut config_value: serde_json::Value = serde_json::from_slice(&bs)?;
        // VL models nest text params inside "text_config"
        let text_config = config_value
            .get("text_config")
            .and_then(|v| v.as_object())
            .cloned();
        if let Some(tc) = text_config {
            if let Some(obj) = config_value.as_object_mut() {
                for (k, v) in tc {
                    if !obj.contains_key(&k) {
                        obj.insert(k, v);
                    }
                }
            }
        }
        serde_json::from_value(config_value)?
    };
    pln!("Loaded model parameters from {}.", param_path);

    let gen_config_path = Path::new(&model_path).join("generation_config.json");
    let gen_config = std::fs::read(&gen_config_path).ok().and_then(|bs| {
        serde_json::from_slice::<serde_json::Value>(&bs).ok()
    });
    let default_temperature: f32 = gen_config
        .as_ref()
        .and_then(|g| g.get("temperature").and_then(|v| v.as_f64()))
        .map(|v| v as f32)
        .unwrap_or(crate::config::inference::DEFAULT_TEMPERATURE);
    let default_top_p: f32 = gen_config
        .as_ref()
        .and_then(|g| g.get("top_p").and_then(|v| v.as_f64()))
        .map(|v| v as f32)
        .unwrap_or(crate::config::inference::DEFAULT_TOP_P);
    let default_top_k: usize = gen_config
        .as_ref()
        .and_then(|g| g.get("top_k").and_then(|v| v.as_i64()))
        .map(|v| v as usize)
        .unwrap_or(crate::config::inference::DEFAULT_TOP_K);
    let default_repetition_penalty: f32 = gen_config
        .as_ref()
        .and_then(|g| g.get("repetition_penalty").and_then(|v| v.as_f64()))
        .map(|v| v as f32)
        .unwrap_or(crate::config::inference::DEFAULT_REPETITION_PENALTY);
    let do_sample: bool = gen_config
        .as_ref()
        .and_then(|g| g.get("do_sample").and_then(|v| v.as_bool()))
        .unwrap_or(true);
    // When do_sample is false, force greedy (top_k=1) unless user explicitly overrides
    let default_top_k = if !do_sample { 1 } else { default_top_k };

    pln!("Loading embeddings from {}", model_path);
    let emb = Embedding::from_unpickled(model_data_source.clone())?;

    let max_seq_len = cfg
        .max_seq_len
        .unwrap_or(crate::config::inference::DEFAULT_MAX_SEQ_LEN);

    #[cfg(feature = "opencl")]
    let has_opencl;

    let f16_enabled = cfg.f16.unwrap_or(false);

    let data_settings = {
        #[cfg(feature = "opencl")]
        {
            let opencl_device = cli.opencl_device.unwrap_or(0);
            let opencl: Option<OpenCL> = match OpenCL::new(!be_quiet, opencl_device) {
                Err(openclerr) => {
                    eprintln!("OpenCL error: {}", openclerr);
                    eprintln!("OpenCL is disabled because it failed to initialize.");
                    None
                }
                Ok(opencl) => {
                    println!("OpenCL initialized.");
                    Some(opencl)
                }
            };
            has_opencl = opencl.is_some();
            let ds = if let Some(ocl) = opencl {
                let ds = DataSettings::new(Some(ocl));
                ds.percentage_to_gpu(percentage_to_gpu).use_opencl()
            } else {
                DataSettings::new(None)
            };
            let ds = if f16_enabled || has_opencl {
                ds.force_f16()
            } else {
                ds
            };
            ds
        }
        #[cfg(not(feature = "opencl"))]
        {
            let mut ds = DataSettings::new();
            if f16_enabled {
                ds = ds.force_f16();
            }
            ds
        }
    };

    #[cfg(not(feature = "opencl"))]
    let has_opencl = false;

    pln!("Loading transformer weights from {}", model_path);
    let tr = Transformer::from_unpickled(
        emb,
        params.dim,
        params.n_layers,
        params.n_heads,
        max_seq_len,
        params.norm_eps,
        data_settings,
        model_data_source,
        params.head_dim,
        params.n_kv_heads,
        params.rope_theta,
    )?;
    pln!("All is loaded. Starting inference.");

    let tr: Arc<Transformer> = Arc::new(tr);
    let tok: Arc<Tokenizer> = Arc::new(tok);

    if cli_mode {
        command_line_inference(
            cli.clone(),
            tr.clone(),
            tok.clone(),
            prompt.clone(),
            interactive_stop.clone(),
            interactive_system_prompt.clone(),
            interactive_prompt_prefix.clone(),
            interactive_prompt_postfix.clone(),
            start_interactive,
            be_quiet,
            max_seq_len,
            params.clone(),
            max_threads,
            chatml,
            system_prompt.clone(),
            default_temperature,
            default_top_p,
            default_top_k,
            default_repetition_penalty,
        )
    } else {
        {
            let server_cli = Cli {
                inference_server_port: cfg.inference_server_port,
                inference_server_host: cfg.inference_server_host.clone(),
                inference_server_max_concurrent_inferences: cfg.inference_server_max_concurrent_inferences,
                inference_server_prompt_cache_size: cfg.inference_server_prompt_cache_size,
                inference_server_exit_after_one_query: cfg.inference_server_exit_after_one_query,
                ..cli
            };
            server_inference(
                server_cli,
                tr,
                tok,
                be_quiet,
                max_seq_len,
                params,
                max_threads,
                default_temperature,
                default_top_p,
                default_top_k,
                default_repetition_penalty,
            )
            .await
        }
        #[cfg(not(feature = "rocket"))]
        {
            eprintln!("The inference server feature is not enabled.");
            eprintln!("Please enable it with the \"rocket\" feature.");
            Err("The inference server feature is not enabled.".into())
        }
    }
}

#[cfg(feature = "rocket")]
async fn server_inference(
    cli: Cli,
    tr: Arc<Transformer>,
    tok: Arc<Tokenizer>,
    be_quiet: bool,
    max_seq_len: usize,
    _params: ModelParams,
    _max_threads: usize,
    default_temperature: f32,
    default_top_p: f32,
    default_top_k: usize,
    default_repetition_penalty: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    macro_rules! pln {
        ($($arg:tt)*) => {
            if !be_quiet {
                std::println!($($arg)*);
            }
        };
    }

    let inference_server_port = cli
        .inference_server_port
        .unwrap_or(crate::config::server::DEFAULT_PORT);
    let inference_server_host = cli
        .inference_server_host
        .clone()
        .unwrap_or(crate::config::server::DEFAULT_HOST.to_string());
    let inference_server_max_concurrent_inferences = cli
        .inference_server_max_concurrent_inferences
        .unwrap_or(crate::config::server::DEFAULT_MAX_CONCURRENT_INFERENCES);
    let inference_server_prompt_cache_size = cli
        .inference_server_prompt_cache_size
        .unwrap_or(crate::config::server::DEFAULT_PROMPT_CACHE_SIZE);

    pln!(
        "Maximum concurrent inferences: {}",
        inference_server_max_concurrent_inferences
    );
    pln!("Prompt cache size: {}", inference_server_prompt_cache_size);
    pln!("Maximum sequence length: {}", max_seq_len);
    pln!(
        "--- Starting HTTP server on {}:{} ---",
        inference_server_host,
        inference_server_port
    );
    pln!("Endpoints:");
    pln!("  POST /v1/chat/completions  (OpenAI chat, SSE streaming)");
    pln!("  POST /v1/completions       (OpenAI completions, SSE streaming)");
    pln!("  GET  /v1/models            (list available models)");

    let rocket_conf = rocket::Config::figment()
        .merge(("address", inference_server_host))
        .merge(("port", inference_server_port));

    let model_id = cli.model_path.clone()
        .unwrap_or_default()
        .rsplit(|c| c == '/' || c == '\\')
        .next()
        .unwrap_or("unknown")
        .to_string();

    let mut available_models = scan_available_models(&cli, be_quiet);
    if !available_models.contains(&model_id) {
        available_models.push(model_id.clone());
    }

    let app = rocket::custom(rocket_conf)
        .mount("/", routes![list_models, openai_chat_handler, openai_completions_handler])
        .manage(InferenceServerState {
            transformer: tr,
            tokenizer: tok,
            max_seq_len,
            model_id,
            available_models,
            eos_token_ids: _params.eos_token_ids(),
            attention_cache_repository: Arc::new(RwLock::new(AttentionCacheRepository::empty(
                inference_server_prompt_cache_size,
            ))),
            exit_after_one_query: cli.inference_server_exit_after_one_query.unwrap_or(false),
            default_temperature,
            default_top_p,
            default_top_k,
            default_repetition_penalty,
        });

    let _ = app.launch().await;
    panic!("Starting web server failed.");
}

#[cfg(feature = "rocket")]
struct GeneratingSession {
    transformer: Arc<Transformer>,
    token_sampler: TokenSampler,
    tokenizer: Arc<Tokenizer>,
    attention_cache_repository: Arc<RwLock<AttentionCacheRepository>>,
    tokens: Vec<TokenId>,
    req_max_seq_len: usize,
    req_max_new_tokens: usize,
    new_tokens_generated: usize,
    prev_pos: usize,
    no_token_sampling: bool,
    stop_at_end_token: bool,
    sent_stuff_last_time: bool,
    exit_after_one_query: bool,
    result: Vec<u8>,
    eos_token_ids: Vec<i64>,
}

#[cfg(feature = "rocket")]
impl<'r> Responder<'r, 'static> for GeneratingSession {
    fn respond_to(self, _: &'r Request<'_>) -> response::Result<'static> {
        let ct = if self.no_token_sampling {
            ContentType::JSON
        } else {
            ContentType::Plain
        };
        Response::build()
            .header(ct)
            .streamed_body(self)
            .ok()
    }
}

#[cfg(feature = "rocket")]
impl rocket::tokio::io::AsyncRead for GeneratingSession {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut rocket::tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let mut b = vec![0u8; buf.remaining()];
        match std::io::Read::read(self.get_mut(), &mut b) {
            Ok(n) => {
                buf.put_slice(&b[..n]);
                std::task::Poll::Ready(Ok(()))
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
            Err(e) => std::task::Poll::Ready(Err(e)),
        }
    }
}

#[cfg(feature = "rocket")]
impl GeneratingSession {
    fn read_from_result(&mut self, buf: &mut [u8]) -> usize {
        if !self.result.is_empty() {
            if self.result.len() <= buf.len() {
                for idx in 0..self.result.len() {
                    buf[idx] = self.result[idx];
                }
                let len = self.result.len();
                self.sent_stuff_last_time = true;
                self.result.truncate(0);
                return len;
            } else {
                for idx in 0..buf.len() {
                    buf[idx] = self.result[idx];
                }
                self.result = self.result[buf.len()..].to_vec();
                self.sent_stuff_last_time = true;
                return buf.len();
            }
        }
        return 0;
    }
}

#[cfg(feature = "rocket")]
impl GeneratingSession {
    fn generate_one_token(&mut self) -> std::io::Result<Option<String>> {
        if self.tokens.len() >= self.req_max_seq_len {
            if self.exit_after_one_query {
                std::process::exit(0);
            }
            return Ok(None);
        }
        if self.new_tokens_generated >= self.req_max_new_tokens {
            if self.exit_after_one_query {
                std::process::exit(0);
            }
            return Ok(None);
        }

        let (mut caches, update_pos) = {
            let mut ac = self.attention_cache_repository.write().unwrap();
            match ac.get(&self.tokens) {
                Some((c, pos)) if pos >= self.prev_pos => (c.true_clone(), pos),
                Some(_) => {
                    std::mem::drop(ac);
                    (self.transformer.make_caches(), 0)
                }
                None => {
                    let caches = self.transformer.make_caches();
                    ac.put(self.tokens.clone(), caches.true_clone(), self.prev_pos);
                    (caches, self.prev_pos)
                }
            }
        };
        if update_pos > self.prev_pos {
            self.prev_pos = update_pos;
        }

        let predictions =
            self.transformer
                .forward(&self.tokens[self.prev_pos..], self.prev_pos, &mut caches);
        self.prev_pos = self.tokens.len();
        let (highest_pred_idx, _token_prob) =
            self.token_sampler
                .sample(&predictions, self.tokenizer.as_ref(), &self.tokens);
        self.tokens.push(highest_pred_idx as TokenId);
        {
            let mut ac = self.attention_cache_repository.write().unwrap();
            ac.put(self.tokens.clone(), caches, self.prev_pos);
        }
        self.new_tokens_generated += 1;
        let token: String = self.tokenizer.decode_token(highest_pred_idx as TokenId);

        if self.stop_at_end_token && self.eos_token_ids.contains(&(highest_pred_idx as i64)) {
            if self.exit_after_one_query {
                std::process::exit(0);
            }
            return Ok(None);
        }

        Ok(Some(token))
    }
}

#[cfg(feature = "rocket")]
impl Read for GeneratingSession {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.sent_stuff_last_time && self.result.is_empty() {
            self.sent_stuff_last_time = false;
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "WouldBlock",
            ));
        }

        let bytes_read = self.read_from_result(buf);
        if bytes_read > 0 {
            return Ok(bytes_read);
        }

        match self.generate_one_token()? {
            Some(token) => {
                self.result.extend(token.as_bytes());
                Ok(self.read_from_result(buf))
            }
            None => Ok(0),
        }
    }
}

#[cfg(feature = "rocket")]
struct AttentionCacheRepository {
    caches: BTreeMap<Vec<TokenId>, (TransformerCaches, usize, std::time::Instant)>,
    max_sz: usize,
}

#[cfg(feature = "rocket")]
impl AttentionCacheRepository {
    fn empty(max_size: usize) -> AttentionCacheRepository {
        AttentionCacheRepository {
            caches: BTreeMap::new(),
            max_sz: max_size,
        }
    }

    fn limit_size(&mut self, sz: usize) {
        if sz == 0 {
            self.caches = BTreeMap::new();
            return;
        }
        while self.caches.len() > sz {
            let mut oldest_time = None;
            let mut oldest_key: Option<&Vec<TokenId>> = None;
            for (k, (_, _, time)) in self.caches.iter() {
                if oldest_time.is_none() || time < oldest_time.unwrap() {
                    oldest_time = Some(time);
                    oldest_key = Some(k);
                }
            }
            let oldest_key = oldest_key.unwrap().clone();
            self.caches.remove(&oldest_key);
        }
    }

    fn get(&self, tokens: &[TokenId]) -> Option<(&TransformerCaches, usize)> {
        if let Some((caches, pos, _)) = self.caches.get(tokens) {
            Some((caches, *pos))
        } else {
            None
        }
    }

    fn put(&mut self, tokens: Vec<TokenId>, caches: TransformerCaches, prev_pos: usize) {
        self.caches
            .insert(tokens, (caches, prev_pos, std::time::Instant::now()));
        self.limit_size(self.max_sz);
    }
}

#[cfg(feature = "rocket")]
fn scan_available_models(cli: &Cli, be_quiet: bool) -> Vec<String> {
    let scan_dir = cli.models_dir.clone().or_else(|| {
        cli.model_path.as_ref().and_then(|p| {
            let parent = Path::new(p).parent()?;
            parent.to_str().map(|s| s.to_string())
        })
    });

    let mut ids: Vec<String> = Vec::new();

    if let Some(dir) = scan_dir {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let has_config = path.join("config.json").exists();
                let has_params = path.join("params.json").exists();
                let has_safetensors = {
                    let mut found = false;
                    if let Ok(files) = std::fs::read_dir(&path) {
                        for f in files.flatten() {
                            let name = f.file_name();
                            if let Some(n) = name.to_str() {
                                if n.ends_with(".safetensors") {
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                    found
                };
                if has_config || has_params || has_safetensors {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        ids.push(name.to_string());
                        if !be_quiet {
                            eprintln!("Found model: {}", name);
                        }
                    }
                }
            }
        }
    }

    ids.sort();
    ids.dedup();
    ids
}

#[cfg(feature = "rocket")]
struct InferenceServerState {
    transformer: Arc<Transformer>,
    tokenizer: Arc<Tokenizer>,
    max_seq_len: usize,
    model_id: String,
    available_models: Vec<String>,
    eos_token_ids: Vec<i64>,
    attention_cache_repository: Arc<RwLock<AttentionCacheRepository>>,
    exit_after_one_query: bool,
    default_temperature: f32,
    default_top_p: f32,
    default_top_k: usize,
    default_repetition_penalty: f32,
}

#[cfg(feature = "rocket")]
#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[cfg(feature = "rocket")]
#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelEntry>,
}

#[cfg(feature = "rocket")]
#[get("/v1/models")]
fn list_models(state: &State<InferenceServerState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: state.available_models.iter().map(|id| ModelEntry {
            id: id.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "user".to_string(),
        }).collect(),
    })
}

// ── OpenAI-compatible API ──────────────────────────────────────────

#[cfg(feature = "rocket")]
#[derive(Deserialize)]
struct OpenAIRequest {
    model: Option<String>,
    messages: Option<Vec<OpenAIMessage>>,
    prompt: Option<String>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    stream: bool,
}

#[cfg(feature = "rocket")]
#[derive(Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[cfg(feature = "rocket")]
#[derive(Serialize)]
struct OpenAIChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAIChoice>,
}

#[cfg(feature = "rocket")]
#[derive(Serialize)]
struct OpenAIChoice {
    index: usize,
    delta: OpenAIDelta,
    finish_reason: Option<String>,
}

#[cfg(feature = "rocket")]
#[derive(Serialize)]
struct OpenAIDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[cfg(feature = "rocket")]
fn build_openai_prompt(messages: &[OpenAIMessage]) -> Result<String, status::BadRequest<String>> {
    let mut buf = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => buf.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", msg.content)),
            "user" => buf.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content)),
            "assistant" => buf.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", msg.content)),
            _ => return Err(status::BadRequest(format!("Unknown role: {}", msg.role))),
        }
    }
    buf.push_str("<|im_start|>assistant\n");
    Ok(buf)
}

#[cfg(feature = "rocket")]
struct OpenAISession {
    inner: GeneratingSession,
    model: String,
    id: String,
    created: u64,
    first_chunk: bool,
    done: bool,
}

#[cfg(feature = "rocket")]
impl<'r> Responder<'r, 'static> for OpenAISession {
    fn respond_to(self, _: &'r Request<'_>) -> response::Result<'static> {
        Response::build()
            .header(ContentType::new("text", "event-stream"))
            .raw_header("Cache-Control", "no-cache")
            .raw_header("Connection", "keep-alive")
            .streamed_body(self)
            .ok()
    }
}

#[cfg(feature = "rocket")]
impl rocket::tokio::io::AsyncRead for OpenAISession {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut rocket::tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let this = self.get_mut();
        let mut b = vec![0u8; buf.remaining()];
        match std::io::Read::read(this, &mut b) {
            Ok(n) => {
                buf.put_slice(&b[..n]);
                std::task::Poll::Ready(Ok(()))
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
            Err(e) => std::task::Poll::Ready(Err(e)),
        }
    }
}

#[cfg(feature = "rocket")]
impl Read for OpenAISession {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.done {
            return Ok(0);
        }
        if !self.inner.result.is_empty() {
            let n = self.inner.read_from_result(buf);
            if n > 0 {
                return Ok(n);
            }
        }

        // Generate one token from inner session
        match self.inner.generate_one_token() {
            Ok(Some(token_text)) => {
                let chunk = if self.first_chunk {
                    self.first_chunk = false;
                    OpenAIChunk {
                        id: self.id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: self.created,
                        model: self.model.clone(),
                        choices: vec![OpenAIChoice {
                            index: 0,
                            delta: OpenAIDelta {
                                role: Some("assistant".to_string()),
                                content: Some(token_text),
                            },
                            finish_reason: None,
                        }],
                    }
                } else {
                    OpenAIChunk {
                        id: self.id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: self.created,
                        model: self.model.clone(),
                        choices: vec![OpenAIChoice {
                            index: 0,
                            delta: OpenAIDelta {
                                role: None,
                                content: Some(token_text),
                            },
                            finish_reason: None,
                        }],
                    }
                };
                let json = serde_json::to_string(&chunk).unwrap();
                let line = format!("data: {}\n\n", json);
                self.inner.result.extend(line.as_bytes());
                return Ok(self.inner.read_from_result(buf));
            }
            Ok(None) => {
                // Generation complete — send final chunk and [DONE]
                self.done = true;
                let chunk = OpenAIChunk {
                    id: self.id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: self.created,
                    model: self.model.clone(),
                    choices: vec![OpenAIChoice {
                        index: 0,
                        delta: OpenAIDelta { role: None, content: None },
                        finish_reason: Some("stop".to_string()),
                    }],
                };
                let json = serde_json::to_string(&chunk).unwrap();
                let line = format!("data: {}\n\ndata: [DONE]\n", json);
                self.inner.result.extend(line.as_bytes());
                return Ok(self.inner.read_from_result(buf));
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(feature = "rocket")]
#[post("/v1/chat/completions", data = "<input>")]
async fn openai_chat_handler(
    state: &State<InferenceServerState>,
    input: Data<'_>,
) -> Result<OpenAISession, status::BadRequest<String>> {
    let mut data = input.open(128.megabytes());
    let mut databuf: Vec<u8> = Vec::new();
    data.read_to_end(&mut databuf)
        .await
        .map_err(|e| status::BadRequest(format!("Read error: {}", e)))?;

    let req: OpenAIRequest = serde_json::from_slice(&databuf)
        .map_err(|e| status::BadRequest(format!("Invalid JSON: {}", e)))?;

    let prompt = match req.prompt {
        Some(p) => p,
        None => {
            match req.messages.as_ref() {
                Some(msgs) => build_openai_prompt(msgs)?,
                None => return Err(status::BadRequest("Missing 'messages' or 'prompt'.".to_string())),
            }
        }
    };

    let max_tokens = req.max_tokens.unwrap_or(256);
    let temperature = req.temperature.unwrap_or(state.default_temperature);
    let top_p = req.top_p.unwrap_or(state.default_top_p);
    let top_k = req.top_k.unwrap_or(state.default_top_k);
    let repetition_penalty = req.repetition_penalty.unwrap_or(state.default_repetition_penalty);
    let model = req.model.clone().unwrap_or_else(|| state.model_id.clone());

    let token_sampler = TokenSampler::new()
        .temperature(temperature)
        .top_p(top_p)
        .top_k(top_k)
        .repetition_penalty(repetition_penalty);

    let toks_id: Vec<TokenId> = state.tokenizer.tokenize_to_ids(prompt.clone());

    let inner = GeneratingSession {
        transformer: state.transformer.clone(),
        tokenizer: state.tokenizer.clone(),
        attention_cache_repository: state.attention_cache_repository.clone(),
        token_sampler,
        tokens: toks_id,
        req_max_seq_len: state.max_seq_len,
        req_max_new_tokens: max_tokens,
        new_tokens_generated: 0,
        prev_pos: 0,
        no_token_sampling: false,
        stop_at_end_token: true,
        sent_stuff_last_time: false,
        exit_after_one_query: state.exit_after_one_query,
        result: Vec::new(),
        eos_token_ids: state.eos_token_ids.clone(),
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(OpenAISession {
        inner,
        model,
        id: format!("chatcmpl-{:016x}", rand::random::<u64>()),
        created: now,
        first_chunk: true,
        done: false,
    })
}

#[cfg(feature = "rocket")]
#[post("/v1/completions", data = "<input>")]
async fn openai_completions_handler(
    state: &State<InferenceServerState>,
    input: Data<'_>,
) -> Result<OpenAISession, status::BadRequest<String>> {
    // Reuse chat handler by wrapping prompt in messages format
    let mut data = input.open(128.megabytes());
    let mut databuf: Vec<u8> = Vec::new();
    data.read_to_end(&mut databuf)
        .await
        .map_err(|e| status::BadRequest(format!("Read error: {}", e)))?;

    let req: OpenAIRequest = serde_json::from_slice(&databuf)
        .map_err(|e| status::BadRequest(format!("Invalid JSON: {}", e)))?;

    let prompt = req.prompt.ok_or_else(|| status::BadRequest("Missing 'prompt'.".to_string()))?;

    let max_tokens = req.max_tokens.unwrap_or(256);
    let temperature = req.temperature.unwrap_or(state.default_temperature);
    let top_p = req.top_p.unwrap_or(state.default_top_p);
    let top_k = req.top_k.unwrap_or(state.default_top_k);
    let repetition_penalty = req.repetition_penalty.unwrap_or(state.default_repetition_penalty);
    let model = req.model.clone().unwrap_or_else(|| state.model_id.clone());

    let token_sampler = TokenSampler::new()
        .temperature(temperature)
        .top_p(top_p)
        .top_k(top_k)
        .repetition_penalty(repetition_penalty);

    let toks_id: Vec<TokenId> = state.tokenizer.tokenize_to_ids(prompt.clone());

    let inner = GeneratingSession {
        transformer: state.transformer.clone(),
        tokenizer: state.tokenizer.clone(),
        attention_cache_repository: state.attention_cache_repository.clone(),
        token_sampler,
        tokens: toks_id,
        req_max_seq_len: state.max_seq_len,
        req_max_new_tokens: max_tokens,
        new_tokens_generated: 0,
        prev_pos: 0,
        no_token_sampling: false,
        stop_at_end_token: true,
        sent_stuff_last_time: false,
        exit_after_one_query: state.exit_after_one_query,
        result: Vec::new(),
        eos_token_ids: state.eos_token_ids.clone(),
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(OpenAISession {
        inner,
        model,
        id: format!("cmpl-{:016x}", rand::random::<u64>()),
        created: now,
        first_chunk: true,
        done: false,
    })
}

#[cfg(feature = "rocket")]
fn command_line_inference(
    cli: Cli,
    tr: Arc<Transformer>,
    tok: Arc<Tokenizer>,
    prompt: String,
    interactive_stop: Vec<String>,
    interactive_system_prompt: String,
    interactive_prompt_prefix: String,
    interactive_prompt_postfix: String,
    start_interactive: bool,
    be_quiet: bool,
    max_seq_len: usize,
    params: ModelParams,
    max_threads: usize,
    chatml: bool,
    system_prompt: String,
    default_temperature: f32,
    default_top_p: f32,
    default_top_k: usize,
    default_repetition_penalty: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    macro_rules! pln {
        ($($arg:tt)*) => {
            if !be_quiet {
                std::println!($($arg)*);
            }
        };
    }

    let mut prompt = prompt;

    if start_interactive && !prompt.is_empty() {
        return Err(
            "Cannot start interactive mode with a prompt. Use --interactive-system-prompt instead."
                .into(),
        );
    }
    if start_interactive {
        prompt = interactive_system_prompt.clone();
    }
    if chatml {
        let user_msg = std::mem::take(&mut prompt);
        prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, user_msg
        );
    }

    let mut toks_id: Vec<TokenId> = tok.tokenize_to_ids(prompt.clone());
    if let Some(bos_id) = params.bos_token_id {
        if bos_id >= 0 && !toks_id.is_empty() && toks_id[0] != bos_id as i32 {
            toks_id.insert(0, bos_id as TokenId);
        }
    }
    let mut toks_str: String = prompt.clone();
    let mut prev_pos = 0;
    let mut token_sampler = TokenSampler::new()
        .temperature(default_temperature)
        .top_p(default_top_p)
        .top_k(default_top_k)
        .repetition_penalty(default_repetition_penalty);

    if let Some(temperature) = cli.temperature {
        token_sampler = token_sampler.temperature(temperature);
    }
    if let Some(top_p) = cli.top_p {
        token_sampler = token_sampler.top_p(top_p);
    }
    if let Some(top_k) = cli.top_k {
        token_sampler = token_sampler.top_k(top_k as usize);
    }
    if let Some(repetition_penalty) = cli.repetition_penalty {
        token_sampler = token_sampler.repetition_penalty(repetition_penalty);
    }
    pln!("---");
    pln!(" dim: {}", params.dim);
    pln!(" n_heads: {}", params.n_heads);
    pln!(" n_layers: {}", params.n_layers);
    pln!(" norm_eps: {}", params.norm_eps);
    pln!(" vocab_size: {}", params.vocab_size);
    pln!("---");
    pln!(" maximum number of threads: {}", max_threads);
    pln!("---");
    pln!("Max sequence length: {}", max_seq_len);
    pln!("Temperature: {}", token_sampler.get_temperature());
    pln!("Top P: {}", token_sampler.get_top_p());
    pln!("Top K: {}", token_sampler.get_top_k());
    pln!(
        "Repetition penalty: {}",
        token_sampler.get_repetition_penalty()
    );
    if start_interactive {
        pln!(
            "  Interactive mode stop token sequences: {:?}",
            interactive_stop
        );
        pln!("---");
        pln!("System prompt:");
        pln!("  {}", interactive_system_prompt);
        pln!("---");
        pln!("Interactive prompt prefix: {}", interactive_prompt_prefix);
        pln!("Interactive prompt postfix: {}", interactive_prompt_postfix);
    }
    pln!("---");
    pln!(
        "{}",
        "  This is the color of the initial prompt".truecolor(128, 128, 255)
    );
    pln!(
        "{}",
        "  This is the color of the generated text".truecolor(128, 255, 128)
    );
    pln!("---");
    print!("{}", prompt.as_str().truecolor(128, 128, 255));

    let _ = std::io::stdout().flush();

    let mut first_token_time: std::time::Duration = std::time::Duration::new(0, 0);
    let mut times_per_token: Vec<std::time::Duration> = vec![];
    let mut caches = tr.make_caches();
    let mut first: bool = true;
    let mut stop_seen: bool = false;
    let mut interactive = start_interactive;
    let mut user_token: Vec<TokenId> = vec![];
    let eos_ids: Vec<i64> = params.eos_token_ids();
    while toks_id.len() < max_seq_len {
        let now = std::time::Instant::now();
        let preds = tr.forward(&toks_id[prev_pos..], prev_pos, &mut caches);
        if interactive {
            let mut newinput = String::new();
            std::io::stdin().read_line(&mut newinput)?;
            if newinput.ends_with('\n') {
                let _ = newinput.pop();
            }
            newinput = interactive_prompt_prefix.clone() + &newinput;
            newinput += &interactive_prompt_postfix;
            user_token = tok.tokenize_to_ids(newinput.clone());

            let _ = user_token.remove(0);
            interactive = false;
        }
        let (highest_pred_idx, token_prob) = if user_token.len() > 0 {
            (user_token.remove(0), 0.0)
        } else {
            token_sampler.sample(&preds, &tok, &toks_id)
        };
        toks_id.push(highest_pred_idx as TokenId);

        for (tok_idx, tok_id) in toks_id[prev_pos + 1..].iter().enumerate() {
            if *tok_id == 1 {
                continue;
            }
            let mut tok_print: String = "".to_string();
            let is_last = tok_idx == toks_id[prev_pos + 1..].len() - 1;
            if eos_ids.contains(&(*tok_id as i64)) && is_last {
                stop_seen = true;
                break;
            }
            let tok_str = tok.id_to_str(*tok_id);
            if tok_str == "<0x0A>" {
                tok_print += "\n";
            } else {
                tok_print += &tok.decode_token(*tok_id);
            }
            toks_str += tok_print.as_str();
            if first && tok_idx < toks_id.len() - 2 {
            } else {
                let redness: f32 = token_prob * 255.0;
                let redness = if redness > 255.0 {
                    255
                } else if redness < 0.0 {
                    0
                } else {
                    redness as u8
                };
                print!(
                    "{}",
                    tok_print.truecolor(128 + redness / 2, 255 - redness / 2, 128)
                );
            }
            for stop_str in interactive_stop.iter() {
                if !first && toks_str.ends_with(stop_str.as_str()) {
                    if start_interactive {
                        interactive = true;
                    }
                    break;
                }
            }
        }
        if first {
            first_token_time = now.elapsed();
        } else {
            times_per_token.push(now.elapsed());
        }
        let _ = std::io::stdout().flush();
        prev_pos = toks_id.len() - 1;
        first = false;
        if stop_seen {
            break;
        }
    }
    println!();
    if stop_seen && !be_quiet {
        println!("Stop token seen. Stopping.");
    }
    eprintln!("---");
    eprintln!(
        "Time taken to generate first token: {:?}ms",
        first_token_time.as_millis()
    );
    let num_gen = times_per_token.len();
    if num_gen > 0 {
        let per_token_ms = times_per_token.iter().map(|t| t.as_millis()).sum::<u128>() / num_gen as u128;
        eprintln!(
            "Time taken per token (excluding first token): {:?}ms",
            per_token_ms
        );
    } else {
        eprintln!("No token generated");
    }
    Ok(())
}
