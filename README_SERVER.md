# Tensor Engine — Inference Server (engine binary)

The `engine` binary starts an HTTP inference server by default, serving LLaMA-compatible models via OpenAI-compatible endpoints.

## Quick Start

```bash
# Build the engine binary
cargo build --bin engine --features compat

# Start the inference server
cargo run --bin engine --features compat -- \
  --model-path /path/to/model
```

The server starts on `http://127.0.0.1:8080` by default. The model directory should contain `config.json`, `tokenizer.json` (or `tokenizer.model`), and model weights (`.safetensors`).

## Configuration

Server settings are passed as CLI flags:

| Argument | Default | Description |
|---|---|---|
| `--model-path` | (required) | Path to model directory |
| `--inference-server-port` | `8080` | HTTP port |
| `--inference-server-host` | `127.0.0.1` | Bind address |
| `--inference-server-max-concurrent-inferences` | `5` | Max concurrent requests |
| `--inference-server-prompt-cache-size` | `50` | Number of prompt cache slots |
| `--inference-server-exit-after-one-query` | — | Shut down after first request |
| `--max-seq-len` | `1024` | Maximum sequence length |
| `--max-threads` | CPU count | Thread pool size |
| `--f16` | — | Use half-precision storage |
| `--opencl-device` | `0` | OpenCL device index |
| `-q, --quiet` | — | Suppress startup output |

### Sampling Parameters

The server reads sampling defaults from `generation_config.json` in the model directory. The request body can override each parameter individually. The default values (used when neither config file nor request body provides a value) are:

| Parameter | Default |
|---|---|
| `temperature` | `1.0` |
| `top_p` | `1.0` |
| `top_k` | `20` |
| `repetition_penalty` | `1.0` |

If `generation_config.json` contains `"do_sample": false`, the server forces `top_k=1` (greedy decoding) unless explicitly overridden in the request.

## API

All endpoints return SSE (Server-Sent Events) streams.

### `POST /v1/chat/completions`

OpenAI-compatible chat completions. Request body:

```json
{
  "model": "model-name",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "max_tokens": 256,
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 40,
  "repetition_penalty": 1.1,
  "stream": true
}
```

Response: SSE stream with OpenAI chat completion chunks.

#### Example (curl)

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "stream": true
  }'
```

### `POST /v1/completions`

OpenAI-compatible text completions. Request body:

```json
{
  "model": "model-name",
  "prompt": "The meaning of life is",
  "max_tokens": 200,
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 40,
  "repetition_penalty": 1.1,
  "stream": true
}
```

#### Example (curl)

```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The meaning of life is",
    "max_tokens": 50,
    "stream": true
  }'
```

### `GET /v1/models`

Lists available models (scans the parent directory of `--model-path` for model subdirectories).

#### Example (curl)

```bash
curl http://127.0.0.1:8080/v1/models
```

## CLI Mode

To run a one-shot prompt on the command line instead of starting the server, pass `--cli-mode`:

```bash
cargo run --bin engine --features compat -- \
  --model-path /path/to/model \
  --cli-mode --prompt "Hello, world!"
```

Interactive chat is available with `--cli-mode --start-interactive`. ChatML formatting is supported with `--chatml` and a system prompt with `--system-prompt`.

### CLI Sampling Flags

In CLI mode, sampling parameters can be overridden via flags:

| Flag | Default (from `generation_config.json`) |
|---|---|
| `--temperature` | from config or `1.0` |
| `--top-p` | from config or `1.0` |
| `--top-k` | from config or `20` |
| `--repetition-penalty` | from config or `1.0` |
