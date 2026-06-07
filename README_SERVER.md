# Tensor Engine — Inference Server (engine binary)

The `engine` binary starts an HTTP inference server by default, serving LLaMA-compatible models via a Rocket-based JSON API.

## Quick Start

```bash
# Build the engine binary
cargo build --bin engine --features compat

# Start the inference server
cargo run --bin engine --features compat -- \
  --model-path /path/to/model \
  --tokenizer-path /path/to/tokenizer.model \
  --param-path /path/to/params.json
```

The server starts on `http://0.0.0.0:8080` by default.

## Configuration

All server settings are passed as CLI flags to the `engine` binary:

| Argument | Default | Description |
|---|---|---|
| `--model-path` | (required) | Path to model directory |
| `--tokenizer-path` | (required) | Path to tokenizer file |
| `--param-path` | (required) | Path to params.json |
| `--inference-server-port` | `8080` | HTTP port |
| `--inference-server-host` | `0.0.0.0` | Bind address |
| `--inference-server-api-path` | `/` | API route path |
| `--inference-server-max-concurrent-inferences` | `4` | Max concurrent requests |
| `--inference-server-prompt-cache-size` | `128` | Number of prompt cache slots |
| `--inference-server-exit-after-one-query` | — | Shut down after first request |
| `--max-seq-len` | model default | Maximum sequence length |
| `--max-threads` | CPU count | Thread pool size |
| `--f16` | — | Use half-precision storage |
| `-q, --quiet` | — | Suppress startup output |

## API

### POST `<api-path>` (default `/`)

**Request** — `application/json`:

```json
{
  "prompt": "Your input text",
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.9,
  "repetition_penalty": 1.1,
  "max_seq_len": 2048,
  "max_new_tokens": 200,
  "no_token_sampling": false,
  "stop_at_end_token": true
}
```

**Response** — Newline-delimited JSON stream, one object per token:

```json
{"token": {"p": 0.85, "is_end_token": false}}
```

The stream ends with `is_end_token: true` or when `max_new_tokens` is reached.

### Example request (curl)

```bash
curl -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_new_tokens": 50}'
```

## CLI Mode

To run a one-shot prompt on the command line instead of starting the server, pass `--cli-mode`:

```bash
cargo run --bin engine --features compat -- \
  --model-path /path/to/model \
  --tokenizer-path /path/to/tokenizer.model \
  --param-path /path/to/params.json \
  --cli-mode --prompt "Hello, world!"
```

Interactive chat is available with `--cli-mode --start-interactive`.

## Backward Compatibility

The `rllama` binary is maintained as an alias:

```bash
cargo run --bin rllama --features compat -- [same flags]
```
