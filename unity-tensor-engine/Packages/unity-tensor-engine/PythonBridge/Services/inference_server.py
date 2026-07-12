# Tensor Engine Unity Bridge
#
# The Python Flask bridge has been replaced by direct communication
# with the Rust engine.exe HTTP server.
#
# PythonBridgeService.cs now spawns engine.exe directly and talks
# to its OpenAI-compatible REST API on the configured port (default 9090).
#
# Endpoints used:
#   GET  /v1/models              — list available models
#   POST /v1/chat/completions    — streaming chat (SSE)
#   POST /v1/completions         — streaming text completion (SSE)
#
# This file is kept for reference only and is no longer loaded at runtime.
