#!/usr/bin/env python3
"""
Tensor Engine Production Server Example

This example demonstrates the enhanced production model serving capabilities
including dynamic batching, model registry, health checks, and streaming inference.
"""

import argparse
import asyncio
import json
import queue
import requests
import threading
import time
import websockets


def inference_request(server_url, model_id, prompt, max_tokens=100,
                      ord=None, map=None):
    """Send inference request to the server"""
    request_data = {
        "model_id": model_id,
        "input": list(map(ord, prompt)),
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(
            f"{server_url}/inference",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()

        if response.status_code == 200:
            result = response.json()
            print(f"Inference completed in {result['inference_time_ms']}ms")
            print(f"Tokens generated: {result['tokens_generated']}")
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def health_check(server_url):
    """Check server health status"""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        response.raise_for_status()

        if response.status_code == 200:
            result = response.json()
            print(f"Server health: {result.get('status', 'unknown')}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False


def list_models(server_url):
    """List available models"""
    try:
        response = requests.get(f"{server_url}/models", timeout=5)
        response.raise_for_status()

        if response.status_code == 200:
            models = response.json()
            print(f"Available models: {models}")
            return models
        else:
            print(f"Failed to list models: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Failed to list models: {e}")
        return None
async def streaming_inference(server_url, model_id, prompt):
    """Test streaming inference with WebSocket"""
    ws_url = f"ws://{server_url.replace('http://', 'ws://')}/inference/stream"

    try:
        async with websockets.connect(ws_url) as websocket:
            await websocket.send(json.dumps({"type": "start", "model_id": model_id, "prompt": prompt}))
            async for message in websocket:
                payload = json.loads(message)
                if payload.get("type") == "token":
                    print(f"Token: {payload.get('token')}")
                elif payload.get("type") == "completed":
                    print(f"Completed: {payload}")
                    break
    except Exception as e:
        print(f"WebSocket connection error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Tensor Engine Server Example")
    parser.add_argument("--server", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--model", default="demo", help="Model ID to use")
    parser.add_argument("--prompt", required=True, help="Text prompt for inference")
    parser.add_argument("--health-check", action="store_true", help="Run health check only")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--streaming", action="store_true", help="Test streaming inference")

    args = parser.parse_args()

    print(f"Tensor Engine Server Example")
    print(f"Server URL: {args.server}")

    if args.health_check:
        health_check(args.server)
    elif args.list_models:
        list_models(args.server)
    elif args.streaming:
        asyncio.run(streaming_inference(args.server, args.model, args.prompt))
    else:
        result = inference_request(args.server, args.model, args.prompt)
        print(f"Inference result: {result}")


if __name__ == "__main__":
    main()
