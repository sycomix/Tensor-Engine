#!/usr/bin/env python3
"""
Simple Tensor Engine Server Example

Demonstrates basic inference server functionality with health checks.
"""

import asyncio
import json
import queue
import requests
import threading
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


def main():
    print("Tensor Engine Server Example")
    server_url = "http://localhost:8080"

    # Test health check
    print("Testing health check...")
    if health_check(server_url):
        print("✅ Health check passed")
    else:
        print("❌ Health check failed")

    # Test inference
    print("Testing inference...")
    result = inference_request(server_url, "demo", "Hello, Tensor Engine!")
    if result:
        print("✅ Inference successful")
    else:
        print("❌ Inference failed")


if __name__ == "__main__":
    main()
