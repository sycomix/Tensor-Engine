#!/usr/bin/env python3
"""
Tensor-Engine Unity Bridge - Inference Server
Serves as the Python backend for Unity's Tensor Engine plugin.
Provides HTTP API for model loading, inference, and tensor operations.
"""

import argparse
import json
import sys
import os
import threading
import numpy as np

# Add the parent directory to path for tensor_engine import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import tensor_engine as te
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("[WARN] tensor_engine not found. Install with: pip install -e ..")

# Try to import transformer support
try:
    from tensor_engine.nn import TransformerBlock, MultimodalLLM
    from tensor_engine.io import load_safetensors_from_bytes
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("[WARN] Transformer modules not available. Some features will be limited.")

# Model registry
model_registry = {}
model_lock = threading.Lock()


class InferenceServer:
    def __init__(self, port=8765, host='0.0.0.0'):
        self.port = port
        self.host = host
        self.app = None

    def start(self):
        try:
            from flask import Flask, request, jsonify
            self.app = Flask(__name__)
            self._register_routes()
            print(f"[Server] Starting on {self.host}:{self.port}")
            self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
        except ImportError:
            print("[ERROR] Flask not installed. Run: pip install flask")
            sys.exit(1)

    def _register_routes(self):
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'ok',
                'tensor_engine': TE_AVAILABLE,
                'transformer': TRANSFORMER_AVAILABLE,
                'models_loaded': len(model_registry)
            })

        @self.app.route('/models', methods=['GET'])
        def list_models():
            with model_lock:
                return jsonify({'models': list(model_registry.keys())})

        @self.app.route('/models/<model_id>', methods=['DELETE'])
        def unload_model(model_id):
            with model_lock:
                if model_id in model_registry:
                    del model_registry[model_id]
                    return jsonify({'status': 'unloaded', 'model_id': model_id})
                return jsonify({'error': 'model not found'}), 404

        @self.app.route('/models/load', methods=['POST'])
        def load_model():
            data = request.json
            model_id = data.get('model_id', 'default')
            model_path = data.get('model_path', '')
            config_path = data.get('config_path', '')

            try:
                if not model_path or not os.path.exists(model_path):
                    return jsonify({'error': f'Model file not found: {model_path}'}), 404

                # Load model using tensor_engine
                model = self._load_model_from_path(model_path, config_path)
                if model is None:
                    return jsonify({'error': 'Failed to load model'}), 500

                with model_lock:
                    model_registry[model_id] = model

                return jsonify({
                    'status': 'loaded',
                    'model_id': model_id,
                    'path': model_path
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/inference', methods=['POST'])
        def inference():
            data = request.json
            model_id = data.get('model_id', 'default')
            input_data = data.get('input', {})

            try:
                if isinstance(input_data, str):
                    input_tensor = te.Tensor(json.loads(input_data)['data'],
                                             json.loads(input_data)['shape'])
                elif isinstance(input_data, dict):
                    input_tensor = te.Tensor(input_data.get('data', []),
                                             input_data.get('shape', []))
                else:
                    input_tensor = te.Tensor(input_data, [len(input_data)])

                max_tokens = data.get('max_tokens', 100)
                temperature = data.get('temperature', 0.8)
                top_k = data.get('top_k', None)
                top_p = data.get('top_p', None)

                with model_lock:
                    model = model_registry.get(model_id)

                if model is None:
                    return jsonify({'error': f'Model {model_id} not loaded'}), 404

                # Run inference
                result = self._run_inference(model, input_tensor, max_tokens, temperature, top_k, top_p)

                return jsonify({
                    'data': result.get_data(),
                    'shape': result.shape()
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

        @self.app.route('/compute', methods=['POST'])
        def compute():
            data = request.json
            operation = data.get('operation', '')

            try:
                # Parse input tensor(s)
                inputs = []
                if data.get('input'):
                    inp = data['input']
                    if isinstance(inp, str):
                        inp = json.loads(inp)
                    inputs.append(te.Tensor(inp['data'], inp['shape']))

                float_args = data.get('float_args', [])
                if isinstance(float_args, str):
                    float_args = json.loads(float_args)

                result = self._run_operation(operation, inputs, float_args)
                return jsonify({
                    'data': result.get_data(),
                    'shape': result.shape()
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

    def _load_model_from_path(self, model_path, config_path=''):
        """Load a model from a SafeTensors file or config."""
        if not model_path.endswith('.safetensors'):
            print(f"[WARN] Only SafeTensors models supported. Got: {model_path}")
            return None

        try:
            with open(model_path, 'rb') as f:
                state_dict = load_safetensors_from_bytes(f.read(), transpose_two_dim_weights=True)
            print(f"[Server] Loaded {len(state_dict)} tensors from {model_path}")
            return state_dict
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return None

    def _run_inference(self, model, input_tensor, max_tokens, temperature, top_k, top_p):
        """Run inference with the loaded model."""
        # For now, return the input tensor (model-specific inference would go here)
        # This is a placeholder - actual implementation depends on model type
        return input_tensor

    def _run_operation(self, operation, inputs, float_args):
        """Run a tensor operation."""
        if not inputs:
            raise ValueError("No input tensors provided")

        if operation == 'add':
            return te.Tensor.add(inputs[0], inputs[1]) if len(inputs) > 1 else inputs[0]
        elif operation == 'sub':
            return te.Tensor.sub(inputs[0], inputs[1]) if len(inputs) > 1 else inputs[0]
        elif operation == 'mul':
            return te.Tensor.mul(inputs[0], inputs[1]) if len(inputs) > 1 else inputs[0]
        elif operation == 'div':
            return te.Tensor.div(inputs[0], inputs[1]) if len(inputs) > 1 else inputs[0]
        elif operation == 'matmul':
            return te.Tensor.matmul(inputs[0], inputs[1]) if len(inputs) > 1 else inputs[0]
        elif operation == 'relu':
            return inputs[0].relu()
        elif operation == 'sigmoid':
            return inputs[0].sigmoid()
        elif operation == 'tanh':
            return inputs[0].tanh()
        elif operation == 'gelu':
            return inputs[0].gelu()
        elif operation == 'softmax':
            axis = int(float_args[0]) if float_args else 0
            return inputs[0].softmax(axis)
        elif operation == 'log_softmax':
            axis = int(float_args[0]) if float_args else 0
            return inputs[0].log_softmax(axis)
        elif operation == 'mean':
            return inputs[0].mean()
        elif operation == 'sum':
            return inputs[0].sum()
        elif operation == 'reshape':
            shape = [int(x) for x in float_args]
            return inputs[0].reshape(shape)
        elif operation == 'transpose':
            axes = [int(x) for x in float_args]
            return inputs[0].transpose(axes)
        elif operation == 'slice':
            axis = int(float_args[0])
            start = int(float_args[1])
            length = int(float_args[2])
            return inputs[0].slice(axis, start, length)
        elif operation == 'concat':
            return te.Tensor.concat(inputs, int(float_args[0]) if float_args else 0)
        elif operation == 'stack':
            return te.Tensor.stack(inputs, int(float_args[0]) if float_args else 0)
        else:
            raise ValueError(f"Unknown operation: {operation}")


def main():
    parser = argparse.ArgumentParser(description='Tensor-Engine Unity Bridge Server')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    server = InferenceServer(port=args.port, host=args.host)
    server.start()


if __name__ == '__main__':
    main()
