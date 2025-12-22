import tensor_engine as te
import sys
import time

print("Creating TransformerBlock...")
block = te.TransformerBlock(
    d_model=2048,
    d_ff=8192,
    num_heads=32,
    use_rope=True,
    llama_style=True,
    llama_bias=False
)
print("  Created successfully")

print("\nLoading SafeTensors bytes...")
with open('d:\\models\\model.safetensors', 'rb') as f:
    model_bytes = f.read()
print(f"  Loaded {len(model_bytes) / 1e9:.2f}GB")

print("\nTrying py_load_safetensors_into_module for layer 0...")
print("  This may take a while or hang...")
sys.stdout.flush()

start = time.time()
try:
    te.py_load_safetensors_into_module(
        model_bytes,
        transpose=False,
        module=block,
        root="model.layers.0."
    )
    elapsed = time.time() - start
    print(f"  SUCCESS! Took {elapsed:.2f}s")
    
    print("\nVerifying weights changed:")
    params = list(block.named_parameters(""))
    for name, param in params[:3]:
        data = param.get_data()
        print(f"  {name}: first value = {data[0]}")
        
except Exception as exc:
    elapsed = time.time() - start
    print(f"  FAILED after {elapsed:.2f}s: {exc}")
