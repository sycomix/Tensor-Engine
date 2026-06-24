import glob

files = [
    "src/compat/engine/transformer.rs",
    "src/compat/engine/tensor_opencl_support.rs",
    "src/monitoring.rs",
    "src/compat/engine/protomodels/sentencepiece_model.rs",
    "src/nn/moe.rs",
    "src/nn/paged_attention.rs",
    "src/nn/paged_kv_cache.rs",
    "src/nn/model_visualization.rs",
    "src/nn/clip.rs",
    "src/nn/embedding.rs",
]

for f in files:
    content = open(f, encoding="utf-8", errors="replace").read()
    lines = content.splitlines()
    test_starts = {i for i,l in enumerate(lines) if "#[cfg(test)]" in l}
    prod = [(i+1,l.rstrip()) for i,l in enumerate(lines) if ".unwrap()" in l and not any(i >= ts for ts in test_starts)]
    if prod:
        print(f"\n=== {f} ({len(prod)}) ===")
        for n,l in prod:
            print(f"  L{n}: {l.strip()[:120]}")
