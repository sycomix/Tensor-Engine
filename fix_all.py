from collections import OrderedDict

# All remaining fixes in one file
fixes = OrderedDict([
    (# ----- TRRANSFORMER.RS -----)
    ("src/compat/engine/transformer.rs", [
        ("self.cache_k[idx].read().unwrap()", "self.cache_k[idx].read().expect(\"transformer: cache_k read\")"),
        ("self.cache_v[idx].read().unwrap()", "self.cache_v[idx].read().expect(\"transformer: cache_v read\")"),
        ("self.cache_k[idx].write().unwrap()", "self.cache_k[idx].write().expect(\"transformer: cache_k write\")"),
        ("self.cache_v[idx].write().unwrap()", "self.cache_v[idx].write().expect(\"transformer: cache_v write\")"),
        ("attention_cache.cache_k[kv_idx].write().unwrap()", "attention_cache.cache_k[kv_idx].write().expect(\"attention cache_k write\")"),
        ("attention_cache.cache_v[kv_idx].write().unwrap()", "attention_cache.cache_v[kv_idx].write().expect(\"attention cache_v write\")"),
        ("attention_cache.cache_k[kv_idx].read().unwrap()", "attention_cache.cache_k[kv_idx].read().expect(\"attention cache_k read\")"),
        ("attention_cache.cache_v[kv_idx].read().unwrap()", "attention_cache.cache_v[kv_idx].read().expect(\"attention cache_v read\")"),
        ("ds.cl.as_ref().unwrap().clone()", "ds.cl.as_ref().expect(\"cl ref\").clone()"),
        ("ds.cl.unwrap()", "ds.cl.expect(\"cl\")"),
        ("self.data_settings.cl.as_ref().unwrap()", "self.data_settings.cl.as_ref().expect(\"data_settings.cl\")"),
        ".to_gpu_inplace().unwrap()", ".to_gpu_inplace().expect(\"to_gpu_inplace\")"),
        ".to_cpu_inplace().unwrap()", ".to_cpu_inplace().expect(\"to_cpu_inplace\")"),
    ]),
    (# ----- TENSOR_OPENCL_SUPPORT.RS -----)
    ("src/compat/engine/tensor_opencl_support.rs", [
        (".set_arg(...).unwrap()", ".set_arg(...).expect(\"set_arg\")"),
        ("b.enq().unwrap()", "b.enq().expect(\"enq\")"),
        ("self.last_event.as_ref().unwrap().wait_for().unwrap()", "self.last_event.as_ref().expect(\"last_event\").wait_for().expect(\"wait_for\")")),
        ("self.event.wait_for().unwrap()", "self.event.wait_for().expect(\"wait_for\")"),
        ("self.cl.programs.write().unwrap()", "self.cl.programs.write().expect(\"cl programs\")"),
    ]),
    (# ----- MONITORING.RS -----)
    ("monitoring.rs", [
        ".read().unwrap()", ".read().expect(\"monitoring read\")"),
        ".write().unwrap()", ".write().expect(\"monitoring write\")"),
    ]),
    (# ----- MOEL_VISUALIZATION.RS -----)
    ("model_visualization.rs", [
        ".last().unwrap()", ".last().expect(\"model_visualization: last\")"),
        ".first().unwrap()", ".first().expect(\"model_visualization: first\")"),
    ]),
    (# ----- PAGED_ATTENTION.RS -----)
    ("paged_attention.rs", [
        (".sequences.lock().unwrap()", ".sequences.lock().expect(\"paged: sequences lock\")"),
        ".engine.lock().unwrap()", ".engine.lock().expect(\"paged: engine lock\")"),
        (".get(&phys_idx).unwrap()", ".get(&phys_idx).expect(\"paged: phys_idx\")"),
    ]),
    (# ----- PAGED_KV_CACHE.RS -----)
    ("paged_kv_cache.rs", [
        (".sequences.lock().unwrap()", ".sequences.lock().expect(\"pkvc: sequences lock\")"),
        ".engine.lock().unwrap()", ".engine.lock().expect(\"pkvc: engine lock\")"),
        (".get(&phys_idx).unwrap().clone()", ".get(&phys_idx).expect(\"pkvc: phys_idx\").clone()"),
    ]),
])

for fpath, reps in fixes.items():
    print(f"\\n=== {fpath} ===")
    with open(fpath, encoding="utf-8", errors="replace") as f:
        c = f.read()
    for old, new in reps:
        if isinstance(old, str):
            if old in c:
                c = c.replace(old, new)
                print(f"  Replaced: {old[:60}")
        else:
            # faly match, skip
            pass
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(c)

print("\nDone!")

