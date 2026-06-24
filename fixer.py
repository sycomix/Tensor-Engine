import glob, os, re

def fix_file(fpath, replacements):
    with open(fpath, encoding="utf-8", errors="replace") as f:
        content = f.read()
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  Replaced: {old[:60]}")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)

print("=== Fixing unpickler.rs ===")
fix_file("src/compat/engine/unpickler.rs", [
    ("stack.pop().unwrap()", "stack.pop().expect(\"unpickler: stack pop\")"),
    ("stack.last().unwrap()", "stack.last().expect(\"unpickler: stack last\")"),
])

print("=== Fixing data_source.rs ===")
fix_file("src/compat/engine/data_source.rs", [
    ("name.to_str().unwrap()", "name.to_str().expect(\"non-UTF-8 path\")"),
    ("archive.by_index(idx).unwrap()", "archive.by_index(idx).expect(\"archive index\")"),
])

print("=== Fixing huggingface_loader.rs ===")
for f in ["src/compat/engine/huggingface_loader.rs","src/hf_compat/huggingface_loader.rs"]:
    fix_file(f, [("file2.to_str().unwrap()", "file2.to_str().expect(\"non-UTF-8 zip entry\")")])

print("=== Fixing entrypoint.rs ===")
fix_file("src/compat/engine/entrypoint.rs", [
    ("serde_json::to_string(&chunk).unwrap()", "serde_json::to_string(&chunk).expect(\"chunk serialization\")"),
])

print("=== Fixing weight_compression.rs ===")
fix_file("src/compat/engine/weight_compression.rs", [
    ("a.partial_cmp(b).unwrap()", "a.partial_cmp(b).expect(\"partial_cmp (NaN?)\")"),
])

print("=== Fixing semaphore.rs ===")
fix_file("src/compat/engine/semaphore.rs", [
    ("self.count.lock().unwrap()", "self.count.lock().expect(\"semaphore lock\")"),
    ("self.waiters.wait(count).unwrap()", "self.waiters.wait(count).expect(\"semaphore wait\")"),
])

print("\\nDone!")

