import json

idx = json.load(
    open("models/Qwen3.5-0.8B/model.safetensors.index.json", encoding="utf-8")
)
keys = list(idx["weight_map"].keys())
lines = ["total keys: %d" % len(keys)]
for k in keys[:10]:
    lines.append("  " + k)
lines.append(
    "plain model.*: %r"
    % [
        k
        for k in keys
        if k.startswith("model.") and not k.startswith("model.language_model.")
    ][:5]
)
lines.append("lm_head.weight: %r" % ("lm_head.weight" in keys))
lines.append("model.lm_head.weight: %r" % ("model.lm_head.weight" in keys))
lines.append("prefixes: %r" % sorted(set(k.split(".")[0] for k in keys)))
open("tmp_keys.txt", "w", encoding="utf-8").write("\n".join(lines))
print("wrote tmp_keys.txt")
