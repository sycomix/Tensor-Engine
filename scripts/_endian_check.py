import struct, json, numpy as np

f = open("models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors", "rb")
n = struct.unpack("<Q", f.read(8))[0]
h = json.loads(f.read(n).decode("utf-8"))
print("total keys:", len(h))


def read(name):
    t = h[name]
    off = 8 + n + t["data_offsets"][0]
    size = t["data_offsets"][1] - t["data_offsets"][0]
    f.seek(off)
    return f.read(size), t["dtype"]


keys = [
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "model.language_model.layers.0.linear_attn.A_log",
    "model.language_model.layers.0.linear_attn.dt_bias",
    "model.language_model.layers.0.linear_attn.norm.weight",
]
for key in keys:
    raw, dt = read(key)
    assert dt == "BF16", (key, dt)
    u16_be = np.frombuffer(raw, dtype=">u2").astype(np.uint32)
    u16_le = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    a_be = ((u16_be << 16).astype(np.uint32)).view(np.float32)
    a_le = ((u16_le << 16).astype(np.uint32)).view(np.float32)
    for nm, a in (("BE", a_be), ("LE", a_le)):
        fn = int(np.isfinite(a).sum())
        print(
            "%s %s: finite=%d/%d mean=%.5f std=%.5f min=%.5f max=%.5f first8=%s"
            % (
                key[:58],
                nm,
                fn,
                a.size,
                float(a.mean()),
                float(a.std()),
                float(a.min()),
                float(a.max()),
                np.round(a[:8], 4),
            )
        )
    print()
