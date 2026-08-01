import struct, json, numpy as np

f = open("models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors", "rb")
n = struct.unpack("<Q", f.read(8))[0]
h = json.loads(f.read(n).decode("utf-8"))


def read_tensor(name):
    t = h[name]
    off = 8 + n + t["data_offsets"][0]
    size = t["data_offsets"][1] - t["data_offsets"][0]
    f.seek(off)
    raw = f.read(size)
    if t["dtype"] == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        a = ((u16 << 16).astype(np.uint32)).view(np.float32)
    elif t["dtype"] == "F32":
        a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        a = np.frombuffer(raw, dtype="<f8").astype(np.float32)
    return a, t["dtype"]


keys = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.linear_attn.A_log",
    "model.language_model.layers.0.linear_attn.dt_bias",
    "model.language_model.layers.0.linear_attn.conv1d.weight",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.3.self_attn.q_norm.weight",
    "model.language_model.norm.weight",
]
for key in keys:
    a, dt = read_tensor(key)
    if key.endswith("embed_tokens.weight"):
        a = a.reshape(248320, 1024)
        print("%s %s: row0[:8]=%s" % (key, dt, np.round(a[0, :8], 6)))
    elif key.endswith("A_log") or key.endswith("dt_bias"):
        print("%s %s: first4=%s" % (key, dt, np.round(a[:4], 6)))
    elif key.endswith("conv1d.weight"):
        a = a.reshape(6144, 1, 4)
        print("%s %s: [0][0][0:2]=%s" % (key, dt, np.round(a[0, 0, :2], 6)))
    elif key.endswith("in_proj_qkv.weight"):
        a = a.reshape(6144, 1024)
        print("%s %s: [0][0:2]=%s" % (key, dt, np.round(a[0, :2], 6)))
    else:
        print(
            "%s %s: mean=%.4f median=%.4f std=%.4f min=%.4f max=%.4f"
            % (key, dt, a.mean(), np.median(a), a.std(), a.min(), a.max())
        )
