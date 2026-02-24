import numpy as np
import tensor_engine as te

# This script exercises the new Phase 3 components: continuous thought,
# latent operations and decoders.  It can be executed after building the
# Python wheel with `pip install .`.

# 1. latent arithmetic / interpolation
vec_a = te.Tensor(np.array([1.0, 0.0, -1.0], dtype=np.float32))
vec_b = te.Tensor(np.array([0.0, 1.0, 0.0], dtype=np.float32))
print("vector arithmetic", te.vector_arithmetic(vec_a, vec_b, vec_a).to_list())
print("interp", te.linear_interpolate(vec_a, vec_b, 0.5).to_list())
print("slerp", te.spherical_interpolate(vec_a, vec_b, 0.5).to_list())

# 2. continuous thought
ctm = te.ContinuousThoughtModule(3)
inp = te.Tensor(np.zeros((1, 3), dtype=np.float32))
state1 = ctm.forward(inp)
state2 = ctm.forward(inp)
print("thought state shapes", state1.shape, state2.shape)

# 3. text decoder demo
textdec = te.TextDecoder(vocab_size=50, d_model=16, d_ff=64, num_heads=2, depth=2)
latent = te.Tensor(np.zeros((1, 5, 16), dtype=np.float32))
logits = textdec.forward(latent)
print("text logits shape", logits.shape)

# 4. image / video decoder demo
imgdec = te.ImageDecoder(in_channels=3, hidden=8, layers=2)
img_in = te.Tensor(np.zeros((1, 3, 8, 8), dtype=np.float32))
img_out = imgdec.forward(img_in)
print("image out shape", img_out.shape)

viddec = te.VideoDecoder(in_channels=3, hidden=8, layers=2)
vid_in = te.Tensor(np.zeros((1, 3, 4, 8, 8), dtype=np.float32))
vid_out = viddec.forward(vid_in)
print("video out shape", vid_out.shape)
