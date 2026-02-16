import sys
import os
import math
import numpy as np
from PIL import Image
import tensor_engine as te
import time

def load_image(path, size=224):
    if not os.path.exists(path):
        # Create a dummy image if not exists
        print(f"Image {path} not found, creating dummy noise image.")
        arr = np.random.rand(1, 3, size, size).astype(np.float32)
        return te.Tensor(arr, requires_grad=True)
        
    img = Image.open(path).convert('RGB')
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    # [H, W, C] -> [1, C, H, W]
    arr = arr.transpose(2, 0, 1)[None, ...]
    # Normalize with ImageNet mean/std if typical, but CLIP uses specific mean/std
    # OpenAI CLIP mean: [0.48145466, 0.4578275, 0.40821073]
    # std: [0.26862954, 0.26130258, 0.27577711]
    # We'll just return raw 0-1 for optimization and normalize inside loss if needed, 
    # but simplest is to optimize in 0-1 space.
    return te.Tensor(arr, requires_grad=True)

def save_image(tensor, path):
    arr = tensor.numpy() # [1, C, H, W]
    if len(arr.shape) == 4:
        arr = arr[0]
    arr = np.transpose(arr, (1, 2, 0)) # [H, W, C]
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

# Affine grid generator
def affine_grid(theta, size):
    # theta: [N, 2, 3]
    # size: [N, C, H, W]
    N, C, H, W = size
    
    # Create normalized grid [-1, 1]
    # x: -1 to 1 (W)
    # y: -1 to 1 (H)
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xv, yv = np.meshgrid(x, y) # [H, W]
    
    # [H, W, 3] homogenous coords
    ones = np.ones_like(xv)
    grid = np.stack([xv, yv, ones], axis=-1) # [H, W, 3]
    grid = grid.reshape(H * W, 3).T # [3, H*W]
    
    # Apply theta
    # theta [N, 2, 3] x grid [3, H*W] -> [N, 2, H*W]
    # We output [N, H, W, 2]
    
    grids = []
    for i in range(N):
        t = theta[i] # [2, 3]
        g = np.dot(t, grid) # [2, H*W]
        g = g.T.reshape(H, W, 2)
        grids.append(g)
        
    return np.stack(grids).astype(np.float32)

class MakeCutouts:
    def __init__(self, cut_size, num_cutouts, cut_pow=1.0):
        self.cut_size = cut_size
        self.num_cutouts = num_cutouts
        self.cut_pow = cut_pow
        
    def __call__(self, input_tensor):
        # input: [1, 3, H, W]
        # output: [num_cutouts, 3, cut_size, cut_size]
        
        # We generate random affine params for random crops/transforms
        # For simplicity, let's just do random crops via affine translation/scaling
        
        sideY, sideX = input_tensor.shape()[2:]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        
        thetas = []
        for _ in range(self.num_cutouts):
            # Random scaling
            size = min_size + (max_size - min_size) * (np.random.rand() ** self.cut_pow)
            scale = self.cut_size / size
            
            # Random translation
            # Map center of crop (cx, cy) to center of output
            # Affine matrix logic is tricky. 
            # Simple identity grid scaled and shifted.
            # sx 0 tx
            # 0 sy ty
            
            # Keep aspect ratio
            sx = scale
            sy = scale
            
            # Random translation in [-1, 1] range relative to input
            # Available range for center is derived from scale
            # If scale=1 (full crop), t=0.
            # If scale < 1 (zoom out?), no we zoom in usually for cutouts.
            # GridSample maps output [-1,1] to input [-1,1].
            # If we want a crop, we want output [-1,1] to map to a SMALL region in input.
            # So sx, sy should be < 1?
            # E.g. if we want 0.5 of input, we iterate -0.5 to 0.5 in input space.
            # Output grid is -1 to 1.
            # input = grid * matrix? No.
            # grid_sample(input, grid): value at grid[x,y] is sampled from input.
            # If grid[x,y] = 0, we sample center.
            # If we want to zoom in (crop), we want grid values to be within [-0.5, 0.5] (for example).
            # So we multiply output grid coordinates (initially [-1, 1]) by a factor < 1.
            # So sx, sy should be < 1.
            
            # crop_size_in_input_space = self.cut_size / scale ?? 
            # No, size is the size in input pixels.
            # ratio = size / min(sideX, sideY).
            
            ratio = size / float(min(sideX, sideY))
            # ratio is e.g. 0.5 (half image).
            # We want grid range to be [-ratio, ratio] + offset.
            
            s = ratio
            
            # offset
            # max offset is 1 - s
            # random in [-(1-s), 1-s]
            max_offset = 1.0 - s
            tx = (np.random.rand() * 2 - 1) * max_offset
            ty = (np.random.rand() * 2 - 1) * max_offset
            
            # Matrix:
            # x_in = s * x_out + tx
            # y_in = s * y_out + ty
            
            theta = np.array([
                [s, 0, tx],
                [0, s, ty]
            ])
            thetas.append(theta)
            
        thetas = np.stack(thetas)
        
        # Generate grids
        grid_shape = (self.num_cutouts, 3, self.cut_size, self.cut_size)
        grids_np = affine_grid(thetas, grid_shape) # [N, H, W, 2]
        
        # Convert to tensor
        grid_tensor = te.Tensor(grids_np, requires_grad=False)
        
        # Expand input to batch?
        # GridSample supports broadcasting input [1, C, H, W] against grid [N, H_out, W_out, 2]? 
        # Usually yes in PyTorch. Our implementation?
        # Let's verify our GridSample impl. 
        # But if not, we can repeat input.
        # Let's assume broadcasting works effectively or repeat.
        # Actually GridSample implementation likely expects matching Batch or B=1 broadcast.
        # Let's use B=1 input and B=N grid.
        
        cuts = te.grid_sample(input_tensor, grid_tensor, align_corners=False, padding_mode="border")
        
        return cuts

def main():
    target_text = "a painting of a starry night"
    image_path = "input.jpg"
    out_dir = "dream_frames"
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize CLIP
    print("Initializing CLIP...")
    # Config for ViT-B/32
    clip = te.CLIP(
        embed_dim=512,
        image_size=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        vision_heads=12,
        context_length=77,
        vocab_size=49408,
        text_width=512,
        text_heads=8,
        text_layers=12
    )
    
    # In a real script we would load weights here.
    # clip.load_state_dict(...)
    print("CLIP initialized (random weights).")
    
    # Load Image
    img = load_image(image_path, size=224)
    
    # Text embedding
    # We need a tokenizer. For now use dummy tokens or random.
    # We exposed Tokenizer in lib.rs? Yes PyTokenizer.
    # But usually we need BPE files.
    # Let's just create a random text tensor for demonstration of the loop.
    print("Encoding text...")
    text_input = te.Tensor(np.random.randint(0, 49408, (1, 77)).astype(np.float32), requires_grad=False)
    
    # Forward pass to get target features
    # Note: Using random weights, this is meaningless, but tests the graph.
    _, target_features = clip.forward(img, text_input)
    target_features = target_features.detach() # Don't optimize text features if constant
    
    # Optimizer
    # We optimize the image.
    lr = 0.05
    optimizer = te.Adam(lr, 0.9, 0.999, 1e-8)
    
    cutouts = MakeCutouts(224, 4)
    
    print("Starting optimization...")
    start_time = time.time()
    
    for i in range(10):
        optimizer.zero_grad([img])
        
        # Cutouts
        # cuts = cutouts(img)
        # Note: GridSample might require gradients? Yes we want gradients to flow back to img.
        # But for now let's just optimize the full image to match text features directly 
        # to simplify if cutouts are tricky without weights.
        
        # Simple mode: optimize full image
        img_features, _ = clip.forward(img, text_input)
        
        # Loss: Cosine distance
        # 1 - cosine_similarity(img, text)
        # Features are normalized in CLIP forward.
        # dot product is enough.
        
        # loss = 1 - (img * target).sum()
        # But wait, target_features came from 'text_input'.
        
        similarity = img_features.mul(target_features).sum() # dot product
        # Minimize -similarity
        loss = similarity.mul(te.Tensor.from_scalar(-1.0))
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.step([img])
        
        # Clamp
        # img.data().clamp_(0, 1)? 
        # We don't have clamp_ exposed easily on tensor?
        # We can just let it drift or implement clamp op.
        # Or numpy clamp.
        
        arr = img.numpy()
        arr = np.clip(arr, 0, 1)
        # Assign back?
        # te.Tensor doesn't have assign_from_numpy easily exposed?
        # We can implement a simple clamp op or just ignore for simple test.
        
        print(f"Iter {i}: Loss {loss.numpy()}")
        
        if i % 10 == 0:
            save_image(img, os.path.join(out_dir, f"frame_{i:03d}.png"))
            
    print(f"Finished in {time.time() - start_time:.2f}s")
    print(f"Saved frames to {out_dir}")

if __name__ == "__main__":
    main()
