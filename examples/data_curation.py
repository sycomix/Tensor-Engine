import numpy as np
import os
import sys
from PIL import Image

from tensor_engine import Tensor


def make_color_tensor(r, g, b, shape=(224, 224, 1)):
    r_tsr = Tensor(np.full(shape, r, dtype=np.float32), requires_grad=False)
    g_tsr = Tensor(np.full(shape, g, dtype=np.float32), requires_grad=False)
    b_tsr = Tensor(np.full(shape, b, dtype=np.float32), requires_grad=False)
    return Tensor.cat([r_tsr, g_tsr, b_tsr], 2)


def create_dataset_curation_pipeline():
    """
    Creates a real on-disk dataset for ImageTextDataLoader to consume.
    Procedurally generates geometrically accurate images with exact semantic descriptions
    leveraging the native tensor_engine API for all logical masking and arithmetic coloring.
    """
    print("🚀 Initializing Multimodal Dataset Curation Pipeline using tensor_engine logic...")
    base_dir = "data/multimodal_phase2"
    img_dir = os.path.join(base_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest_path = os.path.join(base_dir, "manifest.tsv")

    # Grid for masking logic
    y, x = np.ogrid[0:224, 0:224]

    data_pairs = [
        {"text": "A large red circle centered on a black background.", "bg": (0, 0, 0), "color": (255, 0, 0),
         "mask_np": ((x - 112) ** 2 + (y - 112) ** 2 <= 50 ** 2)},
        {"text": "A blue square in the top left corner of a white background.", "bg": (255, 255, 255),
         "color": (0, 0, 255), "mask_np": ((x > 10) & (x < 100) & (y > 10) & (y < 100))},
        {"text": "A solid green background with no shapes.", "bg": (0, 255, 0), "color": (0, 0, 0),
         "mask_np": np.zeros((224, 224), dtype=bool)},
        {"text": "A yellow rectangle stretching across the bottom of a dark grey background.", "bg": (50, 50, 50),
         "color": (255, 255, 0), "mask_np": (y > 150)},
        {"text": "A white circle on a bright red background.", "bg": (255, 0, 0), "color": (255, 255, 255),
         "mask_np": ((x - 112) ** 2 + (y - 112) ** 2 <= 50 ** 2)},
        {"text": "A magenta diamond centered on a black background.", "bg": (0, 0, 0), "color": (255, 0, 255),
         "mask_np": (np.abs(x - 112) + np.abs(y - 112) <= 62)},
        {"text": "A cyan square in the bottom right corner of a white background.", "bg": (255, 255, 255),
         "color": (0, 255, 255), "mask_np": ((x > 124) & (x < 214) & (y > 124) & (y < 214))},
        {"text": "A completely black image.", "bg": (0, 0, 0), "color": (0, 0, 0),
         "mask_np": np.zeros((224, 224), dtype=bool)}
    ]

    with open(manifest_path, 'w') as f:
        for i, item in enumerate(data_pairs):
            img_filepath = os.path.join(img_dir, f"geometric_{i:04d}.jpg")

            # Use tensor engine arithmetic to literally construct the image
            # Create masks
            mask_tsr_raw = Tensor(item["mask_np"].astype(np.float32)[:, :, np.newaxis], requires_grad=False)
            one_tsr = Tensor(np.ones((224, 224, 1), dtype=np.float32), requires_grad=False)
            inv_mask_tsr = one_tsr.sub(mask_tsr_raw)

            # Broadcast mask onto 3 channels for multiplication
            mask_3c = Tensor.cat([mask_tsr_raw, mask_tsr_raw, mask_tsr_raw], 2)
            inv_mask_3c = Tensor.cat([inv_mask_tsr, inv_mask_tsr, inv_mask_tsr], 2)

            # Colors from tensor logic
            fg = make_color_tensor(*item["color"])
            bg = make_color_tensor(*item["bg"])

            # final = (FG * mask) + (BG * (1-mask))
            fg_masked = fg.mul(mask_3c)
            bg_masked = bg.mul(inv_mask_3c)
            final_img_tsr = fg_masked.add(bg_masked)

            # Save final tensor output using base format library
            img_array = final_img_tsr.numpy().astype(np.uint8)
            img = Image.fromarray(img_array)
            img.save(img_filepath)

            # Write to manifest (TSV format required by ImageTextDataLoader: <path>\t<caption>)
            f.write(f"{img_filepath}\t{item['text']}\n")

    print(f"✅ Generated {len(data_pairs)} tensor-processed geometrically precise image-text pairs")
    print(f"✅ Manifest written to '{manifest_path}'")
    return manifest_path


if __name__ == "__main__":
    create_dataset_curation_pipeline()
