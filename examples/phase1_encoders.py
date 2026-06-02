import numpy as np
import torch
from typing import List

import tensor_engine as te


class TextDataPipeline:
    def __init__(self, vocab_size: int = 50257, max_len: int = 128):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def process(self, batch_texts: List[str]) -> te.Tensor:
        batch_size = len(batch_texts)
        one_hot = np.zeros((batch_size, self.max_len, self.vocab_size), dtype=np.float32)

        for i, text in enumerate(batch_texts):
            words = text.split()[:self.max_len]
            for j, word in enumerate(words):
                token_id = abs(hash(word)) % self.vocab_size
                one_hot[i, j, token_id] = 1.0

        return te.Tensor(one_hot, requires_grad=False)


class ImageDataPipeline:
    def __init__(self, height: int = 224, width: int = 224, channels: int = 3):
        self.h = height
        self.w = width
        self.c = channels

    def process(self, batch_images_np: List[np.ndarray]) -> te.Tensor:
        batch_size = len(batch_images_np)
        # Using [B, C, D=1, H, W] to utilize Conv3D as Conv2D
        batch = np.zeros((batch_size, self.c, 1, self.h, self.w), dtype=np.float32)
        for i, img in enumerate(batch_images_np):
            img_normalized = img.astype(np.float32) / 255.0
            img_c_first = np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), axis=1)
            batch[i] = img_c_first

        return te.Tensor(batch, requires_grad=False)


class AudioDataPipeline:
    def __init__(self, sample_rate: int = 16000, duration_sec: float = 1.0):
        self.sr = sample_rate
        self.length = int(sample_rate * duration_sec)

    def process(self, batch_waveforms: List[np.ndarray]) -> te.Tensor:
        batch_size = len(batch_waveforms)
        # Using [B, C=1, D=1, H=1, W=L] to utilize Conv3D as Conv1D
        batch = np.zeros((batch_size, 1, 1, 1, self.length), dtype=np.float32)
        for i, wave in enumerate(batch_waveforms):
            truncated_len = min(self.length, wave.shape[0])
            batch[i, 0, 0, 0, :truncated_len] = wave[:truncated_len]

        return te.Tensor(batch, requires_grad=False)


class VideoDataPipeline:
    def __init__(self, frames: int = 16, height: int = 112, width: int = 112, channels: int = 3):
        self.f = frames
        self.h = height
        self.w = width
        self.c = channels

    def process(self, batch_videos: List[np.ndarray]) -> te.Tensor:
        batch_size = len(batch_videos)
        batch = np.zeros((batch_size, self.c, self.f, self.h, self.w), dtype=np.float32)
        for i, vid in enumerate(batch_videos):
            vid_normalized = vid.astype(np.float32) / 255.0
            vid_c_first = np.transpose(vid_normalized, (3, 0, 1, 2))
            batch[i] = vid_c_first

        return te.Tensor(batch, requires_grad=False)


class TextEncoder:
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, max_seq_len: int, depth: int):
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = te.Linear(vocab_size, d_model, bias=False)
        self.pos_emb = te.Tensor(np.random.normal(0, 0.02, (1, max_seq_len, d_model)).astype(np.float32),
                                 requires_grad=True)

        self.blocks = []
        for _ in range(depth):
            self.blocks.append(te.TransformerBlock(d_model, d_model * 4, num_heads))

    def forward(self, one_hot_x: te.Tensor) -> te.Tensor:
        x = self.embedding.forward(one_hot_x)
        x = x.add(self.pos_emb)

        for block in self.blocks:
            x = block.forward(x)

        return x

    def parameters(self) -> List[te.Tensor]:
        params = self.embedding.parameters() + [self.pos_emb]
        for b in self.blocks:
            params.extend(b.parameters())
        return params


class ImageEncoder:
    def __init__(self, in_channels: int, patch_size: int, d_model: int, num_heads: int, max_patches: int, depth: int):
        # We use Conv3D with depth=1 to simulate Conv2D
        self.patch_conv = te.Conv3D(in_channels, d_model, 1, patch_size, patch_size, patch_size, 0, True)
        self.pos_emb = te.Tensor(np.random.normal(0, 0.02, (1, max_patches, d_model)).astype(np.float32),
                                 requires_grad=True)

        self.blocks = []
        for _ in range(depth):
            self.blocks.append(te.TransformerBlock(d_model, d_model * 4, num_heads))

    def forward(self, img_tsr: te.Tensor) -> te.Tensor:
        x = self.patch_conv.forward(img_tsr)

        arr = x.numpy()
        b, d, d_pr, hp, wp = arr.shape
        seq_len = d_pr * hp * wp

        arr_swapped = np.transpose(arr, (0, 2, 3, 4, 1)).reshape((b, seq_len, d))
        x = te.Tensor(arr_swapped, requires_grad=True)

        x = x.add(self.pos_emb)

        for block in self.blocks:
            x = block.forward(x)

        return x

    def parameters(self) -> List[te.Tensor]:
        params = self.patch_conv.parameters() + [self.pos_emb]
        for b in self.blocks:
            params.extend(b.parameters())
        return params


class AudioEncoder:
    def __init__(self, in_channels: int, d_model: int, num_heads: int, depth: int, max_seq_len: int):
        # We use Conv3D with depth=1, height=1 to simulate 1D Conv over audio sequences
        self.conv1 = te.Conv3D(in_channels, d_model // 2, 1, 1, 16, 4, 0, True)
        self.conv2 = te.Conv3D(d_model // 2, d_model, 1, 1, 16, 4, 0, True)

        self.pos_emb = te.Tensor(np.random.normal(0, 0.02, (1, max_seq_len, d_model)).astype(np.float32),
                                 requires_grad=True)

        self.blocks = []
        for _ in range(depth):
            self.blocks.append(te.TransformerBlock(d_model, d_model * 4, num_heads))

    def forward(self, audio_tsr: te.Tensor) -> te.Tensor:
        x = self.conv1.forward(audio_tsr)
        x = x.relu()
        x = self.conv2.forward(x)
        x = x.relu()

        arr = x.numpy()
        b, d, d_pr, h, w = arr.shape
        seq_len = d_pr * h * w

        arr_swapped = np.transpose(arr, (0, 2, 3, 4, 1)).reshape((b, seq_len, d))
        x = te.Tensor(arr_swapped, requires_grad=True)

        x = x.add(self.pos_emb)

        for block in self.blocks:
            x = block.forward(x)

        return x


class VideoEncoder:
    def __init__(self, in_channels: int, d_model: int, num_heads: int, max_tokens: int, depth: int):
        # 3D CNN Patch embedding
        # Conv3D parameters: in, out, kd, kh, kw, stride, padding, bias
        self.patch_conv3d = te.Conv3D(in_channels, d_model, 2, 16, 16, 16, 0, True)
        self.pos_emb = te.Tensor(np.random.normal(0, 0.02, (1, max_tokens, d_model)).astype(np.float32),
                                 requires_grad=True)

        self.blocks = []
        for _ in range(depth):
            self.blocks.append(te.TransformerBlock(d_model, d_model * 4, num_heads))

    def forward(self, vid_tsr: te.Tensor) -> te.Tensor:
        x = self.patch_conv3d.forward(vid_tsr)

        arr = x.numpy()
        b, d, d_pr, h_pr, w_pr = arr.shape
        seq_len = d_pr * h_pr * w_pr

        arr_swapped = np.transpose(arr, (0, 2, 3, 4, 1)).reshape((b, seq_len, d))
        x = te.Tensor(arr_swapped, requires_grad=True)

        x = x.add(self.pos_emb)

        for block in self.blocks:
            x = block.forward(x)

        return x


def test_multimodal_encoders():
    print("🚀 Initializing Phase 1 Multimodal Encoders Integration Test...")

    batch_size = 2
    d_model = 128
    num_heads = 4
    depth = 2

    print("--------------------------------------------------")
    print("1. Text Encoder Test")
    vocab_size = 50257
    max_len = 32
    text_pipe = TextDataPipeline(vocab_size=vocab_size, max_len=max_len)
    batch_texts = ["A picture of a dog", "The quick brown fox jumps over the lazy dog"]
    text_tensor = text_pipe.process(batch_texts)
    print(f"   [Pipeline] output shape: {text_tensor.numpy().shape}")

    text_enc = TextEncoder(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, max_seq_len=max_len,
                           depth=depth)
    text_out = text_enc.forward(text_tensor)
    print(f"   [Encoder] output shape: {text_out.numpy().shape}")
    assert text_out.numpy().shape == (batch_size, max_len,
                                      d_model), f"Text Encoder shape mismatch! Got {text_out.numpy().shape}"
    print("   ✅ TextEncoder sequence executed successfully.")

    print("--------------------------------------------------")
    print("2. Image Encoder Test")
    img_h, img_w, img_c = 224, 224, 3
    patch_size = 16
    max_patches = (img_h // patch_size) * (img_w // patch_size)

    img_pipe = ImageDataPipeline(height=img_h, width=img_w, channels=img_c)
    batch_imgs = [np.random.randint(0, 256, (img_h, img_w, img_c), dtype=np.uint8) for _ in range(batch_size)]
    img_tensor = img_pipe.process(batch_imgs)
    print(f"   [Pipeline] output shape: {img_tensor.numpy().shape}")

    img_enc = ImageEncoder(in_channels=img_c, patch_size=patch_size, d_model=d_model, num_heads=num_heads,
                           max_patches=max_patches, depth=depth)
    img_out = img_enc.forward(img_tensor)
    print(f"   [Encoder] output shape: {img_out.numpy().shape}")
    assert img_out.numpy().shape == (batch_size, max_patches,
                                     d_model), f"Image Encoder shape mismatch! Got {img_out.numpy().shape}"
    print("   ✅ ImageEncoder sequence executed successfully.")

    print("--------------------------------------------------")
    print("3. Audio Encoder Test")
    # L = 16000
    # Conv3D 1: stride=4, padding=0, kernel_w=16. L1 = (16000 + 0 - 16)/4 + 1 = 3997
    # Conv3D 2: L2 = (3997 + 0 - 16)/4 + 1 = 996
    max_audio_seq = 996

    audio_pipe = AudioDataPipeline(sample_rate=16000, duration_sec=1.0)
    batch_audio = [np.random.randn(16000).astype(np.float32) for _ in range(batch_size)]
    audio_tensor = audio_pipe.process(batch_audio)
    print(f"   [Pipeline] output shape: {audio_tensor.numpy().shape}")

    audio_enc = AudioEncoder(in_channels=1, d_model=d_model, num_heads=num_heads, depth=depth,
                             max_seq_len=max_audio_seq)
    audio_out = audio_enc.forward(audio_tensor)
    print(f"   [Encoder] output shape: {audio_out.numpy().shape}")
    assert audio_out.numpy().shape == (batch_size, max_audio_seq,
                                       d_model), f"Audio Encoder shape mismatch! Got {audio_out.numpy().shape}"
    print("   ✅ AudioEncoder sequence executed successfully.")

    print("--------------------------------------------------")
    print("4. Video Encoder Test")
    vid_f, vid_h, vid_w, vid_c = 16, 112, 112, 3
    # Conv3D: kd=2, kh=16, kw=16, stride=16.
    # D: (16 - 2)/16 + 1 = 14 / 16 + 1 = 1
    # H: (112 - 16)/16 + 1 = 96 / 16 + 1 = 7
    # W: (112 - 16)/16 + 1 = 7
    # Out seq len: 1 * 7 * 7 = 49
    max_vid_seq = 49
    vid_pipe = VideoDataPipeline(frames=vid_f, height=vid_h, width=vid_w, channels=vid_c)
    batch_vid = [np.random.randint(0, 256, (vid_f, vid_h, vid_w, vid_c), dtype=np.uint8) for _ in range(batch_size)]
    vid_tensor = vid_pipe.process(batch_vid)
    print(f"   [Pipeline] output shape: {vid_tensor.numpy().shape}")

    vid_enc = VideoEncoder(in_channels=vid_c, d_model=d_model, num_heads=num_heads, max_tokens=max_vid_seq, depth=depth)
    vid_out = vid_enc.forward(vid_tensor)
    print(f"   [Encoder] output shape: {vid_out.numpy().shape}")
    assert vid_out.numpy().shape == (batch_size, max_vid_seq,
                                     d_model), f"Video Encoder shape mismatch! Got {vid_out.numpy().shape}"
    print("   [SUCCESS] ✅ VideoEncoder sequence executed successfully.")

    print("--------------------------------------------------")
    print("ALL TESTS PASSED: Multimodal encoders and pipelines are fully functional.")


if __name__ == "__main__":
    test_multimodal_encoders()
