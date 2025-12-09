# Resampling in WavDataLoader

This document explains resampling choices available in `WavDataLoader` and how to enable high-quality resampling.

Options:
- `resample` flag in `WavDataLoader::new(dir, sample_rate, chunk_len, batch_size, resample)`: When true, WAV files with mismatched sample rates will be resampled to the requested sample rate.
- By default, `resample` uses a simple linear resampler implemented in `src/io/dataloader.rs::resample_linear` as a fast fallback.
- When the library is compiled with the `audio` feature, and the `rubato` feature is also enabled (e.g. `--features "audio openblas"`), `WavDataLoader` will use `rubato`-based high-quality resampling via `resample_high_quality`. This resampler implements band-limited sinc interpolation for good audio quality.

How to enable:
- Build with `audio` feature: `cargo build --features audio`.
- For high-quality (rubato), use: `cargo build --features "audio"` (rubato is included under `audio` feature as an optional dep).

Notes & recommendations:
- High-quality resampling results in better audio fidelity but increases CPU cost — set `resample=true` only when needed.
- For large training datasets, preprocess resampling offline or use a streaming resampling strategy in the data pipeline.

See `src/io/dataloader.rs` for implementation details and `examples/train_codec.rs` for the training example using resampling.
