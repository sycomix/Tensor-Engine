use std::iter;
use std::time::{Duration, Instant};

/// Controls how the iterator handles the final partial window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalWindowPolicy {
    /// Skip the last window if it is shorter than `window_size`.
    Drop,
    /// Yield the final short window without padding.
    KeepShort,
    /// Yield the final short window and logically pad with `pad_id`.
    ///
    /// Padding is logical (exposed by iterators), so the underlying token slice
    /// is still borrowed and no padded buffer is allocated.
    Pad { pad_id: u32 },
}

/// Basic configuration/argument errors for window creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowError {
    EmptyInput,
    ZeroWindowSize,
    ZeroStride,
    InvalidShard,
}

/// Context passed to the dynamic-stride hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrideContext {
    pub window_index: usize,
    pub start: usize,
    pub end: usize,
    pub current_stride: usize,
}

/// Optional sharding configuration for multi-GPU/distributed training.
///
/// Windows are assigned by global window index: `index % world_size == rank`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchShard {
    pub rank: usize,
    pub world_size: usize,
}

impl BatchShard {
    pub fn new(rank: usize, world_size: usize) -> Result<Self, WindowError> {
        if world_size == 0 || rank >= world_size {
            return Err(WindowError::InvalidShard);
        }
        Ok(Self { rank, world_size })
    }
}

/// A single training window backed by a borrowed token slice.
///
/// This struct avoids per-window allocation and supports optional logical padding.
#[derive(Debug, Clone, Copy)]
pub struct TokenWindow<'a> {
    tokens: &'a [u32],
    window_size: usize,
    pad_id: Option<u32>,
    start: usize,
}

impl<'a> TokenWindow<'a> {
    /// Borrowed token slice (never cloned).
    pub fn tokens(&self) -> &'a [u32] {
        self.tokens
    }

    /// Start offset in the original token sequence.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Number of real (non-padding) tokens in this window.
    pub fn real_len(&self) -> usize {
        self.tokens.len()
    }

    /// Logical window length used by training (`window_size`).
    pub fn target_len(&self) -> usize {
        self.window_size
    }

    /// Number of padding tokens needed to reach `target_len()`.
    pub fn pad_len(&self) -> usize {
        self.window_size.saturating_sub(self.tokens.len())
    }

    /// True if this window includes logical padding.
    pub fn is_padded(&self) -> bool {
        self.pad_id.is_some() && self.pad_len() > 0
    }

    /// Iterate tokens as a logically padded stream without allocating a new vector.
    pub fn iter_padded(&self) -> impl Iterator<Item = u32> + 'a {
        let pad_count = if self.pad_id.is_some() {
            self.pad_len()
        } else {
            0
        };
        let pad_value = self.pad_id.unwrap_or(0);

        self.tokens
            .iter()
            .copied()
            .chain(iter::repeat(pad_value).take(pad_count))
    }

    /// Optional masking hook for augmentation/pretraining objectives.
    ///
    /// `mask_fn` receives `(absolute_token_index, token_id)` and returns the mapped token.
    pub fn iter_padded_masked<'b, F>(&'b self, mut mask_fn: F) -> impl Iterator<Item = u32> + 'b
    where
        F: FnMut(usize, u32) -> u32 + 'b,
    {
        self.iter_padded()
            .enumerate()
            .map(move |(offset, token)| mask_fn(self.start + offset, token))
    }
}

/// Lazy sliding-window iterator over a token slice.
///
/// Designed for large-scale training loops where allocating all windows at once is prohibitive.
pub struct SlidingWindowIter<'a> {
    token_ids: &'a [u32],
    window_size: usize,
    stride: usize,
    policy: FinalWindowPolicy,
    position: usize,
    finished: bool,
    window_index: usize,
    dynamic_stride_hook: Option<Box<dyn FnMut(StrideContext) -> usize + 'a>>,
    shard: Option<BatchShard>,
}

impl<'a> std::fmt::Debug for SlidingWindowIter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlidingWindowIter")
            .field("token_len", &self.token_ids.len())
            .field("window_size", &self.window_size)
            .field("stride", &self.stride)
            .field("policy", &self.policy)
            .field("position", &self.position)
            .field("finished", &self.finished)
            .field("window_index", &self.window_index)
            .field(
                "has_dynamic_stride_hook",
                &self.dynamic_stride_hook.is_some(),
            )
            .field("shard", &self.shard)
            .finish()
    }
}

impl<'a> SlidingWindowIter<'a> {
    /// Optional hook to adjust stride dynamically between windows.
    ///
    /// Return `0` from the hook to keep the current stride.
    pub fn with_dynamic_stride_hook<F>(mut self, hook: F) -> Self
    where
        F: FnMut(StrideContext) -> usize + 'a,
    {
        self.dynamic_stride_hook = Some(Box::new(hook));
        self
    }

    /// Optional multi-GPU/distributed sharding.
    pub fn with_shard(mut self, shard: BatchShard) -> Self {
        self.shard = Some(shard);
        self
    }

    /// Advance by up to `n` windows without yielding items.
    ///
    /// Returns the number of windows actually skipped (may be smaller at EOF).
    pub fn skip_windows(&mut self, n: usize) -> usize {
        let mut skipped = 0;
        while skipped < n {
            if self.next().is_some() {
                skipped += 1;
            } else {
                break;
            }
        }
        skipped
    }

    fn should_emit_for_shard(&self, index: usize) -> bool {
        match self.shard {
            Some(shard) => index % shard.world_size == shard.rank,
            None => true,
        }
    }
}

impl<'a> Iterator for SlidingWindowIter<'a> {
    type Item = TokenWindow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.finished && self.position < self.token_ids.len() {
            let start = self.position;
            let end = (start + self.window_size).min(self.token_ids.len());
            let slice = &self.token_ids[start..end];
            let is_short = slice.len() < self.window_size;

            let maybe_window = if is_short {
                self.finished = true;
                match self.policy {
                    FinalWindowPolicy::Drop => None,
                    FinalWindowPolicy::KeepShort => Some(TokenWindow {
                        tokens: slice,
                        window_size: self.window_size,
                        pad_id: None,
                        start,
                    }),
                    FinalWindowPolicy::Pad { pad_id } => Some(TokenWindow {
                        tokens: slice,
                        window_size: self.window_size,
                        pad_id: Some(pad_id),
                        start,
                    }),
                }
            } else {
                let window = TokenWindow {
                    tokens: slice,
                    window_size: self.window_size,
                    pad_id: None,
                    start,
                };

                let mut next_stride = self.stride;
                if let Some(hook) = self.dynamic_stride_hook.as_mut() {
                    let suggested = hook(StrideContext {
                        window_index: self.window_index,
                        start,
                        end,
                        current_stride: self.stride,
                    });
                    if suggested > 0 {
                        next_stride = suggested;
                    }
                }

                if end == self.token_ids.len() {
                    self.finished = true;
                } else {
                    self.position += next_stride;
                }

                Some(window)
            };

            let current_index = self.window_index;
            self.window_index += 1;

            if let Some(window) = maybe_window {
                if self.should_emit_for_shard(current_index) {
                    return Some(window);
                }
            }
        }

        None
    }
}

/// Create a lazy iterator over overlapping windows.
///
/// - Returns borrowed slices to minimize memory usage.
/// - Supports dropping, keeping, or logically padding the final short window.
/// - Validates common configuration mistakes for long-running training jobs.
pub fn overlapping_windows<'a>(
    token_ids: &'a [u32],
    window_size: usize,
    stride: usize,
    policy: FinalWindowPolicy,
) -> Result<SlidingWindowIter<'a>, WindowError> {
    if token_ids.is_empty() {
        return Err(WindowError::EmptyInput);
    }
    if window_size == 0 {
        return Err(WindowError::ZeroWindowSize);
    }
    if stride == 0 {
        return Err(WindowError::ZeroStride);
    }

    Ok(SlidingWindowIter {
        token_ids,
        window_size,
        stride,
        policy,
        position: 0,
        finished: false,
        window_index: 0,
        dynamic_stride_hook: None,
        shard: None,
    })
}

/// Eager baseline for comparison/debugging: materializes all windows.
pub fn overlapping_windows_eager(
    token_ids: &[u32],
    window_size: usize,
    stride: usize,
    policy: FinalWindowPolicy,
) -> Result<Vec<Vec<u32>>, WindowError> {
    let windows = overlapping_windows(token_ids, window_size, stride, policy)?
        .map(|w| w.iter_padded().collect::<Vec<u32>>())
        .collect::<Vec<Vec<u32>>>();
    Ok(windows)
}

/// Simple benchmark summary for lazy vs eager window generation.
#[derive(Debug, Clone, Copy)]
pub struct WindowBenchmark {
    pub lazy_duration: Duration,
    pub eager_duration: Duration,
    pub lazy_checksum: u64,
    pub eager_checksum: u64,
}

/// Benchmark lazy zero-copy iteration against eager `Vec<Vec<u32>>` materialization.
///
/// `repeats` should be > 0 for stable timing.
pub fn benchmark_windows(
    token_ids: &[u32],
    window_size: usize,
    stride: usize,
    policy: FinalWindowPolicy,
    repeats: usize,
) -> Result<WindowBenchmark, WindowError> {
    if repeats == 0 {
        return Ok(WindowBenchmark {
            lazy_duration: Duration::ZERO,
            eager_duration: Duration::ZERO,
            lazy_checksum: 0,
            eager_checksum: 0,
        });
    }

    let mut lazy_checksum = 0u64;
    let mut eager_checksum = 0u64;

    let lazy_start = Instant::now();
    for _ in 0..repeats {
        let iter = overlapping_windows(token_ids, window_size, stride, policy)?;
        for window in iter {
            for token in window.iter_padded() {
                lazy_checksum = lazy_checksum.wrapping_add(token as u64);
            }
        }
    }
    let lazy_duration = lazy_start.elapsed();

    let eager_start = Instant::now();
    for _ in 0..repeats {
        let windows = overlapping_windows_eager(token_ids, window_size, stride, policy)?;
        for window in windows {
            for token in window {
                eager_checksum = eager_checksum.wrapping_add(token as u64);
            }
        }
    }
    let eager_duration = eager_start.elapsed();

    Ok(WindowBenchmark {
        lazy_duration,
        eager_duration,
        lazy_checksum,
        eager_checksum,
    })
}
