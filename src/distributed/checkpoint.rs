//! Distributed Checkpointing
//!
//! This module provides robust checkpoint save/load functionality for
//! distributed training, including atomic writes and partial recovery.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::context::DistributedContext;

/// Configuration for distributed checkpointing.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Base directory for checkpoint storage.
    pub checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep.
    pub max_checkpoints: usize,
    /// Save optimizer state along with model.
    pub save_optimizer: bool,
    /// Use atomic writes (write to temp, then rename).
    pub atomic_writes: bool,
    /// Only rank 0 saves (data parallel scenario).
    pub master_only: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("./checkpoints"),
            max_checkpoints: 5,
            save_optimizer: true,
            atomic_writes: true,
            master_only: true,
        }
    }
}

impl CheckpointConfig {
    /// Create configuration with a specific directory.
    pub fn with_dir<P: Into<PathBuf>>(dir: P) -> Self {
        Self {
            checkpoint_dir: dir.into(),
            ..Default::default()
        }
    }

    /// Set maximum checkpoints to keep.
    pub fn with_max_checkpoints(mut self, max: usize) -> Self {
        self.max_checkpoints = max;
        self
    }

    /// Enable/disable optimizer saving.
    pub fn save_optimizer(mut self, save: bool) -> Self {
        self.save_optimizer = save;
        self
    }

    /// Enable/disable atomic writes.
    pub fn atomic(mut self, atomic: bool) -> Self {
        self.atomic_writes = atomic;
        self
    }

    /// Set whether only master rank saves.
    pub fn master_only(mut self, master_only: bool) -> Self {
        self.master_only = master_only;
        self
    }
}

/// Metadata for a saved checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Checkpoint filename.
    pub filename: String,
    /// Training step/epoch when saved.
    pub step: usize,
    /// Epoch number.
    pub epoch: usize,
    /// Timestamp when saved.
    pub timestamp: u64,
    /// Loss value at checkpoint.
    pub loss: Option<f32>,
    /// Additional metrics.
    pub metrics: HashMap<String, f32>,
}

impl CheckpointMetadata {
    /// Create new metadata for a checkpoint.
    pub fn new(step: usize, epoch: usize) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            filename: format!("checkpoint_step{}_epoch{}.bin", step, epoch),
            step,
            epoch,
            timestamp,
            loss: None,
            metrics: HashMap::new(),
        }
    }

    /// Add loss to metadata.
    pub fn with_loss(mut self, loss: f32) -> Self {
        self.loss = Some(loss);
        self
    }

    /// Add a metric to metadata.
    pub fn with_metric(mut self, name: &str, value: f32) -> Self {
        self.metrics.insert(name.to_string(), value);
        self
    }
}

/// Serializable state for checkpointing.
pub trait CheckpointState {
    /// Serialize state to bytes.
    fn to_bytes(&self) -> Vec<u8>;

    /// Deserialize state from bytes.
    fn from_bytes(bytes: &[u8]) -> Result<Self, String>
    where
        Self: Sized;
}

/// Simple key-value checkpoint state.
#[derive(Debug, Clone, Default)]
pub struct SimpleState {
    data: HashMap<String, Vec<u8>>,
}

impl SimpleState {
    /// Create a new empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add bytes to the state.
    pub fn put(&mut self, key: &str, value: Vec<u8>) {
        self.data.insert(key.to_string(), value);
    }

    /// Add f32 slice as bytes.
    pub fn put_f32_slice(&mut self, key: &str, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.put(key, bytes);
    }

    /// Get bytes from the state.
    pub fn get(&self, key: &str) -> Option<&Vec<u8>> {
        self.data.get(key)
    }

    /// Get f32 slice from bytes.
    pub fn get_f32_slice(&self, key: &str) -> Option<Vec<f32>> {
        self.get(key).map(|bytes| {
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        })
    }

    /// Check if key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Get all keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }
}

impl CheckpointState for SimpleState {
    fn to_bytes(&self) -> Vec<u8> {
        // Simple format: [num_entries: u32][key_len: u32][key][value_len: u32][value]...
        let mut bytes = Vec::new();

        // Number of entries
        let num_entries = self.data.len() as u32;
        bytes.extend(num_entries.to_le_bytes());

        for (key, value) in &self.data {
            // Key length and key
            let key_bytes = key.as_bytes();
            bytes.extend((key_bytes.len() as u32).to_le_bytes());
            bytes.extend(key_bytes);

            // Value length and value
            bytes.extend((value.len() as u32).to_le_bytes());
            bytes.extend(value);
        }

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 4 {
            return Err("Checkpoint too small".to_string());
        }

        let mut state = SimpleState::new();
        let mut offset = 0;

        // Read number of entries
        let num_entries = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        for _ in 0..num_entries {
            // Read key length
            if offset + 4 > bytes.len() {
                return Err("Truncated checkpoint: key length".to_string());
            }
            let key_len = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            // Read key
            if offset + key_len > bytes.len() {
                return Err("Truncated checkpoint: key".to_string());
            }
            let key = String::from_utf8_lossy(&bytes[offset..offset + key_len]).to_string();
            offset += key_len;

            // Read value length
            if offset + 4 > bytes.len() {
                return Err("Truncated checkpoint: value length".to_string());
            }
            let value_len = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            // Read value
            if offset + value_len > bytes.len() {
                return Err("Truncated checkpoint: value".to_string());
            }
            let value = bytes[offset..offset + value_len].to_vec();
            offset += value_len;

            state.put(&key, value);
        }

        Ok(state)
    }
}

/// Distributed checkpoint manager.
pub struct DistributedCheckpoint {
    ctx: DistributedContext,
    config: CheckpointConfig,
}

impl DistributedCheckpoint {
    /// Create a new checkpoint manager.
    pub fn new(ctx: DistributedContext, config: CheckpointConfig) -> Self {
        Self { ctx, config }
    }

    /// Save a checkpoint with the given state.
    pub fn save(
        &self,
        state: &SimpleState,
        metadata: &CheckpointMetadata,
    ) -> Result<PathBuf, String> {
        // Only master saves if configured
        if self.config.master_only && !self.ctx.is_master() {
            log::debug!("Rank {}: skipping save (master only)", self.ctx.rank());
            return Ok(PathBuf::new());
        }

        // Ensure directory exists
        fs::create_dir_all(&self.config.checkpoint_dir)
            .map_err(|e| format!("Failed to create checkpoint dir: {}", e))?;

        let checkpoint_path = self.config.checkpoint_dir.join(&metadata.filename);

        log::info!(
            "Saving checkpoint to {} (step={}, epoch={})",
            checkpoint_path.display(),
            metadata.step,
            metadata.epoch
        );

        // Serialize state
        let bytes = state.to_bytes();

        if self.config.atomic_writes {
            // Write to temp file first
            let temp_path = checkpoint_path.with_extension("tmp");
            self.write_bytes(&temp_path, &bytes)?;

            // Atomic rename
            fs::rename(&temp_path, &checkpoint_path)
                .map_err(|e| format!("Failed to rename checkpoint: {}", e))?;
        } else {
            self.write_bytes(&checkpoint_path, &bytes)?;
        }

        // Cleanup old checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(checkpoint_path)
    }

    /// Load a checkpoint from file.
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<SimpleState, String> {
        let path = path.as_ref();
        log::info!("Loading checkpoint from {}", path.display());

        let bytes = self.read_bytes(path)?;
        SimpleState::from_bytes(&bytes)
    }

    /// Load the latest checkpoint from the checkpoint directory.
    pub fn load_latest(&self) -> Result<Option<(SimpleState, CheckpointMetadata)>, String> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.is_empty() {
            return Ok(None);
        }

        // Get the latest by step number
        let latest = checkpoints
            .into_iter()
            .max_by_key(|m| (m.epoch, m.step))
            .expect("Non-empty vec");

        let path = self.config.checkpoint_dir.join(&latest.filename);
        let state = self.load(&path)?;

        Ok(Some((state, latest)))
    }

    /// List available checkpoints.
    pub fn list_checkpoints(&self) -> Result<Vec<CheckpointMetadata>, String> {
        if !self.config.checkpoint_dir.exists() {
            return Ok(Vec::new());
        }

        let entries = fs::read_dir(&self.config.checkpoint_dir)
            .map_err(|e| format!("Failed to read checkpoint dir: {}", e))?;

        let mut checkpoints = Vec::new();

        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "bin" {
                    if let Some(meta) = self.parse_checkpoint_filename(&path) {
                        checkpoints.push(meta);
                    }
                }
            }
        }

        // Sort by step
        checkpoints.sort_by_key(|m| (m.epoch, m.step));

        Ok(checkpoints)
    }

    /// Barrier to synchronize all ranks at checkpoint.
    pub fn barrier(&self) {
        self.ctx.barrier();
    }

    /// Write bytes to file.
    fn write_bytes(&self, path: &Path, bytes: &[u8]) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(bytes)
            .map_err(|e| format!("Failed to write checkpoint: {}", e))?;
        writer
            .flush()
            .map_err(|e| format!("Failed to flush checkpoint: {}", e))?;
        Ok(())
    }

    /// Read bytes from file.
    fn read_bytes(&self, path: &Path) -> Result<Vec<u8>, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .map_err(|e| format!("Failed to read checkpoint: {}", e))?;
        Ok(bytes)
    }

    /// Parse checkpoint metadata from filename.
    fn parse_checkpoint_filename(&self, path: &Path) -> Option<CheckpointMetadata> {
        let filename = path.file_name()?.to_str()?;

        // Expected format: checkpoint_step{N}_epoch{M}.bin
        let parts: Vec<&str> = filename
            .strip_prefix("checkpoint_step")?
            .strip_suffix(".bin")?
            .split("_epoch")
            .collect();

        if parts.len() != 2 {
            return None;
        }

        let step: usize = parts[0].parse().ok()?;
        let epoch: usize = parts[1].parse().ok()?;

        let metadata = path.metadata().ok()?;
        let timestamp = metadata
            .modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_secs();

        Some(CheckpointMetadata {
            filename: filename.to_string(),
            step,
            epoch,
            timestamp,
            loss: None,
            metrics: HashMap::new(),
        })
    }

    /// Remove old checkpoints beyond max_checkpoints.
    fn cleanup_old_checkpoints(&self) -> Result<(), String> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.len() <= self.config.max_checkpoints {
            return Ok(());
        }

        let to_remove = checkpoints.len() - self.config.max_checkpoints;
        log::info!("Cleaning up {} old checkpoints", to_remove);

        for meta in checkpoints.into_iter().take(to_remove) {
            let path = self.config.checkpoint_dir.join(&meta.filename);
            if let Err(e) = fs::remove_file(&path) {
                log::warn!("Failed to remove old checkpoint {}: {}", path.display(), e);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn temp_checkpoint_dir() -> PathBuf {
        static CNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let cnt = CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut dir = env::temp_dir();
        dir.push(format!(
            "tensor_engine_test_checkpoints_{}_{}",
            std::process::id(),
            cnt
        ));
        dir
    }

    fn cleanup_temp_dir(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_simple_state_roundtrip() {
        let mut state = SimpleState::new();
        state.put("model_weights", vec![1, 2, 3, 4]);
        state.put_f32_slice("learning_rate", &[0.001]);

        let bytes = state.to_bytes();
        let restored = SimpleState::from_bytes(&bytes).expect("Valid state");

        assert_eq!(restored.get("model_weights"), Some(&vec![1, 2, 3, 4]));
        let lr = restored.get_f32_slice("learning_rate").expect("Has LR");
        assert!((lr[0] - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = temp_checkpoint_dir();
        let ctx = DistributedContext::single();
        let config = CheckpointConfig::with_dir(&dir);
        let ckpt = DistributedCheckpoint::new(ctx, config);

        // Create state
        let mut state = SimpleState::new();
        state.put_f32_slice("weights", &[1.0, 2.0, 3.0]);

        // Save
        let meta = CheckpointMetadata::new(100, 5);
        let path = ckpt.save(&state, &meta).expect("Save works");
        assert!(path.exists());

        // Load
        let loaded = ckpt.load(&path).expect("Load works");
        let weights = loaded.get_f32_slice("weights").expect("Has weights");
        assert_eq!(weights, vec![1.0, 2.0, 3.0]);

        cleanup_temp_dir(&dir);
    }

    #[test]
    fn test_load_latest() {
        let dir = temp_checkpoint_dir();
        let ctx = DistributedContext::single();
        let config = CheckpointConfig::with_dir(&dir).with_max_checkpoints(10);
        let ckpt = DistributedCheckpoint::new(ctx, config);

        // Save multiple checkpoints
        for i in 0..3 {
            let mut state = SimpleState::new();
            state.put_f32_slice("step", &[i as f32]);

            let meta = CheckpointMetadata::new(i * 100, i);
            ckpt.save(&state, &meta).expect("Save works");
        }

        // Load latest
        let (state, meta) = ckpt.load_latest().expect("Works").expect("Has checkpoints");
        assert_eq!(meta.step, 200);
        assert_eq!(meta.epoch, 2);

        cleanup_temp_dir(&dir);
    }

    #[test]
    fn test_checkpoint_cleanup() {
        let dir = temp_checkpoint_dir();
        let ctx = DistributedContext::single();
        let config = CheckpointConfig::with_dir(&dir).with_max_checkpoints(2);
        let ckpt = DistributedCheckpoint::new(ctx, config);

        // Save 4 checkpoints
        for i in 0..4 {
            let state = SimpleState::new();
            let meta = CheckpointMetadata::new(i * 100, i);
            ckpt.save(&state, &meta).expect("Save works");
        }

        // Should only have 2 checkpoints
        let checkpoints = ckpt.list_checkpoints().expect("List works");
        assert_eq!(checkpoints.len(), 2);

        // Should be the latest ones
        assert_eq!(checkpoints[0].step, 200);
        assert_eq!(checkpoints[1].step, 300);

        cleanup_temp_dir(&dir);
    }
}
