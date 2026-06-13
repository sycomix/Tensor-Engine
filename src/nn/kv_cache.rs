use crate::tensor::Tensor;

/// Simple KV cache scaffolding for incremental decoding.
///
/// Two storage modes are supported:
/// - 'vector' mode: vectors of per-token `Tensor` entries (simple, used earlier)
/// - 'packed' mode: single `Tensor` per side (keys/values) with shape (batch, seq, dim)
///
/// When packed storage is present it is the canonical source; vector entries are kept
/// only for legacy compatibility. `seq_len()` always returns the authoritative count.
#[derive(Clone)]
pub struct KVCache {
    // vector-backed single-token entries (legacy / fallback)
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    // packed storage (optional). When present, this is the canonical storage for incremental use.
    packed_keys: Option<Tensor>,
    packed_values: Option<Tensor>,
}

impl KVCache {
    /// Create an empty KV cache
    pub fn new() -> Self {
        KVCache {
            keys: Vec::new(),
            values: Vec::new(),
            packed_keys: None,
            packed_values: None,
        }
    }

    /// Append a single key/value pair to the cache (vector mode)
    pub fn append(&mut self, key: Tensor, value: Tensor) {
        // If packed storage exists, merge existing vector entries into packed storage first.
        if !self.keys.is_empty() && self.packed_keys.is_some() {
            let keys_to_merge = std::mem::take(&mut self.keys);
            let values_to_merge = std::mem::take(&mut self.values);
            for (k, v) in keys_to_merge.into_iter().zip(values_to_merge.into_iter()) {
                let _ = self.append_packed(&k, &v);
            }
        }
        // If packed storage doesn't exist but we have vector entries, convert them.
        if !self.keys.is_empty() && self.packed_keys.is_none() {
            let keys_to_convert = std::mem::take(&mut self.keys);
            let values_to_convert = std::mem::take(&mut self.values);
            if !keys_to_convert.is_empty() {
                let first_k = &keys_to_convert[0];
                let first_v = &values_to_convert[0];
                self.packed_keys = Some(first_k.clone());
                self.packed_values = Some(first_v.clone());
                for (k, v) in keys_to_convert.into_iter().skip(1).zip(values_to_convert.into_iter().skip(1)) {
                    let _ = self.append_packed(&k, &v);
                }
            }
        }
        self.keys.push(key);
        self.values.push(value);
    }

    /// Initialize packed storage from single per-step tensors
    /// Expects `keys` and `values` to be tensors with shape (batch, seq, dim)
    pub fn set_packed(&mut self, keys: Tensor, values: Tensor) {
        self.packed_keys = Some(keys);
        self.packed_values = Some(values);
    }

    /// Append new packed keys/values along the sequence axis (axis=1)
    /// Returns Err if shapes are incompatible.
    pub fn append_packed(&mut self, new_keys: &Tensor, new_values: &Tensor) -> Result<(), String> {
        // If no packed storage exists, initialize it with the new keys/values
        if self.packed_keys.is_none() {
            self.packed_keys = Some(new_keys.clone());
            self.packed_values = Some(new_values.clone());
            return Ok(());
        }

        let a_keys = self
            .packed_keys
            .as_ref()
            .ok_or_else(|| "Internal error: packed_keys should be Some".to_string())?
            .lock()
            .storage
            .to_f32_array();
        let b_keys = new_keys.lock().storage.to_f32_array();
        let a_vals = self
            .packed_values
            .as_ref()
            .ok_or_else(|| "Internal error: packed_values should be Some".to_string())?
            .lock()
            .storage
            .to_f32_array();
        let b_vals = new_values.lock().storage.to_f32_array();

        // quick shape checks: both must be 3D and match on axes 0 & 2
        if a_keys.ndim() != 3 || b_keys.ndim() != 3 {
            return Err("packed keys must be 3D tensors (batch, seq, dim)".to_string());
        }
        if a_vals.ndim() != 3 || b_vals.ndim() != 3 {
            return Err("packed values must be 3D tensors (batch, seq, dim)".to_string());
        }
        if a_keys.shape()[0] != b_keys.shape()[0] || a_keys.shape()[2] != b_keys.shape()[2] {
            return Err("batch or dim mismatch when appending packed keys".to_string());
        }
        if a_vals.shape() != a_keys.shape() || b_vals.shape() != b_keys.shape() {
            return Err("keys/values packed shapes must match".to_string());
        }

        // concatenate along seq axis (axis=1) using the op-level helper to avoid eager ndarray copies
        let cache_k = self
            .packed_keys
            .as_ref()
            .ok_or_else(|| {
                "Internal error: packed_keys should be Some at concatenation".to_string()
            })?
            .clone();
        let cache_v = self
            .packed_values
            .as_ref()
            .ok_or_else(|| {
                "Internal error: packed_values should be Some at concatenation".to_string()
            })?
            .clone();
        let new_cache_k = Tensor::kvcache_append(&cache_k, new_keys, 1);
        let new_cache_v = Tensor::kvcache_append(&cache_v, new_values, 1);

        self.packed_keys = Some(new_cache_k);
        self.packed_values = Some(new_cache_v);
        Ok(())
    }

    /// Number of entries in vector mode (legacy)
    pub fn len(&self) -> usize {
        if let Some(pk) = &self.packed_keys {
            pk.lock().storage.shape()[1]
        } else {
            self.keys.len()
        }
    }

    /// Sequence length (current number of tokens in cache).
    /// When packed storage is present it returns the authoritative count from the tensor shape.
    pub fn seq_len(&self) -> usize {
        if let Some(pk) = &self.packed_keys {
            pk.lock().storage.shape()[1]
        } else {
            self.keys.len()
        }
    }

    /// Return whether packed storage is present
    pub fn has_packed(&self) -> bool {
        self.packed_keys.is_some() && self.packed_values.is_some()
    }

    /// Get current packed key tensor (clone handle) if available
    pub fn packed_keys(&self) -> Option<Tensor> {
        self.packed_keys.clone()
    }

    /// Get current packed value tensor (clone handle) if available
    pub fn packed_values(&self) -> Option<Tensor> {
        self.packed_values.clone()
    }

    /// Clear cached key/value pairs and any packed storage
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.packed_keys = None;
        self.packed_values = None;
    }

    /// Check whether cache is empty (no packed & no vector entries)
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty() && self.packed_keys.is_none()
    }

    /// Remove n tokens from the end of the cache.
    pub fn truncate(&mut self, n: usize) {
        if n == 0 {
            return;
        }

        // Truncate vectors
        if self.keys.len() >= n {
            self.keys.truncate(self.keys.len() - n);
            self.values.truncate(self.values.len() - n);
        } else {
            self.keys.clear();
            self.values.clear();
        }

        // Truncate packed
        if let Some(pk) = &self.packed_keys {
            let shape = pk.lock().storage.shape().to_vec();
            // [batch, seq, dim]
            let current_len = shape[1];
            if n >= current_len {
                self.packed_keys = None;
                self.packed_values = None;
            } else {
                let new_len = current_len - n;
                let pk_slice = Tensor::apply(
                    std::sync::Arc::new(crate::ops::Slice::new(1, 0, new_len)),
                    &[pk.clone()][..],
                );
                self.packed_keys = Some(pk_slice);

                if let Some(pv) = &self.packed_values {
                    let pv_slice = Tensor::apply(
                        std::sync::Arc::new(crate::ops::Slice::new(1, 0, new_len)),
                        &[pv.clone()][..],
                    );
                    self.packed_values = Some(pv_slice);
                }
            }
        }
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}
