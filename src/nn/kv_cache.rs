use crate::tensor::Tensor;
use ndarray::ArrayD;

/// KV cache for incremental decoding with O(1) append.
///
/// Uses a single pre-allocated packed tensor per side (keys/values) with shape
/// `(batch, capacity, dim)`. A `filled_len` field tracks how many positions are
/// occupied. Appending a new token writes directly into the buffer at the next
/// slot — no reallocation, no concatenation, no op dispatch.
///
/// `seq_len()` returns `filled_len`. `packed_keys()` / `packed_values()` return
/// a sliced view of only the filled portion so downstream attention sees exactly
/// the cached tokens.
///
/// Falls back to a growable mode if `set_packed` is called without pre-allocation
/// (legacy path); in that case append grows the buffer by concatenation.
#[derive(Clone)]
pub struct KVCache {
    // packed storage: pre-allocated buffer with shape (batch, capacity, dim)
    packed_keys: Option<Tensor>,
    packed_values: Option<Tensor>,
    // number of token positions actually filled in the packed buffers
    filled_len: usize,
    // capacity (max tokens) of the packed buffers along the seq axis
    capacity: usize,

    // legacy vector-backed single-token entries (kept for backward compat)
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
}

impl KVCache {
    /// Create an empty KV cache.
    pub fn new() -> Self {
        KVCache {
            packed_keys: None,
            packed_values: None,
            filled_len: 0,
            capacity: 0,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Pre-allocate packed storage with the given capacity.
    /// Buffers are zero-initialised `[batch, capacity, dim]`.
    /// `filled_len` is set to 0 so the cache starts empty.
    pub fn set_packed_capacity(&mut self, batch: usize, capacity: usize, dim: usize) {
        use ndarray::IxDyn;
        let k = crate::tensor::Tensor::new(
            ArrayD::<f32>::zeros(IxDyn(&[batch, capacity, dim])),
            false,
        );
        let v = crate::tensor::Tensor::new(
            ArrayD::<f32>::zeros(IxDyn(&[batch, capacity, dim])),
            false,
        );
        self.packed_keys = Some(k);
        self.packed_values = Some(v);
        self.filled_len = 0;
        self.capacity = capacity;
    }

    /// Initialize packed storage from existing tensors (legacy path).
    /// Expects `keys` and `values` to be tensors with shape (batch, seq, dim).
    /// `filled_len` is set to the seq dimension of the provided tensors.
    pub fn set_packed(&mut self, keys: Tensor, values: Tensor) {
        let seq = {
            let s = keys.lock().storage.shape().to_vec();
            if s.len() >= 2 { s[1] } else { 0 }
        };
        self.capacity = seq;
        self.filled_len = seq;
        self.packed_keys = Some(keys);
        self.packed_values = Some(values);
    }

    /// Append new packed keys/values along the sequence axis.
    ///
    /// If pre-allocated buffers exist (capacity > 0), writes directly into the
    /// buffer at `filled_len` — O(1) per token, no reallocation.
    ///
    /// Falls back to concatenation if no pre-allocated capacity is set.
    pub fn append_packed(
        &mut self,
        new_keys: &Tensor,
        new_values: &Tensor,
    ) -> Result<(), String> {
        // --- Pre-allocated fast path ---
        if self.capacity > 0 && self.packed_keys.is_some() {
            let new_k_arr = new_keys.lock().storage.to_f32_array();
            let new_v_arr = new_values.lock().storage.to_f32_array();

            if new_k_arr.ndim() != 3 || new_v_arr.ndim() != 3 {
                return Err("packed keys/values must be 3D (batch, seq, dim)".to_string());
            }

            let batch = new_k_arr.shape()[0];
            let new_seq = new_k_arr.shape()[1];
            let dim = new_k_arr.shape()[2];

            if self.filled_len + new_seq > self.capacity {
                return Err(format!(
                    "KV cache overflow: filled={} + new={} > capacity={}",
                    self.filled_len, new_seq, self.capacity
                ));
            }

            // Validate batch/dim against existing buffer
            {
                let pk = self.packed_keys.as_ref().unwrap();
                let pk_shape = pk.lock().storage.shape().to_vec();
                if pk_shape.len() != 3 || pk_shape[0] != batch || pk_shape[2] != dim {
                    return Err(format!(
                        "KV cache batch/dim mismatch: buffer {:?} vs new [{}, {}, {}]",
                        pk_shape, batch, new_seq, dim
                    ));
                }
            }

            let offset = self.filled_len;

            // Write keys directly into the pre-allocated buffer
            {
                let pk = self.packed_keys.as_ref().unwrap();
                let mut pk_lock = pk.lock();
                let storage = &mut pk_lock.storage;
                let arr = storage.to_f32_array_mut();
                for b in 0..batch {
                    for s in 0..new_seq {
                        for d in 0..dim {
                            arr[[b, offset + s, d]] = new_k_arr[[b, s, d]];
                        }
                    }
                }
            }

            // Write values directly into the pre-allocated buffer
            {
                let pv = self.packed_values.as_ref().unwrap();
                let mut pv_lock = pv.lock();
                let storage = &mut pv_lock.storage;
                let arr = storage.to_f32_array_mut();
                for b in 0..batch {
                    for s in 0..new_seq {
                        for d in 0..dim {
                            arr[[b, offset + s, d]] = new_v_arr[[b, s, d]];
                        }
                    }
                }
            }

            self.filled_len += new_seq;
            return Ok(());
        }

        // --- Legacy / growable path ---
        if self.packed_keys.is_none() {
            self.packed_keys = Some(new_keys.clone());
            self.packed_values = Some(new_values.clone());
            let seq = {
                let s = new_keys.lock().storage.shape().to_vec();
                if s.len() >= 2 { s[1] } else { 0 }
            };
            self.filled_len = seq;
            self.capacity = seq;
            return Ok(());
        }

        // Validate shapes
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

        // concatenate along seq axis (axis=1)
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

        let new_seq = {
            let s = new_cache_k.lock().storage.shape().to_vec();
            if s.len() >= 2 { s[1] } else { 0 }
        };
        self.packed_keys = Some(new_cache_k);
        self.packed_values = Some(new_cache_v);
        self.filled_len = new_seq;
        self.capacity = new_seq;
        Ok(())
    }

    /// Append a single key/value pair to the cache (vector mode — legacy).
    pub fn append(&mut self, key: Tensor, value: Tensor) {
        if !self.keys.is_empty() && self.packed_keys.is_some() {
            let keys_to_merge = std::mem::take(&mut self.keys);
            let values_to_merge = std::mem::take(&mut self.values);
            for (k, v) in keys_to_merge.into_iter().zip(values_to_merge.into_iter()) {
                let _ = self.append_packed(&k, &v);
            }
        }
        if self.packed_keys.is_some() {
            let _ = self.append_packed(&key, &value);
        }
        self.keys.push(key);
        self.values.push(value);
    }

    /// Number of entries in vector mode (legacy).
    pub fn len(&self) -> usize {
        self.seq_len()
    }

    /// Sequence length (current number of tokens in cache).
    pub fn seq_len(&self) -> usize {
        if self.packed_keys.is_some() {
            self.filled_len
        } else {
            self.keys.len()
        }
    }

    /// Return whether packed storage is present.
    pub fn has_packed(&self) -> bool {
        self.packed_keys.is_some() && self.packed_values.is_some()
    }

    /// Get current packed key tensor — returns a sliced view of only the filled
    /// portion so downstream attention sees exactly the cached tokens.
    pub fn packed_keys(&self) -> Option<Tensor> {
        let pk = self.packed_keys.as_ref()?;
        if self.filled_len == 0 {
            // Return empty 3D tensor with seq=0
            let shape = pk.lock().storage.shape().to_vec();
            let empty_shape = if shape.len() >= 3 {
                vec![shape[0], 0, shape[2]]
            } else {
                vec![0]
            };
            return Some(Tensor::new(
                ArrayD::<f32>::zeros(ndarray::IxDyn(&empty_shape)),
                false,
            ));
        }
        if self.filled_len == self.capacity {
            return Some(pk.clone());
        }
        // Slice along axis 1: [0..filled_len]
        Some(Tensor::apply(
            std::sync::Arc::new(crate::ops::Slice::new(1, 0, self.filled_len)),
            &[pk.clone()][..],
        ))
    }

    /// Get current packed value tensor — sliced view of filled portion.
    pub fn packed_values(&self) -> Option<Tensor> {
        let pv = self.packed_values.as_ref()?;
        if self.filled_len == 0 {
            let shape = pv.lock().storage.shape().to_vec();
            let empty_shape = if shape.len() >= 3 {
                vec![shape[0], 0, shape[2]]
            } else {
                vec![0]
            };
            return Some(Tensor::new(
                ArrayD::<f32>::zeros(ndarray::IxDyn(&empty_shape)),
                false,
            ));
        }
        if self.filled_len == self.capacity {
            return Some(pv.clone());
        }
        Some(Tensor::apply(
            std::sync::Arc::new(crate::ops::Slice::new(1, 0, self.filled_len)),
            &[pv.clone()][..],
        ))
    }

    /// Clear cached key/value pairs and any packed storage.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.packed_keys = None;
        self.packed_values = None;
        self.filled_len = 0;
        self.capacity = 0;
    }

    /// Check whether cache is empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty() && self.filled_len == 0
    }

    /// Remove n tokens from the end of the cache.
    pub fn truncate(&mut self, n: usize) {
        if n == 0 {
            return;
        }

        // Truncate vectors (legacy)
        if self.keys.len() >= n {
            self.keys.truncate(self.keys.len() - n);
            self.values.truncate(self.values.len() - n);
        } else {
            self.keys.clear();
            self.values.clear();
        }

        // Truncate packed: just reduce filled_len (no reallocation needed)
        if self.filled_len > 0 {
            if n >= self.filled_len {
                self.filled_len = 0;
            } else {
                self.filled_len -= n;
            }
        }
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}