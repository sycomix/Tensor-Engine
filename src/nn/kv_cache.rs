use crate::tensor::Tensor;

/// Simple KV cache scaffolding for incremental decoding.
///
/// This is intentionally minimal for the first iteration: a vector-backed cache that stores
/// keys and values as `Tensor` objects. Later iterations will focus on packing into efficient
/// contiguous storage and adding ops for append/concat without cloning.
#[derive(Clone)]
pub struct KVCache {
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
}

impl KVCache {
    /// Create an empty KV cache
    pub fn new() -> Self {
        KVCache {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Append a single key/value pair to the cache
    pub fn append(&mut self, key: Tensor, value: Tensor) {
        self.keys.push(key);
        self.values.push(value);
    }

    /// Number of entries in the cache
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Clear cached key/value pairs
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }

    /// Check whether cache is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    // TODO: add concatenation helpers, efficient storage, and serialization support
}
