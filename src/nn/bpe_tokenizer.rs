use regex::Regex;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;
use std::sync::Arc;
use unicode_normalization::UnicodeNormalization;

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Default)]
pub struct FnvHasher(u64);

impl Hasher for FnvHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut hash = if self.0 == 0 {
            FNV_OFFSET_BASIS
        } else {
            self.0
        };
        for &byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        self.0 = hash;
    }

    fn finish(&self) -> u64 {
        if self.0 == 0 {
            FNV_OFFSET_BASIS
        } else {
            self.0
        }
    }
}

type FnvBuildHasher = BuildHasherDefault<FnvHasher>;
type FnvHashMap<K, V> = HashMap<K, V, FnvBuildHasher>;
type FnvHashSet<T> = HashSet<T, FnvBuildHasher>;

#[derive(Default)]
struct PairHeap(BinaryHeap<(usize, Reverse<(u32, u32)>)>);

impl PairHeap {
    fn new() -> Self {
        Self(BinaryHeap::new())
    }

    fn push(&mut self, pair: (u32, u32), count: usize) {
        if count > 0 {
            self.0.push((count, Reverse(pair)));
        }
    }

    fn pop_best(
        &mut self,
        pair_freq: &FnvHashMap<(u32, u32), usize>,
    ) -> Option<((u32, u32), usize)> {
        while let Some((count, Reverse(pair))) = self.0.pop() {
            if pair_freq.get(&pair).copied().unwrap_or(0) == count {
                return Some((pair, count));
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct TrainProgress {
    pub merges_done: usize,
    pub total_merges: usize,
    pub vocab_size: usize,
    pub best_pair: (u32, u32),
    pub best_pair_frequency: usize,
}

#[derive(Clone)]
pub struct BPETokenizer {
    pub special_tokens: Vec<String>,
    pub pad_token: String,
    pub bos_token: String,
    pub eos_token: String,

    pub pad_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,

    pub merges: FnvHashMap<(u32, u32), u32>,
    pub merge_rules: Vec<((u32, u32), u32)>,
    pub merge_ranks: FnvHashMap<(u32, u32), usize>,

    pub token_to_id: FnvHashMap<String, u32>,
    pub id_to_token: FnvHashMap<u32, String>,
    pub id_to_bytes: FnvHashMap<u32, Vec<u8>>,
    pub byte_to_id: [u32; 256],

    pub unk_token: String,
    pub unk_id: Option<u32>,
    pub use_unk_token: bool,
    pub unknown_decode_fallback: String,
    pub unknown_decode_fallbacks: FnvHashMap<u32, Vec<u8>>,

    pub lowercase: bool,
    pub normalization_form: Option<String>,
    pub keep_whitespace_tokens: bool,

    pub log_every_merges: Option<usize>,
    pub compact_every_merges: usize,
    pub small_sequence_merge_threshold: usize,

    pub train_progress_callback: Option<Arc<dyn Fn(&TrainProgress) + Send + Sync>>,

    pretoken_pattern: Regex,
}

#[derive(Debug, Serialize, Deserialize)]
struct VocabPayload {
    special_tokens: Vec<String>,
    token_to_id: FnvHashMap<String, u32>,
    id_to_token: FnvHashMap<u32, String>,
    merges: Vec<[u32; 3]>,
    use_unk_token: bool,
    unk_token: String,
    lowercase: bool,
    normalization_form: Option<String>,
    keep_whitespace_tokens: bool,
    unknown_decode_fallback: String,
    unknown_decode_fallbacks: FnvHashMap<u32, Vec<u8>>,
    log_every_merges: Option<usize>,
    compact_every_merges: usize,
    small_sequence_merge_threshold: usize,
}

impl BPETokenizer {
    pub fn new() -> Self {
        let special_tokens = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
        ];
        let pretoken_pattern = Regex::new(
            r"\d+\.\d+|[\p{Alphabetic}]+(?:'[\p{Alphabetic}]+)?(?:-[\p{Alphabetic}]+(?:'[\p{Alphabetic}]+)?)*|\s+|.",
        )
        .expect("invalid pretoken regex");

        let mut tokenizer = Self {
            special_tokens: special_tokens.clone(),
            pad_token: "<pad>".to_string(),
            bos_token: "<bos>".to_string(),
            eos_token: "<eos>".to_string(),
            pad_id: 0,
            bos_id: 1,
            eos_id: 2,
            merges: FnvHashMap::default(),
            merge_rules: Vec::new(),
            merge_ranks: FnvHashMap::default(),
            token_to_id: FnvHashMap::default(),
            id_to_token: FnvHashMap::default(),
            id_to_bytes: FnvHashMap::default(),
            byte_to_id: [0; 256],
            unk_token: "<unk>".to_string(),
            unk_id: None,
            use_unk_token: false,
            unknown_decode_fallback: "?".to_string(),
            unknown_decode_fallbacks: FnvHashMap::default(),
            lowercase: false,
            normalization_form: None,
            keep_whitespace_tokens: true,
            log_every_merges: None,
            compact_every_merges: 1000,
            small_sequence_merge_threshold: 24,
            train_progress_callback: None,
            pretoken_pattern,
        };
        tokenizer.initialize_base_vocab();
        tokenizer
    }

    pub fn configure_text_processing(
        &mut self,
        lowercase: Option<bool>,
        normalization_form: Option<String>,
        keep_whitespace_tokens: Option<bool>,
    ) {
        if let Some(v) = lowercase {
            self.lowercase = v;
        }
        self.normalization_form = normalization_form;
        if let Some(v) = keep_whitespace_tokens {
            self.keep_whitespace_tokens = v;
        }
    }

    pub fn enable_unk_token(&mut self) {
        if self.use_unk_token {
            return;
        }
        self.use_unk_token = true;
        self.ensure_unk_token();
    }

    pub fn disable_unk_token(&mut self) {
        self.use_unk_token = false;
    }

    pub fn set_unknown_decode_fallback(&mut self, fallback: String) {
        self.unknown_decode_fallback = fallback;
    }

    pub fn set_unknown_decode_fallback_for_token(&mut self, token_id: u32, fallback: Vec<u8>) {
        self.unknown_decode_fallbacks.insert(token_id, fallback);
    }

    pub fn clear_unknown_decode_fallback_for_token(&mut self, token_id: u32) {
        self.unknown_decode_fallbacks.remove(&token_id);
    }

    pub fn set_train_progress_callback(
        &mut self,
        callback: Option<Arc<dyn Fn(&TrainProgress) + Send + Sync>>,
    ) {
        self.train_progress_callback = callback;
    }

    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    fn ensure_unk_token(&mut self) {
        if let Some(existing) = self.token_to_id.get(&self.unk_token).copied() {
            self.unk_id = Some(existing);
            return;
        }

        let mut unk_id = self.id_to_token.keys().max().map_or(0, |v| v + 1);
        while self.id_to_token.contains_key(&unk_id) {
            unk_id += 1;
        }

        self.token_to_id.insert(self.unk_token.clone(), unk_id);
        self.id_to_token.insert(unk_id, self.unk_token.clone());
        self.unk_id = Some(unk_id);
    }

    fn normalize_text(&self, text: &str) -> String {
        let normalized = match self.normalization_form.as_deref() {
            Some("NFC") => text.nfc().collect::<String>(),
            Some("NFD") => text.nfd().collect::<String>(),
            Some("NFKC") => text.nfkc().collect::<String>(),
            Some("NFKD") => text.nfkd().collect::<String>(),
            _ => text.to_string(),
        };

        if self.lowercase {
            normalized.to_lowercase()
        } else {
            normalized
        }
    }

    fn byte_to_token_id_or_unk(&mut self, byte_value: u8) -> Option<u32> {
        let token_id = self.byte_to_id[byte_value as usize];
        if token_id != u32::MAX {
            return Some(token_id);
        }

        if self.use_unk_token {
            if self.unk_id.is_none() {
                self.ensure_unk_token();
            }
            return self.unk_id;
        }

        None
    }

    fn initialize_base_vocab(&mut self) {
        self.token_to_id.clear();
        self.id_to_token.clear();
        self.id_to_bytes.clear();
        self.byte_to_id = [u32::MAX; 256];

        for (idx, tok) in self.special_tokens.iter().enumerate() {
            let id = idx as u32;
            self.token_to_id.insert(tok.clone(), id);
            self.id_to_token.insert(id, tok.clone());
        }

        let start = self.special_tokens.len() as u32;
        for byte_value in 0u32..=255 {
            let token_id = start + byte_value;
            self.byte_to_id[byte_value as usize] = token_id;
            self.id_to_bytes.insert(token_id, vec![byte_value as u8]);
            let token_str = format!("<0x{:02X}>", byte_value);
            self.token_to_id.insert(token_str.clone(), token_id);
            self.id_to_token.insert(token_id, token_str);
        }
    }

    fn pretokenize(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        for m in self.pretoken_pattern.find_iter(text) {
            let piece = m.as_str().to_string();
            if self.keep_whitespace_tokens || !piece.chars().all(char::is_whitespace) {
                out.push(piece);
            }
        }
        out
    }

    fn count_pairs_in_sequence(sequence: &[u32]) -> FnvHashMap<(u32, u32), usize> {
        let mut counts: FnvHashMap<(u32, u32), usize> = FnvHashMap::default();
        if sequence.len() < 2 {
            return counts;
        }
        for i in 0..(sequence.len() - 1) {
            *counts.entry((sequence[i], sequence[i + 1])).or_insert(0) += 1;
        }
        counts
    }

    fn merge_pair_in_sequence(sequence: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
        let mut merged = Vec::with_capacity(sequence.len());
        let mut i = 0usize;
        while i < sequence.len() {
            if i + 1 < sequence.len() && sequence[i] == pair.0 && sequence[i + 1] == pair.1 {
                merged.push(new_id);
                i += 2;
            } else {
                merged.push(sequence[i]);
                i += 1;
            }
        }
        merged
    }

    fn pop_best_pair(
        heap: &mut PairHeap,
        pair_freq: &FnvHashMap<(u32, u32), usize>,
    ) -> Option<((u32, u32), usize)> {
        heap.pop_best(pair_freq)
    }

    fn register_sequence(
        sequence: Vec<u32>,
        seq_to_id: &mut FnvHashMap<Vec<u32>, usize>,
        id_to_seq: &mut Vec<Vec<u32>>,
    ) -> usize {
        if let Some(id) = seq_to_id.get(&sequence).copied() {
            return id;
        }
        let seq_id = id_to_seq.len();
        id_to_seq.push(sequence.clone());
        seq_to_id.insert(sequence, seq_id);
        seq_id
    }

    fn build_sequence_store<I, S>(
        &self,
        corpus: I,
    ) -> (
        FnvHashMap<Vec<u32>, usize>,
        Vec<Vec<u32>>,
        FnvHashMap<usize, usize>,
    )
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut seq_to_id: FnvHashMap<Vec<u32>, usize> = FnvHashMap::default();
        let mut id_to_seq: Vec<Vec<u32>> = Vec::new();
        let mut seq_freq: FnvHashMap<usize, usize> = FnvHashMap::default();

        for text in corpus {
            let text = self.normalize_text(text.as_ref());
            for piece in self.pretokenize(&text) {
                let bytes = piece.as_bytes();
                if bytes.is_empty() {
                    continue;
                }
                let sequence: Vec<u32> =
                    bytes.iter().map(|b| self.byte_to_id[*b as usize]).collect();
                if sequence.is_empty() {
                    continue;
                }
                let seq_id = Self::register_sequence(sequence, &mut seq_to_id, &mut id_to_seq);
                *seq_freq.entry(seq_id).or_insert(0) += 1;
            }
        }

        (seq_to_id, id_to_seq, seq_freq)
    }

    fn initialize_pair_stats(
        id_to_seq: &[Vec<u32>],
        seq_freq: &FnvHashMap<usize, usize>,
    ) -> (
        FnvHashMap<(u32, u32), usize>,
        FnvHashMap<(u32, u32), FnvHashSet<usize>>,
        PairHeap,
    ) {
        let mut pair_freq: FnvHashMap<(u32, u32), usize> = FnvHashMap::default();
        let mut pair_to_seq_ids: FnvHashMap<(u32, u32), FnvHashSet<usize>> = FnvHashMap::default();

        for (&seq_id, &freq) in seq_freq {
            let sequence = &id_to_seq[seq_id];
            if sequence.len() < 2 {
                continue;
            }
            let local_pairs = Self::count_pairs_in_sequence(sequence);
            for (pair, count) in local_pairs {
                *pair_freq.entry(pair).or_insert(0) += count * freq;
                pair_to_seq_ids.entry(pair).or_default().insert(seq_id);
            }
        }

        let mut heap = PairHeap::new();
        for (&pair, &count) in &pair_freq {
            heap.push(pair, count);
        }

        (pair_freq, pair_to_seq_ids, heap)
    }

    fn subtract_sequence_from_pair_stats(
        seq_id: usize,
        sequence: &[u32],
        freq: usize,
        pair_freq: &mut FnvHashMap<(u32, u32), usize>,
        pair_to_seq_ids: &mut FnvHashMap<(u32, u32), FnvHashSet<usize>>,
        changed_pairs: &mut FnvHashSet<(u32, u32)>,
    ) {
        let local_pairs = Self::count_pairs_in_sequence(sequence);
        for (pair, count) in local_pairs {
            if let Some(current) = pair_freq.get(&pair).copied() {
                let updated = current.saturating_sub(count * freq);
                if updated > 0 {
                    pair_freq.insert(pair, updated);
                } else {
                    pair_freq.remove(&pair);
                }
            }

            if let Some(seq_ids) = pair_to_seq_ids.get_mut(&pair) {
                seq_ids.remove(&seq_id);
                if seq_ids.is_empty() {
                    pair_to_seq_ids.remove(&pair);
                }
            }

            changed_pairs.insert(pair);
        }
    }

    fn add_sequence_to_pair_stats(
        seq_id: usize,
        sequence: &[u32],
        freq: usize,
        pair_freq: &mut FnvHashMap<(u32, u32), usize>,
        pair_to_seq_ids: &mut FnvHashMap<(u32, u32), FnvHashSet<usize>>,
        changed_pairs: &mut FnvHashSet<(u32, u32)>,
    ) {
        if sequence.len() < 2 {
            return;
        }
        let local_pairs = Self::count_pairs_in_sequence(sequence);
        for (pair, count) in local_pairs {
            *pair_freq.entry(pair).or_insert(0) += count * freq;
            pair_to_seq_ids.entry(pair).or_default().insert(seq_id);
            changed_pairs.insert(pair);
        }
    }

    fn compact_training_state(
        id_to_seq: &mut Vec<Vec<u32>>,
        seq_freq: &FnvHashMap<usize, usize>,
        pair_to_seq_ids: &mut FnvHashMap<(u32, u32), FnvHashSet<usize>>,
    ) {
        let alive_ids: FnvHashSet<usize> = seq_freq.keys().copied().collect();
        if alive_ids.is_empty() {
            return;
        }

        let pairs: Vec<(u32, u32)> = pair_to_seq_ids.keys().copied().collect();
        for pair in pairs {
            if let Some(seq_ids) = pair_to_seq_ids.get_mut(&pair) {
                seq_ids.retain(|id| alive_ids.contains(id));
                if seq_ids.is_empty() {
                    pair_to_seq_ids.remove(&pair);
                }
            }
        }

        while let Some(last_idx) = id_to_seq.len().checked_sub(1) {
            if alive_ids.contains(&last_idx) {
                break;
            }
            id_to_seq.pop();
        }
    }

    fn merge_affected_sequences(
        best_pair: (u32, u32),
        new_id: u32,
        seq_to_id: &mut FnvHashMap<Vec<u32>, usize>,
        id_to_seq: &mut Vec<Vec<u32>>,
        seq_freq: &mut FnvHashMap<usize, usize>,
        pair_freq: &mut FnvHashMap<(u32, u32), usize>,
        pair_to_seq_ids: &mut FnvHashMap<(u32, u32), FnvHashSet<usize>>,
        heap: &mut PairHeap,
    ) {
        let affected_seq_ids: Vec<usize> = pair_to_seq_ids
            .get(&best_pair)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        if affected_seq_ids.is_empty() {
            return;
        }

        let mut changed_pairs: FnvHashSet<(u32, u32)> = FnvHashSet::default();
        let mut pending_additions: FnvHashMap<usize, usize> = FnvHashMap::default();

        for seq_id in affected_seq_ids {
            let freq = seq_freq.get(&seq_id).copied().unwrap_or(0);
            if freq == 0 {
                continue;
            }

            let old_sequence = id_to_seq[seq_id].clone();
            Self::subtract_sequence_from_pair_stats(
                seq_id,
                &old_sequence,
                freq,
                pair_freq,
                pair_to_seq_ids,
                &mut changed_pairs,
            );

            seq_freq.remove(&seq_id);

            let merged_sequence = Self::merge_pair_in_sequence(&old_sequence, best_pair, new_id);
            let merged_seq_id = Self::register_sequence(merged_sequence, seq_to_id, id_to_seq);
            *pending_additions.entry(merged_seq_id).or_insert(0) += freq;
        }

        for (seq_id, add_freq) in pending_additions {
            *seq_freq.entry(seq_id).or_insert(0) += add_freq;
            let seq = id_to_seq[seq_id].clone();
            Self::add_sequence_to_pair_stats(
                seq_id,
                &seq,
                add_freq,
                pair_freq,
                pair_to_seq_ids,
                &mut changed_pairs,
            );
        }

        for pair in changed_pairs {
            let count = pair_freq.get(&pair).copied().unwrap_or(0);
            heap.push(pair, count);
        }
    }

    fn apply_all_merges(&self, token_ids: &[u32]) -> Vec<u32> {
        if token_ids.len() < 2 || self.merge_ranks.is_empty() {
            return token_ids.to_vec();
        }

        if token_ids.len() <= self.small_sequence_merge_threshold {
            let mut result = token_ids.to_vec();
            for &((left, right), new_id) in &self.merge_rules {
                if result.len() < 2 {
                    break;
                }
                result = Self::merge_pair_in_sequence(&result, (left, right), new_id);
            }
            return result;
        }

        let mut tokens = token_ids.to_vec();
        let n = tokens.len();

        let mut prev = vec![usize::MAX; n];
        let mut next = vec![usize::MAX; n];
        let mut alive = vec![true; n];

        for i in 0..n {
            if i > 0 {
                prev[i] = i - 1;
            }
            if i + 1 < n {
                next[i] = i + 1;
            }
        }

        let mut heap: BinaryHeap<(Reverse<usize>, usize, u32, u32)> = BinaryHeap::new();

        let push_pair = |left_idx: usize,
                         heap: &mut BinaryHeap<(Reverse<usize>, usize, u32, u32)>,
                         tokens: &[u32],
                         next: &[usize],
                         merge_ranks: &FnvHashMap<(u32, u32), usize>| {
            let right_idx = next[left_idx];
            if right_idx == usize::MAX {
                return;
            }
            let pair = (tokens[left_idx], tokens[right_idx]);
            if let Some(&rank) = merge_ranks.get(&pair) {
                heap.push((Reverse(rank), left_idx, pair.0, pair.1));
            }
        };

        for i in 0..(n - 1) {
            push_pair(i, &mut heap, &tokens, &next, &self.merge_ranks);
        }

        while let Some((Reverse(rank), left_idx, left_token, right_token)) = heap.pop() {
            if !alive[left_idx] {
                continue;
            }
            let right_idx = next[left_idx];
            if right_idx == usize::MAX || !alive[right_idx] {
                continue;
            }
            if tokens[left_idx] != left_token || tokens[right_idx] != right_token {
                continue;
            }

            let pair = (left_token, right_token);
            let Some(&current_rank) = self.merge_ranks.get(&pair) else {
                continue;
            };
            if current_rank != rank {
                continue;
            }

            let merged_id = self.merges[&pair];
            tokens[left_idx] = merged_id;

            let after_right = next[right_idx];
            next[left_idx] = after_right;
            if after_right != usize::MAX {
                prev[after_right] = left_idx;
            }
            alive[right_idx] = false;

            let before_left = prev[left_idx];
            if before_left != usize::MAX && alive[before_left] {
                push_pair(before_left, &mut heap, &tokens, &next, &self.merge_ranks);
            }
            push_pair(left_idx, &mut heap, &tokens, &next, &self.merge_ranks);
        }

        let mut out = Vec::new();
        let mut idx = 0usize;
        while idx < n && !alive[idx] {
            idx = next[idx];
            if idx == usize::MAX {
                break;
            }
        }

        while idx != usize::MAX {
            if alive[idx] {
                out.push(tokens[idx]);
            }
            idx = next[idx];
        }

        out
    }

    pub fn train<I, S>(&mut self, corpus: I, vocab_size: usize)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let min_vocab = self.special_tokens.len() + 256;
        if vocab_size < min_vocab {
            panic!("vocab_size must be at least {}", min_vocab);
        }

        self.merges.clear();
        self.merge_rules.clear();
        self.merge_ranks.clear();
        self.initialize_base_vocab();

        let (mut seq_to_id, mut id_to_seq, mut seq_freq) = self.build_sequence_store(corpus);

        let mut next_id = (self.special_tokens.len() + 256) as u32;
        let target_vocab = vocab_size.max(next_id as usize);

        let (mut pair_freq, mut pair_to_seq_ids, mut heap) =
            Self::initialize_pair_stats(&id_to_seq, &seq_freq);

        if self.use_unk_token {
            self.ensure_unk_token();
        }

        let mut merge_steps = 0usize;
        let total_target_merges = target_vocab.saturating_sub(next_id as usize);
        let log_interval = self.log_every_merges.filter(|v| *v > 0).unwrap_or_else(|| {
            if total_target_merges > 0 {
                (total_target_merges / 100).max(100)
            } else {
                100
            }
        });

        while (next_id as usize) < target_vocab {
            let Some((best_pair, max_count)) = Self::pop_best_pair(&mut heap, &pair_freq) else {
                break;
            };

            if max_count < 2 {
                break;
            }

            let left_bytes = self.id_to_bytes[&best_pair.0].clone();
            let right_bytes = self.id_to_bytes[&best_pair.1].clone();
            let mut merged_bytes = left_bytes;
            merged_bytes.extend(right_bytes);

            self.merges.insert(best_pair, next_id);
            self.merge_rules.push((best_pair, next_id));
            self.merge_ranks
                .insert(best_pair, self.merge_rules.len() - 1);
            self.id_to_bytes.insert(next_id, merged_bytes);

            let token_str = format!("<bpe:{}>", next_id);
            self.token_to_id.insert(token_str.clone(), next_id);
            self.id_to_token.insert(next_id, token_str);

            Self::merge_affected_sequences(
                best_pair,
                next_id,
                &mut seq_to_id,
                &mut id_to_seq,
                &mut seq_freq,
                &mut pair_freq,
                &mut pair_to_seq_ids,
                &mut heap,
            );

            next_id += 1;
            merge_steps += 1;

            if merge_steps % log_interval == 0 {
                let pct = if total_target_merges > 0 {
                    (merge_steps as f64 / total_target_merges as f64) * 100.0
                } else {
                    100.0
                };
                let progress = TrainProgress {
                    merges_done: merge_steps,
                    total_merges: total_target_merges,
                    vocab_size: self.token_to_id.len(),
                    best_pair,
                    best_pair_frequency: max_count,
                };

                if let Some(callback) = &self.train_progress_callback {
                    callback(&progress);
                }

                eprintln!(
                    "BPE train progress: merges={}/{} ({:.2}%) vocab={} top_pair={:?} freq={}",
                    progress.merges_done,
                    progress.total_merges,
                    pct,
                    progress.vocab_size,
                    progress.best_pair,
                    progress.best_pair_frequency
                );
            }

            if self.compact_every_merges > 0 && merge_steps % self.compact_every_merges == 0 {
                Self::compact_training_state(&mut id_to_seq, &seq_freq, &mut pair_to_seq_ids);
            }
        }
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let mut encoded = vec![self.bos_id];
        let text = self.normalize_text(text);

        if self.use_unk_token && self.unk_id.is_none() {
            self.ensure_unk_token();
        }

        for piece in self.pretokenize(&text) {
            let bytes = piece.as_bytes();
            if bytes.is_empty() {
                continue;
            }
            let token_ids: Vec<u32> = bytes
                .iter()
                .map(|b| {
                    self.byte_to_token_id_or_unk(*b).unwrap_or_else(|| {
                        panic!(
                            "Byte value {} missing from base vocabulary and <unk> is disabled",
                            *b
                        )
                    })
                })
                .collect();
            let token_ids = self.apply_all_merges(&token_ids);
            encoded.extend(token_ids);
        }

        encoded.push(self.eos_id);
        encoded
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, String> {
        let mut decoded_bytes: Vec<u8> = Vec::new();
        let fallback_bytes = self.unknown_decode_fallback.as_bytes();

        for &token_id in token_ids {
            if token_id == self.pad_id || token_id == self.bos_id || token_id == self.eos_id {
                continue;
            }

            if self.use_unk_token && self.unk_id == Some(token_id) {
                let per_token = self.unknown_decode_fallbacks.get(&token_id);
                decoded_bytes.extend_from_slice(per_token.map_or(fallback_bytes, |v| v.as_slice()));
                continue;
            }

            if let Some(token_bytes) = self.id_to_bytes.get(&token_id) {
                decoded_bytes.extend_from_slice(token_bytes);
            } else if self.use_unk_token && self.unk_id.is_some() {
                let per_token = self.unknown_decode_fallbacks.get(&token_id);
                decoded_bytes.extend_from_slice(per_token.map_or(fallback_bytes, |v| v.as_slice()));
            } else {
                return Err(format!("Unknown token id: {}", token_id));
            }
        }

        String::from_utf8(decoded_bytes).map_err(|e| format!("UTF-8 decode error: {}", e))
    }

    pub fn decode_lossy(&self, token_ids: &[u32]) -> String {
        let mut decoded_bytes: Vec<u8> = Vec::new();
        let fallback_bytes = self.unknown_decode_fallback.as_bytes();

        for &token_id in token_ids {
            if token_id == self.pad_id || token_id == self.bos_id || token_id == self.eos_id {
                continue;
            }

            if self.use_unk_token && self.unk_id == Some(token_id) {
                let per_token = self.unknown_decode_fallbacks.get(&token_id);
                decoded_bytes.extend_from_slice(per_token.map_or(fallback_bytes, |v| v.as_slice()));
                continue;
            }

            if let Some(token_bytes) = self.id_to_bytes.get(&token_id) {
                decoded_bytes.extend_from_slice(token_bytes);
            } else if self.use_unk_token && self.unk_id.is_some() {
                let per_token = self.unknown_decode_fallbacks.get(&token_id);
                decoded_bytes.extend_from_slice(per_token.map_or(fallback_bytes, |v| v.as_slice()));
            } else {
                decoded_bytes.extend_from_slice(fallback_bytes);
            }
        }

        String::from_utf8_lossy(&decoded_bytes).into_owned()
    }

    pub fn save_vocab<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let merges: Vec<[u32; 3]> = self
            .merge_rules
            .iter()
            .map(|((l, r), new_id)| [*l, *r, *new_id])
            .collect();

        let payload = VocabPayload {
            special_tokens: self.special_tokens.clone(),
            token_to_id: self.token_to_id.clone(),
            id_to_token: self.id_to_token.clone(),
            merges,
            use_unk_token: self.use_unk_token,
            unk_token: self.unk_token.clone(),
            lowercase: self.lowercase,
            normalization_form: self.normalization_form.clone(),
            keep_whitespace_tokens: self.keep_whitespace_tokens,
            unknown_decode_fallback: self.unknown_decode_fallback.clone(),
            unknown_decode_fallbacks: self.unknown_decode_fallbacks.clone(),
            log_every_merges: self.log_every_merges,
            compact_every_merges: self.compact_every_merges,
            small_sequence_merge_threshold: self.small_sequence_merge_threshold,
        };

        let json =
            serde_json::to_string(&payload).map_err(|e| format!("serialize error: {}", e))?;
        fs::write(path, json).map_err(|e| format!("write error: {}", e))
    }

    pub fn load_vocab<P: AsRef<Path>>(&mut self, path: P) -> Result<(), String> {
        let content = fs::read_to_string(path).map_err(|e| format!("read error: {}", e))?;
        let payload: VocabPayload =
            serde_json::from_str(&content).map_err(|e| format!("parse error: {}", e))?;

        self.special_tokens = payload.special_tokens;
        self.pad_token = self
            .special_tokens
            .get(0)
            .cloned()
            .unwrap_or_else(|| "<pad>".to_string());
        self.bos_token = self
            .special_tokens
            .get(1)
            .cloned()
            .unwrap_or_else(|| "<bos>".to_string());
        self.eos_token = self
            .special_tokens
            .get(2)
            .cloned()
            .unwrap_or_else(|| "<eos>".to_string());

        self.pad_id = self
            .special_tokens
            .iter()
            .position(|x| x == &self.pad_token)
            .unwrap_or(0) as u32;
        self.bos_id = self
            .special_tokens
            .iter()
            .position(|x| x == &self.bos_token)
            .unwrap_or(1) as u32;
        self.eos_id = self
            .special_tokens
            .iter()
            .position(|x| x == &self.eos_token)
            .unwrap_or(2) as u32;

        self.token_to_id = payload.token_to_id;
        self.id_to_token = payload.id_to_token;

        self.unk_token = payload.unk_token;
        self.use_unk_token = payload.use_unk_token;
        self.unk_id = self.token_to_id.get(&self.unk_token).copied();
        self.unknown_decode_fallback = payload.unknown_decode_fallback;
        self.unknown_decode_fallbacks = payload.unknown_decode_fallbacks;
        self.lowercase = payload.lowercase;
        self.normalization_form = payload.normalization_form;
        self.keep_whitespace_tokens = payload.keep_whitespace_tokens;
        self.log_every_merges = payload.log_every_merges;
        self.compact_every_merges = payload.compact_every_merges;
        self.small_sequence_merge_threshold = payload.small_sequence_merge_threshold;

        self.merges.clear();
        self.merge_rules.clear();
        self.merge_ranks.clear();

        for (rank, item) in payload.merges.iter().enumerate() {
            let pair = (item[0], item[1]);
            let new_id = item[2];
            self.merges.insert(pair, new_id);
            self.merge_rules.push((pair, new_id));
            self.merge_ranks.insert(pair, rank);
        }

        self.id_to_bytes.clear();
        self.byte_to_id = [u32::MAX; 256];

        let start = self.special_tokens.len() as u32;
        for byte_value in 0u32..=255 {
            let token_id = start + byte_value;
            self.byte_to_id[byte_value as usize] = token_id;
            self.id_to_bytes.insert(token_id, vec![byte_value as u8]);
        }

        for ((left, right), new_id) in &self.merge_rules {
            let mut bytes = self
                .id_to_bytes
                .get(left)
                .cloned()
                .ok_or_else(|| format!("missing bytes for token {}", left))?;
            let right_bytes = self
                .id_to_bytes
                .get(right)
                .cloned()
                .ok_or_else(|| format!("missing bytes for token {}", right))?;
            bytes.extend(right_bytes);
            self.id_to_bytes.insert(*new_id, bytes);
        }

        if self.use_unk_token && self.unk_id.is_none() {
            self.ensure_unk_token();
        }

        Ok(())
    }
}
