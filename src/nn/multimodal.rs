use crate::nn::{Module, VisionTransformer, AudioEncoder, Sequential, Linear};
use crate::tensor::Tensor;
use ndarray::IxDyn;
// Projector helper methods will be implemented below
use std::time::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize};
use std::sync::atomic::Ordering as AtomicOrdering;

static DECODE_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn reset_decode_count() { DECODE_CALL_COUNT.store(0, AtomicOrdering::SeqCst); }

pub fn get_decode_count() -> usize { DECODE_CALL_COUNT.load(AtomicOrdering::SeqCst) }


pub struct MultimodalLLM {
    pub vision_encoder: VisionTransformer,
    pub text_embedding: Tensor, // [vocab, d_model]
    pub projector: Option<Projector>,
    pub audio_encoder: Option<AudioEncoder>,
    pub decoder_blocks: Vec<crate::nn::TransformerBlock>,
    pub head: Linear,
}

// Kronos native types: data variant and memory context for multimodal generation.
/// KronosData: lightweight wrappers used to represent modality-specific payloads
/// in the shared Multimodal pipeline.
pub enum KronosData {
    Dense(Tensor),
    TextIndices(Tensor),
    ImageBatch(Tensor),
    Embedding(Tensor),
}

/// Projector variants supported by MultimodalLLM.
pub enum Projector {
    Linear(Linear),
    MLP(Sequential),
}

impl Projector {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Projector::Linear(l) => l.forward(x),
            Projector::MLP(m) => m.forward(x),
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        match self {
            Projector::Linear(l) => l.parameters(),
            Projector::MLP(m) => m.parameters(),
        }
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        match self {
            Projector::Linear(l) => l.named_parameters(prefix),
            Projector::MLP(m) => m.named_parameters(prefix),
        }
    }
}

impl Module for Projector {
    fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            Projector::Linear(l) => l.forward(input),
            Projector::MLP(m) => m.forward(input),
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.named_parameters(prefix)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// ModalMemoryContext stores projected encodings for a single modality and an
/// optional attention mask. Used by the Multimodal runtime to maintain history
/// and context across decoding steps.
#[derive(Clone)]
pub struct ModalMemoryContext {
    pub modality: String,
    pub encoding: Tensor,
    pub attention_mask: Option<Tensor>,
    pub timestamp: Instant,
    /// Number of image tokens that were present in the prefill encoding. Used as causal offset during decode.
    pub prefill_image_tokens: usize,
}

impl MultimodalLLM {
    pub fn new(vision: VisionTransformer, vocab_size: usize, d_model: usize, d_ff: usize, num_heads: usize, depth: usize) -> Self {
        let text_embedding = Tensor::new(ndarray::Array::zeros(IxDyn(&[vocab_size, d_model])), true);
        let projector = None::<Projector>;
        let audio_encoder = None::<AudioEncoder>;
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth { blocks.push(crate::nn::TransformerBlock::new_decoder(d_model, d_ff, num_heads)); }
        let head = Linear::new(d_model, vocab_size, true);
        MultimodalLLM { vision_encoder: vision, text_embedding, projector, decoder_blocks: blocks, head, audio_encoder }
    }

    /// Forward pass: images -> img_tokens, text indices -> txt_tokens, concat and decode.
    /// images: [B, C, H, W], input_ids: [B, seq]
    pub fn forward(&self, images: &Tensor, input_ids: &Tensor) -> Tensor {
        let img_feats = self.vision_encoder.forward(images); // [B, Np, d_model]
        let mut img_proj = img_feats.clone();
        if let Some(p) = &self.projector {
            img_proj = match p {
                Projector::Linear(l) => l.forward(&img_feats),
                Projector::MLP(m) => m.forward(&img_feats),
            };
        }
        let txt_tokens = crate::tensor::Tensor::embedding_lookup(&self.text_embedding, input_ids);
        // Compute the causal offset (number of image tokens) BEFORE moving tensors into concat
        let offset = {
            let shape = img_proj.lock().storage.shape();
            if shape.len() == 3 { Some(shape[1]) } else { None }
        };
        let mut combined = crate::tensor::Tensor::concat(&[img_proj.clone(), txt_tokens], 1);
        for blk in &self.decoder_blocks {
            combined = blk.forward_block_with_causal_offset(&combined, offset);
        }
        let logits = self.head.forward(&combined);
        logits
    }

    /// Set a linear projector with the given output dim (inferred from vision d_model -> out_dim).
    pub fn set_projector_linear(&mut self, linear: Linear) {
        self.projector = Some(Projector::Linear(linear));
    }

    /// Set an MLP projector with a two-layer MLP: d_model->hidden->d_model
    pub fn set_projector_mlp(&mut self, hidden_dim: usize) {
        // Construct a small Sequential: Linear(d_model, hidden_dim) -> Linear(hidden_dim, d_model)
        let d_model = {
            let tmp = self.vision_encoder.patch_embed.conv.weight.lock().storage.shape().to_vec();
            // weight shape: [out_channels, in_channels, k_h, k_w]; out_channels == d_model
            let dm = if tmp.len() >= 1 { tmp[0] } else { 0 };
            dm
        };
        let mut seq = Sequential::new();
        seq = seq.add(Linear::new(d_model, hidden_dim, true));
        seq = seq.add(Linear::new(hidden_dim, d_model, true));
        self.projector = Some(Projector::MLP(seq));
    }

    /// Attach an audio encoder to the MultimodalLLM.
    pub fn set_audio_encoder(&mut self, enc: AudioEncoder) {
        self.audio_encoder = Some(enc);
    }

    /// Prefill image features and optional text prefix into a ModalMemoryContext for efficient decoding.
    /// If `input_ids` is Some, the text embeddings for the prefix will be appended after the image tokens.
    pub fn prefill(&self, images: &Tensor, input_ids: Option<&Tensor>) -> Result<ModalMemoryContext, String> {
        let img_feats = self.vision_encoder.forward(images); // [B, Np, d_model]
        let mut img_proj = img_feats.clone();
        if let Some(p) = &self.projector {
            img_proj = p.forward(&img_feats);
        }
        let image_tokens = {
            let shape = img_proj.lock().storage.shape();
            if shape.len() == 3 { shape[1] } else { 0usize }
        };
        let mut combined = img_proj.clone();
        if let Some(ids) = input_ids {
            let txt_tokens = crate::tensor::Tensor::embedding_lookup(&self.text_embedding, ids);
            combined = crate::tensor::Tensor::concat(&[img_proj.clone(), txt_tokens], 1);
        }
        // Pass through decoder blocks to produce initial hidden states (cached)
        let mut hidden = combined.clone();
        for blk in &self.decoder_blocks {
            hidden = blk.forward_block_with_causal_offset(&hidden, Some(image_tokens));
        }
        let mem = ModalMemoryContext {
            modality: "multimodal".to_string(),
            encoding: hidden,
            attention_mask: None,
            timestamp: Instant::now(),
            prefill_image_tokens: image_tokens,
        };
        Ok(mem)
    }

    /// Prefill from audio input tensor. The audio tensor is expected to be [B, C, L].
    pub fn prefill_audio(&self, audio: &Tensor, input_ids: Option<&Tensor>) -> Result<ModalMemoryContext, String> {
        let enc = match &self.audio_encoder {
            Some(e) => e.forward(audio),
            None => return Err("No audio_encoder configured for MultimodalLLM".to_string()),
        };
        // enc: [B, channels, T] -> reshape to [B, T, d_model] by mapping channels to d_model via linear if projector exists
        // For now, collapse channel and time to tokens by permuting to [B, T, C] and then project
        let shape = enc.lock().storage.shape();
        if shape.len() != 3 { return Err("Audio encoder output must be 3D [B, C, T]".to_string()); }
        let b = shape[0];
        let c = shape[1];
        let t = shape[2];
        // permute to [B, T, C]
        let audio_tokens = match enc.permute(vec![0, 2, 1]).reshape(vec![b, t, c]) {
            Ok(t) => t,
            Err(e) => { log::error!("prefill_audio: reshape audio tokens failed: {}", e); return Err("reshape audio tokens failed".to_string()); }
        };
        // Project to d_model if needed
        let mut proj = audio_tokens.clone();
        if let Some(p) = &self.projector {
            proj = match p {
                Projector::Linear(l) => l.forward(&audio_tokens),
                Projector::MLP(m) => m.forward(&audio_tokens),
            };
        }
        let image_tokens = { let shape = proj.lock().storage.shape(); if shape.len() == 3 { shape[1] } else { 0usize } };
        let mut combined = proj.clone();
        if let Some(ids) = input_ids {
            let txt_tokens = crate::tensor::Tensor::embedding_lookup(&self.text_embedding, ids);
            combined = crate::tensor::Tensor::concat(&[proj.clone(), txt_tokens], 1);
        }
        let mut hidden = combined.clone();
        for blk in &self.decoder_blocks {
            hidden = blk.forward_block_with_causal_offset(&hidden, Some(image_tokens));
        }
        let mem = ModalMemoryContext { modality: "audio".to_string(), encoding: hidden, attention_mask: None, timestamp: Instant::now(), prefill_image_tokens: image_tokens };
        Ok(mem)
    }

    /// Perform a decoding step by appending `new_input_ids` embeddings to the provided memory context,
    /// running the decoder and returning logits for the full sequence as well as an updated memory context.
    /// NOTE: Currently returns logits for the full sequence; callers can slice to recent tokens if desired.
    pub fn decode_step(&self, memory: &ModalMemoryContext, new_input_ids: &Tensor) -> Result<(Tensor, ModalMemoryContext), String> {
        DECODE_CALL_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
        // Create token embeddings for new_input_ids
        let token_emb = crate::tensor::Tensor::embedding_lookup(&self.text_embedding, new_input_ids);
        // Append to cache along sequence axis (1)
        let new_encoding = crate::tensor::Tensor::kvcache_append(&memory.encoding, &token_emb, 1);
        // Run decoder blocks with causal offset equal to number of image tokens
        let mut hidden = new_encoding.clone();
        for blk in &self.decoder_blocks {
            hidden = blk.forward_block_with_causal_offset(&hidden, Some(memory.prefill_image_tokens));
        }
        let logits = self.head.forward(&hidden);
        let new_mem = ModalMemoryContext {
            modality: memory.modality.clone(),
            encoding: hidden,
            attention_mask: memory.attention_mask.clone(),
            timestamp: Instant::now(),
            prefill_image_tokens: memory.prefill_image_tokens,
        };
        Ok((logits, new_mem))
    }

    /// Compute logits for current memory encoding without appending new tokens.
    pub fn logits_from_memory(&self, memory: &ModalMemoryContext) -> Result<Tensor, String> {
        // Re-run head on current memory encoding
        // NOTE: We could optionally re-run decoder blocks but assume memory.encoding is already the final hidden states.
        Ok(self.head.forward(&memory.encoding))
    }

    /// Sample the next token id from the logits in `memory` using temperature, top_k and top_p sampling.
    pub fn sample_next_token(&self, memory: &ModalMemoryContext, temperature: f32, top_k: Option<usize>, top_p: Option<f32>) -> Result<usize, String> {
        // Delegate to sample_candidates helper then sample
        let cands = self.sample_candidates(memory, temperature, top_k, top_p)?; // Vec<(idx, prob)>
        if cands.is_empty() { return Err("No candidates available for sampling".to_string()); }
        // Extract probs and build sampler
        let probs: Vec<f32> = cands.iter().map(|(_, p)| *p).collect();
        use rand::Rng;
        let mut rng = rand::rng();
        let sum_p: f32 = probs.iter().sum();
        if sum_p <= 0.0 { return Err("Invalid probabilities for sampling".to_string()); }
        let r = rng.random_range(0.0..sum_p);
        let mut acc = 0.0f32;
        let mut idx = 0usize;
        for (i, p) in probs.iter().enumerate() {
            acc += *p;
            if r <= acc {
                idx = i;
                break;
            }
        }
        Ok(cands[idx].0)
    }

        /// Return top candidate tokens and normalized probabilities for the next step, after applying
        /// temperature, top_k and top_p truncation. Useful for testing and deterministic inspection.
        pub fn sample_candidates(&self, memory: &ModalMemoryContext, temperature: f32, top_k: Option<usize>, top_p: Option<f32>) -> Result<Vec<(usize, f32)>, String> {
            let logits = self.logits_from_memory(memory)?; // [B, seq, vocab]
            let arr = logits.lock().storage.to_f32_array();
            if arr.ndim() != 3 { return Err("logits must be 3D [B, seq, vocab]".to_string()); }
            let b = arr.shape()[0];
            if b != 1 { return Err("batch size >1 not supported for sample_candidates".to_string()); }
            let seq = arr.shape()[1];
            let vocab = arr.shape()[2];
            let last = arr.index_axis(ndarray::Axis(1), seq - 1);
            let last0 = last.index_axis(ndarray::Axis(0), 0).to_owned(); // 1D array of len vocab
            // Apply temperature and global stability
            let mut logits_vec: Vec<f32> = last0.iter().map(|v| *v / temperature).collect();
            let global_max = logits_vec.iter().cloned().fold(std::f32::NEG_INFINITY, f32::max);
            for v in logits_vec.iter_mut() { *v -= global_max; }
            // Top-k truncation
            let mut candidate_idx: Vec<usize> = (0..vocab).collect();
            if let Some(k) = top_k {
                if k < vocab {
                    candidate_idx.sort_by(|&i, &j| logits_vec[j].partial_cmp(&logits_vec[i]).unwrap_or(std::cmp::Ordering::Equal));
                    candidate_idx.truncate(k);
                }
            }
            // Build candidate logits and sort descending
            let mut cand_logits: Vec<(usize, f32)> = candidate_idx.iter().map(|&i| (i, logits_vec[i])).collect();
            cand_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            // Top-p truncation using numerically stable local softmax
            if let Some(p_cut) = top_p {
                // compute local exps with local max for stability
                if !cand_logits.is_empty() {
                    let local_max = cand_logits.iter().map(|(_, l)| *l).fold(std::f32::NEG_INFINITY, f32::max);
                    let exps_local: Vec<f32> = cand_logits.iter().map(|(_, l)| (l - local_max).exp()).collect();
                    let sum_local: f32 = exps_local.iter().sum();
                    if sum_local == 0.0 {
                        // degenerate: keep only the top candidate
                        cand_logits.truncate(1);
                    } else {
                        let mut cum_prob = 0.0f32;
                        let mut keep: Vec<(usize, f32)> = Vec::new();
                        for (idx, (i, l)) in cand_logits.iter().enumerate() {
                            let p = exps_local[idx] / sum_local;
                            cum_prob += p;
                            keep.push((*i, *l));
                            if cum_prob >= p_cut { break; }
                        }
                        if keep.is_empty() { keep.push(cand_logits[0].clone()); }
                        cand_logits = keep;
                    }
                }
            }
            // compute normalized probabilities over final candidate set
            if cand_logits.is_empty() { return Ok(vec![]); }
            let local_max = cand_logits.iter().map(|(_, l)| *l).fold(std::f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = cand_logits.iter().map(|(_, l)| (l - local_max).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            if sum_exp == 0.0 { return Ok(vec![]); }
            let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
            let out = cand_logits.into_iter().enumerate().map(|(i, (tok, _l))| (tok, probs[i])).collect();
            Ok(out)
        }

    /// Beam search generation for a single example. Returns the best token sequence found.
    /// length_penalty: if > 0, normalized_score = raw_logprob / (seq_len ^ length_penalty)
    /// eos_token: optional end-of-sequence token id; beam will be finished if EOS is generated.
    pub fn beam_search_with_options(&self, mem: &ModalMemoryContext, max_len: usize, beam_size: usize, length_penalty: f32, eos_token: Option<usize>) -> Result<Vec<usize>, String> {
        if beam_size == 0 { return Err("beam_size must be >= 1".to_string()); }
        // Beam represented as (score, seq, memory)
        use std::cmp::Ordering;
        #[derive(Clone)]
        struct Beam { score: f32, seq: Vec<usize>, mem: ModalMemoryContext }
        // initial beam
        let init = Beam { score: 0.0, seq: vec![], mem: mem.clone() };
        let mut beams = vec![init];
        let mut completed: Vec<Beam> = Vec::new();
        for _step in 0..max_len {
            // Collect candidates from each beam without performing decode_step.
            // We will decode only the top global `beam_size` candidates to avoid redundant work.
            struct Candidate { parent_idx: usize, token: usize, logp: f32 }
            let mut all_cands: Vec<Candidate> = Vec::new();
            for (bi, b) in beams.iter().enumerate() {
                // get logits for current beam
                let logits = self.logits_from_memory(&b.mem)?; // [1, seq, vocab]
                let arr = logits.lock().storage.to_f32_array();
                let seq_len = arr.shape()[1];
                let vocab = arr.shape()[2];
                let last = arr.index_axis(ndarray::Axis(1), seq_len - 1).index_axis(ndarray::Axis(0), 0).to_owned();
                // compute softmax probs (logp) for stability
                let maxv = last.iter().cloned().fold(std::f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = last.iter().map(|v| (v - maxv).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                if sum_exp == 0.0 { continue; }
                let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
                // get top K candidates per beam; K = beam_size by default
                let mut idxs: Vec<usize> = (0..vocab).collect();
                idxs.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap_or(Ordering::Equal));
                idxs.truncate(beam_size);
                for &cand in idxs.iter() {
                    let logp = (probs[cand] + 1e-12).ln();
                    all_cands.push(Candidate { parent_idx: bi, token: cand, logp });
                }
            }
            // No candidates (e.g. all sequences had zero-sum probs) -> stop
            if all_cands.is_empty() { break; }
            // Convert to global scored list and select top `beam_size` candidates
            all_cands.sort_by(|a, b| {
                let score_a = beams[a.parent_idx].score + a.logp;
                let score_b = beams[b.parent_idx].score + b.logp;
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            if all_cands.len() > beam_size { all_cands.truncate(beam_size); }

            // For selected candidates, perform decode_step and build the new beam list.
            let mut new_beams: Vec<Beam> = Vec::new();
            for sel in all_cands.into_iter() {
                let parent = &beams[sel.parent_idx];
                let new_score = parent.score + sel.logp;
                let mut new_seq = parent.seq.clone();
                new_seq.push(sel.token);
                // If token is EOS, add to completed without decoding
                if let Some(eos) = eos_token {
                    if sel.token == eos {
                        let completed_beam = Beam { score: new_score, seq: new_seq, mem: parent.mem.clone() };
                        completed.push(completed_beam);
                        continue;
                    }
                }
                // For non-EOS, perform decode only for selected candidate to fetch the new mem
                let token_t = crate::tensor::Tensor::new(ndarray::Array::from_elem(IxDyn(&[1,1]), sel.token as f32), true);
                #[cfg(test)]
                    DECODE_CALL_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
                let (_logits2, new_mem) = self.decode_step(&parent.mem, &token_t)?;
                let beam_item = Beam { score: new_score, seq: new_seq, mem: new_mem };
                new_beams.push(beam_item);
            }
            // prune to beam_size best (consider length penalty when sorting)
            // We already selected the top `beam_size` candidates globally; but sorting by length-penalty can change selection.
            // To keep length_penalty, re-sort here by normalized score then truncate to beam_size.
            new_beams.sort_by(|a,b| {
                let norm_a = if length_penalty > 0.0 { a.score / ((a.seq.len() as f32).powf(length_penalty)) } else { a.score };
                let norm_b = if length_penalty > 0.0 { b.score / ((b.seq.len() as f32).powf(length_penalty)) } else { b.score };
                norm_b.partial_cmp(&norm_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            if new_beams.len() > beam_size { new_beams.truncate(beam_size); }
            beams = new_beams;
            // if all beams are empty and we have completed beams, we can stop
            if beams.is_empty() && !completed.is_empty() { break; }
        }
        // If we have completed beams, choose the best among completed using length_penalty.
        if !completed.is_empty() {
            completed.sort_by(|a,b| {
                let norm_a = if length_penalty > 0.0 { a.score / ((a.seq.len() as f32).powf(length_penalty)) } else { a.score };
                let norm_b = if length_penalty > 0.0 { b.score / ((b.seq.len() as f32).powf(length_penalty)) } else { b.score };
                norm_b.partial_cmp(&norm_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            return Ok(completed.first().map(|b| b.seq.clone()).unwrap_or_else(|| vec![]));
        }
        // otherwise return best current beam
        beams.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(beams.first().map(|b| b.seq.clone()).unwrap_or_else(|| vec![]))
    }
    /// For backward compatibility, keep a simple wrapper that calls options with defaults.
    pub fn beam_search(&self, mem: &ModalMemoryContext, max_len: usize, beam_size: usize) -> Result<Vec<usize>, String> {
        self.beam_search_with_options(mem, max_len, beam_size, 0.0, None)
    }

    /// Beam search for batched ModalMemoryContext; returns a vector of sequences for each batch.
    pub fn beam_search_batch_with_options(&self, mem: &ModalMemoryContext, max_len: usize, beam_size: usize, length_penalty: f32, eos_token: Option<usize>) -> Result<Vec<Vec<usize>>, String> {
        // Validate encoding shape
        let shape = mem.encoding.lock().storage.shape();
        if shape.len() != 3 { return Err("ModalMemoryContext encoding must be 3D [B, seq, d]".to_string()); }
        let batch = shape[0];
        if batch == 1 { return Ok(vec![self.beam_search_with_options(mem, max_len, beam_size, length_penalty, eos_token)?]); }
        // Initialize beams per batch
        #[derive(Clone)]
        struct Beam { score: f32, seq: Vec<usize>, mem: ModalMemoryContext }
        let mut beams_per_batch: Vec<Vec<Beam>> = Vec::with_capacity(batch);
        // Build single-batch ModalMemoryContexts for each batch row
        let enc_arr = mem.encoding.lock().storage.to_f32_array();
        for i in 0..batch {
            let single_arr = enc_arr.index_axis(ndarray::Axis(0), i).to_owned(); // [seq, d]
            let single_with_batch = single_arr.insert_axis(ndarray::Axis(0)).into_dyn();
            let single_tensor = crate::tensor::Tensor::new(single_with_batch, false);
            let mut single = mem.clone(); single.encoding = single_tensor;
            beams_per_batch.push(vec![Beam { score: 0.0, seq: vec![], mem: single }]);
        }
        let mut completed: Vec<Vec<Beam>> = (0..batch).map(|_| Vec::new()).collect();

        for _step in 0..max_len {
            // Collect all parent beam encodings and indexes across all batches
            let mut parent_encodings: Vec<ndarray::ArrayD<f32>> = Vec::new();
            let mut parent_infos: Vec<(usize, usize, usize)> = Vec::new(); // (batch_idx, beam_idx, prefill)
            for bi in 0..batch {
                for (bj, b) in beams_per_batch[bi].iter().enumerate() {
                    let mut arr = b.mem.encoding.lock().storage.to_f32_array();
                    // Normalize arr to 2D [seq, d] when pushing parent encodings for stacking
                    if arr.ndim() == 3 && arr.shape()[0] == 1 {
                        arr = arr.index_axis(ndarray::Axis(0), 0).to_owned();
                    }
                    parent_encodings.push(arr);
                    parent_infos.push((bi, bj, b.mem.prefill_image_tokens));
                }
            }
            if parent_encodings.is_empty() { break; }
            // Stack encodings into shape [N, seq, d]
            let views: Vec<ndarray::ArrayViewD<f32>> = parent_encodings.iter().map(|a| a.view()).collect();
            let stacked = ndarray::stack(ndarray::Axis(0), &views[..]).map_err(|e| format!("Failed to stack encodings: {}", e))?;
            if stacked.shape()[0] != parent_infos.len() {
                return Err(format!("stack result N={} does not match parent count {}", stacked.shape()[0], parent_infos.len()));
            }
            // debug
            // println!("stacked dims: {:?}, parent_infos: {}", stacked.shape(), parent_infos.len());
            let enc_tensor = crate::tensor::Tensor::new(stacked, false);
            let mut global_mem = mem.clone(); global_mem.encoding = enc_tensor;
            // Inspect head weights (debug)
            // debug: head weight shape available for troubleshooting
            // Compute logits for all parent beams in one call
            let logits_all = self.logits_from_memory(&global_mem)?; // [N, seq, vocab]
            let arr_all = logits_all.lock().storage.to_f32_array();
            let seq_len = arr_all.shape()[1];
            let vocab = arr_all.shape()[2];
            // debug: logits_all.shape
            // Collect candidates per parent beam
            #[derive(Clone)]
            struct Candidate { batch_idx: usize, parent_bi: usize, token: usize, logp: f32, parent_global_idx: usize }
            let mut all_cands: Vec<Candidate> = Vec::new();
            for (global_idx, (bi, bj, _pref)) in parent_infos.iter().enumerate() {
                // parent_infos.len and stacked shape checked above
                let last = arr_all.index_axis(ndarray::Axis(1), seq_len - 1).index_axis(ndarray::Axis(0), global_idx).to_owned();
                let maxv = last.iter().cloned().fold(std::f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = last.iter().map(|v| (v - maxv).exp()).collect();
                let sum_exp: f32 = exps.iter().sum(); if sum_exp == 0.0 { continue; }
                let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
                let mut idxs: Vec<usize> = (0..vocab).collect();
                idxs.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap_or(std::cmp::Ordering::Equal));
                idxs.truncate(beam_size);
                for &cand in idxs.iter() {
                    let logp = (probs[cand] + 1e-12).ln();
                    all_cands.push(Candidate { batch_idx: *bi, parent_bi: *bj, token: cand, logp, parent_global_idx: global_idx });
                }
            }
            if all_cands.is_empty() { break; }
            // Group candidates per batch and select top per batch
            let mut selected_per_batch: Vec<Vec<Candidate>> = (0..batch).map(|_| Vec::new()).collect();
            for c in all_cands.into_iter() {
                selected_per_batch[c.batch_idx].push(c);
            }
            for bi in 0..batch {
                selected_per_batch[bi].sort_by(|a,b| {
                    let sa = beams_per_batch[bi][a.parent_bi].score + a.logp;
                    let sb = beams_per_batch[bi][b.parent_bi].score + b.logp;
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                if selected_per_batch[bi].len() > beam_size { selected_per_batch[bi].truncate(beam_size); }
            }
            // Now perform vectorized decode across all selected candidates grouped by prefill_image_tokens
            // Build mapping from parent_global_idx to selected candidates
            let mut global_selected_indices: Vec<usize> = Vec::new();
            let mut selected_candidates_flat: Vec<Candidate> = Vec::new();
            // Extract EOS candidates as completed and collect only non-EOS for decoding
            for bi in 0..batch {
                for c in selected_per_batch[bi].drain(..) {
                    if let Some(eos) = eos_token {
                        if c.token == eos {
                            // finalize this completed beam
                            let parent = &beams_per_batch[bi][c.parent_bi];
                            let mut seq = parent.seq.clone(); seq.push(c.token);
                            completed[bi].push(Beam { score: parent.score + c.logp, seq, mem: parent.mem.clone() });
                            continue;
                        }
                    }
                    selected_candidates_flat.push(c.clone());
                    global_selected_indices.push(c.parent_global_idx);
                }
            }
            if selected_candidates_flat.is_empty() { break; }
            // Group by prefill_image_tokens to satisfy forward_block_with_causal_offset
            use std::collections::HashMap;
            let mut groups: HashMap<usize, Vec<(usize, &Candidate)>> = HashMap::new();
            for (i, cand) in selected_candidates_flat.iter().enumerate() {
                let parent_info = parent_infos[cand.parent_global_idx];
                let prefill_tokens = parent_info.2;
                groups.entry(prefill_tokens).or_insert_with(Vec::new).push((i, cand));
            }
            // Prepare new beams placeholder
            let mut new_beams_per_batch: Vec<Vec<Beam>> = vec![Vec::new(); batch];
            // For each group, vectorized decode all candidate parents
            for (prefill, group) in groups.into_iter() {
                // collect parent encodings
                let mut encs: Vec<ndarray::ArrayD<f32>> = Vec::new();
                let mut tokens: Vec<f32> = Vec::new();
                let mut parent_scores: Vec<f32> = Vec::new();
                let mut seqs: Vec<Vec<usize>> = Vec::new();
                let mut target_batch_idxs: Vec<usize> = Vec::new();
                for (_i_idx, cand) in group.iter() {
                    let pgi = cand.parent_global_idx;
                    // parent enc is parent_encodings[pgi]
                    let mut arr = parent_encodings[pgi].clone();
                    // parent_encodings stores 2D [seq,d]; ensure arr is 2D before stacking
                    if arr.ndim() == 3 && arr.shape()[0] == 1 {
                        arr = arr.index_axis(ndarray::Axis(0), 0).to_owned();
                    }
                    encs.push(arr);
                    tokens.push(cand.token as f32);
                    let parent = &beams_per_batch[cand.batch_idx][cand.parent_bi];
                    parent_scores.push(parent.score + cand.logp);
                    let mut seq = parent.seq.clone(); seq.push(cand.token);
                    seqs.push(seq);
                    target_batch_idxs.push(cand.batch_idx);
                }
                // stack encs into [N, seq, d]
                let views: Vec<ndarray::ArrayViewD<f32>> = encs.iter().map(|a| a.view()).collect();
                let stacked = ndarray::stack(ndarray::Axis(0), &views[..]).map_err(|e| format!("Failed to stack encodings: {}", e))?;
                let enc_tensor = crate::tensor::Tensor::new(stacked, false);
                // build token ids [N,1]
                let tokens_len = tokens.len();
                let token_arr = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[tokens_len, 1]), tokens).map_err(|e| format!("Failed to build token array: {}", e))?;
                let token_tensor = crate::tensor::Tensor::new(token_arr.into_dyn(), true);
                // create a grouped memory
                let mut group_mem = mem.clone(); group_mem.encoding = enc_tensor; group_mem.prefill_image_tokens = prefill;
                // vectorized decode step (single call)
                // decoded group info
                DECODE_CALL_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
                let (_logits2, new_mem_group) = self.decode_step(&group_mem, &token_tensor)?;
                // new_mem_group.encoding shape info
                let new_arr = new_mem_group.encoding.lock().storage.to_f32_array();
                // decoded new_mem_group shape info
                for idx in 0..seqs.len() {
                    // extract i-th slice (handle both cases where new_arr is 3D [N, seq, d] or 2D [seq, d])
                    let slice = if new_arr.ndim() == 3 {
                        new_arr.index_axis(ndarray::Axis(0), idx).to_owned()
                    } else if new_arr.ndim() == 2 {
                        // only a single sample returned
                        new_arr.clone()
                    } else { return Err("unexpected new_mem_group encoding ndim".to_string()); };
                    let single_with_batch = if slice.ndim() == 2 { slice.insert_axis(ndarray::Axis(0)).into_dyn() } else { slice.into_dyn() };
                    let t_new = crate::tensor::Tensor::new(single_with_batch, false);
                    let new_beam = Beam { score: parent_scores[idx], seq: seqs[idx].clone(), mem: ModalMemoryContext { modality: mem.modality.clone(), encoding: t_new, attention_mask: mem.attention_mask.clone(), timestamp: Instant::now(), prefill_image_tokens: prefill } };
                    new_beams_per_batch[target_batch_idxs[idx]].push(new_beam);
                }
            }
            // Add completed EOS beams if any were produced earlier (we handled EOS by not decoding them above)
            // Now prune beams per batch and set beams_per_batch for next step
            for bi in 0..batch {
                let mut nb = new_beams_per_batch[bi].clone();
                nb.sort_by(|a,b| {
                    let norm_a = if length_penalty > 0.0 { a.score / ((a.seq.len() as f32).powf(length_penalty)) } else { a.score };
                    let norm_b = if length_penalty > 0.0 { b.score / ((b.seq.len() as f32).powf(length_penalty)) } else { b.score };
                    norm_b.partial_cmp(&norm_a).unwrap_or(std::cmp::Ordering::Equal)
                });
                if nb.len() > beam_size { nb.truncate(beam_size); }
                beams_per_batch[bi] = nb;
            }
            // if all beams empty and completed exists, break
            if beams_per_batch.iter().all(|v| v.is_empty()) && completed.iter().any(|v| !v.is_empty()) { break; }
        }
        // Collect final best sequences per batch
        let mut out_vec: Vec<Vec<usize>> = Vec::with_capacity(batch);
        for bi in 0..batch {
            if !completed[bi].is_empty() {
                completed[bi].sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                out_vec.push(completed[bi][0].seq.clone());
            } else if !beams_per_batch[bi].is_empty() {
                let mut bb = beams_per_batch[bi].clone();
                bb.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                out_vec.push(bb[0].seq.clone());
            } else { out_vec.push(vec![]); }
        }
        Ok(out_vec)
    }

    /// Batched generation API. If beam_size > 1 uses batched beam search; otherwise performs sampling for each batch item.
    pub fn generate_batch(&self, images: &Tensor, prefix: Option<&Tensor>, max_len: usize, temperature: f32, top_k: Option<usize>, top_p: Option<f32>, beam_size: usize, length_penalty: f32, eos_token: Option<usize>) -> Result<Vec<Vec<usize>>, String> {
        let mem = self.prefill(images, prefix)?;
        // Determine batch size
        let shape = mem.encoding.lock().storage.shape();
        let b = shape[0];
        if beam_size > 1 {
            return self.beam_search_batch_with_options(&mem, max_len, beam_size, length_penalty, eos_token);
        }
        // Sampling per batch item
        let mut outputs: Vec<Vec<usize>> = Vec::with_capacity(b);
        for i in 0..b {
            // slice memory
            let sliced = crate::tensor::Tensor::apply(Arc::new(crate::ops::Slice::new(0, i, 1)), &[mem.encoding.clone()]);
            let mut single_mem = mem.clone();
            single_mem.encoding = sliced;
            let mut cur_mem = single_mem;
            let mut out = Vec::new();
            for _ in 0..max_len {
                let tok = self.sample_next_token(&cur_mem, temperature, top_k, top_p)?;
                out.push(tok);
                let token_t = crate::tensor::Tensor::new(ndarray::Array::from_elem(IxDyn(&[1,1]), tok as f32), true);
                DECODE_CALL_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
                let (_logits, new_mem) = self.decode_step(&cur_mem, &token_t)?;
                cur_mem = new_mem;
            }
            outputs.push(out);
        }
        Ok(outputs)
    }

    /// High-level generation API that supports sampling or beam search depending on beam_size.
    pub fn generate(&self, images: &Tensor, prefix: Option<&Tensor>, max_len: usize, temperature: f32, top_k: Option<usize>, top_p: Option<f32>, beam_size: usize) -> Result<Vec<usize>, String> {
        // Prefill with image and optional prefix
        let mem = self.prefill(images, prefix)?;
        if beam_size > 1 {
            return self.beam_search(&mem, max_len, beam_size);
        }
        // iterative sampling
        let mut cur_mem = mem;
        let mut out = Vec::new();
        for _ in 0..max_len {
            let tok = self.sample_next_token(&cur_mem, temperature, top_k, top_p)?;
            out.push(tok);
            // append token into memory
            let token_t = crate::tensor::Tensor::new(ndarray::Array::from_elem(IxDyn(&[1,1]), tok as f32), true);
            #[cfg(test)]
            DECODE_CALL_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
            let (_logits, new_mem) = self.decode_step(&cur_mem, &token_t)?;
            cur_mem = new_mem;
        }
        Ok(out)
    }
}

impl Module for MultimodalLLM {
    fn forward(&self, input: &Tensor) -> Tensor { input.clone() }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.vision_encoder.parameters();
        p.push(self.text_embedding.clone());
        if let Some(proj) = &self.projector { p.extend(proj.parameters()); }
        if let Some(enc) = &self.audio_encoder { p.extend(enc.parameters()); }
        for b in &self.decoder_blocks { p.extend(b.parameters()); }
        p.extend(self.head.parameters());
        p
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(self.vision_encoder.named_parameters(&format!("{}.vision_encoder", prefix)));
        out.push((format!("{}.text_embedding", prefix), self.text_embedding.clone()));
        if let Some(proj) = &self.projector {
            out.extend(proj.named_parameters(&format!("{}.projector", prefix)));
        }
        if let Some(enc) = &self.audio_encoder {
            out.extend(enc.named_parameters(&format!("{}.audio_encoder", prefix)));
        }
        for (i, b) in self.decoder_blocks.iter().enumerate() {
            out.extend(b.named_parameters(&format!("{}.decoder_blocks.{}", prefix, i)));
        }
        out.extend(self.head.named_parameters(&format!("{}.head", prefix)));
        out
    }
}
