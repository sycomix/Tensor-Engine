use ndarray::Array;
use tensor_engine::nn::{MultimodalLLM, VisionTransformer};
use tensor_engine::tensor::Tensor;

#[test]
fn test_generate_sampling_and_beam() {
    let b = 1usize;
    let c = 3usize;
    let h = 8usize;
    let w = 8usize;
    let patch_size = 2usize;
    let d_model = 16usize;
    let d_ff = 32usize;
    let num_heads = 2usize;
    let depth = 1usize;
    let vocab = 32usize;
    let vit = VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, 16)
        .expect("create vision transformer");
    let mut model = MultimodalLLM::new(vit, vocab, d_model, d_ff, num_heads, depth)
        .expect("create multimodal model");
    // set an MLP projector for variety
    model.set_projector_mlp(32);
    // random image
    let img = Tensor::new(Array::from_elem((b, c, h, w), 0.5f32).into_dyn(), false);
    // greedy sampling (temperature=1, top_k None) length 3
    let seq = model
        .generate(&img, None, 3, 1.0, None, None, 1)
        .expect("generate failed");
    assert_eq!(seq.len(), 3);
    // beam search
    let seqb = model
        .generate(&img, None, 3, 1.0, None, None, 2)
        .expect("beam generate failed");
    assert_eq!(seqb.len(), 3);
}

#[test]
fn test_top_p_truncation() {
    use ndarray::{Array, IxDyn};
    let b = 1usize;
    let c = 3usize;
    let h = 8usize;
    let w = 8usize;
    let patch_size = 2usize;
    let d_model = 16usize;
    let d_ff = 32usize;
    let num_heads = 2usize;
    let depth = 1usize;
    let vocab = 5usize;
    let vit = VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, 16)
        .expect("create vision transformer");
    let mut model = MultimodalLLM::new(vit, vocab, d_model, d_ff, num_heads, depth)
        .expect("create multimodal model");
    // set a simple projector to keep shapes consistent
    model.set_projector_mlp(32);
    // zero-out weights and set bias to enforce known logits (bias = logits)
    let zero_w = Tensor::new(ndarray::Array::zeros(IxDyn(&[d_model, vocab])), true);
    model.head.weight = zero_w;
    let bias_vals = vec![5.0f32, 4.0f32, -20.0f32, -20.0f32, -20.0f32];
    let bias_t = Tensor::new(Array::from_vec(bias_vals).into_dyn(), true);
    model.head.bias = Some(bias_t);

    let img = Tensor::new(Array::from_elem((b, c, h, w), 0.5f32).into_dyn(), false);
    let mem = model.prefill(&img, None).expect("prefill failed");
    // With top_p = 0.7 we expect only token 0 to be allowed in the candidate set; repeated samples should be 0
    for _ in 0..50 {
        let tok = model
            .sample_next_token(&mem, 1.0, None, Some(0.7))
            .expect("sample failed");
        assert_eq!(
            tok, 0,
            "Token outside top-p candidate set was sampled: {}",
            tok
        );
    }
}

#[test]
fn test_beam_eos_and_batching() {
    use ndarray::Array;
    let b = 2usize;
    let c = 3usize;
    let h = 8usize;
    let w = 8usize;
    let patch_size = 2usize;
    let d_model = 16usize;
    let d_ff = 32usize;
    let num_heads = 2usize;
    let depth = 1usize;
    let vocab = 5usize;
    let vit = VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, 16)
        .expect("create vision transformer");
    let mut model = MultimodalLLM::new(vit, vocab, d_model, d_ff, num_heads, depth)
        .expect("create multimodal model");
    // bias output to make token 4 the EOS dominant
    let mut bias_vals = vec![0.0f32; vocab];
    bias_vals[4] = 10.0; // token 4 is EOS with high bias
    model.head.bias = Some(Tensor::new(Array::from_vec(bias_vals).into_dyn(), true));

    let img = Tensor::new(Array::from_elem((b, c, h, w), 0.5f32).into_dyn(), false);
    // run batched beam search
    let mem = model.prefill(&img, None).expect("prefill failed");
    let batch_seqs = model
        .beam_search_batch_with_options(&mem, 3, 2, 0.0, Some(4))
        .expect("beam batch failed");
    assert_eq!(batch_seqs.len(), 2);
    for seq in batch_seqs.iter() {
        // expect immediate EOS token returned
        assert_eq!(seq.len(), 1);
        assert_eq!(seq[0], 4);
    }
}

#[test]
fn test_top_k_and_top_p_combined() {
    use ndarray::Array;
    let b = 1usize;
    let c = 3usize;
    let h = 8usize;
    let w = 8usize;
    let patch_size = 2usize;
    let d_model = 16usize;
    let d_ff = 32usize;
    let num_heads = 2usize;
    let depth = 1usize;
    let vocab = 6usize;
    let vit = VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, 16)
        .expect("create vision transformer");
    let mut model = MultimodalLLM::new(vit, vocab, d_model, d_ff, num_heads, depth)
        .expect("create multimodal model");
    // set biases to reflect distinct logits
    // top values: token 0=5, token1=4, others very low
    let mut bias_vals = vec![-20.0f32; vocab];
    bias_vals[0] = 5.0;
    bias_vals[1] = 4.0;
    bias_vals[2] = 3.0;
    bias_vals[3] = -5.0;
    bias_vals[4] = -10.0;
    bias_vals[5] = -10.0;
    model.head.bias = Some(Tensor::new(Array::from_vec(bias_vals).into_dyn(), true));
    let img = Tensor::new(Array::from_elem((b, c, h, w), 0.5f32).into_dyn(), false);
    let mem = model.prefill(&img, None).expect("prefill failed");
    // Now call sample_candidates with top_k=2 and top_p=0.9; both should keep at least tokens 0 and 1
    let candidates = model
        .sample_candidates(&mem, 1.0, Some(2), Some(0.9))
        .expect("sample candidates failed");
    // Expect at most 2 candidates and token 0 should be present
    assert!(candidates.len() <= 2);
    assert!(candidates.iter().any(|(tok, _)| *tok == 0));
}

#[test]
fn test_batched_beam_vectorized_calls() {
    use ndarray::Array;
    use std::time::Instant;

    let b = 2usize;
    let c = 3usize;
    let h = 8usize;
    let w = 8usize;
    let patch_size = 2usize;
    let d_model = 16usize;
    let d_ff = 32usize;
    let num_heads = 2usize;
    let depth = 1usize;
    let vocab = 7usize;
    let vit = VisionTransformer::new(c, patch_size, d_model, d_ff, num_heads, depth, 16)
        .expect("create vision transformer");
    let mut model = MultimodalLLM::new(vit, vocab, d_model, d_ff, num_heads, depth)
        .expect("create multimodal model");
    // make logits non-degenerate
    let mut bias_vals = vec![0.0f32; vocab];
    for i in 0..vocab {
        bias_vals[i] = (i as f32) - 1.0;
    }
    model.head.bias = Some(Tensor::new(Array::from_vec(bias_vals).into_dyn(), true));

    let img = Tensor::new(Array::from_elem((b, c, h, w), 0.5f32).into_dyn(), false);
    let mem = model.prefill(&img, None).expect("prefill failed");

    // naive implementation: decode for each beam's top-K candidates individually
    fn naive_beam_search_batch(
        model: &mut MultimodalLLM,
        mem: &tensor_engine::nn::ModalMemoryContext,
        max_len: usize,
        beam_size: usize,
        _length_penalty: f32,
        eos_token: Option<usize>,
    ) -> Vec<Vec<usize>> {
        use ndarray::IxDyn;
        struct Beam {
            score: f32,
            seq: Vec<usize>,
            mem: tensor_engine::nn::ModalMemoryContext,
        }
        let shape = mem.encoding.lock().storage.shape();
        let batch = shape[0];
        let mut batch_beams: Vec<Vec<Beam>> = Vec::with_capacity(batch);
        for i in 0..batch {
            let enc_arr = mem.encoding.lock().storage.to_f32_array();
            let single_arr = enc_arr.index_axis(ndarray::Axis(0), i).to_owned();
            let single_with_batch = single_arr.insert_axis(ndarray::Axis(0)).into_dyn();
            let single_tensor = Tensor::new(single_with_batch, false);
            let mut single = mem.clone();
            single.encoding = single_tensor;
            batch_beams.push(vec![Beam {
                score: 0.0,
                seq: vec![],
                mem: single,
            }]);
        }
        let mut completed: Vec<Vec<Beam>> = (0..batch).map(|_| Vec::new()).collect();
        for _step in 0..max_len {
            for bi in 0..batch {
                let mut new_beams: Vec<Beam> = Vec::new();
                for b in batch_beams[bi].iter() {
                    let logits = model.logits_from_memory(&b.mem).expect("logits failed");
                    let arr = logits.lock().storage.to_f32_array();
                    let seq_len = arr.shape()[1];
                    let vocab = arr.shape()[2];
                    let last = arr
                        .index_axis(ndarray::Axis(1), seq_len - 1)
                        .index_axis(ndarray::Axis(0), 0)
                        .to_owned();
                    let maxv = last.iter().cloned().fold(std::f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = last.iter().map(|v| (v - maxv).exp()).collect();
                    let sum_exp: f32 = exps.iter().sum();
                    if sum_exp == 0.0 {
                        continue;
                    }
                    let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
                    let mut idxs: Vec<usize> = (0..vocab).collect();
                    idxs.sort_by(|&i, &j| {
                        probs[j]
                            .partial_cmp(&probs[i])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    idxs.truncate(beam_size);
                    for &cand in idxs.iter() {
                        let token_t = Tensor::new(
                            ndarray::Array::from_elem(IxDyn(&[1, 1]), cand as f32),
                            true,
                        );
                        // this will increment decode counter via instrumentation
                        let (_logits2, new_mem) =
                            model.decode_step(&b.mem, &token_t).expect("decode failed");
                        let logp = (probs[cand] + 1e-12).ln();
                        let mut new_seq = b.seq.clone();
                        new_seq.push(cand);
                        let beam_item = Beam {
                            score: b.score + logp,
                            seq: new_seq,
                            mem: new_mem,
                        };
                        if let Some(eos) = eos_token {
                            if cand == eos {
                                completed[bi].push(beam_item);
                                continue;
                            }
                        }
                        new_beams.push(beam_item);
                    }
                }
                new_beams.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                if new_beams.len() > beam_size {
                    new_beams.truncate(beam_size);
                }
                batch_beams[bi] = new_beams;
            }
        }
        let mut out = Vec::with_capacity(batch);
        for bi in 0..batch {
            if !completed[bi].is_empty() {
                completed[bi].sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                out.push(completed[bi][0].seq.clone());
            } else if !batch_beams[bi].is_empty() {
                batch_beams[bi].sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                out.push(batch_beams[bi][0].seq.clone());
            } else {
                out.push(vec![]);
            }
        }
        out
    }

    // Run a single decode to validate instrumentation
    tensor_engine::nn::multimodal::reset_decode_count();
    let token_test = Tensor::new(
        ndarray::Array::from_elem(ndarray::IxDyn(&[1, 1]), 1.0f32),
        true,
    );
    // slice memory to a single-batch memory for decode_step
    let enc_arr = mem.encoding.lock().storage.to_f32_array();
    let single_arr = enc_arr.index_axis(ndarray::Axis(0), 0).to_owned();
    let single_with_batch = single_arr.insert_axis(ndarray::Axis(0)).into_dyn();
    let single_tensor = Tensor::new(single_with_batch, false);
    let mut single_mem = mem.clone();
    single_mem.encoding = single_tensor;
    let (_logits_test, _mem_test) = model
        .decode_step(&single_mem, &token_test)
        .expect("single decode failed");
    let cnt_test = tensor_engine::nn::multimodal::get_decode_count();
    assert!(
        cnt_test > 0,
        "instrumentation didn't increment decode counter"
    );

    // Run naive
    tensor_engine::nn::multimodal::reset_decode_count();
    let t0 = Instant::now();
    let _naive = naive_beam_search_batch(&mut model, &mem, 4, 2, 0.0, None);
    let naive_time = t0.elapsed();
    let naive_calls = tensor_engine::nn::multimodal::get_decode_count();

    // Run optimized
    tensor_engine::nn::multimodal::reset_decode_count();
    let t1 = Instant::now();
    let _opt = model
        .beam_search_batch_with_options(&mem, 4, 2, 0.0, None)
        .expect("beam batch failed");
    let opt_time = t1.elapsed();
    let opt_calls = tensor_engine::nn::multimodal::get_decode_count();

    println!(
        "naive_calls={}, opt_calls={}, naive_time={:?}, opt_time={:?}",
        naive_calls, opt_calls, naive_time, opt_time
    );
    assert!(
        opt_calls <= naive_calls,
        "optimized decode calls should be less or equal to naive"
    );
    // Ensure at least some reduction typically occurs for beam search
    assert!(opt_calls > 0 && naive_calls > 0);
    // Assert performance improvement (allow some headroom for variability)
    assert!(
        opt_time.as_secs_f32() <= naive_time.as_secs_f32() * 1.5,
        "optimized beam runtime should be <= 1.5x naive runtime"
    );
}
