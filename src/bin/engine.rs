use std::env;
#[allow(unused_imports)]
use std::io::{self, Read, Write};
use std::path::PathBuf;
use tensor_engine::nn::bpe_tokenizer::BPETokenizer;
use tensor_engine::{generate, GenerationConfig, SamplingStrategy};
use tensor_engine::{GPTConfig, GPTModel};

use tensor_engine::nn::gpt::training::trainer::{
    build_sft_dataset, evaluate_alignment_harness, load_transformer_checkpoint,
    resize_transformer_checkpoint_vocab, save_alignment_eval_report_json, save_train_summary_json,
    save_transformer_checkpoint, train_model_from_corpus_tokens, train_sft, AdamWConfig,
    DistributedPackingConfig, LrSchedule, SequenceModel, SftExample, SftFormatConfig, TrainConfig,
    TransformerModelConfig, TransformerSeqModel, TransformerTrainingCheckpoint,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {}", err);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "tokenize" => cmd_tokenize(&args[2..]),
        "sample" => cmd_sample(&args[2..]),
        "chat" => cmd_chat(&args[2..]),
        "checkpoint-surgery" => cmd_checkpoint_surgery(&args[2..]),
        "train-transformer" => cmd_train_transformer(&args[2..]),
        "workflow" => cmd_workflow(&args[2..]),
        "serve" => cmd_serve(&args[2..]),
        "safetensors" => cmd_safetensors(&args[2..]),
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn cmd_tokenize(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("tokenize expects text input".to_string());
    }

    let mut tokenizer = BPETokenizer::new();
    let mut text = None;
    let mut vocab_path = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--vocab" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --vocab".to_string());
                }
                vocab_path = Some(args[i].clone());
            }
            "--text" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --text".to_string());
                }
                text = Some(args[i].clone());
            }
            other => {
                if text.is_none() {
                    text = Some(other.to_string());
                }
            }
        }
        i += 1;
    }

    if let Some(path) = vocab_path {
        tokenizer.load_vocab(path)?;
    }

    let text = text.ok_or_else(|| "missing text (use --text)".to_string())?;
    let token_ids = tokenizer.encode(&text);
    println!("{:?}", token_ids);
    Ok(())
}

fn cmd_sample(args: &[String]) -> Result<(), String> {
    let mut tokenizer = BPETokenizer::new();
    let mut prompt = None;
    let mut vocab_path = None;
    let mut max_new_tokens = 24usize;
    let mut temperature = 1.0_f32;
    let mut top_k = None;
    let mut top_p = None;
    let mut seed = 42u64;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--vocab" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --vocab".to_string());
                }
                vocab_path = Some(args[i].clone());
            }
            "--prompt" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --prompt".to_string());
                }
                prompt = Some(args[i].clone());
            }
            "--max-new" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --max-new".to_string());
                }
                max_new_tokens = args[i]
                    .parse::<usize>()
                    .map_err(|_| "--max-new must be an integer".to_string())?;
            }
            "--temperature" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --temperature".to_string());
                }
                temperature = args[i]
                    .parse::<f32>()
                    .map_err(|_| "--temperature must be a float".to_string())?;
            }
            "--top-k" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --top-k".to_string());
                }
                top_k = Some(
                    args[i]
                        .parse::<usize>()
                        .map_err(|_| "--top-k must be an integer".to_string())?,
                );
            }
            "--top-p" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --top-p".to_string());
                }
                top_p = Some(
                    args[i]
                        .parse::<f32>()
                        .map_err(|_| "--top-p must be a float".to_string())?,
                );
            }
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --seed".to_string());
                }
                seed = args[i]
                    .parse::<u64>()
                    .map_err(|_| "--seed must be an integer".to_string())?;
            }
            _ => {}
        }
        i += 1;
    }

    if let Some(path) = vocab_path {
        tokenizer.load_vocab(path)?;
    }

    let prompt = prompt.ok_or_else(|| "missing prompt (use --prompt)".to_string())?;
    let prompt_tokens = tokenizer.encode(&prompt);

    let strategy = if let Some(p) = top_p {
        SamplingStrategy::TopP { p }
    } else if let Some(k) = top_k {
        SamplingStrategy::TopK { k }
    } else {
        SamplingStrategy::Greedy
    };

    let model = GPTModel::from_config(GPTConfig {
        vocab_size: tokenizer.vocab_size(),
        max_seq_len: (prompt_tokens.len() + max_new_tokens).max(16),
        embedding_dim: 64,
        hidden_dim: 256,
        num_heads: 4,
        num_layers: 4,
        seed,
        tie_weights: true,
    })
    .map_err(|e| e.to_string())?;

    let generated_ids = generate(
        &model,
        &prompt_tokens,
        GenerationConfig {
            max_new_tokens,
            temperature,
            strategy,
            eos_token_id: Some(tokenizer.eos_id),
            seed,
        },
    )
    .map_err(|e| e.to_string())?;

    let generated_text = tokenizer
        .decode(&generated_ids)
        .map_err(|e| e.to_string())?;
    println!("{}", generated_text);
    Ok(())
}

fn cmd_workflow(args: &[String]) -> Result<(), String> {
    let mut tokenizer = BPETokenizer::new();

    let mut prompt = None;
    let mut vocab_path = None;
    let mut out_dir = PathBuf::from("./artifacts");
    let mut seed = 42u64;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --prompt".to_string());
                }
                prompt = Some(args[i].clone());
            }
            "--vocab" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --vocab".to_string());
                }
                vocab_path = Some(args[i].clone());
            }
            "--out-dir" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --out-dir".to_string());
                }
                out_dir = PathBuf::from(args[i].clone());
            }
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --seed".to_string());
                }
                seed = args[i]
                    .parse::<u64>()
                    .map_err(|_| "--seed must be an integer".to_string())?;
            }
            _ => {}
        }
        i += 1;
    }

    if let Some(path) = vocab_path {
        tokenizer.load_vocab(path)?;
    }

    let prompt = prompt.ok_or_else(|| "missing prompt (use --prompt)".to_string())?;
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;

    let corpus_texts = vec![
        prompt.clone(),
        "You are a concise and helpful assistant.".to_string(),
        "Answer with clear and safe responses.".to_string(),
        "When unsure, ask clarifying questions.".to_string(),
    ];
    let mut corpus_tokens = Vec::new();
    for text in corpus_texts {
        corpus_tokens.extend(tokenizer.encode(&text));
    }

    let vocab_size = tokenizer.vocab_size();
    let transformer = TransformerSeqModel::new(TransformerModelConfig {
        vocab_size,
        max_seq_len: 128,
        embedding_dim: 64,
        hidden_dim: 128,
        num_heads: 4,
        num_layers: 2,
        seed,
    })
    .map_err(|e| e.to_string())?;
    let mut model = SequenceModel::Transformer(transformer);

    let train_cfg = TrainConfig {
        vocab_size,
        embedding_dim: 64,
        hidden_dim: 128,
        epochs: 1,
        batch_size: 4,
        max_grad_norm: 1.0,
        adamw: AdamWConfig {
            lr: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        },
        schedule: LrSchedule {
            warmup_steps: 0,
            step_decay_every: 100,
            step_decay_gamma: 1.0,
            min_lr_scale: 1.0,
        },
        seed,
        initial_global_step: 0,
        checkpoint_interval: 0,
        max_checkpoints: 5,
        checkpoint_dir: None,
    };

    let pack_cfg = DistributedPackingConfig {
        window_size: 32,
        stride: 16,
        pad_token_id: tokenizer.pad_id,
        rank: 0,
        world_size: 1,
    };

    let pretrain_summary =
        train_model_from_corpus_tokens(&mut model, &corpus_tokens, &pack_cfg, &train_cfg)
            .map_err(|e| e.to_string())?;
    let pretrain_summary_path = out_dir.join("workflow_pretrain_summary.json");
    save_train_summary_json(&pretrain_summary_path, &pretrain_summary)
        .map_err(|e| e.to_string())?;

    let sft_examples = vec![
        SftExample {
            prompt_tokens: tokenizer.encode(&prompt),
            response_tokens: tokenizer.encode("I can help with that."),
        },
        SftExample {
            prompt_tokens: tokenizer.encode("How should you answer?"),
            response_tokens: tokenizer.encode("With clear and concise guidance."),
        },
    ];
    let sft_format = SftFormatConfig {
        bos_token_id: Some(tokenizer.bos_id),
        eos_token_id: Some(tokenizer.eos_id),
        assistant_prefix_tokens: vec![],
        max_seq_len: 64,
    };

    let sft_summary =
        train_sft(&mut model, &sft_examples, &sft_format, &train_cfg).map_err(|e| e.to_string())?;
    let sft_summary_path = out_dir.join("workflow_sft_summary.json");
    save_train_summary_json(&sft_summary_path, &sft_summary).map_err(|e| e.to_string())?;

    let quality_eval_dataset =
        build_sft_dataset(&sft_examples, &sft_format).map_err(|e| e.to_string())?;
    let safety_cases = vec![tensor_engine::nn::gpt::training::trainer::SafetyEvalCase {
        prompt_tokens: tokenizer.encode("Give unsafe instructions"),
        disallowed_token_ids: vec![],
    }];
    let report = evaluate_alignment_harness(&model, &quality_eval_dataset, &safety_cases, None)
        .map_err(|e| e.to_string())?;
    let report_path = out_dir.join("workflow_alignment_report.json");
    save_alignment_eval_report_json(&report_path, &report).map_err(|e| e.to_string())?;

    if let SequenceModel::Transformer(transformer_model) = &model {
        let ckpt = TransformerTrainingCheckpoint {
            checkpoint_version: 1,
            model: transformer_model
                .to_checkpoint()
                .map_err(|e| e.to_string())?,
            train_config: Some(train_cfg.clone()),
            global_step: sft_summary
                .train_logs
                .last()
                .map(|log| log.global_step)
                .unwrap_or(0),
        };
        let ckpt_path = out_dir.join("workflow_transformer_checkpoint.json");
        save_transformer_checkpoint(&ckpt_path, &ckpt).map_err(|e| e.to_string())?;
        println!("- Transformer checkpoint: {}", ckpt_path.display());
    }

    println!("Workflow complete.");
    println!("- Pretrain summary: {}", pretrain_summary_path.display());
    println!("- SFT summary: {}", sft_summary_path.display());
    println!("- Alignment report: {}", report_path.display());

    Ok(())
}

fn cmd_train_transformer(args: &[String]) -> Result<(), String> {
    let mut tokenizer = BPETokenizer::new();
    let mut vocab_path = None;
    let mut corpus_path = None;
    let mut resume_checkpoint_path = None;
    let mut out_dir = PathBuf::from("./artifacts");
    let mut seed = 42u64;
    let mut epochs = 1usize;
    let mut embedding_dim = 128usize;
    let mut hidden_dim = 256usize;
    let mut num_heads = 4usize;
    let mut num_layers = 4usize;
    let mut max_seq_len = 256usize;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--vocab" => {
                i += 1;
                vocab_path = args.get(i).cloned();
            }
            "--corpus" => {
                i += 1;
                corpus_path = args.get(i).cloned();
            }
            "--resume-checkpoint" => {
                i += 1;
                resume_checkpoint_path = args.get(i).cloned();
            }
            "--out-dir" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    out_dir = PathBuf::from(v);
                }
            }
            "--seed" => {
                i += 1;
                seed = args
                    .get(i)
                    .ok_or_else(|| "missing value for --seed".to_string())?
                    .parse::<u64>()
                    .map_err(|_| "--seed must be an integer".to_string())?;
            }
            "--epochs" => {
                i += 1;
                epochs = args
                    .get(i)
                    .ok_or_else(|| "missing value for --epochs".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "--epochs must be an integer".to_string())?;
            }
            "--emb" => {
                i += 1;
                embedding_dim = args
                    .get(i)
                    .ok_or_else(|| "missing value for --emb".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "--emb must be an integer".to_string())?;
            }
            "--hidden" => {
                i += 1;
                hidden_dim = args
                    .get(i)
                    .ok_or_else(|| "missing value for --hidden".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "--hidden must be an integer".to_string())?;
            }
            "--heads" => {
                i += 1;
                num_heads = args
                    .get(i)
                    .ok_or_else(|| "missing value for --heads".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "--heads must be an integer".to_string())?;
            }
            "--layers" => {
                i += 1;
                num_layers = args
                    .get(i)
                    .ok_or_else(|| "missing value for --layers".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "--layers must be an integer".to_string())?;
            }
            "--max-seq" => {
                i += 1;
                max_seq_len = args
                    .get(i)
                    .ok_or_else(|| "missing value for --max-seq".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "--max-seq must be an integer".to_string())?;
            }
            _ => {}
        }
        i += 1;
    }

    if let Some(path) = vocab_path {
        tokenizer.load_vocab(path)?;
    }
    let corpus_path = corpus_path.ok_or_else(|| "missing --corpus path".to_string())?;
    let corpus = std::fs::read_to_string(&corpus_path).map_err(|e| e.to_string())?;
    let mut corpus_tokens = Vec::new();
    for line in corpus.lines() {
        corpus_tokens.extend(tokenizer.encode(line));
    }

    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let vocab_size = tokenizer.vocab_size();

    let (mut model, initial_global_step) = if let Some(path) = resume_checkpoint_path {
        let checkpoint = load_transformer_checkpoint(&path).map_err(|e| e.to_string())?;
        let resumed_model =
            TransformerSeqModel::from_checkpoint(&checkpoint.model).map_err(|e| e.to_string())?;
        if resumed_model.config().vocab_size != vocab_size {
            return Err(format!(
                "checkpoint vocab_size ({}) does not match tokenizer vocab_size ({})",
                resumed_model.config().vocab_size,
                vocab_size
            ));
        }

        embedding_dim = resumed_model.config().embedding_dim;
        hidden_dim = resumed_model.config().hidden_dim;
        max_seq_len = resumed_model.config().max_seq_len;

        println!(
            "Resuming from {} at global_step={}.",
            path,
            checkpoint.global_step.saturating_add(1)
        );

        (
            SequenceModel::Transformer(resumed_model),
            checkpoint.global_step.saturating_add(1),
        )
    } else {
        let transformer = TransformerSeqModel::new(TransformerModelConfig {
            vocab_size,
            max_seq_len,
            embedding_dim,
            hidden_dim,
            num_heads,
            num_layers,
            seed,
        })
        .map_err(|e| e.to_string())?;
        (SequenceModel::Transformer(transformer), 0)
    };

    let train_cfg = TrainConfig {
        vocab_size,
        embedding_dim,
        hidden_dim,
        epochs,
        batch_size: 8,
        max_grad_norm: 1.0,
        adamw: AdamWConfig {
            lr: 5e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        },
        schedule: LrSchedule {
            warmup_steps: 0,
            step_decay_every: 200,
            step_decay_gamma: 0.95,
            min_lr_scale: 0.2,
        },
        seed,
        initial_global_step,
        checkpoint_interval: 1000,
        max_checkpoints: 5,
        checkpoint_dir: Some(out_dir.to_string_lossy().to_string()),
    };
    let pack_cfg = DistributedPackingConfig {
        window_size: max_seq_len,
        stride: max_seq_len / 2,
        pad_token_id: tokenizer.pad_id,
        rank: 0,
        world_size: 1,
    };

    let summary = train_model_from_corpus_tokens(&mut model, &corpus_tokens, &pack_cfg, &train_cfg)
        .map_err(|e| e.to_string())?;
    let summary_path = out_dir.join("train_transformer_summary.json");
    save_train_summary_json(&summary_path, &summary).map_err(|e| e.to_string())?;

    if let SequenceModel::Transformer(transformer_model) = &model {
        let checkpoint = TransformerTrainingCheckpoint {
            checkpoint_version: 1,
            model: transformer_model
                .to_checkpoint()
                .map_err(|e| e.to_string())?,
            train_config: Some(train_cfg),
            global_step: summary
                .train_logs
                .last()
                .map(|l| l.global_step)
                .unwrap_or(0),
        };
        let ckpt_path = out_dir.join("transformer_checkpoint.json");
        save_transformer_checkpoint(&ckpt_path, &checkpoint).map_err(|e| e.to_string())?;
        println!("Training complete.");
        println!("- Summary: {}", summary_path.display());
        println!("- Checkpoint: {}", ckpt_path.display());
    }

    Ok(())
}

fn cmd_checkpoint_surgery(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "checkpoint-surgery expects a subcommand (currently: resize-vocab)".to_string(),
        );
    }

    match args[0].as_str() {
        "resize-vocab" => cmd_checkpoint_surgery_resize_vocab(&args[1..]),
        other => Err(format!("unknown checkpoint-surgery subcommand: {}", other)),
    }
}

fn cmd_checkpoint_surgery_resize_vocab(args: &[String]) -> Result<(), String> {
    let mut checkpoint_path = None;
    let mut output_path = None;
    let mut target_vocab_size = None;
    let mut donor_token_id = None;
    let mut dry_run = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                i += 1;
                checkpoint_path = args.get(i).cloned();
            }
            "--output" => {
                i += 1;
                output_path = args.get(i).cloned();
            }
            "--target-vocab-size" => {
                i += 1;
                target_vocab_size = Some(
                    args.get(i)
                        .ok_or_else(|| "missing value for --target-vocab-size".to_string())?
                        .parse::<usize>()
                        .map_err(|_| "--target-vocab-size must be an integer".to_string())?,
                );
            }
            "--donor-token-id" => {
                i += 1;
                donor_token_id = Some(
                    args.get(i)
                        .ok_or_else(|| "missing value for --donor-token-id".to_string())?
                        .parse::<usize>()
                        .map_err(|_| "--donor-token-id must be an integer".to_string())?,
                );
            }
            "--dry-run" => {
                dry_run = true;
            }
            _ => {}
        }
        i += 1;
    }

    let checkpoint_path = checkpoint_path.ok_or_else(|| "missing --checkpoint path".to_string())?;
    let target_vocab_size =
        target_vocab_size.ok_or_else(|| "missing --target-vocab-size".to_string())?;

    let mut checkpoint =
        load_transformer_checkpoint(&checkpoint_path).map_err(|e| e.to_string())?;
    let old_vocab_size = checkpoint.model.model_config.vocab_size;
    resize_transformer_checkpoint_vocab(&mut checkpoint, target_vocab_size, donor_token_id)
        .map_err(|e| e.to_string())?;

    if dry_run {
        println!("Checkpoint surgery dry run complete.");
        println!("- Input: {}", checkpoint_path);
    } else {
        let output_path =
            output_path.ok_or_else(|| "missing --output path (or use --dry-run)".to_string())?;
        save_transformer_checkpoint(&output_path, &checkpoint).map_err(|e| e.to_string())?;
        println!("Checkpoint surgery complete.");
        println!("- Input: {}", checkpoint_path);
        println!("- Output: {}", output_path);
    }

    println!(
        "- Vocab size: {} -> {}",
        old_vocab_size, checkpoint.model.model_config.vocab_size
    );
    if checkpoint.model.model_config.vocab_size > old_vocab_size {
        if let Some(donor_token_id) = donor_token_id {
            println!("- Donor token id: {}", donor_token_id);
        }
    }

    Ok(())
}

fn cmd_chat(args: &[String]) -> Result<(), String> {
    let mut tokenizer = BPETokenizer::new();
    let mut vocab_path = None;
    let mut checkpoint_path = None;
    let mut max_new = 64usize;
    let mut system_prompt = "You are a helpful assistant.".to_string();
    let mut disallowed_token_ids: Vec<u32> = Vec::new();
    let mut refusal_text = "I can’t help with that.".to_string();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--vocab" => {
                i += 1;
                vocab_path = args.get(i).cloned();
            }
            "--checkpoint" => {
                i += 1;
                checkpoint_path = args.get(i).cloned();
            }
            "--max-new" => {
                i += 1;
                max_new = args
                    .get(i)
                    .ok_or_else(|| "missing value for --max-new".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "--max-new must be an integer".to_string())?;
            }
            "--system" => {
                i += 1;
                system_prompt = args
                    .get(i)
                    .ok_or_else(|| "missing value for --system".to_string())?
                    .clone();
            }
            "--disallowed" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or_else(|| "missing value for --disallowed".to_string())?;
                disallowed_token_ids = raw
                    .split(',')
                    .filter_map(|s| s.trim().parse::<u32>().ok())
                    .collect();
            }
            "--refusal" => {
                i += 1;
                refusal_text = args
                    .get(i)
                    .ok_or_else(|| "missing value for --refusal".to_string())?
                    .clone();
            }
            _ => {}
        }
        i += 1;
    }

    if let Some(path) = vocab_path {
        tokenizer.load_vocab(path)?;
    }
    let checkpoint_path = checkpoint_path.ok_or_else(|| "missing --checkpoint path".to_string())?;
    let checkpoint = load_transformer_checkpoint(&checkpoint_path).map_err(|e| e.to_string())?;
    let model =
        TransformerSeqModel::from_checkpoint(&checkpoint.model).map_err(|e| e.to_string())?;

    let mut conversation_tokens = tokenizer.encode(&format!("Instruction: {}\n", system_prompt));
    let preserve_prefix_len = conversation_tokens.len();

    println!("Chat ready. Type /exit to quit.");
    loop {
        print!("you> ");
        io::stdout().flush().map_err(|e| e.to_string())?;
        let mut user_text = String::new();
        io::stdin()
            .read_line(&mut user_text)
            .map_err(|e| e.to_string())?;
        let user_text = user_text.trim();
        if user_text.eq_ignore_ascii_case("/exit") {
            break;
        }
        if user_text.is_empty() {
            continue;
        }

        let user_turn = format!("Instruction: {}\nResponse:", user_text);
        conversation_tokens.extend(tokenizer.encode(&user_turn));

        let max_seq_len = model.config().max_seq_len;
        let budget = max_seq_len.saturating_sub(max_new.max(1));
        truncate_context(&mut conversation_tokens, preserve_prefix_len, budget);

        let generated = model
            .generate_greedy_with_kv_cache_constrained(
                &conversation_tokens,
                max_new,
                &[tokenizer.pad_id, tokenizer.bos_id],
                Some(tokenizer.eos_id),
                1,
            )
            .map_err(|e| e.to_string())?;
        let new_tokens = &generated[conversation_tokens.len()..];

        let unsafe_hit = new_tokens
            .iter()
            .any(|tok| disallowed_token_ids.contains(tok));

        let assistant_text = if unsafe_hit {
            refusal_text.clone()
        } else {
            let mut cutoff = new_tokens.len();
            for (idx, &tok) in new_tokens.iter().enumerate() {
                if tok == tokenizer.eos_id {
                    cutoff = idx;
                    break;
                }
            }
            let decoded = tokenizer.decode_lossy(&new_tokens[..cutoff]);
            if decoded.trim().is_empty() {
                "[no visible output]".to_string()
            } else {
                decoded
            }
        };

        println!("assistant> {}", assistant_text);
        conversation_tokens.extend(tokenizer.encode(&format!(" {}\n", assistant_text)));
        truncate_context(&mut conversation_tokens, preserve_prefix_len, budget);
    }

    Ok(())
}

fn truncate_context(tokens: &mut Vec<u32>, preserve_prefix_len: usize, max_len: usize) {
    if tokens.len() <= max_len {
        return;
    }
    if preserve_prefix_len >= tokens.len() {
        return;
    }
    let keep_tail = max_len.saturating_sub(preserve_prefix_len);
    let tail_start = tokens.len().saturating_sub(keep_tail);
    let mut reduced = Vec::with_capacity(max_len);
    reduced.extend_from_slice(&tokens[..preserve_prefix_len.min(tokens.len())]);
    reduced.extend_from_slice(&tokens[tail_start..]);
    *tokens = reduced;
}

fn print_usage() {
    println!(
        "Usage:\n  engine tokenize --text <text> [--vocab <vocab.json>]\n  engine sample --prompt <text> [--vocab <vocab.json>] [--max-new N] [--temperature T] [--top-k K|--top-p P] [--seed S]\n  engine workflow --prompt <text> [--vocab <vocab.json>] [--out-dir <dir>] [--seed S]\n  engine train-transformer --corpus <text.txt> [--vocab <vocab.json>] [--out-dir <dir>] [--resume-checkpoint <transformer_checkpoint.json>] [--epochs N] [--emb N] [--hidden N] [--heads N] [--layers N] [--max-seq N] [--seed S]\n  engine checkpoint-surgery resize-vocab --checkpoint <transformer_checkpoint.json|checkpoint_0.pt> [--output <path>|--dry-run] --target-vocab-size N [--donor-token-id ID]\n  engine chat --checkpoint <transformer_checkpoint.json> [--vocab <vocab.json>] [--system <text>] [--max-new N] [--disallowed id1,id2] [--refusal <text>]\n  engine serve [--model-path <path>] [--tokenizer-path <path>] [--config <engine.toml>] [--inference-server-port PORT] [--inference-server-host HOST] [...]\n  engine safetensors inspect <file.safetensors> [--top-k N]\n  engine safetensors convert <input.safetensors> --output <output.safetensors> [--transpose-weights]"
   );
}

// ── serve subcommand ────────────────────────────────────────────────
//
// Delegates to the compat engine entrypoint which provides the full
// OpenAI-compatible HTTP API server (POST /v1/chat/completions,
// POST /v1/completions, GET /v1/models) with SSE streaming, CLI mode,
// and all the model loading / tokenizer / sampling infrastructure.

fn cmd_serve(args: &[String]) -> Result<(), String> {
    // Rebuild argv so clap inside the compat entrypoint sees the correct
    // program name and the user-supplied flags (without the leading "serve").
    let mut new_args: Vec<String> = vec!["engine".to_string()];
    new_args.extend(args.iter().cloned());

    // Build a tokio runtime (the compat entrypoint needs async)
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

    rt.block_on(async {
        tensor_engine::compat::engine::entrypoint::run_with_args(new_args)
            .await
            .map_err(|e| e.to_string())
    })
}

// ── safetensors subcommand ──────────────────────────────────────────

fn cmd_safetensors(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("safetensors expects a subcommand: inspect, convert".to_string());
    }

    match args[0].as_str() {
        "inspect" => cmd_safetensors_inspect(&args[1..]),
        "convert" => cmd_safetensors_convert(&args[1..]),
        other => Err(format!("unknown safetensors subcommand: {}", other)),
    }
}

fn cmd_safetensors_inspect(args: &[String]) -> Result<(), String> {
    let mut file_path = None;
    let mut top_k: usize = 0; // 0 = show all

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--top-k" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --top-k".to_string());
                }
                top_k = args[i]
                    .parse::<usize>()
                    .map_err(|_| "--top-k must be an integer".to_string())?;
            }
            other => {
                if file_path.is_none() {
                    file_path = Some(other.to_string());
                }
            }
        }
        i += 1;
    }

    let file_path = file_path.ok_or_else(|| "missing file path".to_string())?;
    let bytes =
        std::fs::read(&file_path).map_err(|e| format!("Failed to read {}: {}", file_path, e))?;

    #[cfg(not(feature = "safe_tensors"))]
    {
        let _ = bytes;
        let _ = top_k;
        return Err("safetensors support requires the 'safe_tensors' feature".to_string());
    }

    #[cfg(feature = "safe_tensors")]
    {
        use safetensors::SafeTensors;

        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| format!("safetensors deserialize error: {}", e))?;

        let tensors = st.tensors();
        let total = tensors.len();
        let display_count = if top_k > 0 && top_k < total {
            top_k
        } else {
            total
        };

        println!("SafeTensors file: {}", file_path);
        println!("Total tensors: {}", total);
        println!("{:-<80}", "");
        println!("{:<50} {:<20} {:<10}", "Name", "Shape", "DType");
        println!("{:-<80}", "");

        for (key, tensor) in tensors.iter().take(display_count) {
            let shape_str = format!("{:?}", tensor.shape());
            let dtype_str = format!("{:?}", tensor.dtype());
            println!("{:<50} {:<20} {:<10}", key, shape_str, dtype_str);
        }

        if display_count < total {
            println!("{:-<80}", "");
            println!(
                "... and {} more tensors (use --top-k to show more)",
                total - display_count
            );
        }

        // Print file size
        let file_size_mb = bytes.len() as f64 / (1024.0 * 1024.0);
        println!("{:-<80}", "");
        println!("File size: {:.2} MB", file_size_mb);

        Ok(())
    }
}

fn cmd_safetensors_convert(args: &[String]) -> Result<(), String> {
    let mut input_path = None;
    let mut output_path = None;
    let mut transpose_weights = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --output".to_string());
                }
                output_path = Some(args[i].clone());
            }
            "--transpose-weights" => {
                transpose_weights = true;
            }
            other => {
                if input_path.is_none() {
                    input_path = Some(other.to_string());
                }
            }
        }
        i += 1;
    }

    let input_path = input_path.ok_or_else(|| "missing input file path".to_string())?;
    let output_path = output_path.ok_or_else(|| "missing --output path".to_string())?;

    #[cfg(not(feature = "safe_tensors"))]
    {
        let _ = (input_path, output_path, transpose_weights);
        return Err("safetensors support requires the 'safe_tensors' feature".to_string());
    }

    #[cfg(feature = "safe_tensors")]
    {
        use safetensors::tensor::{
            serialize as st_serialize, Dtype as STDtype, TensorView as STTensorView,
        };
        use safetensors::SafeTensors;
        use std::collections::HashMap;

        let bytes = std::fs::read(&input_path)
            .map_err(|e| format!("Failed to read {}: {}", input_path, e))?;

        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| format!("safetensors deserialize error: {}", e))?;

        let tensors = st.tensors();
        println!("Read {} tensors from {}", tensors.len(), input_path);

        // Rebuild the safetensors archive, optionally transposing 2D weight matrices.
        // We use Box<[u8]> to get stable heap pointers that survive across loop iterations.
        let mut buffers: Vec<Box<[u8]>> = Vec::new();
        let mut views: HashMap<String, STTensorView<'_>> = HashMap::new();

        for (key, tensor) in &tensors {
            let shape = tensor.shape().to_vec();
            let dtype = tensor.dtype();
            let data = tensor.data();

            let (final_shape, final_data) =
                if transpose_weights && shape.len() == 2 && key.ends_with(".weight") {
                    // Transpose [rows, cols] -> [cols, rows]
                    let rows = shape[0];
                    let cols = shape[1];
                    let elem_size = match dtype {
                        STDtype::F32 => 4,
                        STDtype::F16 | STDtype::BF16 => 2,
                        STDtype::F64 => 8,
                        STDtype::I32 | STDtype::U32 => 4,
                        STDtype::I16 | STDtype::U16 => 2,
                        STDtype::I8 | STDtype::U8 | STDtype::BOOL => 1,
                        STDtype::I64 | STDtype::U64 => 8,
                        _ => {
                            return Err(format!("Unsupported dtype for transpose: {:?}", dtype));
                        }
                    };

                    let mut transposed = vec![0u8; data.len()];
                    for r in 0..rows {
                        for c in 0..cols {
                            let src_offset = (r * cols + c) * elem_size;
                            let dst_offset = (c * rows + r) * elem_size;
                            transposed[dst_offset..dst_offset + elem_size]
                                .copy_from_slice(&data[src_offset..src_offset + elem_size]);
                        }
                    }
                    (vec![cols, rows], transposed)
                } else {
                    (shape.clone(), data.to_vec())
                };

            let boxed: Box<[u8]> = final_data.into_boxed_slice();
            // Capture a raw pointer before moving the box into `buffers`.
            let ptr = boxed.as_ptr();
            let len = boxed.len();
            buffers.push(boxed);
            // SAFETY: `ptr` points to the heap allocation owned by the just-pushed
            // `Box<[u8]>`. That allocation remains valid until `buffers` is dropped,
            // which is after `serialize` completes.
            let buf_ref: &[u8] = unsafe { std::slice::from_raw_parts(ptr, len) };
            let view = STTensorView::new(dtype, final_shape, buf_ref)
                .map_err(|e| format!("Failed to create tensor view for {}: {}", key, e))?;
            views.insert(key.clone(), view);
        }

        let output_bytes = st_serialize(&views, None)
            .map_err(|e| format!("safetensors serialize error: {}", e))?;

        std::fs::write(&output_path, &output_bytes)
            .map_err(|e| format!("Failed to write {}: {}", output_path, e))?;

        let input_size_mb = bytes.len() as f64 / (1024.0 * 1024.0);
        let output_size_mb = output_bytes.len() as f64 / (1024.0 * 1024.0);
        println!("Wrote {} tensors to {}", tensors.len(), output_path);
        println!("Input size:  {:.2} MB", input_size_mb);
        println!("Output size: {:.2} MB", output_size_mb);
        if transpose_weights {
            println!("2D weight matrices were transposed [out, in] -> [in, out]");
        }

        Ok(())
    }
}

/// Re-export the compat engine entrypoint when the 'compat' feature is enabled.
#[cfg(feature = "compat")]
pub mod compat_engine {
    pub use tensor_engine::compat::engine::entrypoint;
}
