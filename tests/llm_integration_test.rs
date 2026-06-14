//! Integration tests for GPT model and generation

use tensor_engine::{GPTModel, GPTConfig, generate, GenerationConfig, SamplingStrategy};
use tensor_engine::nn::bpe_tokenizer::BPETokenizer;

#[test]
fn test_gpt_model_creation() {
    let config = GPTConfig {
        vocab_size: 1000,
        max_seq_len: 128,
        embedding_dim: 64,
        hidden_dim: 256,
        num_heads: 4,
        num_layers: 2,
        seed: 42,
        tie_weights: true,
    };
    
    let model = GPTModel::from_config(config);
    assert!(model.is_ok());
}

#[test]
fn test_tokenizer_basic() {
    let mut tokenizer = BPETokenizer::new();
    
    // Test encoding
    let text = "Hello, world!";
    let tokens = tokenizer.encode(text);
    assert!(!tokens.is_empty());
    
    // Test decoding
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_generation_greedy() {
    let config = GPTConfig {
        vocab_size: 100,
        max_seq_len: 64,
        embedding_dim: 32,
        hidden_dim: 128,
        num_heads: 2,
        num_layers: 1,
        seed: 42,
        tie_weights: true,
    };
    
    let model = GPTModel::from_config(config).unwrap();
    let prompt = vec![1, 2, 3];
    
    let gen_config = GenerationConfig {
        max_new_tokens: 5,
        temperature: 1.0,
        strategy: SamplingStrategy::Greedy,
        eos_token_id: None,
        seed: 42,
    };
    
    let result = generate(&model, &prompt, gen_config);
    assert!(result.is_ok());
    
    let generated = result.unwrap();
    assert_eq!(generated.len(), prompt.len() + 5);
}

#[test]
fn test_generation_with_eos() {
    let config = GPTConfig {
        vocab_size: 100,
        max_seq_len: 64,
        embedding_dim: 32,
        hidden_dim: 128,
        num_heads: 2,
        num_layers: 1,
        seed: 42,
        tie_weights: true,
    };
    
    let model = GPTModel::from_config(config).unwrap();
    let prompt = vec![1, 2, 3];
    
    let gen_config = GenerationConfig {
        max_new_tokens: 10,
        temperature: 1.0,
        strategy: SamplingStrategy::Greedy,
        eos_token_id: Some(0),
        seed: 42,
    };
    
    let result = generate(&model, &prompt, gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_sampling_strategies() {
    let config = GPTConfig {
        vocab_size: 100,
        max_seq_len: 64,
        embedding_dim: 32,
        hidden_dim: 128,
        num_heads: 2,
        num_layers: 1,
        seed: 42,
        tie_weights: true,
    };
    
    let model = GPTModel::from_config(config).unwrap();
    let prompt = vec![1, 2, 3];
    
    // Test TopK sampling
    let gen_config = GenerationConfig {
        max_new_tokens: 5,
        temperature: 1.0,
        strategy: SamplingStrategy::TopK { k: 10 },
        eos_token_id: None,
        seed: 42,
    };
    
    let result = generate(&model, &prompt, gen_config);
    assert!(result.is_ok());
    
    // Test TopP sampling
    let gen_config = GenerationConfig {
        max_new_tokens: 5,
        temperature: 1.0,
        strategy: SamplingStrategy::TopP { p: 0.9 },
        eos_token_id: None,
        seed: 42,
    };
    
    let result = generate(&model, &prompt, gen_config);
    assert!(result.is_ok());
}
