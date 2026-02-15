use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;
use tensor_engine::generation::sampling::Sampler;
use tensor_engine::generation::speculative::{SpeculativeModel, SpeculativeSampler};
use tensor_engine::nn::transformer_cleaned::Llama;
use tensor_engine::ops::Slice;
use tensor_engine::tensor::Tensor;

fn slice_last(t: &Tensor) -> Tensor {
    let shape = t.lock().storage.shape().to_vec();
    let axis = shape.len() - 1;
    let len = shape[axis];
    if len == 0 {
        return t.clone();
    }
    Tensor::apply(Arc::new(Slice::new(axis, len - 1, 1)), &[t.clone()][..])
}

fn standard_generate(
    model: &mut Llama,
    input: &Tensor,
    max_new_tokens: usize,
    sampler: &mut Sampler,
) -> Tensor {
    let mut all_tokens = input.clone();

    // Prime
    let _ = model.forward_t(&input);

    let mut n_generated = 0;
    while n_generated < max_new_tokens {
        let loop_input = slice_last(&all_tokens);
        let logits = model.forward_t(&loop_input);
        let sample = sampler.sample(&logits);

        let token = Tensor::new(
            ArrayD::from_elem(IxDyn(&[1usize, 1][..]), sample.token as f32),
            false,
        );
        all_tokens = Tensor::concat(&[all_tokens, token.clone()][..], 1);
        n_generated += 1;
    }
    all_tokens
}

fn main() {
    // 1. Setup Models
    // Target Model (Large)
    let mut target_model = Llama::new(
        1000, // vocab
        256,  // d_model
        4,    // num_layers
        1024, // d_ff
        4,    // num_heads
        4,    // kv_heads
    )
    .unwrap();
    // Enable KV cache!
    target_model.set_kv_cache(true);

    // Create a copy for standard generation (re-create to ensure fresh state)
    let mut target_model_base = Llama::new(1000, 256, 4, 1024, 4, 4).unwrap();
    target_model_base.set_kv_cache(true);

    // Draft Model (Small)
    let mut draft_model = Llama::new(
        1000, // vocab
        128,  // d_model
        2,    // num_layers
        512,  // d_ff
        2,    // num_heads
        2,    // kv_heads
    )
    .unwrap();
    draft_model.set_kv_cache(true);

    // 2. Initialize Sampler
    let sampler = Sampler::new(0.7, 10, 0.9, 42);
    let mut sampler2 = Sampler::new(0.7, 10, 0.9, 42);

    let gamma = 4;
    let mut spec_sampler = SpeculativeSampler::new(
        Box::new(draft_model),
        Box::new(target_model),
        gamma,
        sampler,
    );

    // 3. Create Dummy Input
    let input = Tensor::ones(&[1usize, 5][..]);

    // 4. Benchmark Speculative
    println!("Starting Speculative Generation...");
    let start_spec = std::time::Instant::now();
    let output_spec = spec_sampler.generate(&input, 50);
    let duration_spec = start_spec.elapsed();
    println!("Speculative Time: {:?}", duration_spec);
    println!(
        "Speculative Tokens/sec: {:.2}",
        50.0 / duration_spec.as_secs_f64()
    );

    // 5. Benchmark Standard
    println!("Starting Standard Generation...");
    let start_std = std::time::Instant::now();
    let output_std = standard_generate(&mut target_model_base, &input, 50, &mut sampler2);
    let duration_std = start_std.elapsed();
    println!("Standard Time: {:?}", duration_std);
    println!(
        "Standard Tokens/sec: {:.2}",
        50.0 / duration_std.as_secs_f64()
    );

    // Speedup
    let speedup = duration_std.as_secs_f64() / duration_spec.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);

    // Print outputs (first 10)
    let spec_data = output_spec.to_f32_array();
    let std_data = output_std.to_f32_array();
    println!(
        "Spec Output (first 10): {:?}",
        spec_data.iter().take(10).collect::<Vec<_>>()
    );
    println!(
        "Std Output (first 10): {:?}",
        std_data.iter().take(10).collect::<Vec<_>>()
    );
}
