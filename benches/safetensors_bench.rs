use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{ArrayD, IxDyn};
use std::hint::black_box;
use tensor_engine::{
    nn::{KVCache, TransformerBlock, TransformerConfig},
    tensor::Tensor,
};

fn bench_safetensors_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformer_ops");

    // Setup for Generator Step
    let d_model = 256;
    let d_ff = 1024;
    let num_heads = 4;
    let kv_heads = 4;
    // Tiny-Llama style
    let block = TransformerBlock::new_llama_style(TransformerConfig {
        d_model,
        d_ff,
        num_heads,
        kv_heads,
        use_rope: true,
        rope_theta: 10000.0,
        rope_scale: 1.0,
        bias: false,
    })
    .expect("failed to create block");

    // Input: Batch=1, Seq=1 (incremental step), Dim=256
    let input_shape = vec![1, 1, d_model];
    let data = vec![0.0f32; 1 * 1 * d_model];
    // Create ndarray from shape and data
    let array = ArrayD::from_shape_vec(IxDyn(&input_shape), data).expect("failed to create array");
    let input_tensor = Tensor::new(array, false);

    group.bench_function("forward_block_no_cache", |b| {
        b.iter(|| {
            let mut blk = block.clone();
            black_box(blk.forward_block(black_box(&input_tensor), None))
        })
    });

    group.bench_function("forward_block_with_cache", |b| {
        let mut blk_cached = block.clone();
        let kvc = KVCache::new();
        blk_cached.set_kv_cache(kvc);

        b.iter(|| {
            // We clear cache to keep benchmark stable (step 0 -> 1 latency)
            blk_cached.clear_kv_cache();
            black_box(blk_cached.forward_block(black_box(&input_tensor), None))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_safetensors_load);
criterion_main!(benches);
