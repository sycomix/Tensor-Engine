use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2, Array4};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tensor_engine::nn::{Adam, DataLoader, Linear, Module, Optimizer, SGD};
use tensor_engine::tensor::Tensor;

fn bench_matmul(c: &mut Criterion) {
    let _ = env_logger::try_init();
    // Print compile-time feature flags for traceability
    log::info!(
        "bench features: openblas={}, multi_precision={}, dtype_f16={}, dtype_bf16={}, dtype_f8={}, py_abi3={}, python_bindings={}",
        cfg!(feature = "openblas"),
        cfg!(feature = "multi_precision"),
        cfg!(feature = "dtype_f16"),
        cfg!(feature = "dtype_bf16"),
        cfg!(feature = "dtype_f8"),
        cfg!(feature = "py_abi3"),
        cfg!(feature = "python_bindings")
    );
    let mut group = c.benchmark_group("matmul");
    // use CI_BENCH env var to reduce bench times in CI
    let ci_bench = std::env::var("CI_BENCH").is_ok();
    if ci_bench {
        group.measurement_time(std::time::Duration::from_millis(250));
        group.sample_size(10);
        group.warm_up_time(std::time::Duration::from_millis(50));
    } else {
        group.measurement_time(std::time::Duration::from_secs(3));
        group.sample_size(50);
        group.warm_up_time(std::time::Duration::from_millis(200));
    }

    for &size in [10, 50, 100, 200].iter() {
        let mut rng = StdRng::seed_from_u64(42);
        let a_data = Array2::<f32>::from_shape_fn((size, size), |_| rng.gen());
        let b_data = Array2::<f32>::from_shape_fn((size, size), |_| rng.gen());

        let a = Tensor::new(a_data.clone().into_dyn(), false);
        let b_tensor = Tensor::new(b_data.clone().into_dyn(), false);

        group.bench_function(format!("matmul_{}x{}", size, size), |b| {
            b.iter(|| std::hint::black_box(a.matmul(&b_tensor)))
        });

        // Also bench forward+backward for autograd (small networks should use requires_grad)
        group.bench_function(
            format!("matmul_forward_backward_{}x{}", size, size),
            |bencher| {
                bencher.iter(|| {
                    let a_g = Tensor::new(a_data.clone().into_dyn(), true);
                    let b_g = Tensor::new(b_data.clone().into_dyn(), true);
                    let out = a_g.matmul(&b_g);
                    let s = out.sum();
                    s.backward();
                    std::hint::black_box(())
                })
            },
        );
    }

    // Non-square matmul case (rectangular): 64x128 * 128x64
    {
        let m = 64usize;
        let k = 128usize;
        let n = 64usize;
        let mut rng = StdRng::seed_from_u64(42);
        let a_data = Array2::<f32>::from_shape_fn((m, k), |_| rng.gen());
        let b_data = Array2::<f32>::from_shape_fn((k, n), |_| rng.gen());
        let a = Tensor::new(a_data.clone().into_dyn(), false);
        let b_tensor = Tensor::new(b_data.clone().into_dyn(), false);
        group.bench_function(format!("matmul_{}x{}x{}", m, k, n), |bencher| {
            bencher.iter(|| std::hint::black_box(a.matmul(&b_tensor)))
        });
        group.bench_function(
            format!("matmul_forward_backward_{}x{}x{}", m, k, n),
            |bencher| {
                bencher.iter(|| {
                    let a_g = Tensor::new(a_data.clone().into_dyn(), true);
                    let b_g = Tensor::new(b_data.clone().into_dyn(), true);
                    let out = a_g.matmul(&b_g);
                    let s = out.sum();
                    s.backward();
                    std::hint::black_box(())
                })
            },
        );
    }

    group.finish();
}

fn bench_ops(c: &mut Criterion) {
    let _ = env_logger::try_init();
    log::info!(
        "bench features: openblas={}, multi_precision={}, dtype_f16={}, dtype_bf16={}, dtype_f8={}, py_abi3={}, python_bindings={}",
        cfg!(feature = "openblas"),
        cfg!(feature = "multi_precision"),
        cfg!(feature = "dtype_f16"),
        cfg!(feature = "dtype_bf16"),
        cfg!(feature = "dtype_f8"),
        cfg!(feature = "py_abi3"),
        cfg!(feature = "python_bindings")
    );
    let mut group = c.benchmark_group("ops");
    // group configuration
    let ci_bench = std::env::var("CI_BENCH").is_ok();
    if ci_bench {
        group.measurement_time(std::time::Duration::from_millis(250));
        group.sample_size(10);
        group.warm_up_time(std::time::Duration::from_millis(50));
    } else {
        group.measurement_time(std::time::Duration::from_secs(1));
        group.warm_up_time(std::time::Duration::from_millis(200));
        group.noise_threshold(0.05);
    }

    let mut rng = StdRng::seed_from_u64(42);
    let a_data = Array1::<f32>::from_shape_fn(10, |_| rng.gen());
    let b_data = Array1::<f32>::from_shape_fn(10, |_| rng.gen());

    let a = Tensor::new(a_data.clone().into_dyn(), false);
    let b = Tensor::new(b_data.clone().into_dyn(), false);

    group.bench_function("add", |bencher| {
        bencher.iter(|| std::hint::black_box(a.add(&b)))
    });

    group.bench_function("mul", |bencher| {
        bencher.iter(|| std::hint::black_box(a.mul(&b)))
    });

    group.bench_function("relu", |bencher| {
        bencher.iter(|| std::hint::black_box(a.relu()))
    });

    group.bench_function("sigmoid", |bencher| {
        bencher.iter(|| std::hint::black_box(a.sigmoid()))
    });

    group.bench_function("tanh", |bencher| {
        bencher.iter(|| std::hint::black_box(a.tanh()))
    });

    group.bench_function("log", |bencher| {
        bencher.iter(|| std::hint::black_box(a.log()))
    });

    group.bench_function("pow", |bencher| {
        bencher.iter(|| std::hint::black_box(a.pow(2.0)))
    });

    group.bench_function("sum", |bencher| {
        bencher.iter(|| std::hint::black_box(a.sum()))
    });

    group.bench_function("mean", |bencher| {
        bencher.iter(|| std::hint::black_box(a.mean()))
    });

    // stack/concat benches
    let v1 = Tensor::new(a_data.clone().into_dyn(), false);
    let v2 = Tensor::new(b_data.clone().into_dyn(), false);
    group.bench_function("stack", |bencher| {
        bencher.iter(|| std::hint::black_box(Tensor::stack(&[v1.clone(), v2.clone()], 0)))
    });
    group.bench_function("concat", |bencher| {
        bencher.iter(|| std::hint::black_box(Tensor::concat(&[v1.clone(), v2.clone()], 0)))
    });

    // softmax/log-softmax needs a 2D input
    let logits_data = Array2::<f32>::from_shape_fn((10, 10), |_| rng.gen());
    let logits = Tensor::new(logits_data.into_dyn(), false);
    group.bench_function("softmax_axis1", |bencher| {
        bencher.iter(|| std::hint::black_box(logits.softmax(1)))
    });
    group.bench_function("log_softmax_axis1", |bencher| {
        bencher.iter(|| std::hint::black_box(logits.log_softmax(1)))
    });

    // Ternary quantization benches
    let a_small = Tensor::new(a_data.clone().into_dyn(), false);
    group.bench_function("ternary_forward", |bencher| {
        bencher.iter(|| std::hint::black_box(a_small.ternary()))
    });
    group.bench_function("ternary_forward_backward", |bencher| {
        bencher.iter(|| {
            let ag = Tensor::new(a_data.clone().into_dyn(), true);
            let out = ag.ternary();
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
    });

    // forward+backward for small ops
    group.bench_function("add_forward_backward", |bencher| {
        bencher.iter(|| {
            let a_g = Tensor::new(a_data.clone().into_dyn(), true);
            let b_g = Tensor::new(b_data.clone().into_dyn(), true);
            let out = a_g.add(&b_g);
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
    });

    group.finish();
}

fn bench_nn(c: &mut Criterion) {
    let _ = env_logger::try_init();
    log::info!(
        "bench features: openblas={}, multi_precision={}, dtype_f16={}, dtype_bf16={}, dtype_f8={}, py_abi3={}, python_bindings={}",
        cfg!(feature = "openblas"),
        cfg!(feature = "multi_precision"),
        cfg!(feature = "dtype_f16"),
        cfg!(feature = "dtype_bf16"),
        cfg!(feature = "dtype_f8"),
        cfg!(feature = "py_abi3"),
        cfg!(feature = "python_bindings")
    );
    let mut group = c.benchmark_group("nn");
    let ci_bench = std::env::var("CI_BENCH").is_ok();
    if ci_bench {
        group.measurement_time(std::time::Duration::from_millis(250));
        group.sample_size(10);
        group.warm_up_time(std::time::Duration::from_millis(50));
    } else {
        group.measurement_time(std::time::Duration::from_secs(1));
        group.warm_up_time(std::time::Duration::from_millis(200));
        group.noise_threshold(0.05);
    }

    let mut rng = StdRng::seed_from_u64(42);
    let input_data = Array2::<f32>::from_shape_fn((1, 10), |_| rng.gen());
    let input = Tensor::new(input_data.into_dyn(), false);

    let linear1 = Linear::new(10, 5, true);
    let linear2 = Linear::new(5, 1, true);

    group.bench_function("linear_10_5", |bencher| {
        bencher.iter(|| std::hint::black_box(linear1.forward(&input)))
    });

    let hidden = linear1.forward(&input);
    group.bench_function("linear_5_1", |bencher| {
        bencher.iter(|| std::hint::black_box(linear2.forward(&hidden)))
    });
    // linear backward
    group.bench_function("linear_10_5_backward", |bencher| {
        bencher.iter(|| {
            let input_g = Tensor::new(input.lock().storage.to_f32_array(), true);
            let out = linear1.forward(&input_g);
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
    });

    // LayerNorm bench
    let ln = tensor_engine::nn::LayerNorm::new(10, 1, 1e-5);
    group.bench_function("layernorm_1x10", |bencher| {
        bencher.iter(|| std::hint::black_box(ln.forward(&input)))
    });
    // layernorm backward
    group.bench_function("layernorm_1x10_backward", |bencher| {
        bencher.iter(|| {
            let inp = Tensor::new(input.lock().storage.to_f32_array(), true);
            let gp = ln.forward(&inp);
            let s = gp.sum();
            s.backward();
            std::hint::black_box(())
        })
    });

    // Dropout bench (training)
    let dp = tensor_engine::nn::Dropout::new(0.5, true);
    group.bench_function("dropout_training", |bencher| {
        bencher.iter(|| std::hint::black_box(dp.forward(&input)))
    });

    // Dropout bench (eval, should be identity)
    let dp_eval = tensor_engine::nn::Dropout::new(0.5, false);
    group.bench_function("dropout_eval", |bencher| {
        bencher.iter(|| std::hint::black_box(dp_eval.forward(&input)))
    });

    // Conv2D and MaxPool2D benches
    let input_c4 = Array4::<f32>::from_shape_fn((1, 3, 64, 64), |_| rng.gen());
    let input_c4_tensor = Tensor::new(input_c4.clone().into_dyn(), false);
    let conv = tensor_engine::nn::Conv2D::new(3, 8, 3, 1, 1, true);
    group.bench_function("conv2d_3x64x64", |bencher| {
        bencher.iter(|| std::hint::black_box(conv.forward(&input_c4_tensor)))
    });
    // conv2d forward+backward
    group.bench_function("conv2d_3x64x64_backward", |bencher| {
        bencher.iter(|| {
            let xg = Tensor::new(input_c4.clone().into_dyn(), true);
            let out = conv.forward(&xg);
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
    });

    let pool = tensor_engine::nn::MaxPool2D::new(2, 2);
    group.bench_function("maxpool2d_3x64x64", |bencher| {
        bencher.iter(|| std::hint::black_box(pool.forward(&input_c4_tensor)))
    });

    // NLLLoss and softmax_cross_entropy benchmarks
    let logits_data = Array2::<f32>::from_shape_fn((32, 10), |_| rng.gen());
    let logits_t = Tensor::new(logits_data.clone().into_dyn(), true);
    let labels_data = Array1::<f32>::from_shape_fn(32, |_| (rng.gen::<usize>() % 10) as f32);
    let labels_t = Tensor::new(labels_data.into_dyn(), false);
    group.bench_function("softmax_cross_entropy", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(logits_t.softmax_cross_entropy_with_logits(&labels_t, -1))
        })
    });
    group.bench_function("cross_entropy_with_logits", |bencher| {
        bencher.iter(|| std::hint::black_box(logits_t.cross_entropy_with_logits(&labels_t, -1)))
    });
    // For NLLLoss we need log_probs
    let log_probs = logits_t.log_softmax(1);
    group.bench_function("nll_loss", |bencher| {
        bencher.iter(|| std::hint::black_box(log_probs.nll_loss(&labels_t)))
    });
    group.bench_function("nll_loss_backward", |bencher| {
        bencher.iter(|| {
            let logits_g = Tensor::new(logits_data.clone().into_dyn(), true);
            let logp = logits_g.log_softmax(1);
            let loss = logp.nll_loss(&labels_t);
            loss.backward();
            std::hint::black_box(())
        })
    });
    group.bench_function("softmax_cross_entropy_backward", |bencher| {
        bencher.iter(|| {
            let logits_g = Tensor::new(logits_data.clone().into_dyn(), true);
            let loss = logits_g.softmax_cross_entropy_with_logits(&labels_t, -1);
            loss.backward();
            std::hint::black_box(())
        })
    });

    group.finish();
}

fn bench_training_loop(c: &mut Criterion) {
    let _ = env_logger::try_init();
    log::info!(
        "bench features: openblas={}, multi_precision={}, dtype_f16={}, dtype_bf16={}, dtype_f8={}, py_abi3={}, python_bindings={}",
        cfg!(feature = "openblas"),
        cfg!(feature = "multi_precision"),
        cfg!(feature = "dtype_f16"),
        cfg!(feature = "dtype_bf16"),
        cfg!(feature = "dtype_f8"),
        cfg!(feature = "py_abi3"),
        cfg!(feature = "python_bindings")
    );
    let mut group = c.benchmark_group("training");
    let ci_bench = std::env::var("CI_BENCH").is_ok();
    if ci_bench {
        group.measurement_time(std::time::Duration::from_millis(250));
        group.sample_size(10);
        group.warm_up_time(std::time::Duration::from_millis(50));
    } else {
        group.measurement_time(std::time::Duration::from_secs(1));
        group.warm_up_time(std::time::Duration::from_millis(200));
        group.noise_threshold(0.05);
    }

    // Simple training loop microbenchmark
    let mut rng = StdRng::seed_from_u64(42);
    let x_data = Array2::<f32>::from_shape_fn((10, 5), |_| rng.gen());
    let y_data = Array1::<f32>::from_shape_fn(10, |_| rng.gen());

    let x = Tensor::new(x_data.into_dyn(), false);
    let y = Tensor::new(y_data.into_dyn(), false);

    let model = Linear::new(5, 1, true);

    group.bench_function("training_step", |bencher| {
        bencher.iter(|| {
            let pred = std::hint::black_box(model.forward(&x));
            let diff = std::hint::black_box(pred.sub(&y));
            let loss = std::hint::black_box(diff.pow(2.0).sum());
            // Note: backward would modify the graph, so we skip it for bench
            std::hint::black_box(loss)
        })
    });

    // Training step including backward pass (small network)
    group.bench_function("training_step_backward", |bencher| {
        bencher.iter(|| {
            // recreate inputs to avoid mutation of Tensors across iterations
            let x_iter = Tensor::new(x.lock().storage.to_f32_array(), false);
            let y_iter = Tensor::new(y.lock().storage.to_f32_array(), false);
            let pred = model.forward(&x_iter);
            let diff = pred.sub(&y_iter);
            let loss = diff.pow(2.0).sum();
            loss.backward();
            // zero grads
            for p in model.parameters() {
                p.lock().grad = None;
            }
            std::hint::black_box(())
        })
    });

    group.finish();

    // optimizer benches
    let mut opt_group = c.benchmark_group("optimizers");
    if ci_bench {
        opt_group.measurement_time(std::time::Duration::from_millis(250));
        opt_group.sample_size(10);
    } else {
        opt_group.measurement_time(std::time::Duration::from_secs(1));
        opt_group.sample_size(50);
    }

    // small model
    let model = Linear::new(5, 1, true);
    let x_small = Tensor::new(x.lock().storage.to_f32_array(), false);
    let y_small = Tensor::new(y.lock().storage.to_f32_array(), false);

    // SGD step bench
    let mut sgd = SGD::new(0.01, 0.0);
    opt_group.bench_function("sgd_step", |bencher| {
        bencher.iter(|| {
            // recreate grads
            let p = model.parameters();
            // simulate grad population with a forward/backward
            let pred = model.forward(&x_small);
            let diff = pred.sub(&y_small);
            let loss = diff.pow(2.0).sum();
            loss.backward();
            sgd.step(&p);
            sgd.zero_grad(&p);
            std::hint::black_box(())
        })
    });

    // Adam step bench
    let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
    opt_group.bench_function("adam_step", |bencher| {
        bencher.iter(|| {
            let p = model.parameters();
            let pred = model.forward(&x_small);
            let diff = pred.sub(&y_small);
            let loss = diff.pow(2.0).sum();
            loss.backward();
            adam.step(&p);
            adam.zero_grad(&p);
            std::hint::black_box(())
        })
    });

    opt_group.finish();

    // DataLoader bench: measure shuffle and next_batch performance
    let mut dl_group = c.benchmark_group("dataloader");
    if ci_bench {
        dl_group.measurement_time(std::time::Duration::from_millis(250));
        dl_group.sample_size(10);
    } else {
        dl_group.measurement_time(std::time::Duration::from_secs(1));
        dl_group.sample_size(50);
    }
    // Create small dataset
    let mut dataset: Vec<(Tensor, Tensor)> = Vec::new();
    for _ in 0..64 {
        let xi = Tensor::new(x.lock().storage.to_f32_array(), false);
        let yi = Tensor::new(y.lock().storage.to_f32_array(), false);
        dataset.push((xi, yi));
    }
    let mut loader = DataLoader::new(dataset.clone(), 8);
    dl_group.bench_function("dataloader_shuffle", |bencher| {
        bencher.iter(|| {
            loader.shuffle();
            std::hint::black_box(())
        })
    });
    dl_group.bench_function("dataloader_next_batch", |bencher| {
        bencher.iter(|| {
            loader.reset();
            while let Some((_bx, _by)) = loader.next_batch() {
                // iterate across small dataset
            }
            std::hint::black_box(())
        })
    });
    dl_group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_ops,
    bench_nn,
    bench_training_loop
);
criterion_main!(benches);
