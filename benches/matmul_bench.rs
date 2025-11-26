use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2, Array4};
use rand::prelude::*;
use tensor_engine::nn::{Linear, Module};
use tensor_engine::tensor::Tensor;

fn bench_matmul(c: &mut Criterion) {
    let _ = env_logger::try_init();
    let mut group = c.benchmark_group("matmul");

    for &size in [10, 50, 100].iter() {
        let mut rng = rand::thread_rng();
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
        let mut rng = rand::thread_rng();
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
    let mut group = c.benchmark_group("ops");
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.noise_threshold(0.05);

    let mut rng = rand::thread_rng();
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

    group.finish();
}

fn bench_nn(c: &mut Criterion) {
    let _ = env_logger::try_init();
    let mut group = c.benchmark_group("nn");
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.noise_threshold(0.05);

    let mut rng = rand::thread_rng();
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

    // LayerNorm bench
    let ln = tensor_engine::nn::LayerNorm::new(10, 1, 1e-5);
    group.bench_function("layernorm_1x10", |bencher| {
        bencher.iter(|| std::hint::black_box(ln.forward(&input)))
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
    let input_c4_tensor = Tensor::new(input_c4.into_dyn(), false);
    let conv = tensor_engine::nn::Conv2D::new(3, 8, 3, 1, 1, true);
    group.bench_function("conv2d_3x64x64", |bencher| {
        bencher.iter(|| std::hint::black_box(conv.forward(&input_c4_tensor)))
    });

    let pool = tensor_engine::nn::MaxPool2D::new(2, 2);
    group.bench_function("maxpool2d_3x64x64", |bencher| {
        bencher.iter(|| std::hint::black_box(pool.forward(&input_c4_tensor)))
    });

    // NLLLoss and softmax_cross_entropy benchmarks
    let logits_data = Array2::<f32>::from_shape_fn((32, 10), |_| rng.gen());
    let logits_t = Tensor::new(logits_data.into_dyn(), true);
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

    group.finish();
}

fn bench_training_loop(c: &mut Criterion) {
    let _ = env_logger::try_init();
    let mut group = c.benchmark_group("training");
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.noise_threshold(0.05);

    // Simple training loop microbenchmark
    let mut rng = rand::thread_rng();
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
            let x_iter = Tensor::new(x.lock().data.clone(), false);
            let y_iter = Tensor::new(y.lock().data.clone(), false);
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
}

criterion_group!(
    benches,
    bench_matmul,
    bench_ops,
    bench_nn,
    bench_training_loop
);
criterion_main!(benches);
