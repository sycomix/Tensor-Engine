use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2, Array3, Array4, Array5};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tensor_engine::nn::{AbsolutePositionalEmbedding, MultiHeadAttention};
use tensor_engine::nn::transformer::BiasFunction;
use tensor_engine::nn::{Adam, DataLoader, Linear, Module, Optimizer, SGD};
use tensor_engine::ops::{
    AdaptiveAvgPool2D as AdaptiveAvgPool2DOp, AvgPool2D as AvgPool2DOp, Conv1D as Conv1DOp,
    Conv3D as Conv3DOp, ConvTranspose2D as ConvTranspose2DOp,
    DepthwiseSeparableConv2D as DSConv2DOp,
};
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

            // quantized matmul bench: right-hand side is stored as INT8 with scale
            let qbuf = Tensor::new_with_dtype(b_data.clone().into_dyn(), false, tensor_engine::dtype::DType::I8);
            group.bench_function(format!("quantized_matmul_{}x{}", size, size), |bencher| {
                bencher.iter(|| std::hint::black_box(a.quantized_matmul(&qbuf)))
            });
            // rowwise quantized
            let qbuf_row = Tensor::new_with_dtype(b_data.clone().into_dyn(), false, tensor_engine::dtype::DType::I8Rowwise);
            group.bench_function(format!("quantized_matmul_rowwise_{}x{}", size, size), |bencher| {
                bencher.iter(|| std::hint::black_box(a.quantized_matmul(&qbuf_row)))
            });
            // blockwise quantized
            let qbuf_block = Tensor::new_with_dtype(b_data.clone().into_dyn(), false, tensor_engine::dtype::DType::I8Blockwise);
            group.bench_function(format!("quantized_matmul_blockwise_{}x{}", size, size), |bencher| {
                bencher.iter(|| std::hint::black_box(a.quantized_matmul(&qbuf_block)))
            });

            // bench with other dtype representations if features are enabled
            #[cfg(feature = "dtype_f16")]
            {
                let a_f16 = Tensor::new_with_dtype(a_data.clone().into_dyn(), false, tensor_engine::dtype::DType::F16);
                let b_f16 = Tensor::new_with_dtype(b_data.clone().into_dyn(), false, tensor_engine::dtype::DType::F16);
                group.bench_function(format!("matmul_f16_{}x{}", size, size), |bencher| {
                    bencher.iter(|| std::hint::black_box(a_f16.matmul(&b_f16)))
                });
            }
            #[cfg(feature = "dtype_bf16")]
            {
                let a_bf16 = Tensor::new_with_dtype(a_data.clone().into_dyn(), false, tensor_engine::dtype::DType::BF16);
                let b_bf16 = Tensor::new_with_dtype(b_data.clone().into_dyn(), false, tensor_engine::dtype::DType::BF16);
                group.bench_function(format!("matmul_bf16_{}x{}", size, size), |bencher| {
                    bencher.iter(|| std::hint::black_box(a_bf16.matmul(&b_bf16)))
                });
            }
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

    // Large sizes - gate under CI_BENCH=="" (only run when not in CI or when CI_BENCH not set)
    if !ci_bench {
        for &size in [512usize, 1024usize].iter() {
            let mut rng = StdRng::seed_from_u64(42);
            let a_data = Array2::<f32>::from_shape_fn((size, size), |_| rng.gen());
            let b_data = Array2::<f32>::from_shape_fn((size, size), |_| rng.gen());
            let a = Tensor::new(a_data.clone().into_dyn(), false);
            let b_tensor = Tensor::new(b_data.clone().into_dyn(), false);
            group.bench_function(format!("matmul_{}x{}", size, size), |b| {
                b.iter(|| std::hint::black_box(a.matmul(&b_tensor)))
            });
            // quantized case for large sizes
            let qbuf = Tensor::new_with_dtype(b_data.clone().into_dyn(), false, tensor_engine::dtype::DType::I8);
            group.bench_function(format!("quantized_matmul_{}x{}", size, size), |bencher| {
                bencher.iter(|| std::hint::black_box(a.quantized_matmul(&qbuf)))
            });
            // also bench 'dequantized' path to compare
            group.bench_function(format!("dequantized_matmul_{}x{}", size, size), |bencher| {
                bencher.iter(|| {
                    let bf = qbuf.lock().storage.to_f32_array();
                    let bf_t = Tensor::new(bf.into_dyn(), false);
                    std::hint::black_box(a.matmul(&bf_t))
                })
            });
        }
    }

    group.finish();

    // Quantized vs Dequantization path microbench
    let mut q_group = c.benchmark_group("quantized_dequant_compare");
    if ci_bench {
        q_group.measurement_time(std::time::Duration::from_millis(250));
        q_group.sample_size(10);
    } else {
        q_group.measurement_time(std::time::Duration::from_secs(2));
        q_group.sample_size(30);
    }
    // reuse 128x128
    let size = 128usize;
    let mut rng = StdRng::seed_from_u64(42);
    let a_data = Array2::<f32>::from_shape_fn((size, size), |_| rng.gen());
    let b_data = Array2::<f32>::from_shape_fn((size, size), |_| rng.gen());
    let a = Tensor::new(a_data.clone().into_dyn(), false);
    // b_f32 not used directly; dequantized path uses `b_i8` storage to simulate dequantize
    let b_i8 = Tensor::new_with_dtype(b_data.clone().into_dyn(), false, tensor_engine::dtype::DType::I8);
    q_group.bench_function("quantized_matmul_int8_128x128", |bencher| {
        bencher.iter(|| std::hint::black_box(a.quantized_matmul(&b_i8)))
    });
    q_group.bench_function("dequantized_matmul_128x128", |bencher| {
        bencher.iter(|| {
            // emulate dequantize then matmul (float path)
            let bf = b_i8.lock().storage.to_f32_array();
            let bf_t = Tensor::new(bf.into_dyn(), false);
            std::hint::black_box(a.matmul(&bf_t))
        })
    });
    q_group.finish();
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

    // New benches for Conv1D, Conv3D, DepthwiseSeparableConv2D, ConvTranspose2D, AvgPool2D, AdaptiveAvgPool2D
    // Conv1D
    let input_c3_1d = Array3::<f32>::from_shape_fn((1, 3, 128), |_| rng.gen());
    let input_c3_1d_t = Tensor::new(input_c3_1d.clone().into_dyn(), false);
    let conv1d_op = Conv1DOp::new(1, 1); // stride, padding (op-level)
    let conv1d_op_arc: std::sync::Arc<dyn tensor_engine::ops::Operation + Send + Sync> =
        std::sync::Arc::new(conv1d_op);
    let weight_c1 = Tensor::new(
        ndarray::Array::from_shape_fn((8, 3, 3), |_| rng.gen()).into_dyn(),
        true,
    );
    let bias_c1 = Tensor::new(
        ndarray::Array::from_shape_fn((8,), |_| 0.0).into_dyn(),
        true,
    );
    group.bench_function("conv1d_3x128", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&conv1d_op_arc),
                &[input_c3_1d_t.clone(), weight_c1.clone(), bias_c1.clone()],
            ))
        })
    });
    group.bench_function("conv1d_3x128_backward", |bencher| {
        bencher.iter(|| {
            let xg = Tensor::new(input_c3_1d.clone().into_dyn(), true);
            let out = tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&conv1d_op_arc),
                &[xg.clone(), weight_c1.clone(), bias_c1.clone()],
            );
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
    });

    // Conv3D
    let input_c3_3d = Array5::<f32>::from_shape_fn((1, 3, 8, 32, 32), |_| rng.gen());
    let input_c3_3d_t = Tensor::new(input_c3_3d.clone().into_dyn(), false);
    let conv3d_op = Conv3DOp::new(1, 1);
    let conv3d_op_arc: std::sync::Arc<dyn tensor_engine::ops::Operation + Send + Sync> =
        std::sync::Arc::new(conv3d_op);
    let weight_c3 = Tensor::new(
        ndarray::Array::from_shape_fn((8, 3, 3, 3, 3), |_| rng.gen()).into_dyn(),
        true,
    );
    let bias_c3 = Tensor::new(
        ndarray::Array::from_shape_fn((8,), |_| 0.0).into_dyn(),
        true,
    );
    group.bench_function("conv3d_3x8x32x32", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&conv3d_op_arc),
                &[input_c3_3d_t.clone(), weight_c3.clone(), bias_c3.clone()],
            ))
        })
    });
    group.bench_function("conv3d_3x8x32x32_backward", |bencher| {
        bencher.iter(|| {
            let xg = Tensor::new(input_c3_3d.clone().into_dyn(), true);
            let out = tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&conv3d_op_arc),
                &[xg.clone(), weight_c3.clone(), bias_c3.clone()],
            );
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
    });

    // Depthwise Separable Conv2D
    let ds_conv_op = DSConv2DOp::new(1, 1);
    let ds_conv_op_arc: std::sync::Arc<dyn tensor_engine::ops::Operation + Send + Sync> =
        std::sync::Arc::new(ds_conv_op);
    let dw = Tensor::new(
        ndarray::Array::from_shape_fn((3, 1, 3, 3), |_| rng.gen()).into_dyn(),
        true,
    );
    let pw = Tensor::new(
        ndarray::Array::from_shape_fn((8, 3, 1, 1), |_| rng.gen()).into_dyn(),
        true,
    );
    let bias_ds = Tensor::new(
        ndarray::Array::from_shape_fn((8,), |_| 0.0).into_dyn(),
        true,
    );
    group.bench_function("depthwise_separable_conv2d_3x64x64", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&ds_conv_op_arc),
                &[
                    input_c4_tensor.clone(),
                    dw.clone(),
                    pw.clone(),
                    bias_ds.clone(),
                ],
            ))
        })
    });
    group.bench_function("depthwise_separable_conv2d_3x64x64_backward", |bencher| {
        bencher.iter(|| {
            let xg = Tensor::new(input_c4.clone().into_dyn(), true);
            let out = tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&ds_conv_op_arc),
                &[xg.clone(), dw.clone(), pw.clone(), bias_ds.clone()],
            );
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
    });

    // ConvTranspose2D
    let conv_t_op = ConvTranspose2DOp::new(1, 1);
    let conv_t_op_arc: std::sync::Arc<dyn tensor_engine::ops::Operation + Send + Sync> =
        std::sync::Arc::new(conv_t_op);
    let wt = Tensor::new(
        ndarray::Array::from_shape_fn((8, 3, 3, 3), |_| rng.gen()).into_dyn(),
        true,
    );
    let bias_t = Tensor::new(
        ndarray::Array::from_shape_fn((8,), |_| 0.0).into_dyn(),
        true,
    );
    group.bench_function("convtranspose2d_3x64x64", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&conv_t_op_arc),
                &[input_c4_tensor.clone(), wt.clone(), bias_t.clone()],
            ))
        })
    });
    group.bench_function("convtranspose2d_3x64x64_backward", |bencher| {
        bencher.iter(|| {
            let xg = Tensor::new(input_c4.clone().into_dyn(), true);
            let out = tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&conv_t_op_arc),
                &[xg.clone(), wt.clone(), bias_t.clone()],
            );
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
    });

    // AvgPool2D
    let avg_pool_op = AvgPool2DOp {
        kernel_size: 2,
        stride: 2,
    };
    let avg_pool_op_arc: std::sync::Arc<dyn tensor_engine::ops::Operation + Send + Sync> =
        std::sync::Arc::new(avg_pool_op);
    group.bench_function("avgpool2d_3x64x64", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&avg_pool_op_arc),
                &[input_c4_tensor.clone()],
            ))
        })
    });
    group.bench_function("avgpool2d_3x64x64_backward", |bencher| {
        bencher.iter(|| {
            let xg = Tensor::new(input_c4.clone().into_dyn(), true);
            let out = tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&avg_pool_op_arc),
                &[xg.clone()],
            );
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
    });

    // AdaptiveAvgPool2D
    let adaptive_pool_op = AdaptiveAvgPool2DOp::new(32, 32);
    let adaptive_pool_op_arc: std::sync::Arc<dyn tensor_engine::ops::Operation + Send + Sync> =
        std::sync::Arc::new(adaptive_pool_op);
    group.bench_function("adaptive_avgpool2d_3x64x64", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&adaptive_pool_op_arc),
                &[input_c4_tensor.clone()],
            ))
        })
    });
    group.bench_function("adaptive_avgpool2d_3x64x64_backward", |bencher| {
        bencher.iter(|| {
            let xg = Tensor::new(input_c4.clone().into_dyn(), true);
            let out = tensor_engine::tensor::Tensor::apply(
                std::sync::Arc::clone(&adaptive_pool_op_arc),
                &[xg.clone()],
            );
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
    });

    // Absolute positional embedding
    let seq = 128usize;
    let d_model = 64usize;
    let pos_input = Array3::<f32>::from_shape_fn((1, seq, d_model), |_| rng.gen());
    let pos_input_t = Tensor::new(pos_input.clone().into_dyn(), false);
    let ape = AbsolutePositionalEmbedding::new(512, d_model);
    group.bench_function("absolute_positional_embedding_1x128x64", |bencher| {
        bencher.iter(|| std::hint::black_box(ape.forward(&pos_input_t)))
    });

    // MultiHeadAttention, with and without ALiBi
    let seq64 = 64usize;
    let d_m = 128usize;
    let num_heads = 8usize;
    let mha_inp = Array3::<f32>::from_shape_fn((1, seq64, d_m), |_| rng.gen());
    let mha_inp_t = Tensor::new(mha_inp.clone().into_dyn(), false);
    let mha = MultiHeadAttention::new(d_m, num_heads);
    let mha_alibi = MultiHeadAttention::new(d_m, num_heads).with_alibi();
    group.bench_function("mha_forward_64_128", |bencher| {
        bencher.iter(|| std::hint::black_box(mha.forward(&mha_inp_t)))
    });
    group.bench_function("mha_forward_alibi_64_128", |bencher| {
        bencher.iter(|| std::hint::black_box(mha_alibi.forward(&mha_inp_t)))
    });
    // NL-OOB bench: iterate a set of sequence lengths and batch sizes to measure overhead
    let seq_candidates = [64usize, 128usize, 256usize];
    let batch_candidates = [1usize, 4usize, 16usize];
    for &seq_len in seq_candidates.iter() {
        for &batch_size in batch_candidates.iter() {
            // build a new input and distance matrix
            let mha_inp = Array3::<f32>::from_shape_fn((batch_size, seq_len, d_m), |_| rng.gen());
            let mha_inp_t = Tensor::new(mha_inp.clone().into_dyn(), false);
            let mut dist_vec: Vec<f32> = Vec::with_capacity(seq_len * seq_len);
            for i in 0..seq_len {
                for j in 0..seq_len {
                    dist_vec.push(((i as isize - j as isize).abs()) as f32);
                }
            }
            let dist_tensor = Tensor::new(ndarray::Array::from_shape_vec((seq_len, seq_len), dist_vec).unwrap().into_dyn(), false);
            let mha_oob = MultiHeadAttention::new_with_nl_oob(d_m, num_heads, BiasFunction::Logarithmic, 2.0);
            let bench_name = format!("mha_forward_with_distance_{}x{}_{}", batch_size, seq_len, d_m);
            group.bench_function(bench_name, |bencher| {
                bencher.iter(|| std::hint::black_box(mha_oob.forward_with_distance(&mha_inp_t, &dist_tensor)))
            });
            let bench_name2 = format!("mha_forward_backward_with_distance_{}x{}_{}", batch_size, seq_len, d_m);
            group.bench_function(bench_name2, |bencher| {
                bencher.iter(|| {
                    let xg = Tensor::new(mha_inp.clone().into_dyn(), true);
                    let out = mha_oob.forward_with_distance(&xg, &dist_tensor);
                    let s = out.sum();
                    s.backward();
                    std::hint::black_box(())
                })
            });
        }
    }
    group.bench_function("mha_forward_backward_64_128", |bencher| {
        bencher.iter(|| {
            let xg = Tensor::new(mha_inp.clone().into_dyn(), true);
            let out = mha.forward(&xg);
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
    });
    group.bench_function("mha_forward_backward_alibi_64_128", |bencher| {
        bencher.iter(|| {
            let xg = Tensor::new(mha_inp.clone().into_dyn(), true);
            let out = mha_alibi.forward(&xg);
            let s = out.sum();
            s.backward();
            std::hint::black_box(())
        })
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

fn bench_transformers(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("transformers");
    let ci_bench = std::env::var("CI_BENCH").is_ok();
    if ci_bench {
        group.measurement_time(std::time::Duration::from_millis(250));
        group.sample_size(10);
    }

    let mut rng = StdRng::seed_from_u64(42);
    // MultiHeadAttention bench
    {
        let batch = 2usize;
        let seq = 16usize;
        let d_model = 64usize;
        let heads = 8usize;
        let in_data = ndarray::Array::from_shape_fn((batch, seq, d_model), |_| rng.gen());
        let t = Tensor::new(in_data.clone().into_dyn(), false);
        let mha = MultiHeadAttention::new(d_model, heads);
        group.bench_function("mha_forward_2x16x64", |bencher| { bencher.iter(|| std::hint::black_box(mha.forward(&t))) });
        group.bench_function("mha_forward_backward_2x16x64", |bencher| { bencher.iter(|| {
            let tg = Tensor::new(in_data.clone().into_dyn(), true);
            let out = mha.forward(&tg);
            let s = out.sum(); s.backward(); std::hint::black_box(())
        })});
    }

    // TransformerBlock bench
    {
        let batch = 2usize;
        let seq = 16usize;
        let d_model = 64usize;
        let d_ff = 256usize;
        let heads = 8usize;
        let block = tensor_engine::nn::TransformerBlock::new(d_model, d_ff, heads);
        let data = ndarray::Array::from_shape_fn((batch, seq, d_model), |_| rng.gen());
        let t = Tensor::new(data.clone().into_dyn(), false);
        group.bench_function("transformer_block_forward_2x16x64", |bencher| { bencher.iter(|| std::hint::black_box(block.forward(&t))) });
    }

    // VisionTransformer bench
    {
        let vt = tensor_engine::nn::VisionTransformer::new(3, 2, 64, 256, 8, 1, 64);
        let input = ndarray::Array::from_shape_fn((1, 3, 32, 32), |_| rng.gen());
        let t = Tensor::new(input.into_dyn(), false);
        group.bench_function("vision_transformer_forward", |bencher| { bencher.iter(|| std::hint::black_box(vt.forward(&t))) });
    }

    // UNetModel bench (if present)
    {
        let unet = tensor_engine::nn::UNetModel::new(1, 8, 2);
        let x = ndarray::Array::from_shape_fn((1, 8, 16, 16), |_| rng.gen());
        let t = Tensor::new(x.into_dyn(), false);
        let t_emb = Tensor::new(ndarray::Array::from_elem(ndarray::IxDyn(&[1, 16]), 0.1).into_dyn(), false);
        group.bench_function("unet_forward", |bencher| { bencher.iter(|| std::hint::black_box(unet.forward(&t, &t_emb))) });
    }

    // MultimodalLLM bench (image+text concat)
    {
        let vis = tensor_engine::nn::VisionTransformer::new(3, 2, 64, 256, 8, 1, 64);
        let model = tensor_engine::nn::MultimodalLLM::new(vis, 100, 64, 256, 8, 1);
        let image = ndarray::Array::from_shape_fn((1, 3, 32, 32), |_| rng.gen());
        let input_ids = ndarray::Array::from_shape_fn((1, 8), |_| (rng.gen::<u32>() % 100) as f32);
        let image_t = Tensor::new(image.into_dyn(), false);
        let ids_t = Tensor::new(input_ids.into_dyn(), false);
        group.bench_function("multimodal_forward", |bencher| { bencher.iter(|| std::hint::black_box(model.forward(&image_t, &ids_t))) });
    }

    group.finish();
}

fn bench_batched_and_block_quant(c: &mut Criterion) {
    let mut group = c.benchmark_group("batched_block_quant");
    let ci_bench = std::env::var("CI_BENCH").is_ok();
    if ci_bench {
        group.measurement_time(std::time::Duration::from_millis(250));
        group.sample_size(10);
    } else {
        group.measurement_time(std::time::Duration::from_secs(2));
        group.sample_size(50);
    }
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(123);
    // Batched matmul: 16x64x128 * 16x128x64 -> 16x64x64
    let batch = 16usize;
    let m = 64usize;
    let k = 128usize;
    let n = 64usize;
    let a_data = Array3::<f32>::from_shape_fn((batch, m, k), |_| rng.gen());
    let b_data = Array3::<f32>::from_shape_fn((batch, k, n), |_| rng.gen());
    let a = Tensor::new(a_data.clone().into_dyn(), false);
    let b = Tensor::new(b_data.clone().into_dyn(), false);
        group.bench_function("batched_matmul_16_64_128_64", |bencher| {
        bencher.iter(|| std::hint::black_box(tensor_engine::tensor::Tensor::apply(
            std::sync::Arc::new(tensor_engine::ops::BatchedMatMul::new()), &[a.clone(), b.clone()]
        )))
    });
    // Blockwise quantized matmul: split right-hand side into column blocks and quantize each block
    if !ci_bench {
        let size = 512usize;
        let block = 64usize;
        let a_data = Array2::<f32>::from_shape_fn((size, size), |_| rng.gen());
        let a_t = Tensor::new(a_data.clone().into_dyn(), false);
        // generate blocks of b
        let mut blocks: Vec<Tensor> = vec![];
        for start in (0..size).step_by(block) {
            let end = (start + block).min(size);
            let b_block = Array2::<f32>::from_shape_fn((size, end-start), |_| rng.gen());
            let qblock = Tensor::new_with_dtype(b_block.into_dyn(), false, tensor_engine::dtype::DType::I8);
            blocks.push(qblock);
        }
        group.bench_function("block_quantized_matmul_512x512_block64", |bencher| {
            bencher.iter(|| {
                for q in blocks.iter() {
                    let _out = std::hint::black_box(a_t.quantized_matmul(q));
                    // intentionally discard outputs; we only measure execution time
                }
                std::hint::black_box(())
            })
        });
    }
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
    bench_transformers,
    bench_training_loop,
    bench_batched_and_block_quant
);
criterion_main!(benches);
