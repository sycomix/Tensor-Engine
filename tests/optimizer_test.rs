use ndarray::{arr1, Array2};
use tensor_engine::nn::Adam;
use tensor_engine::nn::AdamW;
use tensor_engine::nn::RMSProp;
use tensor_engine::nn::{
    CosineAnnealing, CyclicLR, CyclicLRMode, ExponentialDecay, LRScheduler, LinearWarmup,
    PolynomialDecay, StepDecay,
};
use tensor_engine::nn::{Linear, Module, Optimizer, Sequential, SGD};
use tensor_engine::tensor::Tensor;

#[test]
fn test_linear_forward() {
    let linear = Linear::new(2, 3, true);
    let input = Tensor::new(
        Array2::from_shape_vec((1, 2), vec![1.0, 2.0])
            .unwrap()
            .into_dyn(),
        false,
    );
    let output = linear.forward(&input);

    // Check output shape
    assert_eq!(output.lock().storage.shape(), &[1, 3]);
}

#[test]
fn test_sequential() {
    let seq = Sequential::new()
        .add(Linear::new(2, 3, true))
        .add(Linear::new(3, 1, false));

    let input = Tensor::new(
        Array2::from_shape_vec((1, 2), vec![1.0, 2.0])
            .unwrap()
            .into_dyn(),
        false,
    );
    let output = seq.forward(&input);

    assert_eq!(output.lock().storage.shape(), &[1, 1]);
}

#[test]
fn test_sgd_step() {
    let param = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), true);
    param.lock().grad = Some(arr1(&[0.1, 0.2]).into_dyn());

    let mut sgd = SGD::new(0.1, 0.0);
    sgd.step(&[param.clone()]);

    let data = param.lock().storage.to_f32_array();
    // param -= lr * grad = [1, 2] - 0.1 * [0.1, 0.2] = [0.99, 1.98]
    assert!((data[0] - 0.99).abs() < 1e-6);
    assert!((data[1] - 1.98).abs() < 1e-6);
}

#[test]
fn test_sgd_momentum_behaviour() {
    use ndarray::arr1;
    let param = Tensor::new(arr1(&[1.0]).into_dyn(), true);
    param.lock().grad = Some(arr1(&[0.1]).into_dyn());
    let mut sgd = SGD::new(0.1, 0.9);
    let mut sgd_no_mom = SGD::new(0.1, 0.0);
    // first step
    let param_a = Tensor::new(arr1(&[1.0]).into_dyn(), true);
    let param_b = Tensor::new(arr1(&[1.0]).into_dyn(), true);
    param_a.lock().grad = Some(arr1(&[0.1]).into_dyn());
    param_b.lock().grad = Some(arr1(&[0.1]).into_dyn());
    // step twice
    sgd.step(&[param_a.clone()]);
    sgd.step(&[param_a.clone()]);
    sgd_no_mom.step(&[param_b.clone()]);
    sgd_no_mom.step(&[param_b.clone()]);
    // compare parameter values after two steps
    let a_val = param_a.lock().storage.to_f32_array()[0];
    let b_val = param_b.lock().storage.to_f32_array()[0];
    // With momentum, the parameter update should differ from non-momentum
    assert!((a_val - b_val).abs() > 1e-6);
}

#[test]
fn test_zero_grad() {
    let param = Tensor::new(arr1(&[1.0]).into_dyn(), true);
    param.lock().grad = Some(arr1(&[1.0]).into_dyn());

    param.zero_grad();

    assert!(param.lock().grad.is_none());
}

#[test]
fn test_optimizer_zero_grad() {
    let param = Tensor::new(arr1(&[1.0]).into_dyn(), true);
    param.lock().grad = Some(arr1(&[1.0]).into_dyn());

    let mut sgd = SGD::new(0.1, 0.0);
    sgd.zero_grad(&[param.clone()]);

    assert!(param.lock().grad.is_none());
}

#[test]
fn test_sgd_cast_params_dtype_changes() {
    use tensor_engine::dtype::DType;
    let param = Tensor::new(arr1(&[0.1234567f32, -0.9876543f32]).into_dyn(), true);
    let mut sgd = SGD::new(0.1, 0.0);
    // Cast params to f8 (emulated)
    sgd.cast_params(&[param.clone()], DType::F8);
    assert_eq!(param.dtype(), DType::F8);
    // After cast, data should be changed (roundtrip quantization)
    // The values should not be exactly equal to the original values
    let param_vals: Vec<f32> = param
        .lock()
        .storage
        .to_f32_array()
        .iter()
        .cloned()
        .collect();
    assert!((param_vals[0] - 0.1234567).abs() > 1e-6 || (param_vals[1] + 0.9876543).abs() > 1e-6);
}

#[test]
fn test_adamw_weight_decay_reduces_param_more_than_adam() {
    use ndarray::arr1;
    let param_a = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), true);
    let param_b = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), true);
    // same grads
    param_a.lock().grad = Some(arr1(&[0.1, 0.2]).into_dyn());
    param_b.lock().grad = Some(arr1(&[0.1, 0.2]).into_dyn());

    let mut adam = Adam::new(0.1, 0.9, 0.999, 1e-8);
    let mut adamw = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.01);

    // step once
    adam.step(&[param_a.clone()]);
    adamw.step(&[param_b.clone()]);

    let a_vals = param_a.lock().storage.to_f32_array();
    let b_vals = param_b.lock().storage.to_f32_array();
    // With positive weight decay and positive params, AdamW should decrease params more
    assert!(b_vals[0] < a_vals[0]);
    assert!(b_vals[1] < a_vals[1]);
}

#[test]
fn test_adamw_matches_adam_with_zero_weight_decay() {
    use ndarray::arr1;
    let param_a = Tensor::new(arr1(&[1.0]).into_dyn(), true);
    let param_b = Tensor::new(arr1(&[1.0]).into_dyn(), true);
    param_a.lock().grad = Some(arr1(&[0.1]).into_dyn());
    param_b.lock().grad = Some(arr1(&[0.1]).into_dyn());

    let mut adam = Adam::new(0.1, 0.9, 0.999, 1e-8);
    let mut adamw = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.0);

    adam.step(&[param_a.clone()]);
    adamw.step(&[param_b.clone()]);
    let a_vals = param_a.lock().storage.to_f32_array();
    let b_vals = param_b.lock().storage.to_f32_array();
    // close to equal
    assert!((a_vals[0] - b_vals[0]).abs() < 1e-6);
}

#[test]
fn test_linear_warmup_scheduler() {
    let s = LinearWarmup::new(0.1, 10);
    assert!((s.get_lr(0) - 0.0).abs() < 1e-6);
    assert!((s.get_lr(5) - 0.05).abs() < 1e-6);
    assert!((s.get_lr(10) - 0.1).abs() < 1e-6);
    assert!((s.get_lr(20) - 0.1).abs() < 1e-6);
}

#[test]
fn test_cosine_annealing_scheduler() {
    let s = CosineAnnealing::new(0.1, 100);
    let lr0 = s.get_lr(0);
    let lr_mid = s.get_lr(50);
    let lr_end = s.get_lr(100);
    assert!((lr0 - 0.1).abs() < 1e-6);
    assert!(lr_mid < lr0);
    assert!((lr_end - 0.0).abs() < 1e-6);
}

#[test]
fn test_exponential_decay_scheduler() {
    let s = ExponentialDecay::new(0.1, 0.9);
    assert!((s.get_lr(0) - 0.1).abs() < 1e-6);
    assert!((s.get_lr(1) - 0.09).abs() < 1e-6);
    assert!((s.get_lr(2) - 0.081).abs() < 1e-6);

    let s_min = ExponentialDecay::new_with_min_lr(0.1, 0.0, 0.01);
    assert!((s_min.get_lr(0) - 0.1).abs() < 1e-6);
    // gamma=0 makes lr drop to 0 for step>=1, but min_lr clamps it.
    assert!((s_min.get_lr(1) - 0.01).abs() < 1e-6);
    assert!((s_min.get_lr(100) - 0.01).abs() < 1e-6);
}

#[test]
fn test_step_decay_scheduler() {
    let s = StepDecay::new(0.1, 10, 0.5);
    assert!((s.get_lr(0) - 0.1).abs() < 1e-6);
    assert!((s.get_lr(9) - 0.1).abs() < 1e-6);
    assert!((s.get_lr(10) - 0.05).abs() < 1e-6);
    assert!((s.get_lr(19) - 0.05).abs() < 1e-6);
    assert!((s.get_lr(20) - 0.025).abs() < 1e-6);
}

#[test]
fn test_polynomial_decay_scheduler() {
    let s = PolynomialDecay::new(0.1, 0.0, 100, 1.0);
    assert!((s.get_lr(0) - 0.1).abs() < 1e-6);
    assert!((s.get_lr(50) - 0.05).abs() < 1e-6);
    assert!((s.get_lr(100) - 0.0).abs() < 1e-6);
    assert!((s.get_lr(200) - 0.0).abs() < 1e-6);

    let s2 = PolynomialDecay::new(0.1, 0.0, 100, 2.0);
    // At step 50, (1-0.5)^2 = 0.25
    assert!((s2.get_lr(50) - 0.025).abs() < 1e-6);
}

#[test]
fn test_cyclic_lr_triangular() {
    let s = CyclicLR::new_with_mode(0.001, 0.006, 2, 2, CyclicLRMode::Triangular);
    assert!((s.get_lr(0) - 0.001).abs() < 1e-6);
    assert!((s.get_lr(1) - 0.0035).abs() < 1e-6);
    assert!((s.get_lr(2) - 0.006).abs() < 1e-6);
    assert!((s.get_lr(3) - 0.0035).abs() < 1e-6);
    assert!((s.get_lr(4) - 0.001).abs() < 1e-6);
}

#[test]
fn test_cyclic_lr_triangular2_amplitude_halves() {
    let s = CyclicLR::new_with_mode(0.001, 0.006, 2, 2, CyclicLRMode::Triangular2);
    // First cycle peak.
    assert!((s.get_lr(2) - 0.006).abs() < 1e-6);
    // Second cycle peak: base + (max-base)*0.5
    let expected_peak_cycle2 = 0.001 + (0.006 - 0.001) * 0.5;
    assert!((s.get_lr(6) - expected_peak_cycle2).abs() < 1e-6);
}

#[test]
fn test_rmsprop_step() {
    use ndarray::arr1;
    let param = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), true);
    param.lock().grad = Some(arr1(&[0.1, 0.2]).into_dyn());
    let mut opt = RMSProp::new(0.1, 0.9, 1e-8);
    opt.step(&[param.clone()]);
    let vals = param.lock().storage.to_f32_array();
    // Ensure parameters decreased
    assert!(vals[0] < 1.0);
    assert!(vals[1] < 2.0);
}

#[test]
fn test_clip_gradients_global_norm_scales_in_place() {
    use ndarray::arr1;
    let p1 = Tensor::new(arr1(&[0.0, 0.0]).into_dyn(), true);
    let p2 = Tensor::new(arr1(&[0.0]).into_dyn(), true);
    p1.lock().grad = Some(arr1(&[3.0, 4.0]).into_dyn()); // norm 5
    p2.lock().grad = Some(arr1(&[0.0]).into_dyn());

    let mut sgd = SGD::new(0.1, 0.0);
    // Clip to 2.5 => scale = 0.5
    sgd.clip_gradients(&[p1.clone(), p2.clone()], 2.5);

    let g1 = p1.lock().grad.clone().unwrap();
    assert!((g1.as_slice_memory_order().unwrap()[0] - 1.5).abs() < 1e-6);
    assert!((g1.as_slice_memory_order().unwrap()[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_scale_gradients_supports_gradient_accumulation_averaging() {
    use ndarray::arr1;

    // Simulate two micro-batches with grads 1.0 and 3.0 accumulated into 4.0,
    // then averaged by scaling with 0.5.
    let p = Tensor::new(arr1(&[10.0]).into_dyn(), true);
    p.lock().grad = Some(arr1(&[1.0]).into_dyn());
    {
        let mut lock = p.lock();
        let g = lock.grad.as_mut().unwrap();
        *g += &arr1(&[3.0]).into_dyn();
    }

    let mut sgd = SGD::new(1.0, 0.0);
    sgd.scale_gradients(&[p.clone()], 0.5);
    sgd.step(&[p.clone()]);

    let val = p.lock().storage.to_f32_array()[0];
    // averaged grad is 2.0 => 10 - 2 = 8
    assert!((val - 8.0).abs() < 1e-6);
}

#[test]
fn test_sgd_weight_decay_reduces_params_even_with_zero_grad() {
    use ndarray::arr1;
    let p = Tensor::new(arr1(&[10.0]).into_dyn(), true);
    p.lock().grad = Some(arr1(&[0.0]).into_dyn());
    let mut sgd = SGD::new_with_weight_decay(1.0, 0.0, 0.1);
    sgd.step(&[p.clone()]);
    let v = p.lock().storage.to_f32_array()[0];
    // decoupled wd: p := p - lr*wd*p = 10 - 1*0.1*10 = 9
    assert!((v - 9.0).abs() < 1e-6);
}

#[test]
fn test_rmsprop_weight_decay_reduces_params_even_with_zero_grad() {
    use ndarray::arr1;
    let p = Tensor::new(arr1(&[10.0]).into_dyn(), true);
    p.lock().grad = Some(arr1(&[0.0]).into_dyn());
    let mut opt = RMSProp::new_with_weight_decay(1.0, 0.9, 1e-8, 0.1);
    opt.step(&[p.clone()]);
    let v = p.lock().storage.to_f32_array()[0];
    assert!((v - 9.0).abs() < 1e-6);
}
