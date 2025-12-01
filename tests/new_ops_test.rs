use ndarray::array;
use ndarray::ArrayD;
use tensor_engine::tensor::Tensor as TE;
use tensor_engine::tensor::Tensor as TE2;

#[test]
fn test_rmsnorm_forward_backward_shapes() {
    // 2x4 input
    let x = TE::new(
        array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]].into_dyn(),
        true,
    );
    let gamma = TE::new(array![1.0, 1.0, 1.0, 1.0].into_dyn(), true);
    let y = x.rmsnorm(&gamma, 1, 1e-5);
    // Check shape
    assert_eq!(y.lock().storage.shape(), vec![2, 4]);
    // Backward basic check compute
    y.backward();
    assert!(x.lock().grad.is_some());
}

#[test]
fn test_swiglu_forward_backward_shapes() {
    let x = TE::new(array![[1.0, 2.0, 3.0, 4.0]].into_dyn(), true);
    let y = x.swiglu();
    assert_eq!(y.lock().storage.shape(), vec![1, 2]);
    y.backward();
    assert!(x.lock().grad.is_some());
}

#[test]
fn test_embedding_lookup_forward_backward() {
    let emb = TE::new(
        array![
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ]
        .into_dyn(),
        true,
    );
    let idx = TE::new(array![1.0, 2.0].into_dyn(), false);
    let out = TE::embedding_lookup(&emb, &idx);
    assert_eq!(out.lock().storage.shape(), vec![2, 4]);
    out.backward();
    assert!(emb.lock().grad.is_some());
}

#[test]
fn test_kvcache_append() {
    let cache = TE::new(array![[[0.1, 0.2], [0.3, 0.4]]].into_dyn(), true); // shape [1,2,2]
    let newkv = TE::new(array![[[0.5, 0.6]]].into_dyn(), true); // shape [1,1,2]
    let out = TE::kvcache_append(&cache, &newkv, 1);
    assert_eq!(out.lock().storage.shape(), vec![1, 3, 2]);
    out.backward();
    assert!(cache.lock().grad.is_some());
    assert!(newkv.lock().grad.is_some());
}

// Numeric gradient checks
fn numeric_gradient<F>(f: F, x: &ArrayD<f32>, h: f32) -> ArrayD<f32>
where
    F: Fn(&ArrayD<f32>) -> f32,
{
    let mut grad = ArrayD::zeros(x.dim());
    for i in 0..x.len() {
        let base = x.as_slice().unwrap()[i].abs();
        let h_local = h * (1.0 + base);
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus.as_slice_mut().unwrap()[i] += h_local;
        x_minus.as_slice_mut().unwrap()[i] -= h_local;
        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);
        grad.as_slice_mut().unwrap()[i] = (f_plus - f_minus) / (2.0 * h_local);
    }
    grad
}

#[test]
fn test_numeric_gradient_rmsnorm() {
    let x_data = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]].into_dyn();
    let gamma_data = array![1.0, 1.0, 1.0, 1.0].into_dyn();
    // analytic
    let x = TE::new(x_data.clone(), true);
    let gamma = TE::new(gamma_data.clone(), false);
    let y = x.rmsnorm(&gamma, 1, 1e-5);
    let s = y.sum();
    s.backward();
    let grad_a = x.lock().grad.clone().unwrap();
    // numeric
    let f = |x_arr: &ArrayD<f32>| {
        let tx = TE2::new(x_arr.clone(), false);
        let tg = TE2::new(gamma_data.clone(), false);
        let ty = tx.rmsnorm(&tg, 1, 1e-5);
        let ts = ty.sum();
        let val = ts
            .lock()
            .storage
            .to_f32_array()
            .iter()
            .next()
            .cloned()
            .unwrap();
        val
    };
    let grad_numeric = numeric_gradient(f, &x_data, 1e-3);
    // compare shapes and approx equality
    assert_eq!(grad_numeric.shape(), grad_a.shape());
    for (an, nu) in grad_a.iter().zip(grad_numeric.iter()) {
        assert!((an - nu).abs() < 1e-2);
    }
}

#[test]
fn test_numeric_gradient_swiglu() {
    let x_data = array![[1.0, 2.0, 3.0, 4.0]].into_dyn();
    let x = TE::new(x_data.clone(), true);
    let y = x.swiglu();
    let s = y.sum();
    s.backward();
    let grad_a = x.lock().grad.clone().unwrap();
    // numeric
    let f = |x_arr: &ArrayD<f32>| {
        let tx = TE2::new(x_arr.clone(), false);
        let ty = tx.swiglu();
        let ts = ty.sum();
        let val = ts
            .lock()
            .storage
            .to_f32_array()
            .iter()
            .next()
            .cloned()
            .unwrap();
        val
    };
    let grad_numeric = numeric_gradient(f, &x_data, 1e-3);
    assert_eq!(grad_numeric.shape(), grad_a.shape());
    for (an, nu) in grad_a.iter().zip(grad_numeric.iter()) {
        assert!((an - nu).abs() < 1e-2);
    }
}
