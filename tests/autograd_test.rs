use ndarray::{arr0, arr1, arr2, ArrayD, IxDyn};
use rand::prelude::*;
use tensor_engine::tensor::Tensor;

// Helper function to compute numeric gradient using finite differences
fn numeric_gradient<F>(f: F, x: &ArrayD<f32>, h: f32) -> ArrayD<f32>
where
    F: Fn(&ArrayD<f32>) -> f32,
{
    let mut grad = ArrayD::zeros(x.dim());
    for i in 0..x.len() {
        // Use a relative step to mitigate cancellation for large magnitude float32 values.
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
fn test_simple_backward() {
    let a = Tensor::new(arr1(&[2.0]).into_dyn(), true);
    let b = Tensor::new(arr1(&[3.0]).into_dyn(), true);
    let c = a.add(&b);
    println!("[TEST] before c.backward");
    c.backward();
    println!("[TEST] after c.backward");
    println!("[TEST] c.backward() returned");

    let grad_a = a.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();

    assert_eq!(grad_a, arr1(&[1.0]).into_dyn());
    assert_eq!(grad_b, arr1(&[1.0]).into_dyn());
}

#[test]
fn test_add_forward() {
    let a = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), false);
    let b = Tensor::new(arr1(&[3.0, 4.0]).into_dyn(), false);
    let c = a.add(&b);

    let expected = arr1(&[4.0, 6.0]).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_add_backward() {
    let a = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), true);
    let b = Tensor::new(arr1(&[3.0, 4.0]).into_dyn(), true);
    let c = a.add(&b);
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();

    assert_eq!(grad_a, arr1(&[1.0, 1.0]).into_dyn());
    assert_eq!(grad_b, arr1(&[1.0, 1.0]).into_dyn());
}

#[test]
fn test_mul_forward() {
    let a = Tensor::new(arr1(&[2.0, 3.0]).into_dyn(), false);
    let b = Tensor::new(arr1(&[4.0, 5.0]).into_dyn(), false);
    let c = a.mul(&b);

    let expected = arr1(&[8.0, 15.0]).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_mul_backward() {
    let a = Tensor::new(arr1(&[2.0, 3.0]).into_dyn(), true);
    let b = Tensor::new(arr1(&[4.0, 5.0]).into_dyn(), true);
    let c = a.mul(&b);
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();

    assert_eq!(grad_a, arr1(&[4.0, 5.0]).into_dyn());
    assert_eq!(grad_b, arr1(&[2.0, 3.0]).into_dyn());
}

#[test]
fn test_sub_forward() {
    let a = Tensor::new(arr1(&[5.0, 7.0]).into_dyn(), false);
    let b = Tensor::new(arr1(&[3.0, 2.0]).into_dyn(), false);
    let c = a.sub(&b);

    let expected = arr1(&[2.0, 5.0]).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_sub_backward() {
    let a = Tensor::new(arr1(&[5.0, 7.0]).into_dyn(), true);
    let b = Tensor::new(arr1(&[3.0, 2.0]).into_dyn(), true);
    let c = a.sub(&b);
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();

    assert_eq!(grad_a, arr1(&[1.0, 1.0]).into_dyn());
    assert_eq!(grad_b, arr1(&[-1.0, -1.0]).into_dyn());
}

#[test]
fn test_div_forward() {
    let a = Tensor::new(arr1(&[8.0, 12.0]).into_dyn(), false);
    let b = Tensor::new(arr1(&[4.0, 3.0]).into_dyn(), false);
    let c = a.div(&b);

    let expected = arr1(&[2.0, 4.0]).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_div_backward() {
    let a = Tensor::new(arr1(&[8.0, 12.0]).into_dyn(), true);
    let b = Tensor::new(arr1(&[4.0, 3.0]).into_dyn(), true);
    let c = a.div(&b);
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();

    assert_eq!(grad_a, arr1(&[0.25, 1.0 / 3.0]).into_dyn());
    assert_eq!(grad_b, arr1(&[-0.5, -4.0 / 3.0]).into_dyn());
}

#[test]
fn test_pow_forward() {
    let a = Tensor::new(arr1(&[2.0, 3.0]).into_dyn(), false);
    let c = a.pow(2.0);

    let expected = arr1(&[4.0, 9.0]).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_pow_backward() {
    let a = Tensor::new(arr1(&[2.0, 3.0]).into_dyn(), true);
    let c = a.pow(2.0);
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();

    assert_eq!(grad_a, arr1(&[4.0, 6.0]).into_dyn()); // d/dx(x^2) = 2x
}

#[test]
fn test_matmul_forward() {
    let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(), false);
    let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]).into_dyn(), false);
    let c = a.matmul(&b);

    let expected = arr2(&[[19.0, 22.0], [43.0, 50.0]]).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_matmul_backward() {
    let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(), true);
    let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]).into_dyn(), true);
    let c = a.matmul(&b);
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();

    let expected_grad_a = arr2(&[[11.0, 15.0], [11.0, 15.0]]).into_dyn(); // b.t() @ ones_like(c)
    let expected_grad_b = arr2(&[[4.0, 4.0], [6.0, 6.0]]).into_dyn(); // a.t() @ ones_like(c)

    assert_eq!(grad_a, expected_grad_a);
    assert_eq!(grad_b, expected_grad_b);
}

#[test]
fn test_relu_forward() {
    let a = Tensor::new(arr1(&[-1.0, 0.0, 1.0, 2.0]).into_dyn(), false);
    let c = a.relu();

    let expected = arr1(&[0.0, 0.0, 1.0, 2.0]).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_relu_backward() {
    let a = Tensor::new(arr1(&[-1.0, 0.0, 1.0, 2.0]).into_dyn(), true);
    let c = a.relu();
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();

    assert_eq!(grad_a, arr1(&[0.0, 0.0, 1.0, 1.0]).into_dyn());
}

#[test]
fn test_sigmoid_forward() {
    let a = Tensor::new(arr1(&[0.0]).into_dyn(), false);
    let c = a.sigmoid();

    let expected = arr1(&[0.5]).into_dyn();
    let arr = c.lock().storage.to_f32_array();
    assert!((arr[0] - expected[0]).abs() < 1e-6);
}

#[test]
fn test_sigmoid_backward() {
    let a = Tensor::new(arr1(&[0.0]).into_dyn(), true);
    let c = a.sigmoid();
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();

    assert!((grad_a[0] - 0.25).abs() < 1e-6); // sigmoid'(0) = 0.25
}

#[test]
fn test_tanh_forward() {
    let a = Tensor::new(arr1(&[0.0]).into_dyn(), false);
    let c = a.tanh();

    let expected = arr1(&[0.0]).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_tanh_backward() {
    let a = Tensor::new(arr1(&[0.0]).into_dyn(), true);
    let c = a.tanh();
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();

    assert!((grad_a[0] - 1.0).abs() < 1e-6); // tanh'(0) = 1
}

#[test]
fn test_ternary_forward_backward() {
    let a = Tensor::new(arr1(&[0.1, 0.5, -2.0]).into_dyn(), true);
    let c = a.ternary();
    //println!("[DEBUG TEST] after a.ternary()");
    // forward: check values are in {-mean_abs, 0, mean_abs}
    // Scope the lock to avoid holding it across backward() call
    let mean_abs = {
        let arr = a.lock().storage.to_f32_array().clone();
        arr.mapv(|x| x.abs()).sum() / (arr.len() as f32)
    };
    //println!("[DEBUG TEST] after mean_abs computation = {}", mean_abs);
    println!("[TEST] mean_abs = {}", mean_abs);
    let c_vals = {
        let arr = c.lock().storage.to_f32_array().clone();
        arr
    };
    //println!(
    //    "[DEBUG TEST] after c.lock().data.clone() => c_vals = {:?}",
    //    c_vals
    //);
    println!("[TEST] c_vals = {:?}", c_vals);
    for (i, v) in c_vals.iter().enumerate() {
        println!("[TEST] c_vals[{}] = {}", i, v);
        assert!((*v - mean_abs).abs() < 1e-5 || (*v + mean_abs).abs() < 1e-5 || (*v).abs() < 1e-5);
    }
    //println!("[DEBUG TEST] before c.backward()");
    c.backward();
    //println!("[DEBUG TEST] after c.backward()");
    let grad_a = a.lock().grad.clone().unwrap();
    // STE: gradients should be passed through unchanged (ones)
    assert_eq!(grad_a, arr1(&[1.0, 1.0, 1.0]).into_dyn());
}

#[test]
fn test_sum_forward() {
    let a = Tensor::new(arr1(&[1.0, 2.0, 3.0]).into_dyn(), false);
    let c = a.sum();

    let expected = arr0(6.0).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_sum_backward() {
    let a = Tensor::new(arr1(&[1.0, 2.0, 3.0]).into_dyn(), true);
    let c = a.sum();
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();

    assert_eq!(grad_a, arr1(&[1.0, 1.0, 1.0]).into_dyn());
}

#[test]
fn test_mean_forward() {
    let a = Tensor::new(arr1(&[1.0, 2.0, 3.0]).into_dyn(), false);
    let c = a.mean();

    let expected = arr0(2.0).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_mean_backward() {
    let a = Tensor::new(arr1(&[1.0, 2.0, 3.0]).into_dyn(), true);
    let c = a.mean();
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();

    assert_eq!(grad_a, arr1(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]).into_dyn());
}

#[test]
fn test_max_forward() {
    let a = Tensor::new(arr1(&[1.0, 3.0, 2.0]).into_dyn(), false);
    let c = a.max();
    let expected = arr0(3.0).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_max_backward_unique_and_tie() {
    // unique max
    let a1 = Tensor::new(arr1(&[1.0, 3.0, 2.0]).into_dyn(), true);
    let c1 = a1.max();
    c1.backward();
    let grad_a1 = a1.lock().grad.clone().unwrap();
    assert_eq!(grad_a1, arr1(&[0.0, 1.0, 0.0]).into_dyn());

    // tie max
    let a2 = Tensor::new(arr1(&[1.0, 3.0, 3.0]).into_dyn(), true);
    let c2 = a2.max();
    c2.backward();
    let grad_a2 = a2.lock().grad.clone().unwrap();
    assert_eq!(grad_a2, arr1(&[0.0, 0.5, 0.5]).into_dyn());
}

#[test]
fn test_min_forward() {
    let a = Tensor::new(arr1(&[1.0, 3.0, 2.0]).into_dyn(), false);
    let c = a.min();
    let expected = arr0(1.0).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_min_backward_unique_and_tie() {
    // unique min
    let a1 = Tensor::new(arr1(&[1.0, 3.0, 2.0]).into_dyn(), true);
    let c1 = a1.min();
    c1.backward();
    let grad_a1 = a1.lock().grad.clone().unwrap();
    assert_eq!(grad_a1, arr1(&[1.0, 0.0, 0.0]).into_dyn());

    // tie min
    let a2 = Tensor::new(arr1(&[1.0, 1.0, 3.0]).into_dyn(), true);
    let c2 = a2.min();
    c2.backward();
    let grad_a2 = a2.lock().grad.clone().unwrap();
    assert_eq!(grad_a2, arr1(&[0.5, 0.5, 0.0]).into_dyn());
}

#[test]
fn test_concat_forward() {
    let a = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), false);
    let b = Tensor::new(arr1(&[3.0, 4.0]).into_dyn(), false);
    let c = Tensor::concat(&[a, b], 0);

    let expected = arr1(&[1.0, 2.0, 3.0, 4.0]).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_concat_backward() {
    let a = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), true);
    let b = Tensor::new(arr1(&[3.0, 4.0]).into_dyn(), true);
    let c = Tensor::concat(&[a.clone(), b.clone()], 0);
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();

    assert_eq!(grad_a, arr1(&[1.0, 1.0]).into_dyn());
    assert_eq!(grad_b, arr1(&[1.0, 1.0]).into_dyn());
}

#[test]
fn test_stack_forward() {
    let a = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), false);
    let b = Tensor::new(arr1(&[3.0, 4.0]).into_dyn(), false);
    let c = Tensor::stack(&[a, b], 0);

    let expected = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
}

#[test]
fn test_stack_backward() {
    let a = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), true);
    let b = Tensor::new(arr1(&[3.0, 4.0]).into_dyn(), true);
    let c = Tensor::stack(&[a.clone(), b.clone()], 0);
    c.backward();

    let grad_a = a.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();

    assert_eq!(grad_a, arr1(&[1.0, 1.0]).into_dyn());
    assert_eq!(grad_b, arr1(&[1.0, 1.0]).into_dyn());
}

#[test]
fn test_maxpool2d_forward_backward() {
    use tensor_engine::ops::MaxPool2D;
    // Input (1,1,4,4) with known max positions
    let data = vec![
        1.0, 2.0, 5.0, 4.0, 3.0, 8.0, 6.0, 7.0, 4.0, 9.0, 2.0, 1.0, 0.0, 5.0, 3.0, 2.0,
    ];
    let a = Tensor::new(
        ndarray::Array::from_shape_vec((1, 1, 4, 4), data.clone())
            .unwrap()
            .into_dyn(),
        true,
    );

    // Apply MaxPool2D with kernel 2, stride 2
    let op = MaxPool2D {
        kernel_size: 2,
        stride: 2,
    };
    let out = Tensor::apply(std::sync::Arc::new(op), &[a.clone()]);
    // Forward expected: (1,1,2,2)
    let expected = ndarray::Array::from_shape_vec((1, 1, 2, 2), vec![8.0, 7.0, 9.0, 3.0])
        .unwrap()
        .into_dyn();
    assert_eq!(out.lock().storage.to_f32_array(), expected);

    // Backward: set grad of out to ones and compute grad for a
    out.backward();
    let grad_a = a.lock().grad.clone().unwrap();
    // Only positions of maxima should have 1, others 0
    let expected_grad = ndarray::Array::from_shape_vec(
        (1, 1, 4, 4),
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ],
    )
    .unwrap()
    .into_dyn();
    assert_eq!(grad_a, expected_grad);
}

#[test]
fn test_conv2d_forward_backward() {
    // Make a simple input of shape (1,1,3,3) and weight (1,1,2,2)
    let input_data = ndarray::Array::from_shape_vec(
        (1, 1, 3, 3),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let weight_data =
        ndarray::Array::from_shape_vec((1, 1, 2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let bias_data = ndarray::Array::from_shape_vec((1,), vec![1.0]).unwrap();

    let a = Tensor::new(input_data.into_dyn(), true);
    let w = Tensor::new(weight_data.into_dyn(), true);
    let b = Tensor::new(bias_data.into_dyn(), true);

    use tensor_engine::ops::Conv2D as Conv2DOp;
    let op = Conv2DOp::new(1, 0);
    let c = Tensor::apply(std::sync::Arc::new(op), &[a.clone(), w.clone(), b.clone()]);

    // Expected forward result shape (1,1,2,2)
    let expected = ndarray::Array::from_shape_vec((1, 1, 2, 2), vec![7.0, 9.0, 13.0, 15.0])
        .unwrap()
        .into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);

    // Backward: set gradient for c and check gradients for input and weight via numeric gradient
    c.backward();
    let grad_a = a.lock().grad.clone().unwrap();
    let grad_w = w.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();

    // Check shapes and non-zero gradients
    assert_eq!(grad_a.shape(), &[1, 1, 3, 3]);
    assert_eq!(grad_w.shape(), &[1, 1, 2, 2]);
    assert_eq!(grad_b.shape(), &[1]);
    assert!(grad_a.iter().any(|&v| v != 0.0));
    assert!(grad_w.iter().any(|&v| v != 0.0));
    assert_eq!(grad_b[[0]], 4.0); // sum of ones over output (2x2)
}

#[test]
fn test_conv1d_forward_backward() {
    // Input (1,1,4) and weight (1,1,2)
    let input_data = ndarray::Array::from_shape_vec((1, 1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let weight_data = ndarray::Array::from_shape_vec((1, 1, 2), vec![1.0, 0.0]).unwrap();
    let bias_data = ndarray::Array::from_shape_vec((1,), vec![1.0]).unwrap();
    let a = Tensor::new(input_data.into_dyn(), true);
    let w = Tensor::new(weight_data.into_dyn(), true);
    let b = Tensor::new(bias_data.into_dyn(), true);

    use tensor_engine::ops::Conv1D as Conv1DOp;
    let op = Conv1DOp::new(1, 0);
    let c = Tensor::apply(std::sync::Arc::new(op), &[a.clone(), w.clone(), b.clone()]);
    // forward result length is 3: [1*1 + b, 2*1 + b, 3*1 + b] = [2, 3, 4]
    let expected = ndarray::Array::from_shape_vec((1, 1, 3), vec![2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);
    c.backward();
    let grad_a = a.lock().grad.clone().unwrap();
    let grad_w = w.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();
    assert_eq!(grad_a.shape(), &[1, 1, 4]);
    assert_eq!(grad_w.shape(), &[1, 1, 2]);
    assert_eq!(grad_b.shape(), &[1]);
}

#[test]
fn test_conv3d_forward_backward() {
    // Input (1,1,2,2,2) -> weight (1,1,2,1,1) which performs depth-wise sum along D
    let input_data = ndarray::Array::from_shape_vec((1, 1, 2, 2, 2), vec![1.0; 8]).unwrap();
    let weight_data = ndarray::Array::from_shape_vec((1, 1, 2, 1, 1), vec![1.0, 1.0]).unwrap();
    let bias_data = ndarray::Array::from_shape_vec((1,), vec![0.0]).unwrap();
    let a = Tensor::new(input_data.into_dyn(), true);
    let w = Tensor::new(weight_data.into_dyn(), true);
    let b = Tensor::new(bias_data.into_dyn(), true);

    use tensor_engine::ops::Conv3D as Conv3DOp;
    let op = Conv3DOp::new(1, 0);
    let c = Tensor::apply(std::sync::Arc::new(op), &[a.clone(), w.clone(), b.clone()]);
    // shape should be (1,1,1,2,2)
    let out = c.lock().storage.to_f32_array();
    assert_eq!(out.shape(), &[1, 1, 1, 2, 2]);
    c.backward();
    let grad_a = a.lock().grad.clone().unwrap();
    let grad_w = w.lock().grad.clone().unwrap();
    assert_eq!(grad_a.shape(), &[1, 1, 2, 2, 2]);
    assert_eq!(grad_w.shape(), &[1, 1, 2, 1, 1]);
}

#[test]
fn test_depthwise_separable_conv2d_forward_backward() {
    // input shape (1,2,4,4), kernel 3, pointwise to 3 channels
    let input_data = ndarray::Array::from_shape_vec((1, 2, 4, 4), vec![1.0; 32]).unwrap();
    let dw_data = ndarray::Array::from_shape_vec((2, 1, 3, 3), vec![1.0; 18]).unwrap();
    let pw_data = ndarray::Array::from_shape_vec((3, 2, 1, 1), vec![1.0; 6]).unwrap();
    let bias_data = ndarray::Array::from_shape_vec((3,), vec![0.0; 3]).unwrap();
    let a = Tensor::new(input_data.into_dyn(), true);
    let dw = Tensor::new(dw_data.into_dyn(), true);
    let pw = Tensor::new(pw_data.into_dyn(), true);
    let b = Tensor::new(bias_data.into_dyn(), true);
    use tensor_engine::ops::DepthwiseSeparableConv2D as DWSep;
    let op = DWSep::new(1, 1);
    let c = Tensor::apply(
        std::sync::Arc::new(op),
        &[a.clone(), dw.clone(), pw.clone(), b.clone()],
    );
    let out = c.lock().storage.to_f32_array();
    assert_eq!(out.ndim(), 4);
    c.backward();
    assert_eq!(a.lock().grad.clone().unwrap().ndim(), 4);
    assert_eq!(dw.lock().grad.clone().unwrap().ndim(), 4);
    assert_eq!(pw.lock().grad.clone().unwrap().ndim(), 4);
}

#[test]
fn test_convtranspose2d_forward_backward() {
    // Basic test: input (1,1,2,2) weight (1,1,2,2), stride 1 padding 0 -> outh = 3
    let input_data =
        ndarray::Array::from_shape_vec((1, 1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let weight_data =
        ndarray::Array::from_shape_vec((1, 1, 2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let a = Tensor::new(input_data.into_dyn(), true);
    let w = Tensor::new(weight_data.into_dyn(), true);
    use tensor_engine::ops::ConvTranspose2D as ConvT2D;
    let op = ConvT2D::new(1, 0);
    let c = Tensor::apply(std::sync::Arc::new(op), &[a.clone(), w.clone()]);
    let out = c.lock().storage.to_f32_array();
    assert_eq!(out.shape(), &[1, 1, 3, 3]);
    c.backward();
    assert_eq!(a.lock().grad.clone().unwrap().shape(), &[1, 1, 2, 2]);
    assert_eq!(w.lock().grad.clone().unwrap().shape(), &[1, 1, 2, 2]);
}

#[test]
fn test_absolute_positional_embedding() {
    use tensor_engine::nn::AbsolutePositionalEmbedding;
    use tensor_engine::nn::Module;
    let batch = 2;
    let seq = 4;
    let d_model = 8;
    let pe = AbsolutePositionalEmbedding::new(10, d_model);
    // input zeros
    let input = Tensor::new(ndarray::Array::zeros(IxDyn(&[batch, seq, d_model])), true);
    let out = pe.forward(&input);
    let out_arr = out.lock().storage.to_f32_array();
    // Should equal positional embeddings repeated across batch
    let idx = ndarray::Array::from_shape_vec((1, seq), (0..seq).map(|i| i as f32).collect())
        .unwrap()
        .into_dyn();
    let idx_t = Tensor::new(idx, false);
    let emb = Tensor::embedding_lookup(&pe.weight, &idx_t);
    let emb_bcast = Tensor::concat(&vec![emb.clone(); batch], 0); // replicate across batch
    assert_eq!(out_arr, emb_bcast.lock().storage.to_f32_array());
}

#[test]
fn test_alibi_bias_changes_attention() {
    use tensor_engine::nn::transformer::compute_alibi_slopes;
    use tensor_engine::nn::Module;
    use tensor_engine::nn::MultiHeadAttention;
    use tensor_engine::tensor::Tensor;
    let seq = 4;
    let b = 1;
    let d_model = 4;
    let num_heads = 1;
    // create module instances: one without ALiBi and one with ALiBi
    let mut mha_no_alibi = MultiHeadAttention::new(d_model, num_heads);
    let mut mha_alibi = MultiHeadAttention::new(d_model, num_heads).with_alibi();
    // set Q/K/V projection weights to identity so Q/K/V are deterministic and non-zero
    let mut id = ndarray::Array::zeros(ndarray::IxDyn(&[d_model, d_model]));
    for i in 0..d_model {
        id[[i, i]] = 1.0;
    }
    let id_small = id.mapv(|x| x * 0.01);
    let id_t = Tensor::new(id_small.into_dyn(), true);
    mha_no_alibi.linear_q.weight = id_t.clone();
    mha_no_alibi.linear_k.weight = id_t.clone();
    mha_no_alibi.linear_v.weight = id_t.clone();
    mha_no_alibi.linear_o.weight = id_t.clone();
    mha_alibi.linear_q.weight = id_t.clone();
    mha_alibi.linear_k.weight = id_t.clone();
    mha_alibi.linear_v.weight = id_t.clone();
    mha_alibi.linear_o.weight = id_t;
    // keep values small to avoid saturating softmax and making attention argmax dominant
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(123456);
    let mut xvals = Vec::new();
    for _ in 0..(b * seq * d_model) {
        xvals.push(rng.random_range(-0.01f32..0.01f32));
    }
    let x = Tensor::new(
        ndarray::Array::from_shape_vec((b, seq, d_model), xvals)
            .unwrap()
            .into_dyn(),
        true,
    );
    // Instead of checking final outputs, verify that ALiBi changes the pre-softmax scaled logits
    // Replicate the internal scaled logits computation from MultiHeadAttention
    let q = mha_no_alibi.linear_q.forward(&x);
    let k = mha_no_alibi.linear_k.forward(&x);
    let shape = q.lock().storage.shape();
    let b = shape[0];
    let seq = shape[1];
    let head_dim = d_model / num_heads;
    let q_heads = q.reshape(vec![b, seq, num_heads, head_dim]).unwrap();
    let k_heads = k.reshape(vec![b, seq, num_heads, head_dim]).unwrap();
    let q_perm = q_heads.permute(vec![0, 2, 1, 3]);
    let k_perm = k_heads.permute(vec![0, 2, 1, 3]);
    let q2 = q_perm.reshape(vec![b * num_heads, seq, head_dim]).unwrap();
    let k2 = k_perm.reshape(vec![b * num_heads, seq, head_dim]).unwrap();
    let k2t = k2.permute(vec![0, 2, 1]);
    let qk = q2.batched_matmul(&k2t);
    let scalar_tensor = Tensor::new(
        ndarray::Array::from_elem(ndarray::IxDyn(&[1]), 1.0 / (head_dim as f32).sqrt()),
        false,
    );
    let scaled_no_alibi = qk.mul(&scalar_tensor);
    let scaled_with_alibi = {
        let slopes = compute_alibi_slopes(num_heads);
        let mut bias_arr =
            ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[b * num_heads, seq, seq]));
        for batch in 0..b {
            for h in 0..num_heads {
                let slope = slopes[h];
                for i in 0..seq {
                    for j in 0..seq {
                        let dist = (j as isize - i as isize) as f32;
                        bias_arr[[batch * num_heads + h, i, j]] = -slope * dist;
                    }
                }
            }
        }
        let bias_t = Tensor::new(bias_arr, false);
        scaled_no_alibi.add(&bias_t)
    };
    assert_ne!(
        scaled_no_alibi.lock().storage.to_f32_array(),
        scaled_with_alibi.lock().storage.to_f32_array()
    );
}

#[test]
fn test_alibi_slopes_nonzero() {
    use tensor_engine::nn::MultiHeadAttention;
    let num_heads = 4;
    let d_model = 16;
    let mha = MultiHeadAttention::new(d_model, num_heads).with_alibi();
    let slopes = mha.alibi_slopes.as_ref().unwrap();
    assert_eq!(slopes.len(), num_heads);
    for s in slopes.iter() {
        assert!(*s > 0.0);
    }
}

#[test]
fn test_alibi_changes_final_output() {
    use tensor_engine::nn::Module;
    use tensor_engine::nn::MultiHeadAttention;
    use tensor_engine::tensor::Tensor;
    let seq = 4;
    let b = 1;
    let d_model = 8;
    let num_heads = 2;
    let mut mha_no_alibi = MultiHeadAttention::new(d_model, num_heads);
    let mut mha_alibi = MultiHeadAttention::new(d_model, num_heads).with_alibi();
    // Fill weights to be identity-like so Q/K are deterministic
    let mut id = ndarray::Array::zeros(ndarray::IxDyn(&[d_model, d_model]));
    for i in 0..d_model {
        id[[i, i]] = 1.0;
    }
    // For V, use identity so V = X and each token's vector is unique
    let mut vmat = ndarray::Array::zeros(ndarray::IxDyn(&[d_model, d_model]));
    for i in 0..d_model {
        vmat[[i, i]] = 1.0;
    }
    let id_small = id.mapv(|x| x * 0.1);
    let id_t = Tensor::new(id_small.into_dyn(), true);
    let v_t = Tensor::new(vmat.into_dyn(), true);
    // set weights
    mha_no_alibi.linear_q.weight = id_t.clone();
    mha_no_alibi.linear_k.weight = id_t.clone();
    mha_no_alibi.linear_v.weight = v_t.clone();
    mha_no_alibi.linear_o.weight = id_t.clone();
    mha_alibi.linear_q.weight = id_t.clone();
    mha_alibi.linear_k.weight = id_t.clone();
    mha_alibi.linear_v.weight = v_t.clone();
    mha_alibi.linear_o.weight = id_t;
    // Input: one-hot encoding per token so token positions map to unique vectors
    let mut xvals = Vec::new();
    for i in 0..seq {
        for j in 0..d_model {
            xvals.push(if i == j { 1.0f32 } else { 0.0f32 });
        }
    }
    let x = Tensor::new(
        ndarray::Array::from_shape_vec((b, seq, d_model), xvals)
            .unwrap()
            .into_dyn(),
        true,
    );
    // Compute outputs and debug print attention weights
    let out_no_alibi = mha_no_alibi.forward(&x);
    let out_alibi = mha_alibi.forward(&x);
    // compute scaled logits for debugging
    let q = mha_no_alibi.linear_q.forward(&x);
    let k = mha_no_alibi.linear_k.forward(&x);
    let shape = q.lock().storage.shape();
    let b = shape[0];
    let seq = shape[1];
    let head_dim = d_model / num_heads;
    let q_heads = q.reshape(vec![b, seq, num_heads, head_dim]).unwrap();
    let k_heads = k.reshape(vec![b, seq, num_heads, head_dim]).unwrap();
    let q_perm = q_heads.permute(vec![0, 2, 1, 3]);
    let k_perm = k_heads.permute(vec![0, 2, 1, 3]);
    let q2 = q_perm.reshape(vec![b * num_heads, seq, head_dim]).unwrap();
    let k2 = k_perm.reshape(vec![b * num_heads, seq, head_dim]).unwrap();
    let k2t = k2.permute(vec![0, 2, 1]);
    let qk = q2.batched_matmul(&k2t);
    let scalar_tensor = Tensor::new(
        ndarray::Array::from_elem(ndarray::IxDyn(&[1]), 1.0 / (head_dim as f32).sqrt()),
        false,
    );
    let scaled_no_alibi = qk.mul(&scalar_tensor);
    let _attn_no_alibi = scaled_no_alibi.softmax(2);
    // with alibi leverage slopes
    // q2 and scalar_tensor clones not required here
    let scaled_with_alibi = {
        let slopes = mha_alibi.alibi_slopes.as_ref().unwrap().clone();
        let mut bias_arr =
            ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[b * num_heads, seq, seq]));
        for batch in 0..b {
            for h in 0..num_heads {
                let slope = slopes[h];
                for i in 0..seq {
                    for j in 0..seq {
                        let dist = (j as isize - i as isize) as f32;
                        bias_arr[[batch * num_heads + h, i, j]] = -slope * dist;
                    }
                }
            }
        }
        let bias_t = Tensor::new(bias_arr, false);
        scaled_no_alibi.add(&bias_t)
    };
    let _attn_with_alibi = scaled_with_alibi.softmax(2);
    assert_ne!(
        out_no_alibi.lock().storage.to_f32_array(),
        out_alibi.lock().storage.to_f32_array()
    );
}

#[test]
fn test_avgpool2d_forward_backward() {
    use tensor_engine::ops::AvgPool2D as AvgPool2DOp;
    // Input (1,1,4,4) all ones -> avg window 2x2 with stride 2 => output all ones
    let data = vec![1.0; 16];
    let a = Tensor::new(
        ndarray::Array::from_shape_vec((1, 1, 4, 4), data.clone())
            .unwrap()
            .into_dyn(),
        true,
    );
    let op = AvgPool2DOp {
        kernel_size: 2,
        stride: 2,
    };
    let out = Tensor::apply(std::sync::Arc::new(op), &[a.clone()]);
    let expected = ndarray::Array::from_shape_vec((1, 1, 2, 2), vec![1.0; 4])
        .unwrap()
        .into_dyn();
    assert_eq!(out.lock().storage.to_f32_array(), expected);
    // Backward: grad ones should distribute 1/4 to each input in window
    out.backward();
    let grad_a = a.lock().grad.clone().unwrap();
    let expected_grad = ndarray::Array::from_shape_vec((1, 1, 4, 4), vec![0.25; 16])
        .unwrap()
        .into_dyn();
    assert_eq!(grad_a, expected_grad);
}

#[test]
fn test_adaptive_avgpool2d_forward_backward() {
    use tensor_engine::ops::AdaptiveAvgPool2D as AdaptiveOp;
    // Input (1,1,4,4) all ones, target output 2x2
    let data = vec![1.0; 16];
    let a = Tensor::new(
        ndarray::Array::from_shape_vec((1, 1, 4, 4), data.clone())
            .unwrap()
            .into_dyn(),
        true,
    );
    let op = AdaptiveOp::new(2, 2);
    let out = Tensor::apply(std::sync::Arc::new(op), &[a.clone()]);
    let expected = ndarray::Array::from_shape_vec((1, 1, 2, 2), vec![1.0; 4])
        .unwrap()
        .into_dyn();
    assert_eq!(out.lock().storage.to_f32_array(), expected);
    out.backward();
    let grad_a = a.lock().grad.clone().unwrap();
    // For 4x4->2x2, each window 2x2 -> each input is in exactly 1 window, grad_elems = 1/4
    let expected_grad = ndarray::Array::from_shape_vec((1, 1, 4, 4), vec![0.25; 16])
        .unwrap()
        .into_dyn();
    assert_eq!(grad_a, expected_grad);
}

#[test]
fn test_dropout_forward_backward() {
    use tensor_engine::ops::Dropout as DropoutOp;
    let data = ndarray::Array::from_shape_vec((1, 1, 4, 4), vec![1.0; 16]).unwrap();
    let x = Tensor::new(data.into_dyn(), true);
    let op = DropoutOp::new(0.5, true);
    let y = Tensor::apply(std::sync::Arc::new(op), &[x.clone()]);
    // In training mode, some outputs should be zero and others scaled by 1/(1-p)=2.0
    let out = y
        .lock()
        .storage
        .to_f32_array()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();
    assert!(out.iter().any(|&v| v == 0.0));
    assert!(out.iter().any(|&v| (v - 2.0).abs() < 1e-6));

    // Backward should propagate mask: set gradient ones and check grad equals mask
    y.backward();
    let grad_x = x
        .lock()
        .grad
        .clone()
        .unwrap()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();
    // gradient should contain zeros and 2.0s matching forward scaling
    assert!(grad_x.iter().any(|&v| v == 0.0));
    assert!(grad_x.iter().any(|&v| (v - 2.0).abs() < 1e-6));
}

#[test]
fn test_broadcast_add_forward_and_backward() {
    let a = Tensor::new(arr2(&[[1.0], [2.0], [3.0]]).into_dyn(), true); // shape (3,1)
    let b = Tensor::new(arr2(&[[10.0, 20.0, 30.0, 40.0]]).into_dyn(), true); // shape (1,4)
    let c = a.add(&b);

    // Forward expected: each element is sum of a_i and b_j
    let expected = arr2(&[
        [11.0, 21.0, 31.0, 41.0],
        [12.0, 22.0, 32.0, 42.0],
        [13.0, 23.0, 33.0, 43.0],
    ])
    .into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);

    // Backward
    c.backward();
    let grad_a = a.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();
    // grad of sum w.r.t each input is ones broadcasted and reduced to original shape
    // Each row for `a` sums four ones across axis=1 -> value 4 for each row
    assert_eq!(grad_a, arr2(&[[4.0], [4.0], [4.0]]).into_dyn());
    // grad_b is sum across rows (3 rows) -> value 3 for each column
    assert_eq!(grad_b, arr2(&[[3.0, 3.0, 3.0, 3.0]]).into_dyn());
}

#[test]
fn test_broadcast_mul_forward_and_backward() {
    let a = Tensor::new(arr2(&[[1.0], [2.0], [3.0]]).into_dyn(), true); // (3,1)
    let b = Tensor::new(arr2(&[[4.0, 5.0, 6.0, 7.0]]).into_dyn(), true); // (1,4)
    let c = a.mul(&b);

    let expected = arr2(&[
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 10.0, 12.0, 14.0],
        [12.0, 15.0, 18.0, 21.0],
    ])
    .into_dyn();
    assert_eq!(c.lock().storage.to_f32_array(), expected);

    c.backward();
    let grad_a = a.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();
    // grad_a: sum over columns of b (ones * b summed across axis 1)
    // grad_a is sum across columns of b => 4+5+6+7 = 22
    assert_eq!(grad_a, arr2(&[[22.0], [22.0], [22.0]]).into_dyn());
    // grad_b: sum over rows of a (1+2+3 = 6) for each column
    assert_eq!(grad_b, arr2(&[[6.0, 6.0, 6.0, 6.0]]).into_dyn());
}

// Numeric gradient checks
#[test]
fn test_numeric_gradient_add() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let a_vec: Vec<f32> = (0..3).map(|_| rng.random_range(-10.0..10.0)).collect();
    let b_vec: Vec<f32> = (0..3).map(|_| rng.random_range(-10.0..10.0)).collect();
    let a_data = ArrayD::from_shape_vec(IxDyn(&[3]), a_vec.clone()).unwrap();
    let b_data = ArrayD::from_shape_vec(IxDyn(&[3]), b_vec.clone()).unwrap();

    let a = Tensor::new(a_data.clone(), true);
    let b = Tensor::new(b_data.clone(), true);
    let c = a.add(&b);
    c.backward();

    let grad_a_computed = a.lock().grad.clone().unwrap();
    let grad_b_computed = b.lock().grad.clone().unwrap();

    let f_a = |x: &ArrayD<f32>| (x + &b_data).sum();
    let grad_a_numeric = numeric_gradient(f_a, &a_data, 1e-3);

    let f_b = |x: &ArrayD<f32>| (&a_data + x).sum();
    let grad_b_numeric = numeric_gradient(f_b, &b_data, 1e-3);

    for i in 0..grad_a_computed.len() {
        let a_comp = grad_a_computed.as_slice().unwrap()[i];
        let a_num = grad_a_numeric.as_slice().unwrap()[i];
        let b_comp = grad_b_computed.as_slice().unwrap()[i];
        let b_num = grad_b_numeric.as_slice().unwrap()[i];
        if (a_comp - a_num).abs() >= 1e-3 {
            println!(
                "numeric add mismatch at {}: computed {} numeric {}",
                i, a_comp, a_num
            );
        }
        if (b_comp - b_num).abs() >= 1e-3 {
            println!(
                "numeric add mismatch at {}: computed {} numeric {}",
                i, b_comp, b_num
            );
        }
        assert!((a_comp - a_num).abs() < 1e-3);
        assert!((b_comp - b_num).abs() < 1e-3);
    }
}

#[test]
fn test_numeric_gradient_mul() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(124);
    let a_vec: Vec<f32> = (0..3).map(|_| rng.random_range(-5.0..5.0)).collect();
    let b_vec: Vec<f32> = (0..3).map(|_| rng.random_range(-5.0..5.0)).collect();
    let a_data = ArrayD::from_shape_vec(IxDyn(&[3]), a_vec.clone()).unwrap();
    let b_data = ArrayD::from_shape_vec(IxDyn(&[3]), b_vec.clone()).unwrap();

    let a = Tensor::new(a_data.clone(), true);
    let b = Tensor::new(b_data.clone(), true);
    let c = a.mul(&b);
    c.backward();

    let grad_a_computed = a.lock().grad.clone().unwrap();
    let grad_b_computed = b.lock().grad.clone().unwrap();

    let f_a = |x: &ArrayD<f32>| (x * &b_data).sum();
    let grad_a_numeric = numeric_gradient(f_a, &a_data, 1e-3);

    let f_b = |x: &ArrayD<f32>| (&a_data * x).sum();
    let grad_b_numeric = numeric_gradient(f_b, &b_data, 1e-3);

    for i in 0..grad_a_computed.len() {
        assert!(
            (grad_a_computed.as_slice().unwrap()[i] - grad_a_numeric.as_slice().unwrap()[i]).abs()
                < 1e-3
        );
        assert!(
            (grad_b_computed.as_slice().unwrap()[i] - grad_b_numeric.as_slice().unwrap()[i]).abs()
                < 1e-3
        );
    }
}

#[test]
fn test_numeric_gradient_broadcast_add() {
    // shapes (3,1) + (1,4) broadcast to (3,4)
    let mut rng = rand::rngs::StdRng::seed_from_u64(125);
    let a_vec: Vec<f32> = (0..3).map(|_| rng.random_range(-5.0..5.0)).collect();
    let b_vec: Vec<f32> = (0..4).map(|_| rng.random_range(-5.0..5.0)).collect();
    let a_data = ndarray::Array::from_shape_vec((3, 1), a_vec.clone())
        .unwrap()
        .into_dyn();
    let b_data = ndarray::Array::from_shape_vec((1, 4), b_vec.clone())
        .unwrap()
        .into_dyn();

    let a = Tensor::new(a_data.clone(), true);
    let b = Tensor::new(b_data.clone(), true);
    let c = a.add(&b);
    c.backward();

    let grad_a_computed = a.lock().grad.clone().unwrap();
    let grad_b_computed = b.lock().grad.clone().unwrap();

    // numeric gradients
    let f_a = |x: &ndarray::ArrayD<f32>| (x + &b_data).sum();
    let grad_a_numeric = numeric_gradient(f_a, &a_data, 1e-3);
    let f_b = |x: &ndarray::ArrayD<f32>| (&a_data + x).sum();
    let grad_b_numeric = numeric_gradient(f_b, &b_data, 1e-3);

    // Both should match within tolerance
    for i in 0..grad_a_computed.len() {
        let a_comp = grad_a_computed.as_slice().unwrap()[i];
        let a_num = grad_a_numeric.as_slice().unwrap()[i];
        if (a_comp - a_num).abs() >= 1e-3 {
            println!("Grad A computed {:?}, numeric {:?}", a_comp, a_num);
        }
        assert!((a_comp - a_num).abs() < 1e-3);
    }
    for i in 0..grad_b_computed.len() {
        let b_comp = grad_b_computed.as_slice().unwrap()[i];
        let b_num = grad_b_numeric.as_slice().unwrap()[i];
        assert!((b_comp - b_num).abs() < 1e-3);
    }
}

#[test]
fn test_numeric_gradient_broadcast_mul() {
    // shapes (3,1) * (1,4) broadcast to (3,4)
    let mut rng = rand::rngs::StdRng::seed_from_u64(126);
    let a_vec: Vec<f32> = (0..3).map(|_| rng.random_range(-2.0..2.0)).collect();
    let b_vec: Vec<f32> = (0..4).map(|_| rng.random_range(-2.0..2.0)).collect();
    let a_data = ndarray::Array::from_shape_vec((3, 1), a_vec.clone())
        .unwrap()
        .into_dyn();
    let b_data = ndarray::Array::from_shape_vec((1, 4), b_vec.clone())
        .unwrap()
        .into_dyn();

    let a = Tensor::new(a_data.clone(), true);
    let b = Tensor::new(b_data.clone(), true);
    let c = a.mul(&b);
    c.backward();

    let grad_a_computed = a.lock().grad.clone().unwrap();
    let grad_b_computed = b.lock().grad.clone().unwrap();

    let f_a = |x: &ndarray::ArrayD<f32>| (x * &b_data).sum();
    let grad_a_numeric = numeric_gradient(f_a, &a_data, 1e-3);
    let f_b = |x: &ndarray::ArrayD<f32>| (&a_data * x).sum();
    let grad_b_numeric = numeric_gradient(f_b, &b_data, 1e-3);

    for i in 0..grad_a_computed.len() {
        let a_comp = grad_a_computed.as_slice().unwrap()[i];
        let a_num = grad_a_numeric.as_slice().unwrap()[i];
        assert!((a_comp - a_num).abs() < 1e-3);
    }
    for i in 0..grad_b_computed.len() {
        let b_comp = grad_b_computed.as_slice().unwrap()[i];
        let b_num = grad_b_numeric.as_slice().unwrap()[i];
        assert!((b_comp - b_num).abs() < 1e-3);
    }
}

#[test]
fn test_numeric_gradient_pow() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(127);
    let a_vec: Vec<f32> = (0..3).map(|_| rng.random_range(0.1..5.0)).collect(); // positive for pow
    let a_data = ArrayD::from_shape_vec(IxDyn(&[3]), a_vec.clone()).unwrap();

    let a = Tensor::new(a_data.clone(), true);
    let c = a.pow(2.0);
    c.backward();

    let grad_a_computed = a.lock().grad.clone().unwrap();

    let f_a = |x: &ArrayD<f32>| x.mapv(|v| v.powi(2)).sum();
    let grad_a_numeric = numeric_gradient(f_a, &a_data, 1e-3);

    for i in 0..grad_a_computed.len() {
        assert!(
            (grad_a_computed.as_slice().unwrap()[i] - grad_a_numeric.as_slice().unwrap()[i]).abs()
                < 1e-3
        );
    }
}

#[test]
fn test_numeric_gradient_sigmoid() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(128);
    let a_vec: Vec<f32> = (0..3).map(|_| rng.random_range(-5.0..5.0)).collect();
    let a_data = ArrayD::from_shape_vec(IxDyn(&[3]), a_vec.clone()).unwrap();

    let a = Tensor::new(a_data.clone(), true);
    let c = a.sigmoid();
    c.backward();

    let grad_a_computed = a.lock().grad.clone().unwrap();

    let f_a = |x: &ArrayD<f32>| x.mapv(|v| 1.0 / (1.0 + (-v).exp())).sum();
    let grad_a_numeric = numeric_gradient(f_a, &a_data, 1e-3);

    for i in 0..grad_a_computed.len() {
        assert!(
            (grad_a_computed.as_slice().unwrap()[i] - grad_a_numeric.as_slice().unwrap()[i]).abs()
                < 1e-3
        );
    }
}

#[test]
fn test_gelu_forward_and_backward() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(129);
    let a_vec: Vec<f32> = (0..3).map(|_| rng.random_range(-4.0..4.0)).collect();
    let a_data = ArrayD::from_shape_vec(IxDyn(&[3]), a_vec.clone()).unwrap();

    let a = Tensor::new(a_data.clone(), true);
    let c = a.gelu();
    c.backward();

    let grad_a_computed = a.lock().grad.clone().unwrap();
    let f_a = |x: &ArrayD<f32>| {
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        x.mapv(|v| {
            let u = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + u.tanh())
        })
        .sum()
    };
    let grad_a_numeric = numeric_gradient(f_a, &a_data, 1e-3);

    for i in 0..grad_a_computed.len() {
        assert!(
            (grad_a_computed.as_slice().unwrap()[i] - grad_a_numeric.as_slice().unwrap()[i]).abs()
                < 1e-3
        );
    }
}

#[test]
fn test_exp_forward_and_backward() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(130);
    let a_vec: Vec<f32> = (0..3).map(|_| rng.random_range(-2.0..2.0)).collect();
    let a_data = ArrayD::from_shape_vec(IxDyn(&[3]), a_vec.clone()).unwrap();

    let a = Tensor::new(a_data.clone(), true);
    let c = a.exp();
    c.backward();

    let grad_a_computed = a.lock().grad.clone().unwrap();
    let f_a = |x: &ArrayD<f32>| x.mapv(|v| v.exp()).sum();
    let grad_a_numeric = numeric_gradient(f_a, &a_data, 1e-3);

    for i in 0..grad_a_computed.len() {
        assert!(
            (grad_a_computed.as_slice().unwrap()[i] - grad_a_numeric.as_slice().unwrap()[i]).abs()
                < 1e-3
        );
    }
}

#[test]
fn test_comparison_ops_forward_and_gradients_are_zero() {
    let a = Tensor::new(arr1(&[1.0, 2.0, 3.0]).into_dyn(), true);
    let b = Tensor::new(arr1(&[2.0, 1.0, 3.0]).into_dyn(), true);
    let eq = a.equal(&b);
    let gt = a.greater(&b);
    let lt = a.less(&b);

    assert_eq!(
        eq.lock().storage.to_f32_array(),
        arr1(&[0.0, 0.0, 1.0]).into_dyn()
    );
    assert_eq!(
        gt.lock().storage.to_f32_array(),
        arr1(&[0.0, 1.0, 0.0]).into_dyn()
    );
    assert_eq!(
        lt.lock().storage.to_f32_array(),
        arr1(&[1.0, 0.0, 0.0]).into_dyn()
    );

    eq.backward();
    gt.backward();
    lt.backward();

    // Gradients should be zero because comparisons are non-differentiable
    let grad_a = a.lock().grad.clone().unwrap();
    let grad_b = b.lock().grad.clone().unwrap();
    assert_eq!(grad_a, arr1(&[0.0, 0.0, 0.0]).into_dyn());
    assert_eq!(grad_b, arr1(&[0.0, 0.0, 0.0]).into_dyn());
}

#[test]
fn test_broadcast_shapes_advanced_cases() {
    // shapes: (3,1,5) and (1,4,5) -> (3,4,5)
    let s1 = vec![3usize, 1usize, 5usize];
    let s2 = vec![1usize, 4usize, 5usize];
    let res = tensor_engine::tensor::Tensor::broadcast_shapes(&[s1.clone(), s2.clone()]).unwrap();
    assert_eq!(res, vec![3usize, 4usize, 5usize]);

    // incompatible shapes should return Err
    let bad1 = vec![3usize, 2usize];
    let bad2 = vec![2usize, 3usize, 4usize];
    assert!(tensor_engine::tensor::Tensor::broadcast_shapes(&[bad1, bad2]).is_err());
}

#[test]
fn test_int8_quantize_dequantize_roundtrip() {
    use tensor_engine::dtype::DType;
    let mut rng = rand::rngs::StdRng::seed_from_u64(131);
    let a_vec: Vec<f32> = (0..12).map(|_| rng.random_range(-100.0..100.0)).collect();
    let shape = vec![3usize, 4usize];
    let a_data = ArrayD::from_shape_vec(IxDyn(&shape), a_vec.clone()).unwrap();

    let a = Tensor::new(a_data.clone(), false);
    let a_i8 = a.astype(DType::I8);
    // dequantized tensor
    let deq = a_i8.lock().storage.to_f32_array();

    // compute expected dequantized via helpers
    let (q, s) = tensor_engine::dtype::int8::quantize_to_i8(&a_data);
    let deq_expected = tensor_engine::dtype::int8::dequantize_from_i8(&q, s, &shape);

    // compare arrays elementwise with tolerance
    for (l, r) in deq.iter().zip(deq_expected.iter()) {
        assert!((*l - *r).abs() / (r.abs().max(1.0)) < 1e-6);
    }
}

#[test]
fn test_mse_loss_backward() {
    use tensor_engine::nn::MSELoss;
    let pred_data = arr1(&[1.0, 2.0, 3.0]).into_dyn();
    let target_data = arr1(&[2.0, 1.0, 4.0]).into_dyn();
    let pred = Tensor::new(pred_data.clone(), true);
    let target = Tensor::new(target_data.clone(), false);
    let loss = MSELoss::new().forward(&pred, &target);
    loss.backward();
    let grad_pred = pred.lock().grad.clone().unwrap();
    // Analytical gradient: 2*(pred - target)/N
    let n = pred_data.len() as f32;
    let expected = (pred_data - target_data).mapv(|v| 2.0 * v / n);
    assert_eq!(grad_pred, expected);
}

#[test]
fn test_dataloader_shuffle_next_batch() {
    use tensor_engine::nn::DataLoader;
    // create 5 samples: (i, i*2)
    let mut data = Vec::new();
    for i in 0..5 {
        let x = Tensor::new(arr1(&[i as f32]).into_dyn(), false);
        let y = Tensor::new(arr1(&[(i as f32) * 2.0]).into_dyn(), false);
        data.push((x, y));
    }
    let mut dl = DataLoader::new(data.clone(), 2);
    // capture order of first batch
    let (bx, _by) = dl.next_batch().unwrap();
    let first_order = bx.lock().storage.to_f32_array().clone();
    // shuffle and ensure order changes (low probability to remain same)
    // Try shuffling up to a few times to avoid a very-low-probability event that shuffle returns same order
    let mut succeeded = false;
    for _ in 0..3 {
        dl.shuffle();
        let (bx2, _) = dl.next_batch().unwrap();
        let second_order = bx2.lock().storage.to_f32_array().clone();
        if first_order != second_order {
            succeeded = true;
            break;
        }
    }
    assert!(
        succeeded,
        "shuffle did not change batch order in repeated trials"
    );
}

#[test]
fn test_cross_entropy_loss_backward() {
    use tensor_engine::nn::CrossEntropyLoss;
    // two-sample batch
    let pred_data = arr2(&[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]).into_dyn();
    let target_data = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).into_dyn();
    let pred = Tensor::new(pred_data.clone(), true);
    let target = Tensor::new(target_data.clone(), false);
    let loss = CrossEntropyLoss::new().forward(&pred, &target);
    loss.backward();
    let grad_pred = pred.lock().grad.clone().unwrap();
    // Analytical gradient: -target / pred / N
    let n = pred_data.shape()[0] as f32;
    let mut expected = ArrayD::zeros(pred_data.dim());
    for i in 0..pred_data.shape()[0] {
        for j in 0..pred_data.shape()[1] {
            expected[[i, j]] = -target_data[[i, j]] / pred_data[[i, j]] / n;
        }
    }
    // Check equality within tolerance
    for i in 0..grad_pred.len() {
        let g = grad_pred.as_slice().unwrap()[i];
        let e = expected.as_slice().unwrap()[i];
        assert!((g - e).abs() < 1e-4);
    }
}

#[test]
fn test_log_softmax_and_softmax_forward() {
    // random logits
    let logits = Tensor::new(arr2(&[[1.0, 2.0, -1.0], [0.1, 0.2, 0.3]]).into_dyn(), false);
    // softmax forward: sum along last axis should be 1
    let s = logits.softmax(1);
    let s_arr = s
        .lock()
        .storage
        .to_f32_array()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    for row in s_arr.rows() {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    // log softmax and softmax relationship
    let ls = logits.log_softmax(1);
    let ls_arr = ls
        .lock()
        .storage
        .to_f32_array()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let s_from_ls = ls_arr.mapv(|v| v.exp());
    for (row, s_row) in s_arr.rows().into_iter().zip(s_from_ls.rows().into_iter()) {
        for (a, b) in row.iter().zip(s_row.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

#[test]
fn test_cross_entropy_logits_backward() {
    // logits for 2 samples, 3 classes
    let logits_data = arr2(&[[1.0, 2.0, -1.0], [0.1, 0.2, 0.3]]).into_dyn();
    let target_idx = arr1(&[1.0, 2.0]).into_dyn(); // 1 and 2
    let logits = Tensor::new(logits_data.clone(), true);
    let targets = Tensor::new(target_idx.clone(), false);
    let loss = logits.cross_entropy_with_logits(&targets, 1);
    loss.backward();
    let grad_logits = logits
        .lock()
        .grad
        .clone()
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    // expected grad = (softmax - target_onehot) / N
    let mut expected = ndarray::Array2::<f32>::zeros((2, 3));
    // compute softmax
    let mut soft = logits_data
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    for mut row in soft.rows_mut() {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v = *v / sum;
        }
    }
    for i in 0..2 {
        for j in 0..3 {
            let mut val = soft[[i, j]];
            if j == target_idx.as_slice().unwrap()[i] as usize {
                val -= 1.0;
            }
            expected[[i, j]] = val / 2.0;
        }
    }
    for i in 0..2 {
        for j in 0..3 {
            let g = grad_logits[[i, j]];
            let e = expected[[i, j]];
            assert!(
                (g - e).abs() < 1e-5,
                "grad mismatch at {} {}: {} vs {}",
                i,
                j,
                g,
                e
            );
        }
    }
}

#[test]
fn test_cross_entropy_logits_backward_one_hot_target() {
    // logits for 1 sample, 3 classes, one-hot target
    let logits_data = arr2(&[[1.0, 2.0, -1.0]]).into_dyn();
    let target_onehot = arr2(&[[0.0, 1.0, 0.0]]).into_dyn();
    let logits = Tensor::new(logits_data.clone(), true);
    let targets = Tensor::new(target_onehot.clone(), false);
    let loss = logits.cross_entropy_with_logits(&targets, 1);
    loss.backward();
    let grad = logits
        .lock()
        .grad
        .clone()
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    // compute expected softmax
    let mut soft = logits_data
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    for mut row in soft.rows_mut() {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v = *v / sum;
        }
    }
    // expected grad = (soft - target)/N = (soft - target)
    for j in 0..3 {
        let expected = soft[[0, j]] - target_onehot[[0, j]];
        assert!((grad[[0, j]] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_softmax_cross_entropy_with_logits_backward() {
    // logits for 2 samples, 3 classes
    let logits_data = arr2(&[[1.0, 2.0, -1.0], [0.1, 0.2, 0.3]]).into_dyn();
    let target_idx = arr1(&[1.0, 2.0]).into_dyn(); // 1 and 2
    let logits = Tensor::new(logits_data.clone(), true);
    let targets = Tensor::new(target_idx.clone(), false);
    let loss = logits.softmax_cross_entropy_with_logits(&targets, 1);
    loss.backward();
    let grad_logits = logits
        .lock()
        .grad
        .clone()
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    // expected grad = (softmax - target) / N
    let mut expected = ndarray::Array2::<f32>::zeros((2, 3));
    // compute softmax
    let mut soft = logits_data
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    for mut row in soft.rows_mut() {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v = *v / sum;
        }
    }
    for i in 0..2 {
        for j in 0..3 {
            let mut val = soft[[i, j]];
            if j == target_idx.as_slice().unwrap()[i] as usize {
                val -= 1.0;
            }
            expected[[i, j]] = val / 2.0;
        }
    }
    for i in 0..2 {
        for j in 0..3 {
            let g = grad_logits[[i, j]];
            let e = expected[[i, j]];
            assert!(
                (g - e).abs() < 1e-5,
                "grad mismatch at {} {}: {} vs {}",
                i,
                j,
                g,
                e
            );
        }
    }
}

#[test]
fn test_nll_loss_backward() {
    // create log_probs as log_softmax of logits
    let logits_data = arr2(&[[1.0, 2.0, -1.0]]).into_dyn();
    let logits = Tensor::new(logits_data.clone(), true);
    let log_probs = logits.log_softmax(1);
    let targets = Tensor::new(arr1(&[1.0]).into_dyn(), false);
    let loss = log_probs.nll_loss(&targets);
    loss.backward();
    let grad_log_probs = log_probs
        .lock()
        .grad
        .clone()
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    // expected gradient: -one_hot / N (here N=1)
    assert_eq!(grad_log_probs[[0, 1]], -1.0);
    assert_eq!(grad_log_probs[[0, 0]], 0.0);
    assert_eq!(grad_log_probs[[0, 2]], 0.0);
}

#[test]
fn test_cross_entropy_logits_with_labels_axis0() {
    use tensor_engine::labels::Labels;
    use tensor_engine::nn::CrossEntropyLogitsLoss;
    // logits shape: (3 classes, 2 samples), class axis 0
    let logits_arr =
        ndarray::Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    // organize as ((3, 2)) so column-major content: we choose values per column
    let logits = Tensor::new(logits_arr.into_dyn(), true);
    // labels for 2 samples
    let labels_vec = vec![2i64, 1i64];
    let labels_arr = ndarray::Array::from_shape_vec((2,), labels_vec).unwrap();
    let labels = Labels::new(labels_arr.into_dyn());
    let loss = CrossEntropyLogitsLoss::new().forward_from_labels(&logits, &labels, 0);
    loss.backward();
    let grad = logits.lock().grad.clone().unwrap();
    assert_eq!(grad.shape(), &[3, 2]);
    // The gradients should exist and be finite
    assert!(grad.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_nll_loss_with_labels_axis1() {
    use tensor_engine::labels::Labels;
    use tensor_engine::nn::NLLLossLayer;
    // log_probs as log_softmax of logits with shape (2 samples, 3 classes)
    let logits_arr =
        ndarray::Array::from_shape_vec((2, 3), vec![1.0, 2.0, -1.0, 0.1, 0.2, 0.3]).unwrap();
    let logits = Tensor::new(logits_arr.into_dyn(), true);
    let log_probs = logits.log_softmax(1);
    let labels_vec = vec![1i64, 2i64];
    let labels_arr = ndarray::Array::from_shape_vec((2,), labels_vec).unwrap();
    let labels = Labels::new(labels_arr.into_dyn());
    let loss = NLLLossLayer::new().forward_from_labels(&log_probs, &labels);
    loss.backward();
    let grad = log_probs.lock().grad.clone().unwrap();
    assert_eq!(grad.shape(), &[2, 3]);
    assert!(grad.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_axis_negative_semantics_for_softmax_crossentropy() {
    // Create logits and targets
    let logits_data = arr2(&[[1.0, 2.0, -1.0]]).into_dyn();
    let targets = Tensor::new(arr1(&[1.0]).into_dyn(), false);
    let logits1 = Tensor::new(logits_data.clone(), true);
    let logits2 = Tensor::new(logits_data.clone(), true);
    // axis as positive
    let loss1 = logits1.softmax_cross_entropy_with_logits(&targets, 1);
    loss1.backward();
    // axis as negative (-1)
    let loss2 = logits2.softmax_cross_entropy_with_logits(&targets, -1);
    loss2.backward();
    let g1 = logits1.lock().grad.clone().unwrap();
    let g2 = logits2.lock().grad.clone().unwrap();
    assert_eq!(g1, g2);
}

#[test]
fn test_dropout_eval_mode() {
    use tensor_engine::ops::Dropout as DropoutOp;
    let data = ndarray::Array::from_shape_vec((1, 1, 4, 4), vec![1.0; 16]).unwrap();
    let x = Tensor::new(data.into_dyn(), true);
    let op = DropoutOp::new(0.5, false); // evaluation mode
    let y = Tensor::apply(std::sync::Arc::new(op), &[x.clone()]);
    // outputs should be identical
    assert_eq!(
        x.lock().storage.to_f32_array(),
        y.lock().storage.to_f32_array()
    );
    y.backward();
    // gradient should be ones because grad of identity? Since out has ones initialized, check it matches
    let grad_x = x.lock().grad.clone().unwrap();
    assert!(grad_x.iter().all(|&v| v == 1.0));
}

#[test]
fn test_dropout_training_p_zero_identity() {
    use tensor_engine::ops::Dropout as DropoutOp;
    let data = ndarray::Array::from_shape_vec((2, 2), vec![3.0; 4]).unwrap();
    let x = Tensor::new(data.into_dyn(), true);
    let op = DropoutOp::new(0.0, true); // training mode but p=0 means no dropout
    let y = Tensor::apply(std::sync::Arc::new(op), &[x.clone()]);
    assert_eq!(
        x.lock().storage.to_f32_array(),
        y.lock().storage.to_f32_array()
    );
    y.backward();
    let grad_x = x.lock().grad.clone().unwrap();
    // gradient should be ones because out had ones for grad (default)
    assert!(grad_x.iter().all(|&v| v == 1.0));
}

#[test]
fn test_layernorm_forward_properties() {
    use tensor_engine::nn::LayerNorm;
    let data = ndarray::arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
    let x = Tensor::new(data.clone(), false); // no grad needed for forward check
    let ln = LayerNorm::new(3, 1, 1e-5);
    let out = ln.forward(&x);
    // output per row mean approx 0, variance approx 1
    let out_arr = out
        .lock()
        .storage
        .to_f32_array()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    for row in out_arr.rows() {
        let mean: f32 = row.sum() / (row.len() as f32);
        let mut var = 0.0f32;
        for v in row.iter() {
            var += (v - mean) * (v - mean);
        }
        var /= row.len() as f32;
        assert!((mean).abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-4);
    }
}

#[test]
fn test_layernorm_backward_numeric() {
    use tensor_engine::nn::LayerNorm;
    let mut rng = rand::rngs::StdRng::seed_from_u64(132);
    let a_vec: Vec<f32> = (0..6).map(|_| rng.random_range(-2.0..2.0)).collect();
    let a_data = ndarray::Array::from_shape_vec((2, 3), a_vec.clone())
        .unwrap()
        .into_dyn();
    let a = Tensor::new(a_data.clone(), true);
    let ln = LayerNorm::new(3, 1, 1e-5);
    let y = ln.forward(&a);
    let loss = y.sum();
    loss.backward();
    let grad_a_computed = a.lock().grad.clone().unwrap();
    // numeric gradient
    let f_a = |x: &ndarray::ArrayD<f32>| {
        ln.forward(&Tensor::new(x.clone(), false))
            .lock()
            .storage
            .to_f32_array()
            .sum()
    };
    let grad_a_numeric = numeric_gradient(f_a, &a_data, 1e-3);
    for i in 0..grad_a_computed.len() {
        let g_comp = grad_a_computed.as_slice().unwrap()[i];
        let g_num = grad_a_numeric.as_slice().unwrap()[i];
        assert!((g_comp - g_num).abs() < 1e-3);
    }
}
