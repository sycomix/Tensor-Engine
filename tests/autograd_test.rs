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
    c.backward();

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
    assert_eq!(c.lock().data, expected);
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
    assert_eq!(c.lock().data, expected);
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
    assert_eq!(c.lock().data, expected);
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
    assert_eq!(c.lock().data, expected);
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
    assert_eq!(c.lock().data, expected);
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
    assert_eq!(c.lock().data, expected);
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
    assert_eq!(c.lock().data, expected);
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
    assert!((c.lock().data[0] - expected[0]).abs() < 1e-6);
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
    assert_eq!(c.lock().data, expected);
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
fn test_sum_forward() {
    let a = Tensor::new(arr1(&[1.0, 2.0, 3.0]).into_dyn(), false);
    let c = a.sum();

    let expected = arr0(6.0).into_dyn();
    assert_eq!(c.lock().data, expected);
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
    assert_eq!(c.lock().data, expected);
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
fn test_concat_forward() {
    let a = Tensor::new(arr1(&[1.0, 2.0]).into_dyn(), false);
    let b = Tensor::new(arr1(&[3.0, 4.0]).into_dyn(), false);
    let c = Tensor::concat(&[a, b], 0);

    let expected = arr1(&[1.0, 2.0, 3.0, 4.0]).into_dyn();
    assert_eq!(c.lock().data, expected);
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
    assert_eq!(c.lock().data, expected);
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
    assert_eq!(out.lock().data, expected);

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
    assert_eq!(c.lock().data, expected);

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
fn test_dropout_forward_backward() {
    use tensor_engine::ops::Dropout as DropoutOp;
    let data = ndarray::Array::from_shape_vec((1, 1, 4, 4), vec![1.0; 16]).unwrap();
    let x = Tensor::new(data.into_dyn(), true);
    let op = DropoutOp::new(0.5, true);
    let y = Tensor::apply(std::sync::Arc::new(op), &[x.clone()]);
    // In training mode, some outputs should be zero and others scaled by 1/(1-p)=2.0
    let out = y
        .lock()
        .data
        .clone()
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
    assert_eq!(c.lock().data, expected);

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
    assert_eq!(c.lock().data, expected);

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
    let mut rng = rand::thread_rng();
    let a_vec: Vec<f32> = (0..3).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let b_vec: Vec<f32> = (0..3).map(|_| rng.gen_range(-10.0..10.0)).collect();
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
    let mut rng = rand::thread_rng();
    let a_vec: Vec<f32> = (0..3).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let b_vec: Vec<f32> = (0..3).map(|_| rng.gen_range(-5.0..5.0)).collect();
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
    let mut rng = rand::thread_rng();
    let a_vec: Vec<f32> = (0..3).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let b_vec: Vec<f32> = (0..4).map(|_| rng.gen_range(-5.0..5.0)).collect();
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
    let mut rng = rand::thread_rng();
    let a_vec: Vec<f32> = (0..3).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let b_vec: Vec<f32> = (0..4).map(|_| rng.gen_range(-2.0..2.0)).collect();
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
    let mut rng = rand::thread_rng();
    let a_vec: Vec<f32> = (0..3).map(|_| rng.gen_range(0.1..5.0)).collect(); // positive for pow
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
    let mut rng = rand::thread_rng();
    let a_vec: Vec<f32> = (0..3).map(|_| rng.gen_range(-5.0..5.0)).collect();
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
    let first_order = bx.lock().data.clone();
    // shuffle and ensure order changes (low probability to remain same)
    dl.shuffle();
    let (bx2, _) = dl.next_batch().unwrap();
    let second_order = bx2.lock().data.clone();
    assert_ne!(first_order, second_order);
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
        .data
        .clone()
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
        .data
        .clone()
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
    assert_eq!(x.lock().data, y.lock().data);
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
    assert_eq!(x.lock().data, y.lock().data);
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
    let out_arr = out.lock().data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
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
    let mut rng = rand::thread_rng();
    let a_vec: Vec<f32> = (0..6).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let a_data = ndarray::Array::from_shape_vec((2, 3), a_vec.clone()).unwrap().into_dyn();
    let a = Tensor::new(a_data.clone(), true);
    let ln = LayerNorm::new(3, 1, 1e-5);
    let y = ln.forward(&a);
    let loss = y.sum();
    loss.backward();
    let grad_a_computed = a.lock().grad.clone().unwrap();
    // numeric gradient
    let f_a = |x: &ndarray::ArrayD<f32>| ln.forward(&Tensor::new(x.clone(), false)).lock().data.sum();
    let grad_a_numeric = numeric_gradient(f_a, &a_data, 1e-3);
    for i in 0..grad_a_computed.len() {
        let g_comp = grad_a_computed.as_slice().unwrap()[i];
        let g_num = grad_a_numeric.as_slice().unwrap()[i];
        assert!((g_comp - g_num).abs() < 1e-3);
    }
}
