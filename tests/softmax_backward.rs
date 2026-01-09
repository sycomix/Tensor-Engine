use tensor_engine::ops::{Softmax, Operation};
use tensor_engine::tensor::Tensor;
use ndarray::{array, ArrayD, IxDyn, Axis};

#[test]
fn softmax_backward_sum_zero() {
    let x = array![[0.1f32, 0.2, -0.3, 0.0, 0.5, -0.1],[0.0,0.1,0.2,0.3,0.4,0.5],[0.5,0.4,0.3,0.2,0.1,0.0],[ -0.1,0.0,0.1,0.2,0.3,0.4]].into_dyn();
    let t = Tensor::new(x.clone(), true);
    let soft = Softmax::new(1);
    let mut out = ArrayD::zeros(IxDyn(&[4,6]));
    soft.forward(&[t.clone()], &mut out);
    // print forward softmax first row
    println!("forward out first row: {:?}", out.index_axis(Axis(0), 0).to_owned());
    // recompute softmax using same loop here to compare
    let mut y2 = x.clone();
    let last_axis = 1usize;
    for mut lane in y2.axis_iter_mut(Axis(last_axis)) {
        let max = lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in lane.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in lane.iter_mut() {
            *v = *v / sum;
        }
    }
    let y2_row = y2.index_axis(Axis(0), 0).to_owned();
    println!("recomputed y2 first row: {:?} sum:{}", y2_row, y2_row.sum());
    // output grad ones
    let out_grad = ArrayD::from_elem(IxDyn(&[4,6]), 1.0f32);
    let grads = soft.backward(&[t.clone()], &out_grad);
    let gin = &grads[0];
    for v in gin.iter() {
        assert!(v.abs() < 1e-6, "non-zero grad: {}", v);
    }
}
