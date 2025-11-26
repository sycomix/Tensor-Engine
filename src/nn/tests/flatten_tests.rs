use crate::nn::Flatten;
use crate::nn::{Conv2D, Linear, Module, Sequential};
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};

#[test]
fn flatten_works_on_4d() {
    let batch = 2usize;
    let c = 3usize;
    let h = 8usize;
    let w = 8usize;
    let total = batch * c * h * w;
    let flat: Vec<f32> = vec![1.0f32; total];
    let arr = Array::from_shape_vec(IxDyn(&[batch, c, h, w]), flat).unwrap();
    let input = Tensor::new(arr.into_dyn(), false);
    let fm = Flatten::default();
    let out = fm.forward(&input);
    assert_eq!(out.lock().data.ndim(), 2);
    assert_eq!(out.lock().data.shape(), &[batch, c * h * w]);
}

#[test]
fn flatten_integrates_with_sequential() {
    // Create a small conv network with Flatten and a Linear layer
    let model = Sequential::new()
        .add(Conv2D::new(3, 8, 3, 1, 1, true))
        .add(Flatten::default())
        .add(Linear::new(8 * 28 * 28, 128, true));

    let batch = 2usize;
    let h = 28usize;
    let w = 28usize;
    let flat: Vec<f32> = vec![0f32; batch * 3 * h * w];
    let arr = Array::from_shape_vec(IxDyn(&[batch, 3, h, w]), flat).unwrap();
    let input = Tensor::new(arr.into_dyn(), false);
    let logits = model.forward(&input);
    assert_eq!(logits.lock().data.shape()[0], batch);
}
