use ndarray::IxDyn;
use tensor_engine::nn::*;
use tensor_engine::nn::{
    ContinuousThoughtModule, ImageDecoder, TextDecoder, VideoDecoder,
};
// bring latent helpers via re-export
use tensor_engine::tensor::Tensor;

#[test]
fn test_vector_arithmetic_and_interpolation() {
    let a = Tensor::new(ndarray::Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn(), false);
    let b = Tensor::new(ndarray::Array::from_vec(vec![0.5, 1.0, 1.5]).into_dyn(), false);
    let c = Tensor::new(ndarray::Array::from_vec(vec![2.0, 2.0, 2.0]).into_dyn(), false);
    let res = vector_arithmetic(&a, &b, &c);
    // a - b + c = [1 - 0.5 +2, 2-1+2, 3-1.5+2] = [2.5,3.0,3.5]
    let arr = res.lock().storage.to_f32_array();
    assert_eq!(arr.as_slice().unwrap(), &[2.5, 3.0, 3.5]);

    // linear interpolation halfway should equal average
    let lin = linear_interpolate(&a, &c, 0.5);
    let arr2 = lin.lock().storage.to_f32_array();
    assert_eq!(arr2.as_slice().unwrap(), &[1.5, 2.0, 2.5]);

    // spherical interpolation of identical points returns same
    let slerp_same = spherical_interpolate(&a, &a, 0.3);
    let arr3 = slerp_same.lock().storage.to_f32_array();
    assert_eq!(arr3.as_slice().unwrap(), &[1.0, 2.0, 3.0]);

    // attribute edit: move "a" towards "c" by strength 1 should equal a+c
    let edited = attribute_edit(&a, &c, 1.0);
    let arr4 = edited.lock().storage.to_f32_array();
    assert_eq!(arr4.as_slice().unwrap(), &[3.0, 4.0, 5.0]);
}

#[test]
fn test_continuous_thought_module() {
    let mut ctm = ContinuousThoughtModule::new(4);
    let inp = Tensor::new(ndarray::Array::zeros(IxDyn(&[1, 4])), false);
    // initial state should be None
    assert!(ctm.get_state().is_none());
    let out = ctm.forward(&inp);
    assert_eq!(out.lock().storage.shape(), &[1, 4]);
    assert!(ctm.get_state().is_some());
    // calling again with same input just leaves state defined
    let _ = ctm.forward(&inp);
    assert!(ctm.get_state().is_some());
}

#[test]
fn test_decoders_shapes() {
    // image decoder with 2 layers doubles spatial dims twice
    let img_dec = ImageDecoder::new(3, 8, 2);
    let inp = Tensor::new(ndarray::Array::zeros(IxDyn(&[1, 3, 8, 8])), false);
    let out = img_dec.forward(&inp);
    let shape = out.lock().storage.shape().to_vec();
    // output channels final=3, dims 8*2*2
    assert_eq!(shape, vec![1, 3, 32, 32]);

    // video decoder: preserves time dimension
    let vid_dec = VideoDecoder::new(3, 8, 2);
    let vinp = Tensor::new(ndarray::Array::zeros(IxDyn(&[1, 3, 4, 8, 8])), false);
    let vout = vid_dec.forward(&vinp);
    let vshape = vout.lock().storage.shape().to_vec();
    assert_eq!(vshape, vec![1, 3, 4, 32, 32]);

    // text decoder shape check
    let mut txt_dec = TextDecoder::new(10, 16, 32, 2, 2).expect("create text decoder");
    let latent = Tensor::new(ndarray::Array::zeros(IxDyn(&[1, 5, 16])), false);
    let logits = txt_dec.forward(&latent);
    let sh = logits.lock().storage.shape().to_vec();
    assert_eq!(sh, vec![1, 5, 10]);
}

#[test]
fn test_pipeline_integration_shapes() {
    // simple pipeline: latent->text decoder to verify we can call multiple modules
    let mut txt_dec = TextDecoder::new(20, 8, 16, 2, 1).expect("text decoder");
    let inp = Tensor::new(ndarray::Array::zeros(IxDyn(&[2, 3, 8])), false);
    let _ = txt_dec.forward(&inp);
    // ensure parameters collection works without panic
    let params = txt_dec.parameters();
    assert!(!params.is_empty());
}
