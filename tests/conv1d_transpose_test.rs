use tensor_engine::nn::{Conv1D, ConvTranspose1D, Module};
use tensor_engine::tensor::Tensor;

#[test]
fn conv1d_transpose_shape_recover() {
    let in_channels = 2usize;
    let out_channels = 4usize;
    let kernel = 4usize;
    let stride = 2usize;
    let padding = 1usize;

    let conv = Conv1D::new(in_channels, out_channels, kernel, stride, padding, true);
    let deconv = ConvTranspose1D::new(out_channels, in_channels, kernel, stride, padding, true);

    let len = 32usize;
    let data = ndarray::Array::from_shape_vec((1usize, in_channels, len), vec![0.1f32; 1*in_channels*len]).unwrap().into_dyn();
    let input = Tensor::new(data, false);

    let out = conv.forward(&input);
    let rec = deconv.forward(&out);

    assert_eq!(input.lock().storage.shape(), rec.lock().storage.shape());
}
