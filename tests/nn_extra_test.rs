// use ndarray::arr2;  // unused
use tensor_engine::nn::{
    ConvBlock, Discriminator, Generator, LSTMCell, Linear, Module, RNNCell, SelfAttention,
    TransformerBlock,
};
use tensor_engine::tensor::Tensor;

#[test]
fn test_rnncell_forward_backward() {
    let b = 2usize;
    let in_dim = 3usize;
    let hid = 4usize;
    // batch inputs
    let input = Tensor::new(
        ndarray::Array::from_shape_vec((b, in_dim), vec![1.0; b * in_dim])
            .unwrap()
            .into_dyn(),
        true,
    );
    let h = Tensor::new(
        ndarray::Array::from_shape_vec((b, hid), vec![0.5; b * hid])
            .unwrap()
            .into_dyn(),
        true,
    );
    let rnn = RNNCell::new(in_dim, hid, true);
    let out = rnn.forward_step(&input, &h);
    // out shape should be [b, hid]
    assert_eq!(out.lock().storage.shape(), &[b, hid]);
    // Backward: call backward on sum to propagate
    let s = out.sum();
    s.backward();
    assert!(input.lock().grad.is_some());
    assert!(h.lock().grad.is_some());
}

#[test]
fn test_lstmcell_forward_backward() {
    let b = 1usize;
    let in_dim = 2usize;
    let hid = 3usize;
    let input = Tensor::new(
        ndarray::Array::from_shape_vec((b, in_dim), vec![0.2; b * in_dim])
            .unwrap()
            .into_dyn(),
        true,
    );
    let h = Tensor::new(
        ndarray::Array::from_shape_vec((b, hid), vec![0.0; b * hid])
            .unwrap()
            .into_dyn(),
        true,
    );
    let c = Tensor::new(
        ndarray::Array::from_shape_vec((b, hid), vec![0.0; b * hid])
            .unwrap()
            .into_dyn(),
        true,
    );
    let cell = LSTMCell::new(in_dim, hid, true);
    let (h2, c2) = cell.forward_step(&input, &h, &c);
    assert_eq!(h2.lock().storage.shape(), &[b, hid]);
    assert_eq!(c2.lock().storage.shape(), &[b, hid]);
    let s = h2.sum();
    s.backward();
    assert!(input.lock().grad.is_some());
}

#[test]
fn test_attention_forward_shape_and_grad() {
    let b = 1usize;
    let seq = 2usize;
    let dim = 3usize;
    let data = vec![0.1; b * seq * dim];
    let q = Tensor::new(
        ndarray::Array::from_shape_vec((b, seq, dim), data.clone())
            .unwrap()
            .into_dyn(),
        true,
    );
    let k = Tensor::new(
        ndarray::Array::from_shape_vec((b, seq, dim), data.clone())
            .unwrap()
            .into_dyn(),
        true,
    );
    let v = Tensor::new(
        ndarray::Array::from_shape_vec((b, seq, dim), data.clone())
            .unwrap()
            .into_dyn(),
        true,
    );
    let att = SelfAttention::new(dim);
    let out = att.forward_attention(&q, &k, &v);
    assert_eq!(out.lock().storage.shape(), &[b, seq, dim]);
    let s = out.sum();
    s.backward();
    assert!(q.lock().grad.is_some());
}

#[test]
fn test_transformer_forward_shape() {
    let b = 1usize;
    let seq = 2usize;
    let d = 4usize;
    let x = Tensor::new(
        ndarray::Array::from_shape_vec((b, seq, d), vec![0.3; b * seq * d])
            .unwrap()
            .into_dyn(),
        true,
    );
    let mut block = TransformerBlock::new(d, d * 2, 1).expect("create simple block");
    let out = block.forward_block(&x);
    assert_eq!(out.lock().storage.shape(), &[b, seq, d]);
}

#[test]
fn test_convblock_and_gan_forward() {
    // ConvBlock
    let input = Tensor::new(
        ndarray::Array::from_shape_vec((1, 1, 4, 4), vec![1.0; 16])
            .unwrap()
            .into_dyn(),
        false,
    );
    let block = ConvBlock::new(1, 1, 2, 1, 0, true, Some((2, 2)));
    let out = block.forward(&input);
    // Output shape: batch=1, channels=1, H_out = 3? kernel2 padding 0 stride 1 -> 3; pooled -> 1
    assert_eq!(out.lock().storage.shape().len(), 4);

    // Simple GAN MLP: generator -> Tanh output, discriminator -> sigm
    let g = Generator::new(vec![Box::new(Linear::new(4, 4, true))]);
    let d = Discriminator::new(vec![Box::new(Linear::new(4, 2, true))]);
    let z = Tensor::new(
        ndarray::Array::from_shape_vec((1, 4), vec![0.1; 4])
            .unwrap()
            .into_dyn(),
        true,
    );
    let gen = g.forward(&z);
    let disc = d.forward(&gen);
    assert_eq!(gen.lock().storage.shape(), &[1, 4]);
    assert_eq!(disc.lock().storage.shape(), &[1, 2]);
}
