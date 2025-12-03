use ndarray::Array;
use ndarray::IxDyn;
use rand::Rng;
use rand::SeedableRng;
use tensor_engine::nn::{AttentionVariant, Module, MultiHeadAttention};
use tensor_engine::tensor::Tensor;

fn set_identity_linear(mha: &mut MultiHeadAttention) {
    let d = mha.d_model;
    let mut id = ndarray::Array::zeros(IxDyn(&[d, d]));
    for i in 0..d {
        id[[i, i]] = 1.0;
    }
    let id_t = Tensor::new(id.into_dyn(), true);
    mha.linear_q.weight = id_t.clone();
    mha.linear_k.weight = id_t.clone();
    mha.linear_v.weight = id_t.clone();
    mha.linear_o.weight = id_t;
}

fn compare_grads(a: &Tensor, b: &Tensor, tol: f32) {
    let ga = a.lock().grad.as_ref().unwrap().to_owned();
    let gb = b.lock().grad.as_ref().unwrap().to_owned();
    assert_eq!(ga.shape(), gb.shape());
    for (x, y) in ga.iter().zip(gb.iter()) {
        assert!((x - y).abs() <= tol, "grad mismatch: {} vs {}", x, y);
    }
}

#[test]
fn test_attention_grad_parity_flashref() {
    let b = 1;
    let seq = 4;
    let d_model = 8;
    let num_heads = 2;
    let mut mha_b = MultiHeadAttention::new(d_model, num_heads);
    let mut mha_f = MultiHeadAttention::new(d_model, num_heads);
    set_identity_linear(&mut mha_b);
    // copy weights
    mha_f.linear_q.weight = mha_b.linear_q.weight.clone();
    mha_f.linear_k.weight = mha_b.linear_k.weight.clone();
    mha_f.linear_v.weight = mha_b.linear_v.weight.clone();
    mha_f.linear_o.weight = mha_b.linear_o.weight.clone();
    // set variants
    mha_b.set_attention_variant(AttentionVariant::Baseline);
    mha_f.set_attention_variant(AttentionVariant::FlashRef);

    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(123456);
    let mut xs = Vec::new();
    for _ in 0..(b * seq * d_model) {
        xs.push(rng.gen_range(-0.01f32..0.01f32));
    }
    let x1 = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), xs.clone())
            .unwrap()
            .into_dyn(),
        true,
    );
    let x2 = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), xs)
            .unwrap()
            .into_dyn(),
        true,
    );

    let out1 = mha_b.forward(&x1).sum();
    out1.backward();
    let out2 = mha_f.forward(&x2).sum();
    out2.backward();

    // compare x grads
    compare_grads(&x1, &x2, 1e-4);
}

#[test]
fn test_attention_grad_parity_chunked() {
    let b = 1;
    let seq = 6;
    let d_model = 8;
    let num_heads = 2;
    let chunk_size = 2;
    let mut mha_b = MultiHeadAttention::new(d_model, num_heads);
    let mut mha_c = MultiHeadAttention::new(d_model, num_heads);
    set_identity_linear(&mut mha_b);
    // copy weights
    mha_c.linear_q.weight = mha_b.linear_q.weight.clone();
    mha_c.linear_k.weight = mha_b.linear_k.weight.clone();
    mha_c.linear_v.weight = mha_b.linear_v.weight.clone();
    mha_c.linear_o.weight = mha_b.linear_o.weight.clone();
    // set variants
    mha_b.set_attention_variant(AttentionVariant::Baseline);
    mha_c.set_attention_variant(AttentionVariant::Chunked { chunk_size });

    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(1234567);
    let mut xs = Vec::new();
    for _ in 0..(b * seq * d_model) {
        xs.push(rng.gen_range(-0.01f32..0.01f32));
    }
    let x1 = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), xs.clone())
            .unwrap()
            .into_dyn(),
        true,
    );
    let x2 = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), xs)
            .unwrap()
            .into_dyn(),
        true,
    );

    let out1 = mha_b.forward(&x1).sum();
    out1.backward();
    let out2 = mha_c.forward(&x2).sum();
    out2.backward();

    // compare x grads
    compare_grads(&x1, &x2, 1e-4);
}
