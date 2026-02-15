use crate::nn::looped_transformer::LoopedTransformer;
use crate::tensor::Tensor;
use ndarray::Array;

#[test]
fn looped_forward_consistency() {
    let b = 1usize;
    let seq = 4usize;
    let d_model = 8usize;
    let d_ff = 16usize;
    let num_heads = 2usize;
    let t_max = 3usize;

    let lt = LoopedTransformer::new_with_nl_oob(d_model, d_ff, num_heads, None, None, t_max, 0.05)
        .expect("create looped");

    let x_data: Vec<f32> = (0..(b * seq * d_model))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        false,
    );

    let (outs, p_phi) = lt.forward_looped(&x, None);
    assert_eq!(outs.len(), t_max);

    // repeated application of the inner block should equal the loop outputs
    let mut cur = x.clone();
    for i in 0..t_max {
        let expected = lt.block.forward_block_no_cache(&cur);
        let out = &outs[i];
        // debug prints to diagnose shape mismatch
        println!(
            "iter={} cur.shape={:?} expected.shape={:?} out.shape={:?} d_model={}",
            i,
            cur.lock().storage.shape(),
            expected.lock().storage.shape(),
            out.lock().storage.shape(),
            8
        );
        assert_eq!(expected.lock().storage.shape(), out.lock().storage.shape());
        assert_eq!(expected.to_f32_array(), out.to_f32_array());
        cur = expected;
    }

    // p_phi shape and stochasticity checks
    let p_shape = p_phi.lock().storage.shape().to_vec();
    assert_eq!(p_shape, vec![b, t_max]);
    // rows should sum ~1.0
    let arr = p_phi.to_f32_array();
    let total: f32 = arr.iter().sum();
    // since batch=1, sum over all elements equals 1.0
    assert!((total - 1.0).abs() < 1e-5);
}

#[test]
fn stage2_gate_loss_prefers_lower_cumulative() {
    let b = 1usize;
    let d_model = 4usize;
    let d_ff = 8usize;
    let num_heads = 1usize;
    let t_max = 3usize;

    let lt = LoopedTransformer::new_with_nl_oob(d_model, d_ff, num_heads, None, None, t_max, 0.0)
        .expect("create looped");

    // per-step losses: [0.5, 0.1, 0.0] -> cumulative [0.5, 0.6, 0.6]
    let per_losses = Tensor::new(
        Array::from_shape_vec((b, t_max), vec![0.5f32, 0.1f32, 0.0f32])
            .unwrap()
            .into_dyn(),
        false,
    );

    // p_phi concentrated on t=1 (early) vs t=2 (later)
    let p_early = Tensor::new(
        Array::from_shape_vec((b, t_max), vec![1.0f32, 0.0f32, 0.0f32])
            .unwrap()
            .into_dyn(),
        false,
    );
    let p_mid = Tensor::new(
        Array::from_shape_vec((b, t_max), vec![0.0f32, 1.0f32, 0.0f32])
            .unwrap()
            .into_dyn(),
        false,
    );

    let loss_early = lt.stage2_loss(&p_early, &per_losses);
    let loss_mid = lt.stage2_loss(&p_mid, &per_losses);

    // early gate should yield lower expected cumulative loss
    assert!(loss_early.to_f32_array()[0] < loss_mid.to_f32_array()[0]);
}

#[test]
fn looped_with_distance_changes_outputs() {
    let b = 1usize;
    let seq = 4usize;
    let d_model = 8usize;
    let d_ff = 16usize;
    let num_heads = 2usize;
    let t_max = 2usize;

    let lt_no =
        LoopedTransformer::new_with_nl_oob(d_model, d_ff, num_heads, None, None, t_max, 0.0)
            .unwrap();
    let lt_yes = LoopedTransformer::new_with_nl_oob(
        d_model,
        d_ff,
        num_heads,
        Some(crate::nn::transformer_cleaned::BiasFunction::Logarithmic),
        Some(2.0),
        t_max,
        0.0,
    )
        .unwrap();

    let x_data: Vec<f32> = (0..(b * seq * d_model))
        .map(|i| (i as f32) * 0.02)
        .collect();
    let x = Tensor::new(
        Array::from_shape_vec((b, seq, d_model), x_data)
            .unwrap()
            .into_dyn(),
        false,
    );

    // simple abs(i-j) distance matrix shape [seq, seq]
    let mut dist_data: Vec<f32> = Vec::with_capacity(seq * seq);
    for i in 0..seq {
        for j in 0..seq {
            dist_data.push(((i as isize - j as isize).abs()) as f32);
        }
    }
    let dist = Tensor::new(
        Array::from_shape_vec((seq, seq), dist_data)
            .unwrap()
            .into_dyn(),
        false,
    );

    let (outs_no, _) = lt_no.forward_looped(&x, Some(&dist));
    let (outs_yes, _) = lt_yes.forward_looped(&x, Some(&dist));

    // outputs should differ when NL-OOB slopes are present
    assert!(outs_no[0].to_f32_array() != outs_yes[0].to_f32_array());
}
