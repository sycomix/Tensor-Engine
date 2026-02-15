use std::sync::Arc;
use tensor_engine::ops::{ContrastiveLoss, FocalLoss, KLDivergence, TripletLoss};
use tensor_engine::tensor::Tensor;

fn main() {
    println!("=== Advanced Loss Functions Demonstration ===\n");

    demonstrate_focal_loss();
    demonstrate_kl_divergence();
    demonstrate_contrastive_loss();
    demonstrate_triplet_loss();
}

fn demonstrate_focal_loss() {
    println!("1. Focal Loss");
    println!("   Use case: Object detection, imbalanced classification");
    println!("   Formula: FL(p_t) = -α * (1 - p_t)^γ * log(p_t)\n");

    let focal = FocalLoss::new(1.0, 2.0);

    // Simulate predictions and targets for binary classification
    let preds = Tensor::new(
        ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![0.9, 0.7, 0.3, 0.1], // Predicted probabilities
        )
            .unwrap(),
        true,
    );
    let targets = Tensor::new(
        ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 1.0, 0.0, 0.0], // Ground truth labels
        )
            .unwrap(),
        false,
    );

    let loss = Tensor::apply(Arc::new(focal), &[preds.clone(), targets]);
    let loss_val = *loss.lock().storage.to_f32_array().iter().next().unwrap();

    println!("   Predictions: [0.9, 0.7, 0.3, 0.1]");
    println!("   Targets:     [1.0, 1.0, 0.0, 0.0]");
    println!("   Focal Loss (α=1.0, γ=2.0): {:.6}", loss_val);
    println!("   → Down-weights easy examples (0.9, 0.1)");
    println!("   → Focuses on hard examples (0.7, 0.3)\n");

    // Test backward pass
    loss.backward();
    println!("   ✓ Gradients computed successfully\n");
}

fn demonstrate_kl_divergence() {
    println!("2. KL Divergence");
    println!("   Use case: Knowledge distillation, VAEs, distribution matching");
    println!("   Formula: KL(P || Q) = Σ P(x) * log(P(x) / Q(x))\n");

    let kl = KLDivergence::new("mean".to_string());

    // Log probabilities for two distributions
    let p_log = Tensor::new(
        ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&[3]),
            vec![-0.5, -1.0, -1.5], // Target distribution (log probs)
        )
            .unwrap(),
        true,
    );
    let q_log = Tensor::new(
        ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&[3]),
            vec![-1.0, -1.5, -2.0], // Predicted distribution (log probs)
        )
            .unwrap(),
        true,
    );

    let kl_loss = Tensor::apply(Arc::new(kl), &[p_log.clone(), q_log.clone()]);
    let kl_val = *kl_loss.lock().storage.to_f32_array().iter().next().unwrap();

    println!("   P (log probs): [-0.5, -1.0, -1.5]");
    println!("   Q (log probs): [-1.0, -1.5, -2.0]");
    println!("   KL(P || Q): {:.6}", kl_val);
    println!("   → Measures how Q diverges from P");
    println!("   → KL = 0 when distributions are identical\n");

    // Test backward pass
    kl_loss.backward();
    println!("   ✓ Gradients computed successfully\n");
}

fn demonstrate_contrastive_loss() {
    println!("3. Contrastive Loss");
    println!("   Use case: Siamese networks, metric learning");
    println!("   Formula: L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2\n");

    // Similar pair
    println!("   Example 1: Similar pair (label=0)");
    let emb1_sim = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        true,
    );
    let emb2_sim = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.1, 2.1, 3.1]).unwrap(),
        true,
    );
    let label_sim = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.0]).unwrap(),
        false,
    );

    let loss_sim = Tensor::apply(
        Arc::new(ContrastiveLoss::new(2.0)),
        &[emb1_sim, emb2_sim, label_sim],
    );
    let loss_sim_val = *loss_sim
        .lock()
        .storage
        .to_f32_array()
        .iter()
        .next()
        .unwrap();

    println!("   Embedding 1: [1.0, 2.0, 3.0]");
    println!("   Embedding 2: [1.1, 2.1, 3.1]");
    println!("   Loss (similar): {:.6}", loss_sim_val);
    println!("   → Penalizes distance between similar pairs\n");

    // Dissimilar pair
    println!("   Example 2: Dissimilar pair (label=1)");
    let emb1_dis = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.0, 0.0, 0.0]).unwrap(),
        true,
    );
    let emb2_dis = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 1.0, 1.0]).unwrap(),
        true,
    );
    let label_dis = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1]), vec![1.0]).unwrap(),
        false,
    );

    let loss_dis = Tensor::apply(
        Arc::new(ContrastiveLoss::new(2.0)),
        &[emb1_dis, emb2_dis, label_dis],
    );
    let loss_dis_val = *loss_dis
        .lock()
        .storage
        .to_f32_array()
        .iter()
        .next()
        .unwrap();

    println!("   Embedding 1: [0.0, 0.0, 0.0]");
    println!("   Embedding 2: [1.0, 1.0, 1.0]");
    println!("   Loss (dissimilar, margin=2.0): {:.6}", loss_dis_val);
    println!("   → Penalizes if distance < margin\n");
}

fn demonstrate_triplet_loss() {
    println!("4. Triplet Loss");
    println!("   Use case: Face recognition, image retrieval, embedding learning");
    println!("   Formula: L = max(0, D(a,p) - D(a,n) + margin)\n");

    // Margin violation case
    println!("   Example 1: Margin violation");
    let anchor1 = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.0, 0.0, 0.0]).unwrap(),
        true,
    );
    let positive1 = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.1, 0.1, 0.1]).unwrap(),
        true,
    );
    let negative1 = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.2, 0.2, 0.2]).unwrap(),
        true,
    );

    let loss1 = Tensor::apply(
        Arc::new(TripletLoss::new(0.5)),
        &[anchor1.clone(), positive1.clone(), negative1.clone()],
    );
    let loss1_val = *loss1.lock().storage.to_f32_array().iter().next().unwrap();

    println!("   Anchor:   [0.0, 0.0, 0.0]");
    println!("   Positive: [0.1, 0.1, 0.1]  (same class)");
    println!("   Negative: [0.2, 0.2, 0.2]  (different class)");
    println!("   Loss (margin=0.5): {:.6}", loss1_val);
    println!("   → Positive too close to negative!\n");

    // Margin satisfied case
    println!("   Example 2: Margin satisfied");
    let anchor2 = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.0, 0.0, 0.0]).unwrap(),
        true,
    );
    let positive2 = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.1, 0.1, 0.1]).unwrap(),
        true,
    );
    let negative2 = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[3]), vec![2.0, 2.0, 2.0]).unwrap(),
        true,
    );

    let loss2 = Tensor::apply(
        Arc::new(TripletLoss::new(0.5)),
        &[anchor2, positive2, negative2],
    );
    let loss2_val = *loss2.lock().storage.to_f32_array().iter().next().unwrap();

    println!("   Anchor:   [0.0, 0.0, 0.0]");
    println!("   Positive: [0.1, 0.1, 0.1]  (same class)");
    println!("   Negative: [2.0, 2.0, 2.0]  (different class)");
    println!("   Loss (margin=0.5): {:.6}", loss2_val);
    println!("   → Negative far enough from anchor ✓\n");

    // Test backward pass
    loss1.backward();
    println!("   ✓ Gradients computed successfully");
    println!("   → Anchor gradient pushes away from positive");
    println!("   → Anchor gradient pulls toward negative\n");
}
