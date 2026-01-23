use tensor_engine::nn::{GRUCell, LSTMCell, Module};
use tensor_engine::optim::{Optimizer, SGD};
use tensor_engine::tensor::Tensor;

fn main() {
    println!("=== GRU (Gated Recurrent Unit) Demonstration ===\n");

    // Simple sequence modeling task: learn to predict next value in sequence
    // Sequence: [1, 2, 3, 4, 5] -> predict 6
    println!("Task: Sequence prediction");
    println!("Given: [1, 2, 3, 4, 5]");
    println!("Predict: 6\n");

    demonstrate_gru();
    demonstrate_comparison();
}

fn demonstrate_gru() {
    println!("1. GRU Cell");
    println!("   Architecture: 2 gates (reset, update) vs LSTM's 3 gates");
    println!("   Advantages: Fewer parameters, faster training\n");

    let input_dim = 1;
    let hidden_dim = 8;
    let gru = GRUCell::new(input_dim, hidden_dim, true);

    // Create a simple sequence
    let x1 = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1, 1]), vec![1.0]).unwrap(),
        true,
    );
    let x2 = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1, 1]), vec![2.0]).unwrap(),
        true,
    );
    let x3 = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1, 1]), vec![3.0]).unwrap(),
        true,
    );

    // Initialize hidden state
    let mut h = Tensor::new(
        ndarray::ArrayD::zeros(ndarray::IxDyn(&[1, hidden_dim])),
        false,
    );

    println!("   Processing sequence:");

    // Step through sequence
    h = gru.forward_step(&x1, &h);
    println!("   Step 1: input=1.0, hidden_state updated");

    h = gru.forward_step(&x2, &h);
    println!("   Step 2: input=2.0, hidden_state updated");

    h = gru.forward_step(&x3, &h);
    println!("   Step 3: input=3.0, hidden_state updated");
    let h_arr = h.lock().storage.to_f32_array();
    println!(
        "   Final hidden state (first 3): {:?}",
        h_arr.iter().take(3).collect::<Vec<_>>()
    );

    println!("   ✓ GRU successfully processed sequence\n");

    // Show parameter count
    let params = gru.parameters();
    let total_params: usize = params
        .iter()
        .map(|p| p.lock().storage.shape().iter().product::<usize>())
        .sum();
    println!("   Total parameters: {}", total_params);
    println!("   (input_dim={}, hidden_dim={})", input_dim, hidden_dim);
    println!("   Formula: 3 * hidden_dim * (input_dim + hidden_dim + 1)\n");
}

fn demonstrate_comparison() {
    println!("2. GRU vs LSTM Comparison\n");

    let input_dim = 4;
    let hidden_dim = 16;

    let gru = GRUCell::new(input_dim, hidden_dim, true);
    let lstm = LSTMCell::new(input_dim, hidden_dim, true);

    let gru_params: usize = gru
        .parameters()
        .iter()
        .map(|p| p.lock().storage.shape().iter().product::<usize>())
        .sum();

    let lstm_params: usize = lstm
        .parameters()
        .iter()
        .map(|p| p.lock().storage.shape().iter().product::<usize>())
        .sum();

    println!(
        "   Configuration: input_dim={}, hidden_dim={}",
        input_dim, hidden_dim
    );
    println!("   ┌─────────────┬────────────┬──────────┐");
    println!("   │ Model       │ Parameters │ Gates    │");
    println!("   ├─────────────┼────────────┼──────────┤");
    println!("   │ GRU         │ {:10} │ 2 (r, z) │", gru_params);
    println!("   │ LSTM        │ {:10} │ 3 (i,f,o)│", lstm_params);
    println!("   └─────────────┴────────────┴──────────┘");

    let reduction = ((lstm_params - gru_params) as f32 / lstm_params as f32) * 100.0;
    println!("\n   GRU has {:.1}% fewer parameters than LSTM", reduction);
    println!("   → Faster training and inference");
    println!("   → Lower memory footprint");
    println!("   → Competitive performance on many tasks\n");

    println!("3. GRU Gate Functions\n");
    println!("   Reset gate (r_t):  Controls how much past info to forget");
    println!("   Update gate (z_t): Controls how much new info to add");
    println!("   New gate (n_t):    Candidate hidden state");
    println!("\n   Output: h_t = (1 - z_t) * n_t + z_t * h_{{t-1}}");
    println!("   → Interpolates between new and old hidden state\n");

    println!("4. Use Cases\n");
    println!("   ✓ Time series forecasting");
    println!("   ✓ Natural language processing");
    println!("   ✓ Speech recognition");
    println!("   ✓ Video analysis");
    println!("   ✓ Any sequential data modeling task\n");

    println!("5. Training Example\n");
    train_gru_example();
}

fn train_gru_example() {
    // Simple training example: learn identity function
    let gru = GRUCell::new(1, 4, true);
    let mut optim = SGD::new(gru.parameters(), 0.01);

    println!("   Training GRU to learn simple pattern...");

    for epoch in 0..10 {
        optim.zero_grad();

        // Create input
        let x = Tensor::new(
            ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1, 1]), vec![0.5]).unwrap(),
            true,
        );

        // Forward pass
        let output = gru.forward(&x);

        // Simple loss: encourage non-zero output
        let target = Tensor::new(ndarray::ArrayD::ones(ndarray::IxDyn(&[1, 4])), false);
        let loss = (output.sub(&target)).pow(2.0).mean();

        // Backward pass
        loss.backward();
        optim.step();

        if epoch % 3 == 0 {
            let loss_arr = loss.lock().storage.to_f32_array();
            let loss_val = *loss_arr.iter().next().unwrap();
            println!("   Epoch {}: Loss = {:.6}", epoch, loss_val);
        }
    }

    println!("   ✓ Training completed successfully\n");
}
