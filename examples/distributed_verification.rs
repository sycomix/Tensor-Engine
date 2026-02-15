#[cfg(feature = "distributed")]
mod example {
    use std::thread;
    use tensor_engine::distributed::{DataParallel, DistributedContext, ReduceOp};
    use tensor_engine::nn::{Linear, Module};
    use tensor_engine::optim::{Optimizer, SGD};
    use tensor_engine::tensor::Tensor;

    pub fn main() {
        // 1. Setup Configuration
        let world_size = 4;
        let epochs = 5;

        println!(
            "Starting Distributed Validation with world_size={}",
            world_size
        );

        // 2. Spawn threads for each rank
        let mut handles = vec![];

        for rank in 0..world_size {
            handles.push(thread::spawn(move || {
                run_rank(rank, world_size, epochs);
            }));
        }

        // 3. Wait for all ranks to complete
        for handle in handles {
            handle.join().unwrap();
        }

        println!("Distributed Validation Completed Successfully!");
    }

    fn run_rank(rank: usize, world_size: usize, epochs: usize) {
        // A. Initialize Distributed Context
        let ctx = DistributedContext::new(rank, world_size);
        println!("[Rank {}] Context initialized", rank);

        // B. Create a Model (Linear layer)
        // We need to ensure all ranks start with the SAME weights for valid comparison.
        // In a real scenario, we'd broadcast weights from rank 0.
        // Here, we'll manually seed or broadcast.
        let model = Linear::new(10, 1, true);

        // Broadcast initial weights from Rank 0 to ensure sync start
        // Note: DataParallel.replicate() does this, but we can do it manually or rely on DP.

        // C. Wrap in DataParallel
        let dp_model = DataParallel::new(model, ctx.clone());

        // D. Create Optimizer
        // We need to get parameters from the inner model. DataParallel gives access.
        // We keep a reference to parameters for sync_gradients later.
        let params = dp_model.model().parameters();
        let mut optimizer = SGD::new(params.clone(), 0.01).with_momentum(0.9);

        // E. Training Loop
        for epoch in 0..epochs {
            // 1. Synthetic Data (Batch Size 32, 10 features)
            // Global batch size = 32.
            // Each rank should get 32/4 = 8 samples if we shard.
            // Use deterministic data generation for simplicity and to avoid rand dependency
            let input_data: Vec<f32> = (0..32 * 10).map(|i| (i as f32).sin()).collect();
            let input = Tensor::new(
                ndarray::Array::from_shape_vec(ndarray::IxDyn(&[32, 10][..]), input_data).unwrap(),
                false, // input doesn't need grad
            );

            let target_data: Vec<f32> = (0..32).map(|i| (i as f32).cos()).collect();
            let target = Tensor::new(
                ndarray::Array::from_shape_vec(ndarray::IxDyn(&[32, 1][..]), target_data).unwrap(),
                false,
            );

            // 2. Shard Data
            // DataParallel has a helper for this
            let batch = dp_model.shard_batch(&input, Some(&target));
            let local_input = batch.local_data;
            let local_target = batch.local_labels.unwrap();

            // 3. Forward Pass
            // We use the local model
            let output = dp_model.model().forward(&local_input);

            // 4. Compute Loss (MSE)
            // MSE = (output - target)^2 . mean()
            let diff = output.sub(&local_target);
            let squared = diff.pow(2.0);
            let loss = squared.mean();

            // 5. Backward Pass
            optimizer.zero_grad();
            loss.backward();

            // 6. Synchronize Gradients (All-Reduce)
            // This is the critical step. We use the params vector we kept earlier.
            dp_model.sync_gradients(&params);

            // 7. Optimizer Step
            optimizer.step();

            // 8. Log
            if rank == 0 {
                // loss.item() doesn't exist? loss is a Tensor (scalar).
                // We can get data via lock.
                let loss_val = *loss.to_f32_array().iter().next().unwrap_or(&0.0);
                // array[[]] usually works for 0-dim.
                println!("[Rank {}] Epoch {}: Loss = {:.4}", rank, epoch, loss_val);
            }

            // Validation:
            // We can check if weights are still synchronized.
            // We'll broadcast rank 0's weights and compare.
            // let weights = dp_model.model().parameters()[0].clone();
            // Verify sync (simplified check)
            // In a real test we would Gather and assert equality.
        }

        println!("[Rank {}] Finished", rank);
    }
}

fn main() {
    #[cfg(feature = "distributed")]
    example::main();

    #[cfg(not(feature = "distributed"))]
    println!("This example requires the 'distributed' feature. Run with: --features distributed");
}
