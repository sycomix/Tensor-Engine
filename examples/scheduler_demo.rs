use tensor_engine::lr_scheduler::{ExponentialLR, LRScheduler, PolynomialLR, StepLR};
use tensor_engine::nn::{Linear, Module};
use tensor_engine::optim::{Optimizer, SGD};
use tensor_engine::tensor::Tensor;

fn main() {
    println!("=== Learning Rate Scheduler Demonstration ===\n");

    // Generate simple training data: y = 2x + 1
    let x_val = vec![1.0, 2.0, 3.0, 4.0];
    let y_val = vec![3.0, 5.0, 7.0, 9.0];

    let x = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 1]), x_val).unwrap(),
        false,
    );
    let y = Tensor::new(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 1]), y_val).unwrap(),
        false,
    );

    // Demonstrate each scheduler
    println!("1. ExponentialLR Scheduler");
    println!("   Formula: lr_t = lr_0 * gamma^t");
    demonstrate_exponential(&x, &y);

    println!("\n2. StepLR Scheduler");
    println!("   Formula: lr_t = lr_0 * gamma^(floor(t / step_size))");
    demonstrate_step(&x, &y);

    println!("\n3. PolynomialLR Scheduler");
    println!("   Formula: lr_t = (lr_0 - lr_final) * (1 - t/T)^power + lr_final");
    demonstrate_polynomial(&x, &y);

    println!("\n=== Learning Rate Curves ===\n");
    print_lr_curves();
}

fn demonstrate_exponential(x: &Tensor, y: &Tensor) {
    let model = Linear::new(1, 1, true);
    let initial_lr = 0.1;
    let mut scheduler = ExponentialLR::new(initial_lr, 0.95);

    println!("   Initial LR: {:.6}", scheduler.get_lr());

    for epoch in 0..20 {
        let lr = scheduler.get_lr();
        let mut optim = SGD::new(lr, 0.0);

        // Training step
        let params = model.parameters();
        optim.zero_grad(&params);
        let out = model.forward(x);
        let loss = out.sub(y).pow(2.0).mean();
        loss.backward();
        optim.step(&params);

        if epoch % 5 == 0 {
            let loss_arr = loss.lock().storage.to_f32_array();
            let loss_val = *loss_arr.iter().next().unwrap();
            println!(
                "   Epoch {:2}: LR = {:.6}, Loss = {:.6}",
                epoch, lr, loss_val
            );
        }

        scheduler.step();
    }

    println!("   Final LR: {:.6}", scheduler.get_lr());
}

fn demonstrate_step(x: &Tensor, y: &Tensor) {
    let model = Linear::new(1, 1, true);
    let initial_lr = 0.1;
    let mut scheduler = StepLR::new(initial_lr, 0.5, 10);

    println!("   Initial LR: {:.6}", scheduler.get_lr());
    println!("   Step size: 10 epochs, Gamma: 0.5");

    for epoch in 0..30 {
        let lr = scheduler.get_lr();
        let mut optim = SGD::new(lr, 0.0);

        // Training step
        let params = model.parameters();
        optim.zero_grad(&params);
        let out = model.forward(x);
        let loss = out.sub(y).pow(2.0).mean();
        loss.backward();
        optim.step(&params);

        if epoch % 10 == 0 || epoch == 9 || epoch == 19 {
            let loss_arr = loss.lock().storage.to_f32_array();
            let loss_val = *loss_arr.iter().next().unwrap();
            println!(
                "   Epoch {:2}: LR = {:.6}, Loss = {:.6}",
                epoch, lr, loss_val
            );
        }

        scheduler.step();
    }

    println!("   Final LR: {:.6}", scheduler.get_lr());
}

fn demonstrate_polynomial(x: &Tensor, y: &Tensor) {
    let model = Linear::new(1, 1, true);
    let initial_lr = 0.1;
    let final_lr = 0.001;
    let max_epochs = 50;
    let mut scheduler = PolynomialLR::new(initial_lr, final_lr, max_epochs, 2.0);

    println!("   Initial LR: {:.6}", scheduler.get_lr());
    println!(
        "   Final LR: {:.6}, Max epochs: {}, Power: 2.0",
        final_lr, max_epochs
    );

    for epoch in 0..max_epochs {
        let lr = scheduler.get_lr();
        let mut optim = SGD::new(lr, 0.0);

        // Training step
        let params = model.parameters();
        optim.zero_grad(&params);
        let out = model.forward(x);
        let loss = out.sub(y).pow(2.0).mean();
        loss.backward();
        optim.step(&params);

        if epoch % 10 == 0 || epoch == max_epochs - 1 {
            let loss_arr = loss.lock().storage.to_f32_array();
            let loss_val = *loss_arr.iter().next().unwrap();
            println!(
                "   Epoch {:2}: LR = {:.6}, Loss = {:.6}",
                epoch, lr, loss_val
            );
        }

        scheduler.step();
    }

    println!("   Final LR: {:.6}", scheduler.get_lr());
}

fn print_lr_curves() {
    println!("ExponentialLR (lr=0.1, gamma=0.95):");
    let mut exp_sched = ExponentialLR::new(0.1, 0.95);
    for epoch in 0..20 {
        if epoch % 5 == 0 {
            println!("  Epoch {:2}: {:.6}", epoch, exp_sched.get_lr());
        }
        exp_sched.step();
    }

    println!("\nStepLR (lr=0.1, gamma=0.5, step_size=10):");
    let mut step_sched = StepLR::new(0.1, 0.5, 10);
    for epoch in 0..30 {
        if epoch % 5 == 0 {
            println!("  Epoch {:2}: {:.6}", epoch, step_sched.get_lr());
        }
        step_sched.step();
    }

    println!("\nPolynomialLR (lr=0.1, final=0.001, max=50, power=2.0):");
    let mut poly_sched = PolynomialLR::new(0.1, 0.001, 50, 2.0);
    for epoch in 0..50 {
        if epoch % 10 == 0 {
            println!("  Epoch {:2}: {:.6}", epoch, poly_sched.get_lr());
        }
        poly_sched.step();
    }
}
