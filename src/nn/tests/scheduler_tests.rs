use crate::nn::{ConstantLR, LRScheduler, LinearLR, OneCycleLR};

#[test]
fn constant_lr_applies_factor_then_restores_base() {
    let sched = ConstantLR::new(0.1, 0.5, 3);

    assert!((sched.get_lr(0) - 0.05).abs() < 1e-8);
    assert!((sched.get_lr(1) - 0.05).abs() < 1e-8);
    assert!((sched.get_lr(2) - 0.05).abs() < 1e-8);
    assert!((sched.get_lr(3) - 0.1).abs() < 1e-8);
    assert!((sched.get_lr(10) - 0.1).abs() < 1e-8);
}

#[test]
fn linear_lr_interpolates_and_then_holds_end_factor() {
    let sched = LinearLR::new(0.2, 0.1, 1.0, 4);

    assert!((sched.get_lr(0) - 0.02).abs() < 1e-8);
    assert!((sched.get_lr(2) - 0.11).abs() < 1e-8);
    assert!((sched.get_lr(4) - 0.2).abs() < 1e-8);
    assert!((sched.get_lr(100) - 0.2).abs() < 1e-8);
}

#[test]
fn onecycle_lr_warms_up_and_then_decays() {
    let sched = OneCycleLR::new(
        0.1,   // max_lr
        10,    // total_steps
        0.3,   // pct_start
        10.0,  // div_factor -> initial 0.01
        100.0, // final_div_factor -> final 0.001
    );

    let lr0 = sched.get_lr(0);
    let lr2 = sched.get_lr(2);
    let lr3 = sched.get_lr(3);
    let lr5 = sched.get_lr(5);
    let lr10 = sched.get_lr(10);

    assert!(lr0 > 0.0);
    assert!(lr2 > lr0);
    assert!(lr3 >= lr2);
    assert!(lr5 < lr3);
    assert!(lr10 < lr5);

    let expected_final = 0.1 / 100.0;
    assert!((lr10 - expected_final).abs() < 1e-5);
}
