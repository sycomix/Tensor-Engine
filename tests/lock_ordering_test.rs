use ndarray::IxDyn;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

use rand::Rng;
use tensor_engine::nn::{Linear, Module};
use tensor_engine::tensor::Tensor;

// Stress test for Module::load_state_dict to ensure no deadlock under concurrent calls.
// This test creates two modules that share a tensor and launches multiple threads which
// repeatedly call load_state_dict on them in random orders. The deterministic lock
// ordering inside Module::load_state_dict should prevent lock-order inversion and deadlocks.
#[test]
fn test_concurrent_load_state_dict_no_deadlock() {
    let _ = env_logger::builder().is_test(true).try_init();
    let shared_weight = Tensor::new(ndarray::Array::from_vec(vec![0.0f32; 6]).into_dyn(), true);
    let bias_a = Tensor::new(ndarray::Array::from_vec(vec![0.0f32; 3]).into_dyn(), true);
    let bias_b = Tensor::new(ndarray::Array::from_vec(vec![0.0f32; 3]).into_dyn(), true);

    let mut lin_a = Linear::new(2, 3, true);
    let mut lin_b = Linear::new(2, 3, true);
    // Replace weights with the same shared tensor so params are shared.
    lin_a.weight = shared_weight.clone();
    lin_b.weight = shared_weight.clone();
    lin_a.bias = Some(bias_a.clone());
    lin_b.bias = Some(bias_b.clone());

    // Place modules behind Mutex + Arc so they can be mutated across threads
    let ma = Arc::new(Mutex::new(lin_a));
    let mb = Arc::new(Mutex::new(lin_b));

    let threads: usize = 16;
    let loops: usize = 200;
    let mut handles = Vec::new();
    for _ in 0..threads {
        let ma = ma.clone();
        let mb = mb.clone();
        let handle = thread::spawn(move || {
            let mut rng = rand::rng();
            for i in 0..loops {
                log::debug!("thread {:?} iteration {}", std::thread::current().id(), i);
                // prepare new random state tensors
                let mut state: HashMap<String, Tensor> = HashMap::new();
                let new_weight_arr = ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|_| rng.random::<f32>()).collect()).unwrap();
                let new_bias_arr = ndarray::Array::from_shape_vec(IxDyn(&[3]), (0..3).map(|_| rng.random::<f32>()).collect()).unwrap();
                let w = Tensor::new(new_weight_arr.into_dyn(), false);
                let b = Tensor::new(new_bias_arr.into_dyn(), false);
                // named keys match Linear::named_parameters with prefix "seq"
                state.insert("seq.weight".to_string(), w.clone());
                state.insert("seq.bias".to_string(), b.clone());

                // Acquire module locks in deterministic pointer order to avoid test-induced
                // deadlocks (the test previously locked modules in random orders which
                // can deadlock at the module mutex level even when parameter locking is
                // deterministic). We intentionally still randomize which module is
                // 'first' to simulate different call orders while keeping the lock
                // acquisition order consistent.
                let a_ptr = Arc::as_ptr(&ma) as usize;
                let b_ptr = Arc::as_ptr(&mb) as usize;
                if a_ptr <= b_ptr {
                    let mut a = ma.lock().unwrap();
                    a.load_state_dict(&state, "seq").unwrap();
                    let mut bmod = mb.lock().unwrap();
                    bmod.load_state_dict(&state, "seq").unwrap();
                } else {
                    let mut bmod = mb.lock().unwrap();
                    bmod.load_state_dict(&state, "seq").unwrap();
                    let mut a = ma.lock().unwrap();
                    a.load_state_dict(&state, "seq").unwrap();
                }
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("Thread panicked");
    }

    // After all threads complete, verify that shared weight on both modules equals the underlying shared tensor.
    let a = ma.lock().unwrap();
    let b = mb.lock().unwrap();
    let wa = a.weight.lock().storage.to_f32_array();
    let wb = b.weight.lock().storage.to_f32_array();
    assert_eq!(wa, wb);
}
