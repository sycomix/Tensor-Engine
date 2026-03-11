//! Example demonstrating GPU/CPU backend usage in Tensor Engine

use tensor_engine::backend::{get_global_backend, set_cpu_backend};
use ndarray::{ArrayD, IxDyn};

fn main() {
    println!("=== Tensor Engine Backend Demo ===\n");

    // Set CPU backend (default)
    set_cpu_backend().expect("Failed to set CPU backend");
    let backend = get_global_backend();
    
    println!("Active backend: {}", backend.name());
    println!("Memory info: {:.2}GB / {:.2}GB used\n", 
        backend.memory_info().0 as f64 / 1e9,
        backend.memory_info().1 as f64 / 1e9);

    // Test matrix multiplication
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x4

    let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), a_data).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[3, 4]), b_data).unwrap();

    println!("Matrix A (2x3):");
    println!("{:?}", a);
    
    println!("\nMatrix B (3x4):");
    println!("{:?}", b);

    if let Some(result) = backend.matmul(&a, &b) {
        println!("\nResult of A × B (2x4):");
        println!("{:?}", result);
        
        // Verify against expected values
        let expected = vec![22.0, 28.0, 34.0, 40.0, 49.0, 64.0, 79.0, 94.0];
        println!("\nExpected: {:?}", expected);
        
        let all_close = result.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        if all_close {
            println!("✓ Matrix multiplication verified!");
        } else {
            println!("✗ Result mismatch!");
        }
    } else {
        println!("\nMatrix multiplication not supported by current backend");
    }

    // Test softmax
    let input_data = vec![1.0, 2.0, 3.0];
    let input = ArrayD::from_shape_vec(IxDyn(&[1, 3]), input_data).unwrap();
    
    println!("\nInput for softmax: {:?}", input);
    
    if let Some(result) = backend.softmax(&input, 1) {
        println!("Softmax result: {:?}", result);
        
        // Verify sum is approximately 1.0
        let sum: f32 = result.iter().sum();
        if (sum - 1.0).abs() < 1e-5 {
            println!("✓ Softmax verified (sum ≈ 1.0)");
        } else {
            println!("✗ Softmax verification failed (sum = {})", sum);
        }
    } else {
        println!("\nSoftmax not supported by current backend");
    }

    #[cfg(feature = "backend_wgpu")]
    {
        println!("\n=== WGPU Backend Available ===");
        if tensor_engine::backend::is_wgpu_available() {
            match set_cpu_backend() {
                Ok(_) => println!("CPU backend successfully configured"),
                Err(e) => eprintln!("Error: {}", e),
            }
        } else {
            println!("WGPU not available on this system");
        }
    }

    #[cfg(not(feature = "backend_wgpu"))]
    {
        println!("\n=== WGPU Backend ===");
        println!("Not compiled with backend_wgpu feature");
    }

    println!("\n=== Demo Complete ===");
}
