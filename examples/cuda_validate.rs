//! Simple CUDA validation test for RTX A4000 GPUs

use tensor_engine::backend::{cuda::CudaBackend, traits::Backend};

fn main() {
    println!("=== Tensor Engine CUDA Validation ===\n");
    
    // Test 1: Check if CUDA is available
    println!("1. Checking CUDA availability...");
    if !CudaBackend::is_available() {
        eprintln!("   ✗ CUDA not available on this system");
        return;
    }
    println!("   ✓ CUDA detected!");
    
    // Test 2: Create CUDA backend
    println!("\n2. Creating CUDA backend...");
    let cuda_backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("   ✗ Failed to create CUDA backend: {}", e);
            return;
        }
    };
    println!("   ✓ CUDA backend created!");
    
    // Test 3: Verify matmul works
    println!("\n3. Testing CUDA matrix multiplication...");
    let size = 100;
    let a = ndarray::Array2::<f32>::ones((size, size));
    let b = ndarray::Array2::<f32>::ones((size, size));
    
    // Create output array for CPU verification
    let cpu_result = ndarray::Array2::<f32>::ones((size, size)) * (size as f32);
    
    match cuda_backend.matmul(&a, &b) {
        Some(result) => {
            println!("   ✓ CUDA matmul succeeded! Output shape: {:?}", result.shape());
            
            // Verify result is correct
            let mut all_correct = true;
            for i in 0..size {
                for j in 0..size {
                    let val = result[[i, j]];
                    if (val - cpu_result[[i, j]]).abs() > 1e-3 {
                        eprintln!("   ✗ Unexpected value at [{}, {}]: {} (expected {})", 
                                  i, j, val, cpu_result[[i, j]]);
                        all_correct = false;
                        break;
                    }
                }
                if !all_correct { break; }
            }
            
            if all_correct {
                println!("   ✓ CUDA matmul values verified!");
            } else {
                eprintln!("   ✗ Value mismatch detected");
            }
        }
        None => {
            eprintln!("   ✗ CUDA matmul returned None - falling back to CPU");
        }
    }
    
    println!("\n=== Validation Complete ===");
}
