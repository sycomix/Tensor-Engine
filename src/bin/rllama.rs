// This file is a wrapper to ensure rllama is only compiled when the 'compat' feature is enabled.

#[cfg(not(feature = "compat"))]
fn main() {
    eprintln!("The 'rllama' binary requires the 'compat' feature to be enabled.");
    eprintln!("Please run with: cargo run --bin rllama --features compat");
    std::process::exit(1);
}

#[cfg(feature = "compat")]
fn main() {
    if let Err(e) = tensor_engine::compat::rllama::entrypoint::main() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
