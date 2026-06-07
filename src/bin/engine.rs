fn main() {
    if let Err(e) = tensor_engine::compat::engine::entrypoint::main() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
