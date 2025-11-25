use std::env;
use std::path::Path;

fn main() {
    // If `openblas` feature is enabled (Cargo sets CARGO_FEATURE_<FEATURE>), link with local OpenBLAS if provided.
    if env::var("CARGO_FEATURE_OPENBLAS").is_ok() {
        if let Ok(dir) = env::var("OPENBLAS_DIR") {
            let lib_dir = Path::new(&dir).join("lib");
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            // Try linking to 'openblas'; depending on platform it may have different names
            println!("cargo:rustc-link-lib=openblas");
            // On Windows MSVC, the import library is sometimes named 'libopenblas.lib' so also try that
            if cfg!(target_os = "windows") {
                println!("cargo:rustc-link-lib=libopenblas");
            }
        } else {
            // No OPENBLAS_DIR provided: let dependent crates (e.g., `openblas-src` or system libs) handle it.
            println!("cargo:warning=Feature 'openblas' enabled but OPENBLAS_DIR not set. Ensure OpenBLAS is available on your system.");
        }
    }
}
