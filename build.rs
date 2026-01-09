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

    // If the optional `cffi` feature is enabled on Windows/MSVC, fail early with a helpful message
    let cffi_enabled = env::var_os("CARGO_FEATURE_CFFI").is_some();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if cffi_enabled && target_os == "windows" && target_env == "msvc" {
        let has_vcpkg = env::var_os("VCPKG_ROOT").is_some();
        let has_vs = env::var_os("VCINSTALLDIR").is_some() || env::var_os("VisualStudioVersion").is_some();

        if !has_vcpkg || !has_vs {
            // Allow skipping this protective guard via environment variable for experiments.
            if env::var_os("SKIP_CFFI_GUARD").is_some() {
                println!("cargo:warning=SKIP_CFFI_GUARD set: continuing despite missing vcpkg/VS (experimental override)");
            } else {
                panic!(
                    "Building with the 'cffi' feature on Windows/MSVC typically requires extra setup and can fail with obscure linker errors (e.g., unresolved symbols in cffi-impl/ctor).\n\nPlease do one of the following:\n  * Build without the cffi feature (e.g., avoid enabling python_bindings/cffi on Windows)\n  * Build in WSL or on Linux where cffi/curl tend to build more reliably\n  * Install prerequisites: Visual Studio Build Tools + vcpkg and make sure VCPKG_ROOT and VCINSTALLDIR are set, then install curl via vcpkg (e.g., vcpkg install curl:x64-windows-static-md).\n\nSee docs/windows_full_build.md in the repository for detailed instructions and troubleshooting tips."
                );
            }
        }

        println!(
            "cargo:warning=Building with 'cffi' on MSVC can still fail due to a known issue with cffi-impl/ctor causing unresolved linker symbols; consider using WSL or disabling 'cffi' if you hit linker errors."
        );
    }
}

