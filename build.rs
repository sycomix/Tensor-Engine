use std::env;
use std::path::Path;

fn main() {
    // If `openblas` feature is enabled (Cargo sets CARGO_FEATURE_<FEATURE>), link with local OpenBLAS if provided.
    if env::var("CARGO_FEATURE_OPENBLAS").is_ok() {
        // Prefer an explicit OPENBLAS_DIR environment variable when provided.
        let mut chosen_dir: Option<String> = env::var("OPENBLAS_DIR").ok();

        // If not set, try to detect a bundled OpenBLAS directory inside the repository (e.g., OpenBLAS-*)
        if chosen_dir.is_none() {
            if let Ok(manifest) = env::var("CARGO_MANIFEST_DIR") {
                if let Ok(entries) = std::fs::read_dir(&manifest) {
                    for entry in entries.flatten() {
                        if let Ok(file_type) = entry.file_type() {
                            if file_type.is_dir() {
                                let name = entry.file_name().to_string_lossy().into_owned();
                                if name.starts_with("OpenBLAS") {
                                    chosen_dir = Some(entry.path().to_string_lossy().into_owned());
                                    println!("cargo:warning=Using bundled OpenBLAS directory '{}'", name);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(dir) = chosen_dir {
            // Prefer 'lib' folder, fall back to 'lib64' or the directory itself if needed.
            let mut lib_dir = Path::new(&dir).join("lib");
            if !lib_dir.exists() {
                lib_dir = Path::new(&dir).join("lib64");
            }
            if !lib_dir.exists() {
                lib_dir = Path::new(&dir).to_path_buf();
            }
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-lib=openblas");
            if cfg!(target_os = "windows") {
                println!("cargo:rustc-link-lib=libopenblas");
            }
        } else {
            // No OPENBLAS_DIR and no bundled OpenBLAS detected
            println!("cargo:warning=Feature 'openblas' enabled but OPENBLAS_DIR not set and no bundled OpenBLAS directory found. Ensure OpenBLAS is available on your system.");
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
                // Emit a strong warning but continue instead of failing the build by default.
                println!("cargo:warning=Building with 'cffi' on Windows/MSVC but vcpkg/Visual Studio not detected. This may result in linker errors; see docs/windows_full_build.md for troubleshooting. Continuing build anyway.");
            }
        }

        println!(
            "cargo:warning=Building with 'cffi' on MSVC can still fail due to a known issue with cffi-impl/ctor causing unresolved linker symbols; consider using WSL or disabling 'cffi' if you hit linker errors."
        );
    }
}

