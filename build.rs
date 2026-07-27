use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    if let Some(cuda_dir) = detect_cuda_toolkit() {
        println!(
            "cargo:rustc-env=TENSOR_ENGINE_CUDA_HOME={}",
            cuda_dir.display()
        );
        println!("cargo:metadata=CUDA_HOME={}", cuda_dir.display());
    } else if env::var_os("CARGO_FEATURE_BACKEND_CUDA").is_some() {
        println!(
            "cargo:warning=CUDA backend requested, but no toolkit containing include/cuda.h was found"
        );
    }

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
                                    println!("cargo:metadata=OPENBLAS_SOURCE={}", name);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(dir) = chosen_dir {
            if cfg!(target_os = "macos") {
                println!("cargo:rustc-link-search=native=/usr/local/opt/openblas/lib");
                println!("cargo:rustc-link-lib=openblas");
            } else {
                // Prefer 'lib' folder, fall back to 'lib64' or the directory itself if needed.
                let mut lib_dir = Path::new(&dir).join("lib");
                if !lib_dir.exists() {
                    lib_dir = Path::new(&dir).join("lib64");
                }
                if !lib_dir.exists() {
                    lib_dir = Path::new(&dir).to_path_buf();
                }
                println!("cargo:rustc-link-search=native={}", lib_dir.display());

                // On Windows/MSVC, check if we have openblas.lib or libopenblas.lib
                if cfg!(target_os = "windows") {
                    if lib_dir.join("openblas.lib").exists() {
                        println!("cargo:metadata=OPENBLAS_LIBRARY=openblas.lib");
                        println!("cargo:rustc-link-lib=openblas");
                    } else if lib_dir.join("libopenblas.lib").exists() {
                        println!("cargo:metadata=OPENBLAS_LIBRARY=libopenblas.lib");
                        println!("cargo:rustc-link-lib=libopenblas");
                    } else {
                        // Fallback
                        println!("cargo:rustc-link-lib=openblas");
                    }

                    // Copy the OpenBLAS runtime DLL to the output directory so the binary runs without PATH hacks.
                    let bin_dir = Path::new(&dir).join("bin");
                    let dll_name = if cfg!(target_os = "windows") {
                        "libopenblas.dll"
                    } else {
                        "libopenblas.so"
                    };
                    let dll_src = bin_dir.join(dll_name);
                    if dll_src.exists() {
                        if let Ok(out_dir) = env::var("OUT_DIR") {
                            // OUT_DIR = target/<profile>/build/<crate>-<hash>/out
                            let target_profile = Path::new(&out_dir)
                                .parent()
                                .and_then(|p| p.parent())
                                .and_then(|p| p.parent());
                            if let Some(profile_dir) = target_profile {
                                let dst = profile_dir.join(dll_name);
                                if !dst.exists() {
                                    match std::fs::copy(&dll_src, &dst) {
                                        Ok(_) => println!(
                                            "cargo:metadata=OPENBLAS_RUNTIME={}",
                                            profile_dir.join(dll_name).display()
                                        ),
                                        Err(e) => println!(
                                            "cargo:warning=Failed to copy {}: {}",
                                            dll_name, e
                                        ),
                                    }
                                }
                            }
                        }
                    } else {
                        println!(
                            "cargo:warning={} not found in {}",
                            dll_name,
                            bin_dir.display()
                        );
                    }
                } else {
                    println!("cargo:rustc-link-lib=openblas");
                }
            }
        } else {
            // No OPENBLAS_DIR and no bundled OpenBLAS detected
            println!("cargo:warning=Feature 'openblas' enabled but OPENBLAS_DIR not set and no bundled OpenBLAS directory found. Ensure OpenBLAS is available on your system.");
            if cfg!(target_os = "macos") {
                println!("cargo:rustc-link-search=native=/usr/local/opt/openblas/lib");
            }
            println!("cargo:rustc-link-lib=openblas");
        }
    }

    // If OpenBLAS feature is not enabled, compile a small cblas stub into the crate so imports don't fail at runtime.
    if env::var("CARGO_FEATURE_OPENBLAS").is_err() {
        println!("cargo:metadata=BLAS_BACKEND=pure-rust");
        // println!("cargo:rerun-if-changed=cblas_stub.c");
        // cc::Build::new()
        //     .file("scripts/cblas_stub.c")
        //     .compile("cblas_stub");
    }
}

fn detect_cuda_toolkit() -> Option<PathBuf> {
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(path) = env::var_os(variable).map(PathBuf::from) {
            if valid_cuda_toolkit(&path) {
                return Some(path);
            }
        }
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        let program_files = env::var_os("ProgramFiles")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Program Files"));
        let cuda_root = program_files
            .join("NVIDIA GPU Computing Toolkit")
            .join("CUDA");
        return newest_cuda_toolkit(&cuda_root);
    }

    ["/usr/local/cuda", "/opt/cuda"]
        .into_iter()
        .map(PathBuf::from)
        .find(|path| valid_cuda_toolkit(path))
}

fn newest_cuda_toolkit(root: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| valid_cuda_toolkit(path))
        .collect();
    candidates.sort_by(|left, right| {
        cuda_version_key(right)
            .cmp(&cuda_version_key(left))
            .then_with(|| right.cmp(left))
    });
    candidates.into_iter().next()
}

fn valid_cuda_toolkit(path: &Path) -> bool {
    if !path.join("include").join("cuda.h").is_file() {
        return false;
    }
    if cfg!(target_os = "windows") {
        path.join("lib").join("x64").join("cudart.lib").is_file()
    } else {
        path.join("lib64").is_dir() || path.join("lib").is_dir()
    }
}

fn cuda_version_key(path: &Path) -> Vec<u32> {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .trim_start_matches(['v', 'V'])
        .split('.')
        .map(|part| part.parse::<u32>().unwrap_or(0))
        .collect()
}
