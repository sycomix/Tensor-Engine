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
        // On Linux, prefer the system BLAS library by default.
        // The bundled directory may be Windows-only (.a with mingw imports).
        let use_bundled_dir = match env::var("OPENBLAS_DIR") {
            Ok(val) => Some(PathBuf::from(&val)),
            Err(_) => find_bundled_openblas(),
        };

        if let Some(dir) = use_bundled_dir {
            if cfg!(target_os = "macos") {
                println!("cargo:rustc-link-search=native=/usr/local/opt/openblas/lib");
                println!("cargo:rustc-link-lib=openblas");
            } else {
                // For Linux/Windows, check if the bundled dir has a usable shared library.
                let mut lib_dir = Path::new(&dir).join("lib");
                if !lib_dir.exists() {
                    lib_dir = Path::new(&dir).join("lib64");
                }
                if !lib_dir.exists() {
                    lib_dir = Path::new(&dir).to_path_buf();
                }

                // Accept platform-appropriate shared libraries.
                // On Linux, only accept `.so` (not `.lib` or `.dll` from Windows).
                let has_shared_lib = {
                    let entries: Vec<_> = std::fs::read_dir(&lib_dir)
                        .ok()
                        .into_iter()
                        .flatten()
                        .filter_map(|r| r.ok())
                        .collect();
                    entries.iter().any(|e| {
                        let fname = e.file_name();
                        let n = fname.to_string_lossy();
                        #[cfg(target_os = "linux")]
                        {
                            n.ends_with(".so")
                        }
                        #[cfg(target_os = "windows")]
                        {
                            n.ends_with(".dll") || n.ends_with(".lib")
                        }
                    })
                };

                if has_shared_lib {
                    println!("cargo:rustc-link-search=native={}", lib_dir.display());
                    println!("cargo:rustc-link-lib=openblas");
                } else {
                    println!(
                        "cargo:warning=Bundled OpenBLAS at {} has no shared library; using system BLAS",
                        dir.display()
                    );
                    // Link against the system BLAS. On Linux, Ubuntu packages it as libblas.so.3 in /usr/lib/x86_64-linux-gnu/blas/.
                    if cfg!(target_os = "linux") {
                        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/blas");
                        // Use raw linker arg for exact filename since Ubuntu provides no unversioned symlink.
                        println!("cargo:rustc-link-arg=-L/usr/lib/x86_64-linux-gnu/blas");
                        println!("cargo:rustc-link-arg=-l:libblas.so.3");
                    } else {
                        println!("cargo:rustc-link-lib=blas");
                    }
                }

                // Copy runtime DLL for Windows builds.
                if cfg!(target_os = "windows") {
                    let bin_dir = Path::new(&dir).join("bin");
                    let dll_name = "libopenblas.dll";
                    let dll_src = bin_dir.join(dll_name);
                    if dll_src.exists() {
                        if let Ok(out_dir) = env::var("OUT_DIR") {
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
                    }
                }
            }
        } else {
            // No bundled OpenBLAS — link against system BLAS.
            if cfg!(target_os = "macos") {
                println!("cargo:rustc-link-search=native=/usr/local/opt/openblas/lib");
                println!("cargo:rustc-link-lib=openblas");
            } else {
                println!("cargo:rustc-link-lib=blas");
            }
        }
    }

    // If OpenBLAS feature is not enabled, compile a small cblas stub into the crate so imports don't fail at runtime.
    if env::var("CARGO_FEATURE_OPENBLAS").is_err() {
        println!("cargo:metadata=BLAS_BACKEND=pure-rust");
    }
}

fn find_bundled_openblas() -> Option<PathBuf> {
    let manifest = env::var("CARGO_MANIFEST_DIR").ok()?;
    let entries_iter = std::fs::read_dir(&manifest).ok()?;
    for entry in entries_iter.filter_map(|e| e.ok()) {
        if let Ok(ft) = entry.file_type() {
            if ft.is_dir() {
                let fname = entry.file_name();
                let name = fname.to_string_lossy();
                if name.starts_with("OpenBLAS") {
                    println!("cargo:metadata=OPENBLAS_SOURCE={}", name);
                    return Some(entry.path());
                }
            }
        }
    }
    None
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
