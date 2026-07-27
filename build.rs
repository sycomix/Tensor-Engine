use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

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
                                    println!(
                                        "cargo:warning=Using bundled OpenBLAS directory '{}'",
                                        name
                                    );
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
                        println!("cargo:warning=Linking against openblas.lib");
                        println!("cargo:rustc-link-lib=openblas");
                    } else if lib_dir.join("libopenblas.lib").exists() {
                        println!("cargo:warning=Linking against libopenblas.lib");
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
                                            "cargo:warning=Copied {} to {}",
                                            dll_name,
                                            profile_dir.display()
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
        println!("cargo:warning=OpenBLAS feature not enabled; using pure Rust compat_blas fallback instead of C stub to avoid MSVC stack overflow.");
        // println!("cargo:rerun-if-changed=cblas_stub.c");
        // cc::Build::new()
        //     .file("scripts/cblas_stub.c")
        //     .compile("cblas_stub");
    }

    // If the optional `cffi` feature is enabled on Windows/MSVC, fail early with a helpful message
    let cffi_enabled = env::var_os("CARGO_FEATURE_CFFI").is_some();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if cffi_enabled && target_os == "windows" && target_env == "msvc" {
        let has_vcpkg = env::var_os("VCPKG_ROOT")
            .map(|root| PathBuf::from(root).exists())
            .unwrap_or(false);
        let has_vs = detect_visual_studio_installation();

        if !has_vcpkg {
            // Allow skipping this protective guard via environment variable for experiments.
            if env::var_os("SKIP_CFFI_GUARD").is_some() {
                println!("cargo:warning=SKIP_CFFI_GUARD set: continuing despite missing vcpkg (experimental override)");
            } else {
                // Emit a strong warning but continue instead of failing the build by default.
                println!("cargo:warning=Building with 'cffi' on Windows/MSVC but VCPKG_ROOT does not point to an installed vcpkg directory. This may result in linker errors for native dependencies; see docs/windows_full_build.md for troubleshooting. Continuing build anyway.");
            }
        }

        if !has_vs {
            if env::var_os("SKIP_CFFI_GUARD").is_some() {
                println!("cargo:warning=SKIP_CFFI_GUARD set: continuing despite missing Visual Studio detection (experimental override)");
            } else {
                println!("cargo:warning=Building with 'cffi' on Windows/MSVC but Visual Studio Build Tools were not detected. Set VCINSTALLDIR, VSINSTALLDIR, VisualStudioVersion, or TENSOR_ENGINE_VSINSTALLDIR if Visual Studio is installed in a custom location. Continuing build anyway.");
            }
        }

        println!(
            "cargo:warning=Building with 'cffi' on MSVC can still fail due to a known issue with cffi-impl/ctor causing unresolved linker symbols; consider using WSL or disabling 'cffi' if you hit linker errors."
        );
    }
}

fn detect_visual_studio_installation() -> bool {
    if env_path_exists("VCINSTALLDIR")
        || env_path_exists("VSINSTALLDIR")
        || env_path_exists("TENSOR_ENGINE_VSINSTALLDIR")
        || env::var_os("VisualStudioVersion").is_some()
    {
        return true;
    }

    if let Ok(linker) = env::var("CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER") {
        if linker.contains("Microsoft Visual Studio") && Path::new(&linker).exists() {
            return true;
        }
    }

    if vswhere_detects_visual_studio() {
        return true;
    }

    visual_studio_known_paths()
        .into_iter()
        .any(|path| path.exists())
}

fn env_path_exists(key: &str) -> bool {
    env::var_os(key)
        .map(PathBuf::from)
        .map(|path| path.exists())
        .unwrap_or(false)
}

fn vswhere_detects_visual_studio() -> bool {
    let program_files_x86 = env::var_os("ProgramFiles(x86)")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files (x86)"));
    let vswhere = program_files_x86
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    if !vswhere.exists() {
        return false;
    }

    Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .map(|output| output.status.success() && !output.stdout.is_empty())
        .unwrap_or(false)
}

fn visual_studio_known_paths() -> Vec<PathBuf> {
    let program_files = env::var_os("ProgramFiles")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files"));
    let program_files_x86 = env::var_os("ProgramFiles(x86)")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files (x86)"));
    let roots = [
        program_files.join("Microsoft Visual Studio"),
        program_files_x86.join("Microsoft Visual Studio"),
    ];
    let versions = ["18", "17", "16", "15", "2022", "2019", "2017"];
    let editions = [
        "Insiders",
        "Enterprise",
        "Professional",
        "Community",
        "BuildTools",
    ];

    let mut paths = Vec::new();
    for root in roots {
        for version in versions {
            for edition in editions {
                paths.push(
                    root.join(version)
                        .join(edition)
                        .join("VC")
                        .join("Tools")
                        .join("MSVC"),
                );
            }
        }
    }
    paths
}
