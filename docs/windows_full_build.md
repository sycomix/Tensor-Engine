# Building the project with all features on Windows (MSVC)

This document explains how to build Tensor-Engine with optional features that require C libraries and Python bindings on Windows. Building everything on Windows (for example via `cargo build --all-features`) requires additional steps which, if missing, can produce opaque linker errors (e.g., unresolved symbols related to cffi-impl/ctor or C build errors when compiling vendored curl).

Quick checklist

- Install Visual Studio Build Tools (Desktop development with C++) and the VC++ toolset. Ensure `VCINSTALLDIR` is set (open a "Developer Command Prompt for VS" or follow scripts in the repo).
- Install vcpkg and make sure `VCPKG_ROOT` environment variable is set. Use it to install curl (example:
  `vcpkg install curl:x64-windows-static-md`).
- For OpenBLAS: run `.	ools\scripts\setup_dev_repo.ps1` (or `scripts/setup_dev_repo.ps1` in repo root) and/or set `OPENBLAS_DIR` to a local OpenBLAS bundle.
- Install Python 3.11 and `maturin` (if building Python bindings): `python -m pip install maturin`.

Known issues and workarounds

- cffi-impl / ctor unresolved symbol on MSVC:
  The `cffi-impl` crate uses the `ctor` crate for static constructors and there are known cases where linking fails under MSVC producing an unresolved symbol like:

  `error LNK2001: unresolved external symbol _ZN9cffi_impl23init___rust_ctor___ctor...`

  Workarounds:
  - Build on Linux (or WSL) instead of Windows/MSVC (recommended for full-feature builds).
  - Avoid enabling the `cffi` / `python_bindings` features on MSVC if you don't need them.
  - Ensure vcpkg+curl and Visual Studio Build Tools are installed and properly configured (see the Quick checklist above).

- Vendored C libraries (libcurl) failing to build with `cl.exe`:
  This usually means vcpkg is not available or not configured (the build script attempts to find and use vcpkg-first). Installing vcpkg and the curl package, or using a prebuilt curl, will fix the issue.

Recommendations

- If you only need the core Rust functionality and tests: run `cargo test --workspace` (avoid `--all-features` unless you have the prerequisites).
- For Python bindings and examples that require the Python wheel: build on Linux or in WSL for the smoothest experience (maturin + system dependencies are easier there).

If you still hit a Windows linker failure related to `cffi-impl` or `ctor`, please open an issue and include the full build log. We will track upstream fixes and update this doc with the recommended resolution.
