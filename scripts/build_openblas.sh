#!/usr/bin/env bash
set -euo pipefail

# Build OpenBLAS locally into a directory (no sudo required by default).
# Usage:
#   scripts/build_openblas.sh --install-dir /path/to/openblas-install --jobs 8
# After successful build, set OPENBLAS_DIR to the install dir before building the wheel:
#   export OPENBLAS_DIR=/path/to/openblas-install
#   maturin develop --release --bindings pyo3 --features "python_bindings openblas"

INSTALL_DIR="$(pwd)/openblas-install"
JOBS=$(nproc || echo 1)
SRC_DIR="$(pwd)/openblas-src"
REPO_URL="https://github.com/xianyi/OpenBLAS.git"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --src-dir)
      SRC_DIR="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--install-dir DIR] [--src-dir DIR] [--jobs N]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "Building OpenBLAS into: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required to clone OpenBLAS. Install git and retry." >&2
  exit 1
fi
if ! command -v make >/dev/null 2>&1; then
  echo "Error: make is required to build OpenBLAS. Install build-essential/gmake and retry." >&2
  exit 1
fi
if ! command -v gfortran >/dev/null 2>&1; then
  echo "Warning: gfortran not found. OpenBLAS may require Fortran for some targets. Install gfortran for best results." >&2
fi

# Clone or update source
if [[ -d "$SRC_DIR/.git" ]]; then
  echo "Updating existing OpenBLAS source in $SRC_DIR"
  git -C "$SRC_DIR" fetch --depth=1 origin
  git -C "$SRC_DIR" reset --hard origin/master
else
  echo "Cloning OpenBLAS into $SRC_DIR"
  git clone --depth 1 "$REPO_URL" "$SRC_DIR"
fi

pushd "$SRC_DIR" >/dev/null

# Clean any previous build artifacts
make clean || true

# Build: dynamic/shared lib enabled, simple portable config
echo "Running make DYNAMIC_ARCH=1 NO_SHARED=0 USE_OPENMP=0 -j$JOBS"
make DYNAMIC_ARCH=1 NO_SHARED=0 USE_OPENMP=0 -j"$JOBS"

# Install into the prefix (no sudo)
echo "Installing into $INSTALL_DIR"
make PREFIX="$INSTALL_DIR" install

popd >/dev/null

cat <<'MSG'
OpenBLAS build complete.
Next steps (example):
  export OPENBLAS_DIR="$(pwd)/openblas-install"
  maturin develop --release --bindings pyo3 --features "python_bindings openblas"

If you prefer a system install, run "sudo make install" from the OpenBLAS source instead of setting PREFIX.
MSG

exit 0
