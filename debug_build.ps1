
$env:OPENBLAS_DIR="I:\Tensor-Engine\OpenBLAS-0.3.30-x64-64"
$cmd = "cargo test --features async_ops -vv"
Invoke-Expression $cmd | Select-String "cargo:"
