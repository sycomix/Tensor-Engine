param(
    [switch]$DebugBuild = $false
)

$ReleaseFlag = if (-not $DebugBuild) { "--release" } else { "" }

Write-Host "Building Tensor Engine for Windows (all features)..." -ForegroundColor Green
cargo build $ReleaseFlag --all-features

if ($global:lastExitCode -eq 0) {
    Write-Host "`nBuild successful" -ForegroundColor Green
} else {
    Write-Host "`nBuild failed" -ForegroundColor Red
    exit 1
}
