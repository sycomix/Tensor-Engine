param(
    [switch]$DebugBuild = $false
)

$ReleaseFlag = if (-not $DebugBuild) { "--release" } else { "" }
$VcVars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path -LiteralPath $VcVars)) {
    Write-Host "Visual Studio 2022 C++ environment not found: $VcVars" -ForegroundColor Red
    exit 1
}

Write-Host "Building Tensor Engine for Windows..." -ForegroundColor Green
cmd.exe /d /c "call `"$VcVars`" >nul && cargo build $ReleaseFlag"

if ($global:lastExitCode -eq 0) {
    Write-Host "`nBuild successful" -ForegroundColor Green
} else {
    Write-Host "`nBuild failed" -ForegroundColor Red
    exit 1
}
