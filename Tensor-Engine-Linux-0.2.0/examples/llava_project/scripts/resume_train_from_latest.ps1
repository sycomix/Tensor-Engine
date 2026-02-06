param(
    [string]$Dir = 'examples/models',
    [string]$Ext = '.ckpt.safetensors',
    [switch]$PreferPartial,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]] $ExtraArgs
)

Write-Host "Finding latest checkpoint in $Dir (ext $Ext)"

if ($PreferPartial) {
    $ckpt = python ./scripts/find_latest_checkpoint.py --dir $Dir --ext $Ext --prefer-partial
} else {
    $ckpt = python ./scripts/find_latest_checkpoint.py --dir $Dir --ext $Ext
}
if ([string]::IsNullOrWhiteSpace($ckpt)) {
    Write-Error "No checkpoint found in $Dir with ext $Ext"
    exit 1
}

Write-Host "Resuming training using checkpoint: $ckpt"
python -m examples.train_llava --resume --checkpoint $ckpt @ExtraArgs
